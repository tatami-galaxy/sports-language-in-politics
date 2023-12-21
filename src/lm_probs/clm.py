import argparse
import os
import json
import re
import gdown
import polars as pl
from tqdm.auto import tqdm
from itertools import chain
import math

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import Dataset
from transformers import(
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)


def get_perplexity(data: list, model):

    encodings = tokenizer("\n\n".join(data), return_tensors="pt")
    max_length = model.config.n_positions
    stride = 256
    seq_len = encodings.input_ids.size(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).mean())

    return ppl


def get_sports_probs(model, tokenizer, vocab, dataloader):
    stripped_vocab = {}
    sports2ids = {}
    # convert tokens to strings
    for token, id in tokenizer.get_vocab().items():
        stripped_vocab[tokenizer.convert_tokens_to_string([token]).strip()] = id
    # tokens to ids
    for key, val in vocab.items():
        if key in stripped_vocab:
            sports2ids[key] = stripped_vocab[key] # id
    # invert dict
    ids2sports = {v: k for k, v in sports2ids.items()}

    # for each sample get probabilities of sports tokens and sum
    sports2prob = {key: [] for key, _ in sports2ids.items()}
    eval_bar = tqdm(range(len(eval_dataloader)), position=0,
                    disable=not accelerator.is_local_main_process)
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            logits = model(**batch).logits
            probs = torch.nn.functional.softmax(logits, dim=2)
            for id, token in ids2sports.items():
                num = probs[:, :, id].shape[0] * probs[:, :, id].shape[1]
                sports2prob[token].append(torch.sum(probs[:, :, id])/num)
        eval_bar.update(1)

    sport_probs = {}
    for key, val in sports2prob.items():
        sport_probs[key] = sum(val).detach().cpu()

    return sport_probs


def train(accelerator, model, optimizer, train_dataloader, eval_dataloader, lr_scheduler):

    completed_steps = 0
    total_loss = 0

    progress_bar = tqdm(range(args.train_steps), position=0,
                        disable=not accelerator.is_local_main_process)
    eval_bar = tqdm(range(len(eval_dataloader)), position=1,
                    disable=not accelerator.is_local_main_process)

    while True:
        model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps % args.eval_steps == 0:
                model.eval()
                losses = []
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**batch)

                    loss = outputs.loss
                    losses.append(accelerator.gather_for_metrics(
                        loss.repeat(args.per_device_eval_batch_size)))

                    eval_bar.update(1)

                eval_bar.refresh()
                eval_bar.reset()

                losses = torch.cat(losses)
                try:
                    eval_loss = torch.mean(losses)
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    perplexity = float("inf")

                accelerator.print(
                    f"step {completed_steps}: perplexity: {perplexity} eval_loss: {eval_loss} train_loss: {total_loss.item()/len(train_dataloader)}")

                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir+'gpt2_data_'+str(completed_steps), is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                total_loss = 0
                model.train()

            if completed_steps >= args.train_steps:
                accelerator.print("max train steps reached")
                return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--cloud",
        action="store_true",
    )
    parser.add_argument(
        "--perplexity",
        action="store_true",
    )
    parser.add_argument(
        "--sample_size",
        default=50000,
        type=int,
    )
    parser.add_argument(
        "--output_dir",
        default='/users/ujan/sports-language-in-politics/models/',
        type=str,
    )
    parser.add_argument(
        "--data",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--data_dir",
        default='/users/ujan/sports-language-in-politics/data/processed/',
        type=str,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--model_name_or_path",
        default='gpt2',
        type=str,
    )
    parser.add_argument(
        "--text_column_name",
        default='text',
        type=str,
    )
    parser.add_argument(
        "--min_comment_length",
        default=150,
        type=int,
    )
    parser.add_argument(
        "--num_workers",
        default=os.cpu_count(),
        type=int,
    )
    parser.add_argument(
        "--block_size",
        default=256,
        type=int,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default='linear',
        type=str,
    )
    parser.add_argument(
        "--num_warmup_steps",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--train_steps",
        default=5000,
        type=int,
    )
    parser.add_argument(
        "--eval_steps",
        default=1000,
        type=int,
    )

    # parse args
    args = parser.parse_args()

    # check if data directory is None
    if args.data is None:
        raise ValueError(
            f"pass in data_dir"
        )
    if args.data not in ['politics', 'random']:
        raise ValueError(
            f"data must be `random` or `politics`"
        )
    # check if output directory is None
    if args.output_dir is None:
        raise ValueError(
            f"pass in output_dir"
        )
    elif args.output_dir[-1] != '/':
        args.output_dir = args.output_dir+'/'
    # check if data directory is None
    if args.data_dir is None:
        raise ValueError(
            f"pass in data_dir"
        )
    elif args.data_dir[-1] != '/':
        args.data_dir = args.data_dir+'/'
    # which data to use
    if args.data is None:
        raise ValueError(
            f"pass in data: [politics, random]"
        )
    
    # load data
    if args.cloud:
        # download political and random comments
        if args.data == 'politics':
            gdown.download(id="1EVu3LrPIsHTrJhl8oICvxO8CxoeYbSbo",
                        output='politics_sample.csv', quiet=False)
        #elif args.data == 'random':
            #gdown.download(id="1h3ZUGnbjdORQqonq6eOlLxp9MIg36XcN",
                        #output='random_sample.csv', quiet=False)
        elif args.data == 'random':
            gdown.download(id="14aRNS6weyZJKHwiJ94d0j6RWWmkaYCuG",
                           output='random_sample_no_sports.csv', quiet=False)
        # download sports vocab
        #gdown.download(id="15Yc36d4Bbf_Jr3ICPbR5uxOvB79ggklr",
                       #output='sports_vocab.json', quiet=False)
        gdown.download(id="1tFFzv_M_GwdM5hRBm1vd8NJi1yIWYe4C",
                       output='sports_vocab_now.json', quiet=False)
        
        # load political and random comments
        if args.data == 'politics':
            data_df = pl.read_csv('politics_sample.csv').drop_nulls()
        #elif args.data == 'random':
            #data_df = pl.read_csv('random_sample.csv')  # dont drop nulls
        elif args.data == 'random':
            data_df = pl.read_csv('random_sample_no_sports.csv')  # dont drop nulls
        # load sports vocab
        #with open('sports_vocab.json', 'r') as fp:
            #sports_vocab = json.load(fp)
        with open('sports_vocab_now.json', 'r') as fp:
            sports_vocab = json.load(fp)
    else:
        # load political and random comments
        if args.data == 'politics':
            data_df = pl.read_csv(args.data_dir+'politics_sample.csv').drop_nulls()
        #elif args.data == 'random':
            #data_df = pl.read_csv(args.data_dir+'random_sample.csv')  # dont drop nulls
        elif args.data == 'random':
            data_df = pl.read_csv(
                args.data_dir+'random_sample_no_sports.csv')  # dont drop nulls
        # load sports vocab
        #with open(args.data_dir+'sports_vocab.json', 'r') as fp:
            #sports_vocab = json.load(fp)
        with open(args.data_dir+'sports_vocab_now.json', 'r') as fp:
            sports_vocab = json.load(fp)

    # get comments
    comments = [re.sub(r"[^a-zA-Z0-9]+", ' ', comment).lower() for comment in data_df['body'].to_list()]
    # filter
    comments_long = []
    lens = [len(c.split()) for c in comments]
    for i in range(len(comments)):
        if lens[i] >= args.min_comment_length:
            comments_long.append(comments[i])
    # upto sample size
    # shuffle?
    comments = comments_long[:args.sample_size]

    # create dataset
    data = [{args.text_column_name: t} for t in comments]
    dataset = Dataset.from_list(data)

    # get accelerator
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    # model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # eval perplexity before training
    if args.perplexity:
        ppl = get_perplexity(comments, model)
        print('perplexity : {}'.format(ppl))

    # preprocess dataset
    def tokenize_function(examples):
        return tokenizer(examples[args.text_column_name])
        
    column_names = dataset.column_names
    with accelerator.main_process_first():
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=args.num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )
    # main data processing function that will concatenate all texts from the dataset and generate chunks of block_size
    def group_texts(examples):
        # concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # we drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # we could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // args.block_size) * args.block_size
        # split by chunks of max_len.
        result = {
            k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with accelerator.main_process_first():
        dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=args.num_workers,
            desc=f"Grouping texts in chunks of {args.block_size}",
        )

    # split into train eval
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    # dataloaders
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )

    # optimizer
    # split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.train_steps * args.gradient_accumulation_steps,
    )

    # prepare with accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # probabilites for maps?
    # evaluate sports probabilities prior to training
    sports_probs_before = get_sports_probs(model, tokenizer, sports_vocab, eval_dataloader)
    avg_prob = sum(sports_probs_before.values())/len(sports_probs_before)
    # average sports token probability before training: 0.164 [politics]
    # average sports token probability before training: 0.214 [random]

    # average sports token probability before training (now): 0.180 [politics]
    # average sports token probability before training (now): 0.221 [random]
    accelerator.print('average sports token probability before training: {}'.format(avg_prob))

    # train
    accelerator.print('training')
    train(accelerator, model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)
    accelerator.print('training done')

    # eval perplexity after training
    if args.perplexity:
        ppl = get_perplexity(comments, model)
        print('perplexity : {}'.format(ppl))

    # evaluate sports probabilities after training
    sports_probs_after = get_sports_probs(model, tokenizer, sports_vocab, eval_dataloader)
    avg_prob = sum(sports_probs_after.values())/len(sports_probs_after)
    # average sports token probability after training : 0.163 [politics]
    # average sports token probability after training: 0.194 [random]

    # average sports token probability after training (now): 0.188  [politics]
    # average sports token probability after training (now): 0.2033 [random]
    accelerator.print('average sports token probability after training : {}'.format(avg_prob))

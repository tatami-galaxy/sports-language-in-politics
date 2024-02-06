import argparse
import os
from os.path import dirname, abspath
from functools import partial
import json
import yaml
import numpy as np
import gdown
import polars as pl
import re 

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import torch
import torch.nn as nn
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from datasets import Dataset

CBOW_N_WORDS = 5
#SKIPGRAM_N_WORDS = 4

MIN_WORD_FREQUENCY = 20
MAX_SEQUENCE_LENGTH = 256

EMBED_DIMENSION = 300
EMBED_MAX_NORM = 1


## check if in vocab ##
imp_tokens = [
    'biden', 'trump', 'coach', 'captain', 'politician', 'fan',
    'election', 'party', 'team', 'race', 'voter', 'quarterback',
    'democrats', 'republicans',
]

# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'sports-language-in-politics':
    root = dirname(root)


def yield_tokens(dataset, tokenizer):
    for text in dataset['text']:
        yield tokenizer(text)


def collate_cbow(batch, text_pipeline):
    """
    Collate_fn for CBOW model to be used with Dataloader.
    `batch` is expected to be list of text paragrahs.
    
    Context is represented as N=CBOW_N_WORDS past words 
    and N=CBOW_N_WORDS future words.
    
    Long paragraphs will be truncated to contain
    no more that MAX_SEQUENCE_LENGTH tokens.
    
    Each element in `batch_input` is N=CBOW_N_WORDS*2 context words.
    Each element in `batch_output` is a middle word.
    """
    batch_input, batch_output = [], []

    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < CBOW_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - CBOW_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx: (
                idx + CBOW_N_WORDS * 2 + 1)]
            output = token_id_sequence.pop(CBOW_N_WORDS)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


def get_dataloader_and_vocab(dataset, ds_type, batch_size, tokenizer, vocab):

    dataset = dataset[ds_type]

    if vocab is None:
        vocab = build_vocab_from_iterator(
            yield_tokens(dataset, tokenizer),
            specials=["<unk>"],
            min_freq=MIN_WORD_FREQUENCY
        )
        vocab.set_default_index(vocab["<unk>"])

    def text_pipeline(x): return vocab(tokenizer(x))

    dataloader = DataLoader(
        dataset['text'],
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_cbow, text_pipeline=text_pipeline),
    )

    return dataloader, vocab


def get_lr_scheduler(optimizer, total_epochs: int, verbose: bool = True):
    """
    Scheduler to linearly decrease learning rate, 
    so thatlearning rate after the last epoch is 0.
    """
    def lr_lambda(epoch): return (total_epochs - epoch) / total_epochs
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=verbose)
    return lr_scheduler


def save_config(config: dict, model_dir: str):
    """Save config file to `model_dir` directory"""
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "w") as stream:
        yaml.dump(config, stream)


def save_vocab(vocab, model_dir: str):
    """Save vocab file to `model_dir` directory"""
    vocab_path = os.path.join(model_dir, "vocab.pt")
    torch.save(vocab, vocab_path)


class CBOW_Model(nn.Module):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781
    """

    def __init__(self, vocab_size: int):
        super(CBOW_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, inputs):
        x = self.embeddings(inputs)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


class Trainer:
    """Main class for model training"""
    
    def __init__(
        self,
        model,
        epochs,
        train_dataloader,
        train_steps,
        eval_dataloader,
        eval_steps,
        checkpoint_frequency,
        criterion,
        optimizer,
        lr_scheduler,
        device,
        model_dir,
    ):  
        self.model = model
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.train_steps = train_steps
        self.eval_dataloader = eval_dataloader
        self.eval_steps = eval_steps
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_frequency = checkpoint_frequency
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_dir = model_dir

        self.loss = {"train": [], "val": []}
        self.model.to(self.device)

    def train(self):
        for epoch in range(self.epochs):
            self._train_epoch()
            self._validate_epoch()
            print(
                "Epoch: {}/{}, Train Loss={:.5f}, Val Loss={:.5f}".format(
                    epoch + 1,
                    self.epochs,
                    self.loss["train"][-1],
                    self.loss["val"][-1],
                )
            )

            self.lr_scheduler.step()

            if self.checkpoint_frequency:
                self._save_checkpoint(epoch)

    def _train_epoch(self):
        self.model.train()
        running_loss = []

        for i, batch_data in enumerate(self.train_dataloader, 1):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())

            if i == self.train_steps:
                break

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def _validate_epoch(self):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, batch_data in enumerate(self.eval_dataloader, 1):
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss.append(loss.item())

                if i == self.eval_steps:
                    break

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)

    def _save_checkpoint(self, epoch):
        """Save model checkpoint to `self.model_dir` directory"""
        epoch_num = epoch + 1
        if epoch_num % self.checkpoint_frequency == 0:
            model_path = "checkpoint_{}.pt".format(str(epoch_num).zfill(3))
            model_path = os.path.join(self.model_dir, model_path)
            torch.save(self.model, model_path)

    def save_model(self):
        """Save final model to `self.model_dir` directory"""
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model, model_path)

    def save_loss(self):
        """Save train/val loss as json file to `self.model_dir` directory"""
        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)


if __name__ == '__main__':

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
        "--stop_words",
        action="store_true",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
    )
    parser.add_argument(
        "--sample_size",
        default=5000,
        type=int,
    )
    parser.add_argument(
        "--data_dir",
        default=root+'/data/processed/',
        type=str,
    )
    parser.add_argument(
        "--min_comment_length",
        default=100,  # chars
        type=int,
    )
    parser.add_argument(
        "--train_batch_size",
        default=96,
        type=int,
    )
    parser.add_argument(
        "--eval_batch_size",
        default=96,
        type=int,
    )
    parser.add_argument(
        "--learning_rate",
        default=0.025,
        type=float,
    )
    parser.add_argument(
        "--epochs",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--train_steps",
        default=20000,
        type=int,
    )
    parser.add_argument(
        "--eval_steps",
        default=500,
        type=int,
    )
    parser.add_argument(
        "--model_dir",
        default=root+'/models/cbow/',
        type=str,
    )
    parser.add_argument(
        "--subs",
        default=['politics'],  # The_Donald
    )

    
    args = parser.parse_args()

    # seed
    import random
    random.seed(args.seed)
    from random import shuffle

    # check if data directory is None
    if args.data_dir is None:
        raise ValueError(
            f"pass in data_dir"
        )
    elif args.data_dir[-1] != '/':
        args.data_dir = args.data_dir+'/'

    # load data
    if args.cloud:
        # download political comments
        print('downloading political comments')
        gdown.download(
            id="1EVu3LrPIsHTrJhl8oICvxO8CxoeYbSbo",
            output=args.data_dir+'politics_sample.csv', quiet=False
        )
        # download sports comments
        print('downloading sports comments')
        gdown.download(
            id="1Xc6VXdG8cloh8tdxAaboQewkilgvWxub",
            output=args.data_dir+'sports_sample.csv', quiet=False
        )

        # load political and sports comments
        politics_df = pl.read_csv(args.data_dir+'politics_sample.csv').drop_nulls()
        sports_df = pl.read_csv(args.data_dir+'sports_sample.csv').drop_nulls()

    # local
    else:
        # load political and sport comments
        print('loading political and sports comments')
        politics_df = pl.read_csv(args.data_dir+'politics_sample.csv').drop_nulls()
        sports_df = pl.read_csv(args.data_dir+'sports_sample.csv').drop_nulls()

    # filter out subs
    data_df = politics_df.filter(pl.col('subreddit').is_in(args.subs))
    # shuffle dataframe
    data_df = data_df.sample(fraction=1.0, shuffle=True, seed=args.seed)

    # get sports sample of same length after shuffle
    sports_df = sports_df.sample(fraction=1.0, shuffle=True, seed=args.seed)[:len(data_df)]

    # concat dfs and shuffle again
    data_df = pl.concat([data_df, sports_df]).sample(fraction=1.0, shuffle=True, seed=args.seed)

    # filter comments
    print('filtering comments')
    comments = [comment.replace("'", '')
                for comment in data_df['body'].to_list()]
    comments = [re.sub(r"[^a-zA-Z0-9]+", ' ', comment).lower()
                for comment in comments]

    comments_long = []
    # filter by char
    for c in range(len(comments)):
        if len(comments[c]) >= args.min_comment_length:
            comments_long.append(comments[c])

    # sample comments
    if args.sample:
        print('sampling')
        comments = comments_long[:args.sample_size]
    else: comments = comments_long

    # remove stopwords 
    if args.stop_words:
        print('removing stop words')
        stop_words = set(stopwords.words("english"))
        new_comments = []
        for comment in comments:
            new_tokens = []
            tokens = comment.split()
            for token in tokens:
                if token not in stop_words:
                    new_tokens.append(token)
            new_comment = ' '.join(new_tokens)
            new_comments.append(new_comment)
        comments = new_comments

    # convert into dataset
    data_dict = {"text": comments}
    dataset = Dataset.from_dict(data_dict).train_test_split(test_size=0.1)

    tokenizer = get_tokenizer("basic_english", language="en")

    train_dataloader, vocab = get_dataloader_and_vocab(
        dataset=dataset,
        ds_type="train",
        batch_size=args.train_batch_size,
        tokenizer=tokenizer,
        vocab=None,
    )
    eval_dataloader, _ = get_dataloader_and_vocab(
        dataset=dataset,
        ds_type="test",
        batch_size=args.eval_batch_size,
        tokenizer=tokenizer,
        vocab=vocab,
    )

    vocab_size = len(vocab.get_stoi())
    print(f"Vocabulary size: {vocab_size}")

    # check imp tokens 
    for token in imp_tokens:
        if token not in vocab.get_itos():
            raise ValueError('imp token not in vocab')

    model = CBOW_Model(vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_lr_scheduler(optimizer, args.epochs, verbose=True)

    if torch.backends.mps.is_available(): device = "mps"
    elif torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"

    trainer = Trainer(
        model=model,
        epochs=args.epochs,
        train_dataloader=train_dataloader,
        train_steps=args.train_steps,
        eval_dataloader=eval_dataloader,
        eval_steps=args.eval_steps,
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_frequency=500,
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=args.model_dir,
    )

    trainer.train()
    print("Training finished.")

    trainer.save_model()
    trainer.save_loss()
    save_vocab(vocab, args.model_dir)
    print("Model artifacts saved to folder:", args.model_dir)
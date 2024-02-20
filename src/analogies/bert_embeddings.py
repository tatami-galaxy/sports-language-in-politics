import os
from os.path import dirname, abspath
import argparse

import gdown
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import polars as pl
import re
import json

import torch
from transformers import BertModel, AutoTokenizer
from datasets import Dataset

from tqdm.auto import tqdm

# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'sports-language-in-politics':
    root = dirname(root)


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
        default=16,
        type=int,
    )
    parser.add_argument(
        "--eval_batch_size",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--learning_rate",
        default=0.025,
        type=float,
    )
    parser.add_argument(
        "--epochs",
        default=100,
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
        default=['The_Donald'],  # ['The_Donald', 'Conservative'] -> 14M words
    )
    parser.add_argument(
        "--model_name",
        default='bert-base-uncased',
        type=str,
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
            id="1MFZLxtO7CnE24SzsQXfeoEolgXgeH-gK",
            output=args.data_dir+'politics_sample.csv', quiet=False
        )
        # download sports comments
        print('downloading sports comments')
        gdown.download(
            id="1V3J3g-zoPBMHKIJVKoCje3M04gsCJf55",
            output=args.data_dir+'sports_sample.csv', quiet=False
        )

        # load political and sports comments
        politics_df = pl.read_csv(
            args.data_dir+'politics_sample.csv').drop_nulls()
        sports_df = pl.read_csv(args.data_dir+'sports_sample.csv').drop_nulls()

    # local
    else:
        # load political and sport comments
        print('loading political and sports comments')
        politics_df = pl.read_csv(
            args.data_dir+'politics_sample.csv').drop_nulls()
        sports_df = pl.read_csv(args.data_dir+'sports_sample.csv').drop_nulls()

    # filter out subs
    data_df = politics_df.filter(pl.col('subreddit').is_in(args.subs))
    # shuffle dataframe
    data_df = data_df.sample(fraction=1.0, shuffle=True, seed=args.seed)

    # get sports sample of same length after shuffle
    sports_df = sports_df.sample(fraction=1.0, shuffle=True, seed=args.seed)[
        :len(data_df)]

    # concat dfs and shuffle again
    data_df = pl.concat([data_df, sports_df]).sample(
        fraction=1.0, shuffle=True, seed=args.seed)

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

    else:
        comments = comments_long

    print('total comments : {}'.format(len(comments)))
    word_count = 0
    for comment in comments:
        word_count += len(comment.split())
    print('total words : {}'.format(word_count))

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
    #data_dict = {"text": comments}
    #dataset = Dataset.from_dict(data_dict)  # .train_test_split(test_size=0.1)
        
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # model and tokenizer
    model = BertModel.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model.to(device)

    embedding_dict = {}

    bar = tqdm(range(len(comments)))
    for comment in comments:
        words = comment.split()
        # special tokens added
        tensor_inputs = tokenizer(comment, return_tensors="pt", truncation=True).to(device)
        inputs = tokenizer(comment, truncation=True, add_special_tokens=False)
        w_ids = inputs.word_ids()
        if tensor_inputs['input_ids'].shape[-1] >= 512:
            continue
        with torch.no_grad():
            outputs = model(**tensor_inputs)
            embeddings = outputs.last_hidden_state
            embeddings = embeddings.reshape(-1, 768)
            embeddings = embeddings[1:-1]
            for i in range(len(w_ids)):
                word = words[w_ids[i]]
                if word in embedding_dict:
                    embedding_dict[word]['embedding'] += embeddings[i].cpu().numpy()
                    embedding_dict[word]['count'] += 1
                else:
                    embedding_dict[word] = {'embedding':embeddings[i].cpu().numpy(), 'count':1}
        bar.update(1)

    new_embed_dict = {}
    for key, val in embedding_dict.items():
        embedding = val['embedding']/val['count']
        new_embed_dict[key] = embedding.tolist()
    # save
    with open(args.data_dir+'bert_embed.json', 'w') as f:
        json.dump(new_embed_dict, f)

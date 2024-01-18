import json
import polars as pl
import re
import math
from tqdm.auto import tqdm
import argparse
import gdown
from nltk.util import ngrams
from sentence_transformers import SentenceTransformer, util
import random
import torch

"""

Biden/Trump as captain/coach/quarterback/skipper : NER -> biden, trump
politicians as players : NER -> list of politicians, NER -> list of players?
elections as race/competition : 
parties as teams : republicans, democrats
voters/supporters as fans : [voters, supporters, ...]
media as spectators : [media, news_channel, ..], [cnn, fox, nytimes,0 ...]

"""


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
        "--sample",
        action="store_true",
    )
    parser.add_argument(
        "--sample_size",
        default=50000,
        type=int,
    )
    parser.add_argument(
        "--data_dir",
        default='/users/ujan/sports-language-in-politics/data/processed/',
        type=str,
    )
    parser.add_argument(
        "--min_comment_length",
        default=150,  # chars
        type=int,
    )
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
    )

    # parse args
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
        gdown.download(
            id="1EVu3LrPIsHTrJhl8oICvxO8CxoeYbSbo",
            output='politics_sample.csv', quiet=False
        )
        # load political comments
        print('loading political comments')
        data_df = pl.read_csv('politics_sample.csv').drop_nulls()


    # local
    else:
        # load political and random comments
        print('loading political comments')
        data_df = pl.read_csv(args.data_dir+'politics_sample.csv').drop_nulls()
        print('done')

    # filter comments and ids
    print('filtering comments')
    comments = [comment.replace("'", '') for comment in data_df['body'].to_list()]
    comments = [re.sub(r"[^a-zA-Z0-9]+", ' ', comment).lower()
                for comment in comments]
    ids = data_df['id'].to_list()

    comments_long = []
    ids_long = []
    # filter by char
    for c in range(len(comments)):
        if len(comments[c]) >= args.min_comment_length:
            comments_long.append(comments[c])
            ids_long.append(ids[c])
    print('done')

    # sample comments
    print('sampling')
    if args.sample:
        shuffle(comments_long)
        comments = comments_long[:args.sample_size]

    # only detect biden/trump for now

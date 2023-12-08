import argparse
import gdown
import json
import re
import polars as pl
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE


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
        "--sample_size",
        default=50000,
        type=int,
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
        "--min_comment_length",
        default=150,
        type=int,
    )
    parser.add_argument(
        "--n",
        default=3,
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
        elif args.data == 'random':
            gdown.download(id="1h3ZUGnbjdORQqonq6eOlLxp9MIg36XcN",
                        output='random_sample.csv', quiet=False)
        # download sports vocab
        gdown.download(id="15Yc36d4Bbf_Jr3ICPbR5uxOvB79ggklr",
                       output='sports_vocab.json', quiet=False)
        # load political and random comments
        if args.data == 'politics':
            data_df = pl.read_csv('politics_sample.csv').drop_nulls()
        elif args.data == 'random':
            data_df = pl.read_csv('random_sample.csv')  # dont drop nulls
        # load sports vocab
        with open('sports_vocab.json', 'r') as fp:
            sports_vocab = json.load(fp)
    else:
        # load political and random comments
        if args.data == 'politics':
            data_df = pl.read_csv(
                args.data_dir+'politics_sample.csv').drop_nulls()
        elif args.data == 'random':
            data_df = pl.read_csv(
                args.data_dir+'random_sample.csv')  # dont drop nulls
        # load sports vocab
        with open(args.data_dir+'sports_vocab.json', 'r') as fp:
            sports_vocab = json.load(fp)

    # get comments
    comments = [re.sub(r"[^a-zA-Z0-9]+", ' ', comment).lower()
                for comment in data_df['body'].to_list()]
    # filter
    comments_long = []
    lens = [len(c.split()) for c in comments]
    for i in range(len(comments)):
        if lens[i] >= args.min_comment_length:
            comments_long.append(comments[i])
    # upto sample size
    # shuffle?
    comments = comments_long[:args.sample_size]

    # tokenize
    tokenized_comments = [list(map(str.lower, word_tokenize(sent))) for sent in comments]

    # preprocess the tokenized text for n-gram language modelling
    train_data, padded_sents = padded_everygram_pipeline(args.n, tokenized_comments)

    # train n-gram model
    model = MLE(args.n)
    model.fit(train_data, padded_sents)

    # score sports vocab
    scores = 0
    for key, _ in sports_vocab.items():
        scores += model.score(key)

    print('total {}-gram score for {} data: {}'.format(args.n, args.data, scores))

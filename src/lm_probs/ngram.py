import argparse
import gdown
import json
import re
import polars as pl
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.util import bigrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from tqdm.auto import tqdm


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
        "--manual_vocab",
        action="store_true",
    )
    parser.add_argument(
        "--sample_size",
        default=50000,
        type=int,
    )
    parser.add_argument(
        "--data",
        default=None,  # ['politics', 'random']
        type=str,
    )
    parser.add_argument(
        "--sports_data",
        default=None,  # ['vocab', 'comments']
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

    if args.sports_data is None:
        raise ValueError(
            f"pass in data_dir"
        )
    if args.sports_data not in ['vocab', 'comments']:
        raise ValueError(
            f"sports_data must be `vocab` or `comments`"
        )
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
        #elif args.data == 'random':
            #gdown.download(id="1h3ZUGnbjdORQqonq6eOlLxp9MIg36XcN",
                        #output='random_sample.csv', quiet=False)
        elif args.data == 'random':
            gdown.download(id="14aRNS6weyZJKHwiJ94d0j6RWWmkaYCuG",
                           output='random_sample_no_sports.csv', quiet=False)
        if args.sports_data == 'vocab':
            # download sports vocab
            gdown.download(id="15Yc36d4Bbf_Jr3ICPbR5uxOvB79ggklr",
                        output='sports_vocab.json', quiet=False)
        else:
            # download sports comments
            gdown.download(id="1Xc6VXdG8cloh8tdxAaboQewkilgvWxub",
                        output='sports_sample.json', quiet=False)
        
        # load political and random comments
        if args.data == 'politics':
            data_df = pl.read_csv('politics_sample.csv').drop_nulls()
        #elif args.data == 'random':
            #data_df = pl.read_csv('random_sample.csv')  # dont drop nulls
        elif args.data == 'random':
            data_df = pl.read_csv(
                'random_sample_no_sports.csv')  # dont drop nulls
        # load sports vocab
        with open('sports_vocab.json', 'r') as fp:
            sports_vocab = json.load(fp)
        # load sports comments
            sports_df = pl.read_csv('sports_sample.csv').drop_nulls()
    else:
        # load political and random comments
        if args.data == 'politics':
            print('loading political comments')
            data_df = pl.read_csv(
                args.data_dir+'politics_sample.csv').drop_nulls()
            print('done')
        #elif args.data == 'random':
            #data_df = pl.read_csv(
                #args.data_dir+'random_sample.csv')  # dont drop nulls
        elif args.data == 'random':
            print('loading random comments')
            data_df = pl.read_csv(
                args.data_dir+'random_sample_no_sports.csv')  # dont drop nulls
            print('done')
        # load sports vocab
        if args.sports_data == 'vocab':
            with open(args.data_dir+'sports_vocab.json', 'r') as fp:
                sports_vocab = json.load(fp)
            print('loaded sports vocab')
        else:
            # load sports comments
            print('loading sports comments')
            sports_df = pl.read_csv(
                args.data_dir+'sports_sample.csv').drop_nulls()
            print('done')

    if args.manual_vocab:
        print('overwriting sports vocab with manual vocab')
        manual_vocab = [
            'coach', 'season', 'attack', 'defense', 'defend', 'draft', 'game', 'games',
            'pitch', 'pitched', 'players', 'player', 'playing', 'rookie', 'score', 'scored',
            'roster', 'team', 'teams', 'shoot', 'ballpark', 'fans', 'boomerang', 'knockout',
            'mismatch', 'punch', 'dummy', 'prize', 'captain', 'quarterback',
        ]
        sports_vocab = {v: 0 for v in manual_vocab}
        
    # get comments and filter
    print('filtering')
    comments = [re.sub(r"[^a-zA-Z0-9]+", ' ', comment).lower()
                for comment in data_df['body'].to_list()]
    if args.sports_data == 'comments':
        sports_comments = [re.sub(r"[^a-zA-Z0-9]+", ' ', comment).lower()
                    for comment in sports_df['body'].to_list()]
    # filter political/random comments
    comments_long = []
    lens = [len(c.split()) for c in comments]
    for i in range(len(comments)):
        if lens[i] >= args.min_comment_length:
            comments_long.append(comments[i])
    if args.sports_data == 'comments':
        # filter sports comments
        sports_comments_long = []
        lens = [len(c.split()) for c in sports_comments]
        for i in range(len(sports_comments)):
            if lens[i] >= args.min_comment_length:
                sports_comments_long.append(sports_comments[i])
        sports_comments = sports_comments_long[:args.sample_size]
    # upto sample size
    # shuffle?
    comments = comments_long[:args.sample_size]
    if args.sports_data == 'comments':  
        sports_comments = sports_comments_long[:args.sample_size]

    print('training {}-gram lm on {} comments'.format(args.n, args.data))
    # train n gram lm on comments (not sports comments)
    # tokenize
    tokenized_comments = [list(map(str.lower, word_tokenize(sent))) for sent in comments]
    # preprocess the tokenized text for n-gram language modelling
    train_data, padded_sents = padded_everygram_pipeline(args.n, tokenized_comments)
    # train n-gram model
    model = MLE(args.n)
    model.fit(train_data, padded_sents)
    print('done')

    if args.sports_data == 'vocab':
        print('scoring sports vocab')
        # score sports vocab
        scores = 0
        for key, _ in sports_vocab.items():
            scores += model.score(key)

        print('total {}-gram score for {} data: {}'.format(args.n, args.data, scores))
    # 2-gram politics : 0.0089 (shapley), 0.00094 (manual vocab)
    # 2-gram random : 0.0158 (shapley), 0.00296 (manual vocab)

    # 3-gram politics : 0.0088 (shapley), 0.000934 (manual vocab)
    # 3-gram random : 0.0157 (shapley), 0.0029 (manual vocab)
    else:
        print('calculating comment cross entropy without backoff. need to implement backoff')
        # remove stopwords?
        bar = tqdm(range(len(sports_comments)), position=0)
        ce_list = []
        for comment in sports_comments:
            bi_list = list(bigrams(comment.split()))
            plm = 0
            for item in bi_list:
                plm += model.score(item[0], [item[1]])
            ce_list.append(-(plm/len(bi_list)))
            bar.update(1)

        print(sum(ce_list))

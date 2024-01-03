import json
import polars as pl
import re
from tqdm.auto import tqdm
import argparse
import gdown
from nltk.util import ngrams
from sentence_transformers import SentenceTransformer, util
from random import shuffle
import editdistance

# <= 2
def ngram_edit_distance_match(meta_list, comments, args):
    sim_dict = {}
    meta_bar = tqdm(range(len(meta_list)), position=0)
    comment_bar = tqdm(range(len(comments)), position=1)
    for meta in meta_list:
        meta_len = len(meta.split())
        comment_bar = tqdm(range(len(comments)))
        for comment in comments:
            text = comment.split()
            grams = [' '.join(l) for l in list(ngrams(text, n=meta_len))]
            for gram in grams:
                dist = editdistance.eval(gram, meta)
                if dist <= args.edit_thresh:
                    if meta not in sim_dict:
                        sim_dict[meta] = [(gram, dist)]
                    else:
                        sim_dict[meta].append((gram, dist))

            comment_bar.update(1)

        comment_bar.refresh()
        comment_bar.reset()
        meta_bar.update(1)
    
    with open(args.data_dir+'sim_dict1.json', 'w') as f:
        json.dump(sim_dict, f)

    return sim_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

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
        "--data",
        default=None,  # ['politics', 'random']
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
        "--model_name",
        default='sentence-transformers/all-mpnet-base-v2',
        type=str,
    )
    parser.add_argument(
        "--edit_thresh",
        default=2,
        type=int,
    )

    # parse args
    args = parser.parse_args()

    if args.data is None:
        raise ValueError(
            f"pass in data : ['politics, random]"
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
            gdown.download(
                id="1EVu3LrPIsHTrJhl8oICvxO8CxoeYbSbo",
                output='politics_sample.csv', quiet=False
            )
        # download random comments
        elif args.data == 'random':
            gdown.download(
                id="1ujimjlUgQF28OydOMbRvxnVRHOT5ulry",
                output='random_sample_no_sports.csv', quiet=False
            )  # update random sample 

        # load political and random comments
        if args.data == 'politics':
            data_df = pl.read_csv('politics_sample.csv').drop_nulls()
        elif args.data == 'random':
            data_df = pl.read_csv(
                'random_sample_no_sports.csv')  # dont drop nulls
            
        # download metaphors
        gdown.download(
            id="1ApFaZ8Fw0TlaqMoLJhUwwj3NueKypKlK",
            output='meta_dict.json', quiet=False
        )
        with open('meta_dict.json', 'r') as fp:
            data = json.load(fp)
        meta_list = []
        for key, values in data.items():
            meta_list.extend(values)

    # local
    else:
        # load political and random comments
        if args.data == 'politics':
            print('loading political comments')
            data_df = pl.read_csv(
                args.data_dir+'politics_sample.csv').drop_nulls()
            print('done')
        elif args.data == 'random':
            print('loading random comments')  # update random sample
            data_df = pl.read_csv(
                args.data_dir+'random_sample_no_sports.csv')  # dont drop nulls
            print('done')
            
        # load metaphors
        with open(args.data_dir+'meta_dict.json', 'r') as fp:
            data = json.load(fp)
        meta_list = []
        for key, values in data.items():
            meta_list.extend(values)

    # filter metaphors
    print('filtering metaphors')
    meta_list = [meta.replace("'", '') for meta in meta_list]
    meta_list = [re.sub(r"[^a-zA-Z0-9]+", ' ', meta).lower() for meta in meta_list]

    # filter comments
    print('filtering comments')
    comments = [comment.replace("'", '') for comment in data_df['body'].to_list()]
    comments = [re.sub(r"[^a-zA-Z0-9]+", ' ', comment).lower() for comment in comments]
    comments_long = []
    lens = [len(c.split()) for c in comments]
    for i in range(len(comments)):
        if lens[i] >= args.min_comment_length:
            comments_long.append(comments[i])
    print('done')
    print('sampling')
    # sample
    if args.sample:
        shuffle(comments)
        comments = comments_long[:args.sample_size]

    # semantic search model
    print('loading semantic search model')
    model = SentenceTransformer(args.model_name)

    # match metaphors with political comments
    sim_dict = ngram_edit_distance_match(meta_list, comments, args)

    

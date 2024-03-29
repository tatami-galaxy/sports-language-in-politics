from os.path import dirname, abspath
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
import editdistance
import torch

MILLISECONDS_IN_SECOND = 1000

no_list = ['out', 'up', 'tip', 'check']

# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'sports-language-in-politics':
    root = dirname(root)


def ngram_edit_distance_match(meta_list, comments, ids, args):

    # meta : [total_matches, [comment_ids]]
    exact_meta_matches = {}
    # meta : [(gram, comment_id)..]
    sem_dict = {}
    meta_bar = tqdm(range(len(meta_list)), position=0)
    comment_bar = tqdm(range(len(comments)), position=1)

    for meta in meta_list:

        meta_len = len(meta.split())
        if meta_len <= 2:
            edit_thresh = args.edit_thresh_1_2_gram
        elif meta_len == 3:
            edit_thresh = args.edit_thresh_3_gram
        else: 
            edit_thresh = args.edit_thresh_n_gram

        if 'something' in meta or 'someone' in meta or 'someones' in meta:
            if meta.split()[0] in ['something', 'someone', 'someones']:
                meta = ' '.join(meta.split()[1:])
            elif meta.split()[-1] in ['something', 'someone', 'someones']:
                meta = ' '.join(meta.split()[:-1])
            else:
                edit_thresh += 10  # len of something + max edit thresh

        exact_meta_matches[meta] = [0, []]  # [total_matches, [comment_ids]]
        sem_dict[meta] = []

        comment_bar = tqdm(range(len(comments)))
        for c in range(len(comments)):
            # splitting for ngram
            text = comments[c].split()
            id = ids[c]
            grams = [' '.join(l) for l in list(ngrams(text, n=meta_len))]

            for gram in grams:
                dist = editdistance.eval(gram, meta)
                if dist <= edit_thresh: 
                    if dist == 0:
                        # add to exact match count
                        exact_meta_matches[meta][0] += 1
                        # add comment id
                        exact_meta_matches[meta][1].append(id)
                    else:
                       sem_dict[meta].append((gram, id))  # meta : [(gram, comment_id)..]

            comment_bar.update(1)
            
        comment_bar.refresh()
        comment_bar.reset()
        meta_bar.update(1)

    with open(args.data_dir+'exact_matches_'+str(args.seed)+'_'+str(args.year)+'.json', 'w') as f:
        json.dump(exact_meta_matches, f)
    
    with open(args.data_dir+'sem_dict_'+str(args.seed)+'_'+str(args.year)+'.json', 'w') as f:
        json.dump(sem_dict, f)

    return exact_meta_matches, sem_dict


def semantic_filter(model, sem_dict, args):

    # sem_dict -> meta : [(gram, comment_id)..]
    dup_dict = {}
    yes_dict = {}
    meta_count = 0
    # meta : [total_matches, [(gram, score, comment_id)]..]
    semantic_meta_matches = {}

    meta_bar = tqdm(range(len(sem_dict)), position=0)
    for meta, match_list in sem_dict.items():
        meta_bar.update(1)
        meta_count += 1
        semantic_meta_matches[meta] = [0, []]

        # no semantic matches
        if len(match_list) == 0:
            continue

        dup_dict[meta] = []
        yes_dict[meta] = []
        meta_embedding = model.encode(meta)

        bar = tqdm(range(len(match_list)), position=1)
        for match in match_list:  # match -> (gram, comment_id)
            bar.update(1)
            if match[0] not in dup_dict[meta]: 
                match_embedding = model.encode(match[0])
                score = util.cos_sim(meta_embedding, match_embedding).item()
                semantic_meta_matches[meta][1].append((match[0], score, match[1]))  # meta : [total_matches, [(gram, score, comment_id)]..]
                if score >= args.sem_thresh:
                    semantic_meta_matches[meta][0] += 1
                    yes_dict[meta].append(match[0])
                #if meta not in embed_dict:
                    #embed_dict[meta] = [match]
                #else:
                    #embed_dict[meta].append(match)
                dup_dict[meta].append(match[0])
            elif match[0] in yes_dict[meta]:
                semantic_meta_matches[meta][0] += 1

        #if meta_count > 20:
    with open(args.data_dir+'sem_matches_'+str(args.seed)+'_'+str(args.year)+'.json', 'w') as f:
        json.dump(semantic_meta_matches, f)
            
    return semantic_meta_matches
    


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
        "--max_meta",
        default=None,
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
        "--model_name",
        default='sentence-transformers/all-mpnet-base-v2',
        type=str,
    )
    parser.add_argument(
        "--edit_thresh_1_2_gram",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--edit_thresh_3_gram",
        default=3,
        type=int,
    )
    parser.add_argument(
        "--edit_thresh_n_gram",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--sem_thresh",
        default=0.8,
        type=float,
    )
    parser.add_argument(
        "--subs",
        default=['Conservative'],  # ['The_Donald', 'Conservative'] -> 14M words
    )
    parser.add_argument(
        "--year",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--months",
        default=[9, 10, 11, 12]
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
            id="1GElMbQZ0uxGKBL7oVCaKA6miuCtOv5EZ",
            output=args.data_dir+'politics_merged_13m.csv', quiet=False
        )
        # load political comments
        politics_df = pl.read_csv(args.data_dir+'politics_merged_13m.csv').drop_nulls()       
        # download metaphors
        gdown.download(
            id="1zDdechsAiV2A8EkZWAVclTyV1o8osK1c",
            output=args.data_dir+'meta_dict_full.json', quiet=False
        )
        # load metaphors
        with open(args.data_dir+'meta_dict_full.json', 'r') as fp:
            data = json.load(fp)

    # local
    else:
        # load political comments
        print('loading political comments')
        politics_df = pl.read_csv(args.data_dir+'politics_merged_13m.csv').drop_nulls()            
        # load metaphors
        with open(args.data_dir+'meta_dict_full.json', 'r') as fp:
            data = json.load(fp)


    meta_list = []
    for key, values in data.items():
        meta_list.extend(values)
        
    # remove duplicates
    meta_list = list(set(meta_list))

    # filter metaphors
    print('filtering metaphors')
    meta_list = [meta.replace("'", '') for meta in meta_list]
    meta_list = [re.sub(r"[^a-zA-Z0-9]+", ' ', meta).lower() for meta in meta_list]
    meta_list = [m for m in meta_list if m not in no_list]

    if args.max_meta is not None:
        print('truncating metaphor list')
        meta_list = meta_list[:args.max_meta]

    # filter out subs
    data_df = politics_df.filter(pl.col('subreddit').is_in(args.subs))

    # cast datetime
    datetimes = data_df.select((pl.col("created_utc") * MILLISECONDS_IN_SECOND).cast(
        pl.Datetime).dt.with_time_unit("ms").alias("datetime"))
    data_df.replace("created_utc", datetimes['datetime'].dt.date())
    # filter by time period
    data_df = data_df.filter(
        pl.col('created_utc').dt.year().is_in([args.year]))
    data_df = data_df.filter(
        pl.col('created_utc').dt.month().is_in(args.months))

    # shuffle dataframe
    data_df = data_df.sample(fraction=1.0, shuffle=True, seed=args.seed)

    # filter comments and ids
    print('filtering comments')
    comments = [comment.replace("'", '') for comment in data_df['body'].to_list()]
    comments = [re.sub(r"[^a-zA-Z0-9]+", ' ', comment).lower() for comment in comments]
    ids = data_df['id'].to_list()

    comments_long = []
    ids_long = []
    # filter by char 
    for c in range(len(comments)):
        if len(comments[c]) >= args.min_comment_length:
            comments_long.append(comments[c])
            ids_long.append(ids[c])

    print('done')

    # average comment lengths -> political : 435.14  random : 432.06
    #print(sum([len(comment) for comment in comments_long])/len(comments_long))

    # sample comments
    print('sampling')
    if args.sample:
        comments = comments_long[:args.sample_size]
        ids = ids_long[:args.sample_size]
    else:
        comments = comments_long
        ids = ids_long

    print(len(comments))

    # device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # semantic search model
    print('loading semantic search model')
    model = SentenceTransformer(args.model_name, device=device)

    # match metaphors with political comments
    # consider edit distance <= args.edit_thresh (2)
    print('edit distance match')
    exact_meta_matches, sem_dict = ngram_edit_distance_match(meta_list, comments, ids, args)

    # count exact matches
    exact_count = sum([l[0] for l in list(exact_meta_matches.values())])

    # semantic match remaining >= args.sem_thresh (0.8)
    print('semantic filter')
    semantic_meta_matches = semantic_filter(model, sem_dict, args)

    # count semantic matches
    semantic_count = sum([l[0] for l in list(semantic_meta_matches.values())])

    print('exact count : {}'.format(exact_count))
    print('semantic count : {}'.format(semantic_count))


    

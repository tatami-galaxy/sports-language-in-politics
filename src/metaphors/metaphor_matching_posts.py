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

no_list = ['out', 'up', 'tip', 'check']

# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'sports-language-in-politics':
    root = dirname(root)


def ngram_edit_distance_match(meta_list, posts, ids, args):

    # meta : [total_matches, [post_ids]]
    exact_meta_matches = {}
    # meta : [(gram, comment_id)..]
    sem_dict = {}
    meta_bar = tqdm(range(len(meta_list)))

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

        for p in range(len(posts)):
            # splitting for ngram
            text = posts[p].split()
            id = ids[p]
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
                       # meta : [(gram, comment_id)..]
                       sem_dict[meta].append((gram, id))

        meta_bar.update(1)

    with open(args.data_dir+'exact_matches_posts_'+str(args.sample_size)+'.json', 'w') as f:
        json.dump(exact_meta_matches, f)

    with open(args.data_dir+'sem_dict_posts_'+str(args.sample_size)+'.json', 'w') as f:
        json.dump(sem_dict, f)

    return exact_meta_matches, sem_dict


def semantic_filter(model, sem_dict, args):

    # sem_dict -> meta : [(gram, post_id)..]
    dup_dict = {}
    yes_dict = {}
    meta_count = 0
    # meta : [total_matches, [(gram, score, post_id)]..]
    semantic_meta_matches = {}

    meta_bar = tqdm(range(len(sem_dict)))
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

        for match in match_list:  # match -> (gram, post_id)
            if match[0] not in dup_dict[meta]: 
                match_embedding = model.encode(match[0])
                score = util.cos_sim(meta_embedding, match_embedding).item()
                semantic_meta_matches[meta][1].append((match[0], score, match[1]))  # meta : [total_matches, [(gram, score, post_id)]..]
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
    with open(args.data_dir+'sem_matches_posts_'+str(args.sample_size)+'.json', 'w') as f:
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
        default=10000,
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
        # download data
        gdown.download(
            id="19_cD09HJLpb8w1bkVl1jbOV283YehwQQ",
            output=args.data_dir+'posts_2015-21_ps_min_2c_politics.csv', quiet=False
        )
        data_df = pl.read_csv(args.data_dir+'posts_2015-21_ps_min_2c_politics.csv')
            
        # download metaphors
        gdown.download(
            id="1zDdechsAiV2A8EkZWAVclTyV1o8osK1c",
            output=args.data_dir+'meta_dict_full.json', quiet=False
        )
        with open(args.data_dir+'meta_dict_full.json', 'r') as fp:
            data = json.load(fp)

    # local
    else:
        # load data
        print('loading political posts')
        data_df = pl.read_csv(args.data_dir+'posts_2015-21_ps_min_2c_politics.csv')
            
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

    # shuffle dataframe
    data_df = data_df.sample(fraction=1.0, shuffle=True, seed=args.seed)

    # preprocess posts
    print('preprocessing posts')
    posts = [post.replace("'", '') for post in data_df['title'].to_list()]
    posts = [re.sub(r"[^a-zA-Z0-9]+", ' ', post).lower() for post in posts]
    ids = data_df['id'].to_list()

    # sample comments
    print('sampling')
    if args.sample:
        posts = posts[:args.sample_size]
        ids = ids[:args.sample_size]

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

    # match metaphors with political posts
    print('edit distance match')
    exact_meta_matches, sem_dict = ngram_edit_distance_match(meta_list, posts, ids, args)

    # count exact matches
    exact_count = sum([l[0] for l in list(exact_meta_matches.values())])

    # semantic match remaining >= args.sem_thresh (0.8)
    print('semantic filter')
    semantic_meta_matches = semantic_filter(model, sem_dict, args)

    # count semantic matches
    semantic_count = sum([l[0] for l in list(semantic_meta_matches.values())])

    print('exact count : {}'.format(exact_count))
    print('semantic count : {}'.format(semantic_count))


    

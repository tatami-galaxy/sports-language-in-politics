from os.path import dirname, abspath
import argparse
import gdown
import polars as pl
import json
import re
import random
from tqdm.auto import tqdm
import editdistance
from nltk.util import ngrams

import torch
from sentence_transformers import SentenceTransformer, util

from constants import *
from preprocess import preprocess_posts

# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'sports-language-in-politics':
    root = dirname(root)


def ngram_edit_distance_match(meta_list, posts, ids, args):

    # meta : [total_matches, [post_ids]]
    exact_meta_matches = {}
    # meta : [(gram, comment_id)..]
    sem_dict = {}

    bar = tqdm(range(len(meta_list)))

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

        bar.update(1)

    return exact_meta_matches, sem_dict


def semantic_filter(model, sem_dict, args):

    # sem_dict -> meta : [(gram, post_id)..]
    dup_dict = {}
    yes_dict = {}
    meta_count = 0
    # meta : [total_matches, [(gram, score, post_id)]..]
    semantic_meta_matches = {}

    bar = tqdm(range(len(sem_dict)))

    for meta, match_list in sem_dict.items():
        meta_count += 1
        semantic_meta_matches[meta] = [0, []]

        bar.update(1)

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
                # meta : [total_matches, [(gram, score, post_id)]..]
                semantic_meta_matches[meta][1].append(
                    (match[0], score, match[1]))
                if score >= args.sem_thresh:
                    semantic_meta_matches[meta][0] += 1
                    yes_dict[meta].append(match[0])
                # if meta not in embed_dict:
                    # embed_dict[meta] = [match]
                # else:
                    # embed_dict[meta].append(match)
                dup_dict[meta].append(match[0])
            elif match[0] in yes_dict[meta]:
                semantic_meta_matches[meta][0] += 1

    return semantic_meta_matches


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cloud",
        action="store_true",
    )
    parser.add_argument(
        "--data_dir",
        default='/Volumes/PortableSSD/CSS/data/processed/',
        type=str,
    )
    parser.add_argument(
        "--model_name",
        default='sentence-transformers/all-mpnet-base-v2',
        type=str,
    )
    parser.add_argument(
        "--max_meta",
        default=None,
        type=int,
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
        "--cluster_file",
        default=None,
        type=str,
    )

    # parse args
    args = parser.parse_args()

    # seed
    seed = SEED

    # load data
    samples_df = pl.read_csv(args.data_dir+'gpt3_responses_with_gt_53.csv')
    posts = samples_df['post'].to_list()
    ids = samples_df['id'].to_list()
    gt = samples_df['ground_truth'].to_list()

    # preprocess posts
    posts = preprocess_posts(posts)

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
    meta_list = [re.sub(r"[^a-zA-Z0-9]+", ' ', meta).lower()
                 for meta in meta_list]
    meta_list = [m for m in meta_list if m not in NO_META_LIST]
    # truncate metaphor list
    if args.max_meta is not None:
        print('truncating metaphor list')
        meta_list = meta_list[:args.max_meta]

    # device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # load model
    model = SentenceTransformer(args.model_name, device=device)

    meta_ids = []
    sem_meta_ids = []
    non_meta_ids = []

    # exact matches
    print('exact matches')
    exact_meta_matches, sem_dict = ngram_edit_distance_match(meta_list, posts, ids, args)
    exact_count = sum([l[0] for l in list(exact_meta_matches.values())])

    # if any exact matches
    if exact_count > 0:
        # get meta ids 
        for meta, matches in exact_meta_matches.items():
            # at least one match
            if matches[0] > 0:
                # meta : [total_matches, [post_ids]]
                meta_ids.extend(matches[1])
        #meta_ids = list(set(meta_ids))

    # semantic matches
    print('semantic matches')
    semantic_meta_matches = semantic_filter(model, sem_dict, args)
    semantic_count = sum([l[0] for l in list(semantic_meta_matches.values())])

    # if any semantic matches
    if semantic_count > 0:
        # get meta ids 
        for meta, matches in semantic_meta_matches.items():
            # at least one match
            if matches[0] > 0:
                # meta : [total_matches, [(gram, score, post_id)]..]
                sem_ids  = []
                for tup in matches[1]:
                    if tup[1] >= args.sem_thresh:  # score
                        sem_ids.append(tup[2])
                sem_meta_ids.extend(sem_ids)
        #sem_meta_ids = list(set(sem_meta_ids))

    # combine meta_ids
    meta_ids = set(meta_ids + sem_meta_ids)
    # get non meta ids
    for id in ids:
        if id not in meta_ids:
            non_meta_ids.append(id)

    print(len(meta_ids))
    print(len(non_meta_ids))

    # compile results into df
    contains_sports_meta = []
    for i in range(len(ids)):
        if ids[i] in meta_ids:
            contains_sports_meta.append(True)
        else:
            contains_sports_meta.append(False)

    data_dict = {
        'id': ids,
        'post': posts,
        'ground_truth': gt,
        'result': contains_sports_meta,
    }
    data_df = pl.from_dict(data_dict)

    # write to file
    data_df.write_csv(args.data_dir+'sem_matching_responses_eval.csv', separator=",")

            

            

               
    
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
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering

MILLISECONDS_IN_SECOND = 1000

# lists to exclude during pre-processing
NO_DOMAINS = ["i.imgur.com", "imgur.com", "i.reddituploads.com",
              "i.sli.mg", "i.magaimg.net", "gfycat.com", "pbs.twimg.com", "sli.mg"]
NO_IMAGES = ['jpeg', 'png', 'tiff', 'gif', 'jpg']
NO_META_LIST = ['out', 'up', 'tip', 'check']

# hyperparameters
MIN_CHARS = 10
CLUSTERING_THRESHOLD = 0.4

# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'sports-language-in-politics':
    root = dirname(root)


def ngram_edit_distance_match(meta_list, posts, ids, args):

    # meta : [total_matches, [post_ids]]
    exact_meta_matches = {}
    # meta : [(gram, comment_id)..]
    sem_dict = {}

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

    return exact_meta_matches, sem_dict


def semantic_filter(model, sem_dict, args):

    # sem_dict -> meta : [(gram, post_id)..]
    dup_dict = {}
    yes_dict = {}
    meta_count = 0
    # meta : [total_matches, [(gram, score, post_id)]..]
    semantic_meta_matches = {}

    for meta, match_list in sem_dict.items():
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
        "--seed",
        default=42,
        type=int,
    )
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
        "--day_sample",
        default=None,
        type=int,
    )

    # parse args
    args = parser.parse_args()

    # seed
    seed = args.seed

    # load data
    if args.cloud:
        # download posts
        gdown.download(
            id="19_cD09HJLpb8w1bkVl1jbOV283YehwQQ",
            output=args.data_dir+'posts_2015-21_ps_min_2c_politics.csv', quiet=False
        )
        # download metaphors
        gdown.download(
            id="1zDdechsAiV2A8EkZWAVclTyV1o8osK1c",
            output=args.data_dir+'meta_dict_full.json', quiet=False
        )

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

    # load posts
    print('loading posts')
    data_df = pl.read_csv(args.data_dir+'posts_2015-21_ps_min_2c_politics.csv')

    # filter image domains
    data_df = data_df.filter(~pl.col('domain').is_in(NO_DOMAINS))
    # filter images
    data_df = data_df.filter(~pl.col('url').str.contains_any(NO_IMAGES))
    # filter small posts
    data_df = data_df.filter(pl.col('title').str.len_chars() >= MIN_CHARS)

    # cast datetime
    datetimes = data_df.select((pl.col("created_utc") * MILLISECONDS_IN_SECOND).cast(
        pl.Datetime).dt.with_time_unit("ms").alias("datetime"))
    data_df.replace("created_utc", datetimes['datetime'].dt.date())

    # get all unique days
    all_days = data_df['created_utc'].unique().to_list()
    # shuffle days
    random.Random(seed).shuffle(all_days)
    # sample
    if args.day_sample is not None:
        all_days = all_days[:args.day_sample]

    # device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # load model
    model = SentenceTransformer(args.model_name, device=device)

    # day -> {cluster1 : {'meta' : [id1, id2, ..], 'non_meta' : [id3, id4, ..]}, cluster2 : {..}}
    day_dict = {}

    bar = tqdm(range(len(all_days)))

    for day in all_days:

        # dict entry for day
        day_dict[str(day)] = {}

        # filter to get posts from day
        day_df = data_df.filter(pl.col('created_utc') == day)

        # get posts and ids
        posts = day_df['title'].to_list()
        ids = day_df['id'].to_list()

        # preprocess posts
        posts = [post.replace("'", '') for post in posts]
        posts = [re.sub(r"[^a-zA-Z0-9.]+", ' ', post).lower() for post in posts]

        # remove duplicate posts
        # id2post maps ids to posts for each day
        temp = dict(zip(ids, posts))
        dup_set = set()
        id2post = {}
        for id, post in temp.items():
            if post not in dup_set:
                id2post[id] = post
                dup_set.add(post)
        posts = list(id2post.values())

        # encode posts
        embeddings = model.encode(posts)
        # get pairwise cosine distances between post embeddings
        dists = cosine_distances(embeddings, embeddings)

        # cluster posts
        cluster = AgglomerativeClustering(
            metric='precomputed',
            linkage='average',
            n_clusters=None,
            distance_threshold=CLUSTERING_THRESHOLD
        )
        clusters = cluster.fit(dists)

        # map posts (and ids) to cluster labels
        # label -> [(id1, post1), (id2, post2), ...]
        label_to_id_post = {}
        for l in range(len(clusters.labels_)):
            if clusters.labels_[l] not in label_to_id_post:
                label_to_id_post[clusters.labels_[l]] = [(ids[l], posts[l])]
            else:
                label_to_id_post[clusters.labels_[l]].append((ids[l], posts[l]))

        # get clusters with more than 1 post
        # imp_clusters = [[(id1, post1), (id2, post2), ...], [(id3, post3), (id4, post4), ...],...]
        imp_clusters = []
        for label, id_post in label_to_id_post.items():
            if len(id_post) > 1:
                imp_clusters.append(id_post)

        # metaphor match each non trivial cluster
        # day -> {cluster1 : {'meta' : [id1, id2, ..], 'non_meta' : [id3, id4, ..]}, cluster2 : {..}}
        for c in range(len(imp_clusters)):

            meta_ids = []
            sem_meta_ids = []
            non_meta_ids = []

            # init dict entry for each cluster
            day_dict[str(day)]['cluster'+str(c)] = {'meta':[], 'non_meta':[]}

            c_posts = [x[1] for x in imp_clusters[c]]
            c_ids = [x[0] for x in imp_clusters[c]]

            # exact matches
            exact_meta_matches, sem_dict = ngram_edit_distance_match(meta_list, c_posts, c_ids, args)
            exact_count = sum([l[0] for l in list(exact_meta_matches.values())])

            # if any exact matches
            if exact_count > 0:
                # get meta ids for the cluster
                for meta, matches in exact_meta_matches.items():
                    # at least one match
                    if matches[0] > 0:
                        # meta : [total_matches, [post_ids]]
                        meta_ids.extend(matches[1])
                meta_ids = list(set(meta_ids))

            # semantic matches
            semantic_meta_matches = semantic_filter(model, sem_dict, args)
            semantic_count = sum([l[0] for l in list(semantic_meta_matches.values())])

            # if any semantic matches
            if semantic_count > 0:
                # get meta ids for the cluster
                for meta, matches in semantic_meta_matches.items():
                    # at least one match
                    if matches[0] > 0:
                        # meta : [total_matches, [(gram, score, post_id)]..]
                        sem_ids = [x[2] for x in matches[1]]
                        sem_meta_ids.extend(sem_ids)
                sem_meta_ids = list(set(sem_meta_ids))

            # combine meta_ids
            meta_ids = meta_ids + sem_meta_ids
            # get non meta ids for the cluster
            for id in c_ids:
                if id not in meta_ids:
                    non_meta_ids.append(id)

            # add cluster data to day_dict
            day_dict[str(day)]['cluster'+str(c)] = {'meta':meta_ids, 'non_meta':non_meta_ids}   

        bar.update(1)
         
    #with open(args.data_dir+'post_cluster_matches.json', 'w') as f:
        #json.dump(day_dict, f)

    print(day_dict)

            

            

               
    
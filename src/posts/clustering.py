from os.path import dirname, abspath
import argparse
import gdown
import polars as pl
import json
import random
from tqdm.auto import tqdm


import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering

from constants import *
from preprocess import preprocess_posts, remove_duplicates

# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'sports-language-in-politics':
    root = dirname(root)


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
        "--day_sample",
        default=None,
        type=int,
    )

    # parse args
    args = parser.parse_args()

    # seed
    seed = SEED

    # load data
    if args.cloud:
        # download posts
        gdown.download(
            id="19_cD09HJLpb8w1bkVl1jbOV283YehwQQ",
            output=args.data_dir+'posts_2015-21_ps_min_2c_politics.csv', quiet=False
        )

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

    # mappping for clusters to days
    # each cluster is a list of post ids
    # {day1 : {cluster1 : [id1, id2, ..], cluster2 : [id3, id4, ..]..}, day2 : {...}}
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
        posts = preprocess_posts(posts)

        # remove duplicate posts
        # take post with most comments
        ids, posts = remove_duplicates(day_df, ids, posts)

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

        # day_dict
        # {day1 : {cluster_id1 : [id1, id2, ..], cluster_id2 : [id3, id4, ..]..}, day2 : {...}}
        # map ids to cluster labels
        
        for l in range(len(cluster.labels_)):
            c_id = str(cluster.labels_[l])
            if c_id not in day_dict[str(day)]:
                day_dict[str(day)][c_id] = [ids[l]]
            else:
                day_dict[str(day)][c_id].append(ids[l])

        bar.update(1)

    # write to file
    if args.day_sample is not None:
        with open(args.data_dir+'post_clusters_'+str(args.day_sample)+'.json', 'w') as f:
            json.dump(day_dict, f)
    else:
        with open(args.data_dir+'post_clusters.json', 'w') as f:
            json.dump(day_dict, f)

from os.path import dirname, abspath
import argparse
import gdown
import polars as pl
import json
import re
import random
from tqdm.auto import tqdm

import torch
from openai import OpenAI

from constants import *
from preprocess import preprocess_posts

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
        "--max_meta",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--day_sample",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--cluster_file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--api_key",
        default=None,
        type=str,
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
        # download metaphors
        gdown.download(
            id="1zDdechsAiV2A8EkZWAVclTyV1o8osK1c",
            output=args.data_dir+'meta_dict_full.json', quiet=False
        )
        # download clusters #

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

    # load clusters
    with open(args.cluster_file) as f:
        cluster_data = json.load(f)

    # device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # gpt3.5 client
    client = OpenAI(api_key=args.api_key)

    # day -> {cluster1 : {'meta' : [id1, id2, ..], 'non_meta' : [id3, id4, ..]}, cluster2 : {..}}
    day_dict = {}

    bar = tqdm(range(len(all_days)))

    for day in all_days:

        # dict entry for day
        day_dict[str(day)] = {}

        # filter to get posts from day
        day_df = data_df.filter(pl.col('created_utc') == day)

        # get all clusters for day
        clusters = cluster_data[str(day)]

        cluster_bar = tqdm(range(len(clusters)))
        # for each cluster do metaphor matching 
        for c_id, p_ids in clusters.items():

            cluster_bar.update(1)

            # get non trivial clusters
            if len(p_ids) <= 1:
                continue

            day_dict[str(day)][c_id] = {'meta': [], 'non_meta': []}

            posts = day_df.filter(pl.col('id').is_in(p_ids))['title'].to_list()
            posts = preprocess_posts(posts)

            # metaphor match each non trivial cluster
            # day -> {cluster1 : {'meta' : [(id1,exp1), (id2,exp2), ..], 'non_meta' : [id3, id4, ..]}, cluster2 : {..}}

            meta_ids = []
            non_meta_ids = []

            total_responses = []
            for post in posts:

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    response_format={ "type": "json_object" },
                    seed=20,
                    temperature=0.53,
                    messages=[
                        {
                        "role": "system",
                        "content": "You will be provided with a Reddit post title, and your task is to identify if the post contains a sports metaphor or not. Note that sports related words may be used in a nonmetaphorical way, do not label such cases as sports metaphors. If the text does contain a sports metaphor, identify the sports metaphor word or phrase and provide a max 10 word explanation. Provide the answer in a JSON format with the following keys, contains_sports_metaphor (true/false), sports_metaphor, explanation."
                        },
                        {
                        "role": "user",
                        "content": post
                        }
                    ],
                )
                
                obj = json.loads(response.json())
                resp_json = json.loads(obj["choices"][0]["message"]["content"])
                resp_json["post"] = post

                total_responses.append(resp_json)

            # meta ids vs non meta ids
            for r in range(len(total_responses)):
                response = total_responses[r]
                id = p_ids[r]
                if response['contains_sports_metaphor']:
                    meta_ids.append((id, response['explanation']))
                else:
                    non_meta_ids.append(id)

            # add cluster data to day_dict
            day_dict[str(day)][c_id] = {'meta':meta_ids, 'non_meta':non_meta_ids}   

    bar.update(1)
         
    # write to file
    if args.day_sample is not None:
        with open(args.data_dir+'post_cluster_matches_chatgpt'+str(args.day_sample)+'.json', 'w') as f:
            json.dump(day_dict, f)
    else:
        with open(args.data_dir+'post_cluster_matches_chatgpt.json', 'w') as f:
            json.dump(day_dict, f)

            

            

               
    
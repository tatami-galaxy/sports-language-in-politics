import re
import polars as pl
from constants import *


def preprocess_posts(posts):

    posts = [post.replace("'", '') for post in posts]
    posts = [re.sub(r"[^a-zA-Z0-9.]+", ' ', post).lower() for post in posts]

    return posts

def remove_duplicates(df, ids, posts):
    temp = dict(zip(ids, posts))
    post2id = {}
    post2num_comm = {}
    for id, post in temp.items():
        # num comments of the post
        num_comments = df.filter(pl.col('id') == id)['num_comments'].to_list()[0]
        if post not in post2id or num_comments > post2num_comm[post]:
            post2id[post] = id
            post2num_comm[post] = num_comments

    # posts[0] has id ids[0]
    posts = list(post2id.keys())
    ids = list(post2id.values())

    return ids, posts

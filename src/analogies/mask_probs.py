import json
import re
import math
import argparse
import random

import gdown
from tqdm.auto import tqdm
import polars as pl

import torch

from transformers import AutoTokenizer, BertForMaskedLM
from datasets import Dataset

"""

Biden/Trump as captain/coach/quarterback/skipper : NER -> biden, trump
politicians as players : NER -> list of politicians, NER -> list of players?
elections as race/competition : 
parties as teams : republicans, democrats
voters/supporters as fans : [voters, supporters, ...]
media as spectators : [media, news_channel, ..], [cnn, fox, nytimes,0 ...]

"""


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
        "--data_dir",
        default='/users/ujan/sports-language-in-politics/data/processed/',
        type=str,
    )
    parser.add_argument(
        "--min_comment_length",
        default=150,  # chars
        type=int,
    )
    parser.add_argument(
        "--model_name",
        default="bert-base-uncased",
        type=str,
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
            id="1EVu3LrPIsHTrJhl8oICvxO8CxoeYbSbo",
            output='politics_sample.csv', quiet=False
        )
        # load political comments
        print('loading political comments')
        data_df = pl.read_csv('politics_sample.csv').drop_nulls()


    # local
    else:
        # load political and random comments
        print('loading political comments')
        data_df = pl.read_csv(args.data_dir+'politics_sample.csv').drop_nulls()
        print('done')

    # filter comments and ids
    print('filtering comments')
    comments = [comment.replace("'", '') for comment in data_df['body'].to_list()]
    comments = [re.sub(r"[^a-zA-Z0-9]+", ' ', comment).lower()
                for comment in comments]
    ids = data_df['id'].to_list()

    comments_long = []
    ids_long = []
    # filter by char
    for c in range(len(comments)):
        if len(comments[c]) >= args.min_comment_length:
            comments_long.append(comments[c])
            ids_long.append(ids[c])
    print('done')

    # sample comments
    print('sampling')
    if args.sample:
        shuffle(comments_long)
        shuffle(ids_long)
        comments = comments_long[:args.sample_size]
        ids = ids_long[:args.sample_size]

    # build biden/trump dataset
    president_comments = []
    president_ids = []
    for c in range(len(comments)):
        if 'biden' in comments[c] or 'trump' in comments[c]:
            president_comments.append(comments[c])
            president_ids.append(comments[c])

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BertForMaskedLM.from_pretrained(args.model_name)

    # captain : 2952
    # skipper : 23249
    # quarterback : 9074
    # coach : 2873
    token_dict = {}  # token : {value: total_prob_value, count: total_count}
    comment_dict = {}  # {id: {captain: [val1, val2,...], coach : [val1, val2,...], skipper:...}}

    bar = tqdm(range(len(president_comments)), position=0)
    for c in range(len(president_comments)):

        comment_dict[ids[c]] = {'captain':[], 'coach':[], 'skipper':[], 'quarterback':[]}

        tokens = president_comments[c].split()
        # mask biden/trump
        tokens = ['[MASK]' if token in ['trump', 'trumps', 'biden', 'bidens'] else token for token in tokens]
        masked_comment = ' '.join(tokens)
        # pass through model
        inputs = tokenizer(masked_comment, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        # get mask token ids
        mask_token_ids = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        # for each masked token (multiple biden mentions)
        for mask_token_id in mask_token_ids:
            # get corresponding logits
            mask_logits = logits[0, mask_token_id.item()]
            # softmax to get probs for each token in vocab
            mask_probs = torch.nn.functional.softmax(mask_logits, dim=0)
            # sort in descending order
            prob_values, token_ids = torch.sort(mask_probs, descending=True)

            print(token_ids)

            # for each token in vocab
            for i in range(len(token_ids)):
                # decode to get string
                token = tokenizer.decode(token_ids[i])
                # prob value
                value = prob_values[i].item()

                if token == 'captain':
                    comment_dict[ids[c]]['captain'].append(value)
                if token == 'coach':
                    comment_dict[ids[c]]['coach'].append(value)
                if token == 'skipper':
                    comment_dict[ids[c]]['skipper'].append(value)
                if token == 'quarterback':
                    comment_dict[ids[c]]['quarterback'].append(value)
                
                if token in token_dict:
                    token_dict[token]['value'] += value
                    token_dict[token]['count'] += 1
                else:
                    token_dict[token] = {'value': value, 'count': 1}

        bar.update(1)

    new_token_dict = {}
    new_comment_dict = {}
    # normalize probs
    for key, val in token_dict.items():
        new_token_dict[key] = val['value'] / val['count']
    for key, val in comment_dict.items():
        new_comment_dict[key] = {'captain':0, 'coach':0, 'skipper':0, 'quarterback':0}
        new_comment_dict[key]['captain'] = sum(comment_dict[key]['captain']) / len(comment_dict[key]['captain'])
        new_comment_dict[key]['coach'] = sum(comment_dict[key]['coach']) / len(comment_dict[key]['coach'])
        new_comment_dict[key]['skipper'] = sum(comment_dict[key]['skipper']) / len(comment_dict[key]['skipper'])
        new_comment_dict[key]['quarterback'] = sum(comment_dict[key]['quarterback']) / len(comment_dict[key]['quarterback'])

    # save
    with open(args.data_dir+'token_dict_'+str(args.seed)+'_'+str(args.sample_size)+'.json', 'w') as f:
        json.dump(new_token_dict, f)
    with open(args.data_dir+'comment_dict_'+str(args.seed)+'_'+str(args.sample_size)+'.json', 'w') as f:
        json.dump(new_comment_dict, f)

    




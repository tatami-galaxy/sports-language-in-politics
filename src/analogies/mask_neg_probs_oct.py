import json
import re
import argparse
from os.path import dirname, abspath

import gdown
from tqdm.auto import tqdm
import polars as pl

import torch

from transformers import AutoTokenizer, BertForMaskedLM


#TARGETS = ['election'] # cant have multiple with current logic
#TARGETS = ['biden', 'trump']
#TARGETS = ['democratic party', 'democrats', 'republican party', 'republicans']
#TARGETS = ['voters']
#TARGETS = ['president']
#SUBSTITUTES = ['race', 'competition', 'championship', 'tournament']
#SUBSTITUTES = ['captain', 'coach', 'quarterback', 'skipper']
#SUBSTITUTES = ['team', 'teams']
#SUBSTITUTES = ['fan', 'fans', 'spectator', 'spectators']

MILLISECONDS_IN_SECOND = 1000

# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'sports-language-in-politics':
    root = dirname(root)

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
        "--sample_size",
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
        default=150,  # chars
        type=int,
    )
    parser.add_argument(
        "--model_name",
        default="bert-base-uncased",
        type=str,
    )
    parser.add_argument(
        "--target",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--substitute",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--year",
        default=2015,
        type=int,
    )
    parser.add_argument(
        "--subs",
        nargs="*",
        default=['Conservative'],
    )

    # parse args
    args = parser.parse_args()

    # check if data directory is None
    if args.data_dir is None:
        raise ValueError(
            f"pass in data_dir"
        )
    elif args.data_dir[-1] != '/':
        args.data_dir = args.data_dir+'/'

    # load data
    year_file_map = {
        2015: 'politics_comments_2015_10.csv',
        2016: 'politics_comments_2016_10.csv',
        2017: 'politics_comments_2017_10.csv',
        2018: 'politics_comments_2018_10.csv',
        2019: 'politics_comments_2019_10.csv',
        2020: 'politics_comments_2020_10.csv',
        2021: 'politics_comments_2021_10.csv',
    }
    year_id_map = {
        2015: "1kfQLlhe-w1oRDkFI_9xzKSsqgabhQStL",
        2016: "1mrfLuKLlcx2xz613LuEM4_UG5Cjkve8R",
        2017: "1uU8bGEusLTRGN8Y2qzp3lLfIbtNAGK5V",
        2018: "1uthNuv2U-SvSE0LmyblhwqoWzWjGjF3H",
        2019: "17QqsbCxPB5EMxJh3mggwlAC8Egk4xEGu",
        2020: "1dlxpS-v34jnB2tKYmLzkvB3VqIENi2gL",
        2021: "12aGe2P0hbGW37JVMirlcd2lo5lCWnFPq",
    }
    if args.cloud:
        # download political comments
        print('downloading political comments')
        gdown.download(
            id=year_id_map[args.year],
            output=args.data_dir+year_file_map[args.year], quiet=False
        )

        # load politicalomments
        politics_df = pl.read_csv(
            args.data_dir+year_file_map[args.year]).drop_nulls()

    # local
    else:
        # load political comments
        print('loading political and sports comments')
        politics_df = pl.read_csv(args.data_dir+year_file_map[args.year]).drop_nulls()

    # filter out subs
    data_df = politics_df.filter(pl.col('subreddit').is_in(args.subs))

    # cast datetime
    datetimes = data_df.select((pl.col("created_utc") * MILLISECONDS_IN_SECOND).cast(
        pl.Datetime).dt.with_time_unit("ms").alias("datetime"))
    data_df.replace("created_utc", datetimes['datetime'].dt.date())


    # shuffle dataframe
    data_df = data_df.sample(fraction=1.0, shuffle=True, seed=args.seed)

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

    # select comments with target tokens
    president_comments = []
    president_ids = []
    for c in range(len(comments_long)):
        tokens = comments_long[c].split()
        for token in tokens:
            if token == args.target:
                president_comments.append(comments_long[c])
                president_ids.append(ids_long[c])
                break

    president_comments = president_comments
    president_ids = president_ids

    if len(president_comments) < args.sample_size:
        raise ValueError("comments ({}) less than sample size ({})".format(len(president_comments), args.sample_size))

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BertForMaskedLM.from_pretrained(args.model_name)

    # device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model.to(device)

    # captain : 2952
    # skipper : 23249
    # quarterback : 9074
    # coach : 2873
    token_dict = {}  # token : {value: total_prob_value, count: total_count}
    comment_dict = {}  # {id: {captain: [val1, val2,...], coach : [val1, val2,...], skipper:...}}

    bar = tqdm(range(args.sample_size), position=0)

    act_count = 0

    for c in range(len(president_comments)):

        comment_dict[president_ids[c]] = {s:[] for s in [args.target, args.substitute]}

        tokens = president_comments[c].split()
        # mask biden/trump
        tokens = ['[MASK]' if token == args.target else token for token in tokens]
        masked_comment = ' '.join(tokens)
        # pass through model
        inputs = tokenizer(masked_comment, return_tensors="pt", truncation=True)

        # ignore comment since mask can be truncated out
        if inputs['input_ids'].shape[-1] >= 512:
            continue
        act_count += 1

        with torch.no_grad():
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.cpu()
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

            # for each token in vocab
            for i in range(len(token_ids)):
                # decode to get string
                token = tokenizer.decode(token_ids[i])
                # prob value
                value = prob_values[i].item()

                for sub in [args.target, args.substitute]:
                    if token == sub:
                        comment_dict[president_ids[c]][sub].append(value)
                
                if token in token_dict:
                    token_dict[token]['value'] += value
                    token_dict[token]['count'] += 1
                else:
                    token_dict[token] = {'value': value, 'count': 1}

        bar.update(1)
        if args.sample_size is not None:
            if act_count == args.sample_size:
                print('sample limit reached')
                break

    new_token_dict = {}
    new_comment_dict = {}
    # normalize probs
    for key, val in token_dict.items():
        new_token_dict[key] = val['value'] / val['count']
    for key, val in comment_dict.items():
        if len(val[args.substitute]) < 1:
            continue
        new_comment_dict[key] = {s: 0 for s in [args.target, args.substitute]}
        for sub in [args.target, args.substitute]:
            new_comment_dict[key][sub] = sum(val[sub]) / len(val[sub])

    # save
    with open(args.data_dir+'token_dict_'+args.target+'_'+args.substitute+'_'+str(args.sample_size)+'.json', 'w') as f:
        json.dump(new_token_dict, f)
    with open(args.data_dir+'comment_dict_'+args.target+'_'+args.substitute+'_'+str(args.sample_size)+'.json', 'w') as f:
        json.dump(new_comment_dict, f)

    # count neg prob
    count = 0
    for id, val in comment_dict.items():
        if val[args.target] < val[args.substitute]:
            count += 1
    print(count/len(comment_dict))
    





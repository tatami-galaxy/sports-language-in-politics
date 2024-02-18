import os
from os.path import dirname, abspath
import argparse

from transformers import BertModel, AutoTokenizer

# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'sports-language-in-politics':
    root = dirname(root)


if __name__ == '__main__':

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
        "--stop_words",
        action="store_true",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
    )
    parser.add_argument(
        "--sample_size",
        default=5000,
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
        "--train_batch_size",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--eval_batch_size",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--learning_rate",
        default=0.025,
        type=float,
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--train_steps",
        default=20000,
        type=int,
    )
    parser.add_argument(
        "--eval_steps",
        default=500,
        type=int,
    )
    parser.add_argument(
        "--model_dir",
        default=root+'/models/cbow/',
        type=str,
    )
    parser.add_argument(
        "--subs",
        default=['The_Donald'],  # ['The_Donald', 'Conservative'] -> 14M words
    )


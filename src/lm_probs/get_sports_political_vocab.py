import argparse
import json
import re

import gdown
import nltk
from nltk.corpus import stopwords
import polars as pl
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import shap

import polars as pl
import re
import nltk
from nltk.corpus import stopwords



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
        "--min_comment_length",
        default=150,
        type=int,
    )
    parser.add_argument(
        "--classifier_sample_size",
        default=50000,
        type=int,
    )
    parser.add_argument(
        "--shapley_threshold",
        default=0.35,
        type=float,
    )
    parser.add_argument(
        "--output_dir",
        default='/users/ujan/sports-language-in-politics/data/processed/',
        type=str,
    )

    # parse args
    args = parser.parse_args()

    # check if output directory is None
    if args.output_dir is None:
        raise ValueError(
            f"pass in output_dir"
        )
    elif args.output_dir[-1] != '/':
        args.output_dir = args.output_dir+'/'

    # init nltk, shap
    np.random.seed(args.seed)
    shap.initjs()
    status = False
    while not status:
        status = nltk.download('stopwords')

    # load sports and political sample data
    if args.cloud:
        sports_id = "1Xc6VXdG8cloh8tdxAaboQewkilgvWxub"
        politics_id = "1EVu3LrPIsHTrJhl8oICvxO8CxoeYbSbo"
        sports_output = 'sports_sample.csv'
        politics_output = 'politics_sample.csv'
        gdown.download(id=sports_id, output=sports_output, quiet=False)
        print('sports sample downloaded')
        gdown.download(id=politics_id, output=politics_output, quiet=False)
        print('politics sample downloaded') 
        sports_df = pl.read_csv('sports_sample.csv').drop_nulls()
        politics_df = pl.read_csv('politics_sample.csv').drop_nulls()
    else:
        politics_df = pl.read_csv(
            '~/sports-language-in-politics/data/processed/politics_sample.csv').drop_nulls()
        
    # clean data and get comments
    sports_comments = [re.sub(r"[^a-zA-Z0-9]+", ' ', comment).lower()
                       for comment in sports_df['body'].to_list()]
    political_comments = [re.sub(r"[^a-zA-Z0-9]+", ' ', comment).lower()
                      for comment in politics_df['body'].to_list()]
    
    # filter short comments and take smaller sample of comments
    sports_comments_long = []
    political_comments_long = []

    s_lens = [len(s.split()) for s in sports_comments]
    p_lens = [len(c.split()) for c in political_comments]
    for i in range(len(sports_comments)):
        if s_lens[i] >= args.min_comment_length:
            sports_comments_long.append(sports_comments[i])
    for i in range(len(political_comments)):
        if p_lens[i] >= args.min_comment_length:
            political_comments_long.append(political_comments[i])

    sports_comments = sports_comments_long[:args.classifier_sample_size]
    political_comments = political_comments_long[:args.classifier_sample_size]
    print('data loaded')

    # train classifier to get shapeley values

    # labels
    sports_y = np.ones(len(sports_comments), dtype=bool)
    politics_y = np.zeros(len(political_comments), dtype=bool)
    # create dataset
    sports_comments.extend(political_comments)
    y = np.concatenate((sports_y, politics_y))
    corpus_train, corpus_test, y_train, y_test = train_test_split(
        sports_comments, y, test_size=0.2, random_state=7
    )
    print('train data size : {}'.format(len(corpus_train)))
    print('test data size : {}'.format(len(corpus_test)))

    # fit a linear logistic regression model
    vectorizer = TfidfVectorizer(min_df=10)
    X_train = vectorizer.fit_transform(
        corpus_train
    ).toarray()  # sparse also works but Explanation slicing is not yet supported
    X_test = vectorizer.transform(corpus_test).toarray()
    model = sklearn.linear_model.LogisticRegression(penalty="l2", C=0.1)
    model.fit(X_train, y_train)
    print('classifier trained : ')
    print(classification_report(y_test, model.predict(X_test)))

    # get Shapley values
    explainer = shap.Explainer(
        model, X_train, feature_names=vectorizer.get_feature_names_out()
    )
    shap_values = explainer(X_test)
    print('Shapley values generated')
    # Shapley values tell us how each feature affects the prediction of a data point.
    # We could look at the average contribution of a feature across data points 
    # in terms of impact on model output. However we are interested in features 
    # that strongly predict sports or political content.
    # Therefore it perhaps makes sense to look at maximum and minimum values
    print('using min/max Shapley values')
    min_values = np.min(shap_values.values, axis=0)
    max_values = np.max(shap_values.values, axis=0)

    # get feature names from the Tfidf vectorizer
    feature_names = vectorizer.get_feature_names_out()
    # select features by threshold
    # word : min/max shap value
    sports_words = {}
    political_words = {}
    for f in range(len(feature_names)):
        name = feature_names[f]
        if abs(min_values[f]) + args.shapley_threshold < abs(max_values[f]):
            sports_words[name] = max_values[f]
        elif abs(min_values[f]) > abs(max_values[f]) + args.shapley_threshold:
            political_words[name] = min_values[f]
    print('sports words : {}'.format(len(sports_words)))
    print('political words : {}'.format(len(political_words)))

    # filter vocabs
    sports_vocab = {}
    political_vocab = {}
    cachedStopWords = stopwords.words("english")
    for word, value in sports_words.items():
        if word not in cachedStopWords:
            sports_vocab[word] = value
    for word, value in political_words.items():
        if word not in cachedStopWords:
            political_vocab[word] = value

    print('sports words after filtering : {}'.format(len(sports_vocab)))
    print('politcal words after filtering : {}'.format(len(political_vocab)))

    # save vocabs
    with open(args.output_dir+'sports_vocab.json', 'w') as fp:
        json.dump(sports_vocab, fp)
    with open(args.output_dir+'politcal_vocab.json', 'w') as fp:
        json.dump(political_vocab, fp)

    print('done.')

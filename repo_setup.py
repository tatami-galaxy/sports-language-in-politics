import os
import subprocess
import argparse


def main():

    req = [
        'scikit-learn', 'nltk', 'transformers', 'datasets', 'accelerate', 'evaluate', 
        'nltk', 'polars', 'shap', 'matplotlib', 'gdown', 'sentence_transformers', 'editdistance',
        'torchtext', 'plotly', 'flair',
    ]

    parser = argparse.ArgumentParser()

    # directorries
    os.mkdir('data')
    os.mkdir('data/raw')
    os.mkdir('data/interim')
    os.mkdir('data/processed')

    os.mkdir('models')
    os.mkdir('models/cbow')

    # install packages
    for package in req:
        subprocess.run(["pip", "install", package]) 



if __name__ == "__main__":
    main()

 

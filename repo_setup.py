import os
import subprocess
import argparse


def main():

    req = ['scikit-learn', 'transformers', 'datasets', 'accelerate', 'evaluate' 'nltk', 'polars', 'shap', 'matplotlib', 'gdown']

    parser = argparse.ArgumentParser()

    # directorries
    os.mkdir('data')
    os.mkdir('data/raw')
    os.mkdir('data/interim')
    os.mkdir('data/processed')

    os.mkdir('models')

    # install packages
    for package in req:
        subprocess.run(["pip", "install", package]) 



if __name__ == "__main__":
    main()

 
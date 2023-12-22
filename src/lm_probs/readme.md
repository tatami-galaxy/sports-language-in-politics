# generate sports and political vocab from Shapley values
python get_sports_political_vocab.py --output_dir `output directory path` --cloud

# run clm on political or random data
accelerate launch clm.py --data `[politics, random]` --cloud 

# run ngram on political or random data
python ngram.py --data `[politics, random]` --n `n-gram value` --sports_data comments --cloud 
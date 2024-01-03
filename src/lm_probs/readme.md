# generate sports and political vocab from Shapley values
python get_sports_political_vocab.py --output_dir `output directory path` --cloud

# run clm on political or random data
accelerate launch clm.py --data `[politics, random]` --cloud --manual_vocab

# run ngram on political or random data
python ngram.py --data `[politics, random]` --n `n-gram value` --sports_data `[vocab, comments]` --manual_vocab --cloud 

# match metaphor
python metaphor_matching.py --data `[politics, random]` --sample --sample_size 1000 --cloud
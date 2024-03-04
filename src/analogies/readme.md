python mask_probs_v2.py --sample --sample_size 10000 --cloud

# sample size higher than actual sample size 
python mask_probs_v3.py --sample --sample_size 2000 --cloud  

python word2vec_demo.py 

python word2vec.py --epochs 1000 --sample --sample_size 100000 --cloud --stop_words --subs `[list of subs]`

python bert_embeddings.py --sample --sample_size 10000 --cloud --stop_words --subs `[list of subs]`

python word2vec_temporal.py --epochs 1000 --year 2015 --cloud --subs `[list of subs]`

python word2vec_temporal_oct.py --epochs 1000 --year 2015 --subs `[list of subs]` --sample --sample_size 35000 --data_dir /Volumes/PortableSSD/CSS/data/processed --cloud 

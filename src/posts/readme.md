# clustering

python clustering.py --data_dir /Volumes/PortableSSD/CSS/data/processed/ --day_sample 5

# match metaphor

python meta_match_clusters.py --day_sample 2 --cluster_file /Volumes/PortableSSD/CSS/data/processed/post_clusters_5.json

# eval sample

python meta_match_eval.py --data_dir /Volumes/PortableSSD/CSS/data/processed/
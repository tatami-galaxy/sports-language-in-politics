# match metaphor

python metaphor_matching_v2.py --data `[politics, random]` --data_dir `data directory path` --sample --sample_size 10000  --max_meta 30 --cloud

python metaphor_matching_temporal.py --year 2019 --sample --sample_size 3000 --max_meta 30 --cloud

python metaphor_matching_posts.py --sample --sample_size 10000 --data_dir /Volumes/PortableSSD/CSS/data/processed --cloud

python cluster_posts_metaphors.py --day_sample 10
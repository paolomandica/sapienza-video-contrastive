####################################################################################################
# Data and Cache Paths
####################################################################################################

path_to_kinetics="/data_volume/data/kinetics/"
cache_path="/data_volume/data/cached_data/kinetics.pt"

path_to_kinetics_sample="/data_volume/data/kinetics_sample/"
cache_path_sample="/data_volume/data/cached_data/kinetics_sample.pt"


python -W ignore train.py --data-path $path_to_kinetics_sample \
--cache-dataset --cache-path $cache_path_sample \
--frame-aug grid --dropout 0.1 --clip-len 4 --temp 0.05 \
--model-type "scratch" --workers 30 --batch-size 8 --lr 0.0001 \
--epochs 10  --data-parallel 
# --visualize

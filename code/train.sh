####################################################################################################
# Data and Cache Paths
####################################################################################################

path_to_kinetics="/data_volume/data/kinetics/"
cache_path="/data_volume/data/cached_data/kinetics.pt"

path_to_kinetics_sample="/data_volume/data/kinetics_sample/"
cache_path_sample="/data_volume/data/cached_data/kinetics_sample.pt"

path_to_kinetics_ares="/data_volume/data/kinetics/"
cache_path_ares="/data_volume/data/cached_data/kinetics.pt"

path_to_kinetics_sample_ares="/data_volume/data/kinetics_sample/"
cache_path_sample_ares="/data_volume/data/cached_data/kinetics_sample.pt"

####################################################################################################
# Core {Superpixels | Patches | Mix} Model Training
####################################################################################################

python -W ignore train.py --data-path $path_to_kinetics \
--cache-dataset --cache-path $cache_path \
--frame-aug grid --dropout 0.1 --clip-len 4 --temp 0.05 \
--model-type "scratch" --workers 30 --batch-size 20 --lr 0.0001 \
--epochs 10 --data-parallel \
--sp-method slic --num-sp 20 --prob 0 `# NB Changed prob from 0.7` \
--randomise-superpixels --visualize 
--output-dir ./checkpoints/randomise_sp_unnorm/ 

# --resume ""

####################################################################################################
# Teacher-Student Training
####################################################################################################

# python -W ignore train.py --data-path $path_to_kinetics \
# --cache-dataset --cache-path $cache_path \
# --frame-aug grid --dropout 0.1 --clip-len 4 --temp 0.05 \
# --model-type "scratch" --workers 30 --batch-size 8  --lr 0.0001 \
# --data-parallel \
# --teacher-student --alpha-teacher-student 0.5 
# # --visualize

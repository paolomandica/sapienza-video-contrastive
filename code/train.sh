####################################################################################################
# Data and Cache Paths
####################################################################################################

path_to_kinetics="/data_volume/data/kinetics/"
cache_path="/data_volume/data/cached_data/kinetics.pt"

path_to_kinetics_sample="/data_volume/data/kinetics_sample/"
cache_path_sample="/data_volume/data/cached_data/kinetics_sample.pt"

####################################################################################################
# Core {Superpixels | Patches | Mix} Model Training
####################################################################################################

python -W ignore train.py --data-path $path_to_kinetics_sample \
--cache-dataset --cache-path $cache_path_sample \
--frame-aug grid --dropout 0.1 --clip-len 4 --temp 0.05 \
--model-type "scratch" --workers 60 --batch-size 64 --lr 0.0003 \
--epochs 20 \
--sp-method slic --num-sp 49 --prob 0 \
--compactness 200 --data-parallel --output-dir "./checkpoints/sp_unnorm_scratch/"
# --visualize --port 8094 \
# --partial-reload "../pretrained.pth" 
# --randomise-superpixels --data-parallel --port 8095
# --output-dir "./checkpoints/randomise_sp_unnorm/"
# --resume "./checkpoints/randomise_sp_unnorm/checkpoint.pth"

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

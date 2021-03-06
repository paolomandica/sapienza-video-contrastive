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
--model-type "scratch" --workers 20 --batch-size 6 --lr 0.0003 \
--epochs 20 --data-parallel \
--sp-method slic --num-sp 36 --prob 0 \
--compactness 50 \
--dilate-superpixels --dilation-kernel-size 55 # NB --dilation-kernel-shape is 'L1' by default

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

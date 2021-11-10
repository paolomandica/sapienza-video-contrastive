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

python -W ignore train.py --data-path $path_to_kinetics \
--cache-dataset --cache-path $cache_path \
--frame-aug grid --dropout 0.1 --clip-len 4 --temp 0.05 \
--model-type "scratch" --workers 28 --batch-size 20 --lr 0.0001 \
--epochs 10 \
--sp-method slic --num-sp 30 --prob 0.7 `# NB Changed prob from 0.7` \
--data-parallel --visualize \
--resume "./checkpoints/_drop0.1-len4-ftranscrop-fauggrid-optimadam-temp0.05-fdrop0.0-lr0.0001-mlp0-spslic-nsp30-p0.7/checkpoint.pth"

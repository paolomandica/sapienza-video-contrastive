path_to_kinetics="/data_volume/data/kinetics/"
cache_path="/data_volume/data/cached_data/kinetics.pt"

path_to_kinetics_sample="/data_volume/data/kinetics_sample/"

# python -W ignore train.py --data-path $path_to_kinetics \
# --frame-aug grid --dropout 0.1 --clip-len 4 --temp 0.05 \
# --model-type "scratch" --workers 16 --batch-size 4  --lr 0.0001 \
# --data-parallel --cache-dataset 
# # --cache-dataset $cache_path 
# # --visualize

####################################################################################################
# Teacher-Student Training
# NB visualisation toggled; via flag --visualize
####################################################################################################

path_to_kinetics="/data_volume/data/kinetics/"
cache_path="/data_volume/data/cached_data/kinetics.pt"

path_to_kinetics_sample="/data_volume/data/kinetics_sample/"
cache_path_sample="/data_volume/data/cached_data/kinetics_sample.pt"

python -W ignore train.py --data-path $path_to_kinetics_sample \
--cache-dataset --cache-path $cache_path_sample \
--frame-aug grid --dropout 0.1 --clip-len 4 --temp 0.05 \
--model-type "scratch" --workers 30 --batch-size 8  --lr 0.0001 \
--data-parallel \
--alpha-teacher-student 0.5 # --visualize

####################################################################################################
# train.sh Bash script from sapienza-vc-TS (i.e. modifications from video-constrastive)
####################################################################################################

# path_to_kinetics="/data_volume/kinetics-downloader/dataset/train_sample/"
# output_dir="../checkpoints/ts/"
# model_type="scratch"
# cache_path="../cached_data/cached_train_sample.pt"
# # checkpoint="../checkpoints/scratch_10_fh/checkpoint.pth"

# path_to_teacher="../pretrained.pth"

# python -W ignore train.py --data-path $path_to_kinetics --output-dir $output_dir \
# --frame-aug grid --dropout 0.1 --clip-len 10 --temp 0.07 --model-type $model_type \
# --workers 20 --batch-size 16 --lr 0.0001 --epochs 1 \
# --cache-dataset --cache-path $cache_path \
# --data-parallel \
# --path-to-teacher $path_to_teacher 

# # --visualize
# # --resume $checkpoint
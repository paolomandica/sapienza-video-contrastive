path_to_kinetics="/data_volume/data/kinetics/"
output_dir="../checkpoints/pretrained_slic/"
# output_dir="../checkpoints/scratch_sample/"
model_type="scratch"
cache_path="../cached_data/cached_dataset.pt"
# cache_path="../cached_data/cached_sample.pt"
checkpoint="../pretrained.pth"

python -W ignore train.py --data-path $path_to_kinetics --output-dir $output_dir \
--frame-aug none --dropout 0.1 --clip-len 4 --temp 0.07 --model-type $model_type \
--workers 40 --batch-size 68 --lr 0.0003 --epochs 3 \
--cache-dataset --cache-path $cache_path \
--visualize --data-parallel --partial-reload $checkpoint

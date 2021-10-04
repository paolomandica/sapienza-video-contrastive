path_to_kinetics="/data_volume/kinetics-downloader/dataset/train/"
output_dir="../checkpoints/scratch_10_slic/"
# output_dir="../checkpoints/scratch_sample/"
model_type="scratch"
cache_path="../cached_data/cached_10_fs.pt"
# cache_path="../cached_data/cached_sample.pt"
# checkpoint="../checkpoints/scratch_10_fh/checkpoint.pth"

python -W ignore train.py --data-path $path_to_kinetics --output-dir $output_dir \
--frame-aug none --dropout 0.1 --clip-len 10 --temp 0.07 --model-type $model_type \
--workers 20 --batch-size 16 --lr 0.0001 --epochs 10 \
--cache-dataset --cache-path $cache_path \
--visualize --data-parallel
# --resume $checkpoint
path_to_kinetics="/data_volume/kinetics-downloader/dataset/train/"
output_dir="../checkpoints/scratch_10_fh/"
model_type="scratch"
cache_path="../cached_data/cached_10_fs.pt"
# checkpoint="../checkpoints/scratch_10_fh/checkpoint.pth"

python -W ignore train.py --data-path $path_to_kinetics --output-dir $output_dir \
--frame-aug none --dropout 0.3 --clip-len 10 --temp 0.01 --model-type $model_type \
--workers 20 --batch-size 48 --lr 0.0003 --epochs 10 \
--cache-dataset --data-parallel --visualize --cache-path $cache_path
# --resume $checkpoint
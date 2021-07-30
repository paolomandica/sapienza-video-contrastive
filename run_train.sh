# Set up for local use on WSL

# Data Paths
datapath_partial="/mnt/c/Users/anilk/Desktop/sapienza-video-contrastive/runs/kinetics_sample_3_3/"
output_dir="/mnt/c/Users/anilk/Desktop/sapienza-video-contrastive/runs/checkpoints/"

# Model
model_type="scratch"

# Run
python3 -W ignore ./code/train.py --data-path $datapath_partial --output-dir $output_dir \
--dropout 0.1 --clip-len 4 --temp 0.05 --model-type $model_type \
--workers 8 --batch-size 1 --cache-dataset --lr 0.01 --epochs 1 \
--device cpu

# Run with patches

# python3 -W ignore train.py --data-path $datapath_partial --output-dir $output_dir \
# --dropout 0.1 --clip-len 4 --temp 0.05 --model-type $model_type \
# --workers 20 --batch-size 6 --cache-dataset --lr 0.01 --epochs 2 \
# --frame-aug grid
# Set up for local use on WSL

# Data Paths
datapath_partial="/mnt/c/Users/anilk/Desktop/sapienza-video-contrastive/runs/kinetics_sample_3_3/"
output_dir="/mnt/c/Users/anilk/Desktop/sapienza-video-contrastive/runs/checkpoints/"

# Model
model_type="scratch"

python -W ignore ./code/train.py --data-path $path_to_kinetics --output-dir $output_dir \
--frame-aug grid --dropout 0.1 --clip-len 4 --temp 0.05 --model-type $model_type \
--workers 20 --batch-size 20 --cache-dataset --logs-dir $logs_dir \
--lr 0.0001 --epochs 10 --data-parallel

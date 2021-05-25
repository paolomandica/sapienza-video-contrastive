#!/bin/bash

datapath="/home/paolo/dev/data_science/th_proj/kinetics400/"
datapath_partial="/home/paolo/dev/data_science/th_proj/kinetics400_partial/"
output_dir="/home/paolo/dev/data_science/th_proj/checkpoints/resnet_3d_18/"
model_type="scratch"

python ./sapienza-video-contrastive/code/train.py --data-path $datapath_partial --output-dir $output_dir \
--frame-aug grid --dropout 0.1 --clip-len 10 --temp 0.05 \
--model-type $model_type --workers 16 --batch-size 10 \
--cache-dataset --data-parallel --lr 0.0001

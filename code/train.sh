path_to_kinetics="/data_volume/data/kinetics/"
checkpoint="../checkpoints/scratch/checkpoint.pth"

python -W ignore train.py --data-path $path_to_kinetics \
--frame-aug grid --dropout 0.1 --clip-len 4 --temp 0.05 \
--model-type scratch --workers 20 --batch-size 100 \
--lr 0.0003 --epochs 10 --data-parallel --cache-dataset \
--visualize
# --resume $checkpoint 

cd code

path_to_kinetics="../kinetics/"
output_dir="../checkpoints/scratch/"
model_type="scratch"

python -W ignore train.py --data-path $path_to_kinetics --output-dir $output_dir \
--frame-aug grid --dropout 0.1 --clip-len 4 --temp 0.05 --model-type $model_type \
--workers 32 --batch-size 32 --lr 0.0003 --epochs 15 \
--visualize --cache-dataset --data-parallel

path_to_kinetics="../kinetics_sample/"
output_dir="../checkpoints/scratch_sample/"
model_type="scratch"

python -W ignore train.py --data-path $path_to_kinetics --output-dir $output_dir \
--frame-aug grid --dropout 0.1 --clip-len 4 --temp 0.05 --model-type $model_type \
--workers 20 --batch-size 8 --lr 0.0001 --epochs 1 \
--visualize --cache-dataset --data-parallel

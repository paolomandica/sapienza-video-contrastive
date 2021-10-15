path_to_kinetics="/data_volume/data/kinetics/"
cache_path="/data_volume/data/cached_data/kinetics.pt"

python -W ignore train.py --data-path $path_to_kinetics \
--frame-aug none --dropout 0.1 --clip-len 4 --temp 0.05 --model-type scratch \
--workers 30 --batch-size 52 --lr 0.0003 --epochs 10 \
--sp-method random --num-sp 30 --prob 0.7 \
--cache-dataset --cache-path $cache_path --data-parallel --visualize

# --output-dir ./checkpoints/sample/ --resume ""

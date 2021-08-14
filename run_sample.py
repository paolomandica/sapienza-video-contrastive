# Install accelerate if not already

module_name="accelerate"

if 
    python -c "import $module_name" &> /dev/null
then
    echo "$module_name already installed."
else
    echo "$module_name not installed."
    echo "Running pip install $module_name"
    pip install $module_name
fi

# Training 

path_to_kinetics="kinetics_sample"
output_dir="checkpoints/scratch_sample/"
logs_dir="logs_sample/"

model_type="scratch"

python -W ignore ./code/train.py --data-path $path_to_kinetics --output-dir $output_dir \
--frame-aug grid --dropout 0.1 --clip-len 4 --temp 0.05 --model-type $model_type \
--workers 20 --batch-size 20 --cache-dataset --logs-dir $logs_dir \
--lr 0.0001 --epochs 1 --data-parallel
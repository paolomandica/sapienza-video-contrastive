#!/bin/bash

python ./code/test.py --filelist ./code/eval/davis_vallist.txt \
--model-type scratch --resume ./pretrained.pth --save-path ./eval_results_scratch/ \
--topk 10 --videoLen 5 --radius 12  --temperature 0.05  --cropSize -1

### Convert
python eval/convert_davis.py --in_folder ../results/ --out_folder ../convert_results/ --dataset /home/paolo/dev/data_science/th_proj/davis_val/

### Compute metrics
python /home/paolo/dev/data_science/th_proj/davis2017-evaluation/evaluation_method.py \
--task semi-supervised   --results_path ../convert_results/ --set val \
--davis_path /home/paolo/dev/data_science/th_proj/davis_val/

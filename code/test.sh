vallist="/data_volume/sapienza-video-contrastive/code/eval/davis_vallist.txt"
model_type="scratch"
# checkpoint="../checkpoints/scratch_10_slic/checkpoint.pth"
checkpoint="../pretrained.pth"
savepath="../results/pretrained/"
outpath="../results/pretrained_converted/"
dataset="/data_volume/data/davis_val/"

python test.py --filelist $vallist --model-type $model_type \
--resume $checkpoint --save-path $savepath \
--topk 10 --videoLen 20 --radius 12  --temperature 0.05  --cropSize -1

### Convert
python ./eval/convert_davis.py --in_folder $savepath \
--out_folder $outpath --dataset $dataset

### Compute metrics
python /data_volume/davis2017-evaluation/evaluation_method.py \
--task semi-supervised  --results_path $outpath \
--set val --davis_path $dataset
vallist="/data_volume/sapienza-video-contrastive/code/eval/davis_vallist.txt"
checkpoint="./checkpoints/_drop0.1-len4-ftranscrop-fauggrid-optimadam-temp0.05-fdrop0.0-lr0.0001-mlp0-spslic-nsp30-p0.7/model_9.pth"
savepath="../results/masks_grid_slic/"
outpath="../results/converted_grid_slic/"
dataset="/data_volume/data/davis_val/"

# rm -rf ../results/*

python test.py --filelist $vallist --model-type scratch \
--resume $checkpoint --save-path $savepath \
--topk 10 --videoLen 20 --radius 12  --temperature 0.05  --cropSize -1

### Convert
python ./eval/convert_davis.py --in_folder $savepath \
--out_folder $outpath --dataset $dataset

### Compute metrics
python /data_volume/davis2017-evaluation/evaluation_method.py \
--task semi-supervised  --results_path $outpath \
--set val --davis_path $dataset

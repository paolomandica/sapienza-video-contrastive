from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import torch
import pdb
import os
import subprocess
import utils
import re
import pickle
import sys
import imageio

from pathlib import Path
from tqdm.auto import tqdm
from time import time
from datetime import timedelta
from multiprocessing import Pool
from functools import partial
from multiprocessing import current_process
from collections import defaultdict
from joblib import Parallel, delayed

from utils import arguments
from data.kinetics import Kinetics400

from torch import nn
from torchvision import models
from torchvision.io import read_video
from torchvision.datasets.video_utils import VideoClips
from pathlib import Path
from model import CRW
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset

from fast_slic import Slic
import numpy as np
import multiprocessing as mp

args = arguments.train_args()
# model_ft = models.resnet18(pretrained=True)
# model = nn.Sequential(*list(model_ft.children())[:-2])
# encoder = model.to('cuda')
# model = CRW(args).to('cuda')

transform_train = utils.augs.get_train_transforms(args)

clip_len = 4
frame_skip = 8


def make_sp_masks_clip(video, output_path):
    slic = Slic(num_components=50, compactness=30)
    sp_tensor_time = []

    for t in range(video.shape[1]):

        img = video[:, t, :, :]
        img = img.permute(1, 2, 0).cpu().numpy()
        img = img.astype(dtype='uint8', order='C')
        segments_slic = slic.iterate(img)

        sp_tensor_time.append(torch.from_numpy(segments_slic))

    torch.save(torch.stack(sp_tensor_time), output_path+'.pt')


def makedirs(dir1, dir2):
    subdirs = [f.name for f in os.scandir(dir1) if f.is_dir()]
    for subdir in subdirs:
        Path(os.path.join(dir2, subdir)).mkdir(exist_ok=True)
    return subdirs


def generate_sp_mask(video_clips, video_list, idx):

    output_path = video_list[idx].replace("train_256", "masks")

    if os.path.isfile(output_path[:-4]+'.pt'):
        return
    else:
        video, _, _, _ = video_clips.get_clip(idx)

        video = video.permute(3, 0, 1, 2)  # Shape (C, T  H, W)
        make_sp_masks_clip(video, output_path[:-4])

    # if idx % 10 == 0:
    #     print("clip %d / %d" % (idx, len(video_clips)))


def make_sp_masks(input_dir, output_dir, workers=1):
    print("\n========= Starting segmentation process =========\n")
    start = time()
    # pdb.set_trace()

    subdirs = makedirs(input_dir, output_dir)
    tot_dirs = len(subdirs)

    classes = list(sorted(list_dir(input_dir)))
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    extensions = ('mp4',)

    samples = make_dataset(input_dir, class_to_idx,
                           extensions, is_valid_file=None)

    video_list = [sample[0] for sample in samples]

    import hashlib
    h = hashlib.sha1(input_dir.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision",
                              "datasets", "kinetics", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)

    print("Loading dataset_train from {}".format(cache_path))
    dataset, _ = torch.load(cache_path)
    cached = dict(video_paths=dataset.video_clips.video_paths,
                  video_fps=dataset.video_clips.video_fps,
                  video_pts=dataset.video_clips.video_pts)

    video_clips = VideoClips(
        video_list,
        clip_length_in_frames=4,
        frames_between_clips=1,
        frame_rate=8,
        _precomputed_metadata=cached)

    indices = [idx for idx in range(len(video_clips))]

    for idx in tqdm(indices):
        generate_sp_mask(video_clips, video_list, idx)

    # func = partial(generate_sp_mask, video_clips, video_list)

    # Parallel(n_jobs=20, require='sharedmem')(delayed(func)(i) for i in tqdm(indices))

    end = time()
    tot_time = str(timedelta(seconds=round(end-start)))
    print("\n=========== Completed in %s ===========\n" % (tot_time))


if __name__ == "__main__":
    # input_path is the path to the folder containing the subfolders (applauding, jogging, ...) with the clips
    input_path = "../kinetics/train_256"
    # input_path = /data_volume/kinetics-downloader/dataset/train/
    # the output_path is the path to the output folder where the new subfolders will be stored
    # if the output folder doesn't exists it will be automatically created with all the parent folders
    output_path = "../kinetics/masks/"
    # output_path = "/data_volume/kinetics-downloader/dataset/train_256/"
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # create superpixels segmentation masks for each videoclip
    make_sp_masks(input_path, output_path, workers=1)

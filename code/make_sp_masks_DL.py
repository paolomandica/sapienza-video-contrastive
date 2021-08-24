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




import torchvision.datasets.video_utils

from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset


from torch.utils.data.dataloader import default_collate
from torchvision.datasets.samplers.clip_sampler import RandomClipSampler, UniformClipSampler


import numpy as np
import torch
import pdb
import os

from fast_slic import Slic

from time import time
from pathlib import Path


class Kinetics400(VisionDataset):
    """
    `Kinetics-400 <https://deepmind.com/research/open-source/open-source-datasets/kinetics/>`_
    dataset.

    Kinetics-400 is an action recognition video dataset.
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``.

    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.

    Internally, it uses a VideoClips object to handle clip creation.

    Args:
        root (string): Root directory of the Kinetics-400 Dataset.
        frames_per_clip (int): number of frames in a clip
        step_between_clips (int): number of frames between each clip
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.

    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
            and `L` is the number of points
        label (int): class of the video clip
    """

    def __init__(self, root, frames_per_clip, step_between_clips=1, frame_rate=None, extensions=('mp4',), transform=None, cached=None, _precomputed_metadata=None):
        super(Kinetics400, self).__init__(root)
        # extensions = extensions

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)

        new_video_list = []

        for video_path, strange_idx in self.samples:
            mask_path = video_path.replace("train_256", "masks").replace(".mp4", ".pt")
            if not os.path.isfile(mask_path):
                new_video_list.append(video_path)


        #pdb.set_trace()


        self.classes = classes
        video_list = [x[0] for x in self.samples]

        if _precomputed_metadata is None:
            # Se non si usa la cache legge quali sono fatti e quali no
            self.video_list = new_video_list
        else:
            # Altrimenti se è già stato elaborato VideoClips() va usata la lista originale di video_list
            self.video_list = video_list

        class_readed = []
        for vv in self.video_list:
            cc = vv.split('/')[-2]
            class_readed.append(cc)
        print(len(set(class_readed)))
        #pdb.set_trace()

        # print(len(video_list), frames_per_clip, step_between_clips, frame_rate)
        self.video_clips = VideoClips(
            self.video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
        )
        # print(self.video_clips.num_clips())
        #self.transform = transform

        self.class_checked = set([])
        self.len_class_checked = 0

        output_path = root.replace("train_256", "masks")
        Path(output_path).mkdir(parents=True, exist_ok=True)
        makedirs(root, output_path)


    def __len__(self):
        return self.video_clips.num_clips()

    def _len_(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        success = False
        while not success:
            try:
                video, audio, info, video_idx = self.video_clips.get_clip(idx)

                self.class_checked.add(self.video_list[video_idx].split('/')[-2])

                if len(self.class_checked) > self.len_class_checked:
                    print(len(self.class_checked))
                    self.len_class_checked = len(self.class_checked)

                # save sp mask
                output_path = self.video_list[video_idx].replace("train_256", "masks")
                if not os.path.isfile(output_path[:-4]+'__'+str(idx)+'.pt'):
                    # Shape (C, T  H, W)
                    make_sp_masks_clip(video.permute(3, 0, 1, 2), output_path[:-4]+'__'+str(idx)+'.pt')

                success = True
            except Exception as e:
                print('skipped idx', idx)
                print("Error: ", e)
                idx = np.random.randint(self._len_())

        # label = self.samples[video_idx][1]

        # sp_mask_root = self.samples[video_idx][0].replace('train_256', 'masks')[:-4]
        # sp_mask = torch.load(sp_mask_root + '.pt')
        # sp_mask = torch.randn((50, 4, 256, 256))

        if self.transform is not None:
            video = self.transform(video)

        return video, audio


def makedirs(dir1, dir2):
    subdirs = [f.name for f in os.scandir(dir1) if f.is_dir()]
    for subdir in subdirs:
        Path(os.path.join(dir2, subdir)).mkdir(exist_ok=True)


def make_sp_masks_clip(video, output_path):
    slic = Slic(num_components=50, compactness=30)
    sp_tensor_time = []

    for t in range(video.shape[1]):

        img = video[:, t, :, :]
        img = img.permute(1, 2, 0).cpu().numpy()
        img = img.astype(dtype='uint8', order='C')
        segments_slic = slic.iterate(img)

        sp_tensor_time.append(torch.from_numpy(segments_slic))

    torch.save(torch.stack(sp_tensor_time), output_path)




def make_data_sampler(is_train, dataset, clips_per_video):
        torch.manual_seed(0)
        if hasattr(dataset, 'video_clips'):
            _sampler = UniformClipSampler # RandomClipSampler
            return _sampler(dataset.video_clips, clips_per_video)
        else:
            return torch.utils.data.sampler.RandomSampler(dataset) if is_train else None



def collate_fn(batch):
    # remove audio from the batch
    batch = [(d[0], d[1]) for d in batch]
    return default_collate(batch)


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision",
                              "datasets", "kinetics", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path





if __name__ == "__main__":
    # input_path is the path to the folder containing the subfolders (applauding, jogging, ...) with the clips
    input_path = "../kinetics/train_256"
    # input_path = /data_volume/kinetics-downloader/dataset/train/
    # the output_path is the path to the output folder where the new subfolders will be stored
    # if the output folder doesn't exists it will be automatically created with all the parent folders
    output_path = "../kinetics/masks/"
    # output_path = "/data_volume/kinetics-downloader/dataset/train_256/"
    Path(output_path).mkdir(parents=True, exist_ok=True)


    clip_len = 4
    step_between_clips = 1
    frame_skip = 8


    cache_path = _get_cache_path(input_path)

    if os.path.exists(cache_path):
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
        cached = dict(video_paths=dataset.video_clips.video_paths,
                        video_fps=dataset.video_clips.video_fps,
                        video_pts=dataset.video_clips.video_pts)
    else:
        cached = None


    dataset = Kinetics400(
                root=input_path,
                frames_per_clip=clip_len,
                step_between_clips=step_between_clips,
                transform=None,
                extensions=('mp4'),
                frame_rate=frame_skip,
                _precomputed_metadata=cached)

    clips_per_video = 5


    train_sampler = make_data_sampler(True, dataset, clips_per_video=clips_per_video)


    batch_size = 20
    workers = 20

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,  # shuffle=not args.fast_test,
        sampler=train_sampler, num_workers=workers//2,
        pin_memory=True, collate_fn=collate_fn)

    for bb in tqdm(data_loader):
        pass


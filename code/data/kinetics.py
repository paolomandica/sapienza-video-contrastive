import torchvision.datasets.video_utils

from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset

import numpy as np
import torch
import pdb
import os
import random

from fast_slic import Slic
from skimage.segmentation import felzenszwalb

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

    def __init__(self, root, frames_per_clip, step_between_clips=1, frame_rate=None,
                 extensions=('mp4',), transform=None, cached=None, _precomputed_metadata=None):
        super(Kinetics400, self).__init__(root)
        extensions = extensions

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        self.samples = make_dataset(
            self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        self.video_list = video_list

        self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
        )

        self.transform = transform

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        success = False
        while not success:
            try:
                video, audio, info, video_idx = self.video_clips.get_clip(idx)
                success = True
            except Exception as e:
                print('skipped idx', idx)
                print("Error: ", e)
                idx = np.random.randint(self.__len__())

        label = self.samples[video_idx][1]

        # if self.transform is not None:
        video = self.transform(video)
        # print("******* VIDEO.SHAPE = ", video[0].shape)

        def compute_sp_slic(img):
            slic = Slic(num_components=10, compactness=30)
            img = img.astype(dtype='uint8', order='C')
            seg = slic.iterate(img).astype(dtype='uint8')
            return seg

        def compute_sp_FH(img):
            seg = felzenszwalb(img, scale=5000, sigma=0.5, min_size=1000)
            return seg

        def compute_mask(video):
            sp_tensor_time = []

            # select random method for SP computation
            # methods = ['slic', 'fh']
            # rnd_method = random.choice(methods)
            # if rnd_method == 'slic':
            #     compute_sp = compute_sp_slic
            # elif rnd_method == 'fh':
            #     compute_sp = compute_sp_FH
            compute_sp = compute_sp_slic

            for t in range(video.shape[0]):
                img = video[t, :, :, :]
                img = img.permute(1, 2, 0).cpu().numpy()
                segments = compute_sp(img)
                sp_tensor_time.append(torch.from_numpy(segments))

            mask = torch.stack(sp_tensor_time)
            mask = mask.unsqueeze(3).repeat(1, 1, 1, 3)
            mask = mask.permute(0, 3, 1, 2)

            return mask.numpy()

        # compute mask
        video_mask = compute_mask(torch.Tensor(video[0]))
        # print("******* MASK.SHAPE = ", video_mask.shape)

        return video, video_mask, audio, label

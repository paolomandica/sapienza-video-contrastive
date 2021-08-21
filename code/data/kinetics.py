import torchvision.datasets.video_utils

from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset

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

    def _init_(self, root, frames_per_clip, step_between_clips=1, frame_rate=None, extensions=('mp4',), transform=None, cached=None, _precomputed_metadata=None):
        super(Kinetics400, self)._init_(root)
        extensions = extensions

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        self.samples = make_dataset(
            self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        self.video_list = video_list

        print(len(video_list), frames_per_clip, step_between_clips, frame_rate)
        self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
        )
        print(self.video_clips.num_clips())
        self.transform = transform

        output_path = root.replace("train_256", "masks")
        Path(output_path).mkdir(parents=True, exist_ok=True)
        makedirs(root, output_path)

    def _len_(self):
        return self.video_clips.num_clips()

    def _getitem_(self, idx):
        success = False
        while not success:
            try:
                video, audio, info, video_idx = self.video_clips.get_clip(idx)

                print("VIDEO IDX = ", video_idx)

                # save sp mask
                output_path = self.video_list[video_idx].replace(
                    "train_256", "masks")
                if not os.path.isfile(output_path[:-4]+'.pt'):
                    # Shape (C, T  H, W)
                    make_sp_masks_clip(video.permute(
                        3, 0, 1, 2), output_path[:-4])

                success = True
            except Exception as e:
                print('skipped idx', idx)
                print("Error: ", e)
                idx = np.random.randint(self._len_())

        label = self.samples[video_idx][1]

        sp_mask_root = self.samples[video_idx][0].replace('train_256', 'masks')[
            :-4]
        # sp_mask = torch.load(sp_mask_root + '.pt')
        sp_mask = torch.randn((50, 4, 256, 256))

        if self.transform is not None:
            video = self.transform(video)

        return video, sp_mask, audio, label


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

    torch.save(torch.stack(sp_tensor_time), output_path+'.pt')
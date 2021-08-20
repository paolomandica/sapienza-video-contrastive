import torch
import pdb

from utils import arguments

from torchvision import models
from torchvision.datasets.video_utils import VideoClips
from pathlib import Path
from model import CRW


frames_per_clip = 8
step_between_clips = 1
frame_rate = 8

args = arguments.train_args()
model = CRW(args).to('cuda')


def make_sp_masks_clip(video, output_path):
    q, _ = model.extract_sp_feat(video, video_feat)
    torch.save(q, output_path)


def make_sp_masks_dir(input_dir, output_dir, subdir):
    print("===== Segmenting clips in %s" %
          (os.path.join(input_dir, subdir)))
    clips = os.listdir(os.path.join(input_dir, subdir))
    video_clips = VideoClips(
        clips,
        frames_per_clip,
        step_between_clips,
        frame_rate,
        _precomputed_metadata=None,)

    current = current_process()
    pos = current._identity[0]-1 if len(current._identity) > 0 else None
    for idx in tqdm(range(len(video_clips)), position=pos):

        pdb.set_trace()

        # load video clip
        clip = clips[idx]
        video, _, _, _ = video_clips.get_clip(idx)
        video = video.unsqueeze(0).permute(0, 4, 1, 2, 3)

        input_path = os.path.join(input_dir, subdir, clip)
        output_path = os.path.join(output_dir, subdir, clip)
        if not os.path.exists(output_path):
            try:
                make_sp_masks_clip(video, output_path)
            except:
                print("\nFailed: %s" % (output_path))
                with open(os.path.join(output_dir, "failed_clips.txt"), "a") as txt:
                    txt.write(output_path + "\n")
    print("\n")


def makedirs(dir1, dir2):
    subdirs = [f.name for f in os.scandir(dir1) if f.is_dir()]
    for subdir in subdirs:
        Path(os.path.join(dir2, subdir)).mkdir(exist_ok=True)
    return subdirs


def make_sp_masks(input_dir, output_dir, workers=1):
    print("\n========= Starting segmentation process =========\n")
    start = time()

    subdirs = makedirs(input_dir, output_dir)
    tot_dirs = len(subdirs)

    with Pool(workers) as pool:
        pool.map()


if __name__ == "__main__":
    # input_path is the path to the folder containing the subfolders (applauding, jogging, ...) with the clips
    input_path = "../kinetics400_partial/train_256/"
    # input_path = /data_volume/kinetics-downloader/dataset/train/
    # the output_path is the path to the output folder where the new subfolders will be stored
    # if the output folder doesn't exists it will be automatically created with all the parent folders
    output_path = "../kinetics400_partial/masks/"
    # output_path = "/data_volume/kinetics-downloader/dataset/train_256/"
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # create superpixels segmentation masks for each videoclip
    make_sp_masks(input_path, output_path, workers=1)

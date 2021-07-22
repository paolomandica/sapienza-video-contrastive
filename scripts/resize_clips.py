import os
# import moviepy.editor as mp
import subprocess

from pathlib import Path
from tqdm import tqdm
from time import time
from datetime import timedelta
from multiprocessing import Pool
from functools import partial
from multiprocessing import current_process


def resize_clip_mpy(input_path, output_path, size=256, logger="bar", threads=6):
    clip = mp.VideoFileClip(input_path, audio=False)
    clip_resized = clip.resize(newsize=(size, size))
    clip_resized.write_videofile(output_path, logger=logger, threads=threads)


def resize_clip(input_path, output_path, size=256, logger=None, threads=None):
    size = str(size)+':'+str(size)
    subprocess.call(
        ['ffmpeg', '-y', '-hwaccel', 'cuda',
         '-i', input_path, '-vf', 'scale='+size, '-an',
         '-c:v', 'libopenh264', output_path])


def resize_dir(input_dir, output_dir, size, subdir):
    print("===== Resizing clips in %s" %
          (os.path.join(input_dir, subdir)))
    clips = os.listdir(os.path.join(input_dir, subdir))
    current = current_process()
    pos = current._identity[0]-1 if len(current._identity) > 0 else None
    for clip in tqdm(clips, position=pos):
        input_path = os.path.join(input_dir, subdir, clip)
        output_path = os.path.join(output_dir, subdir, clip)
        if not os.path.exists(output_path):
            try:
                resize_clip(input_path, output_path, size=size, logger=None)
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


def resize_multiple_clips(input_dir, output_dir, size=256, workers=1):

    print("\n=========== Starting resizing ===========\n")

    start = time()

    subdirs = makedirs(input_dir, output_dir)
    tot_dirs = len(subdirs)

    if workers > 1:
        n_workers = workers if tot_dirs > workers else tot_dirs
        func = partial(resize_dir, input_dir, output_dir, size)

        with Pool(n_workers) as pool:
            pool.map(func, subdirs)

    else:
        for subdir in subdirs:
            print("%d folders left\n" % (tot_dirs))
            tot_dirs -= 1
            resize_dir(input_dir, output_dir, size, subdir)

    end = time()
    tot_time = str(timedelta(seconds=round(end-start)))
    print("\n=========== Completed in %s ===========\n" % (tot_time))


if __name__ == "__main__":
    # input_path is the path to the folder containing the subfolders (applauding, jogging, ...) with the clips
    input_path = "C:/Users/paolo/dev/data_science/th_proj/kinetics/train/"
    # input_path = /data_volume/kinetics-downloader/dataset/train/
    # the output_path is the path to the output folder where the new subfolders will be stored
    # if the output folder doesn't exists it will be automatically created with all the parent folders
    output_path = "C:/Users/paolo/dev/data_science/th_proj/kinetics/train_256/"
    # output_path = "/data_volume/kinetics-downloader/dataset/train_256/"

    Path(output_path).mkdir(parents=True, exist_ok=True)

    resize_multiple_clips(input_path, output_path, workers=1)

import os
import moviepy.editor as mp

from pathlib import Path
from tqdm import tqdm
from time import time
from datetime import timedelta


def resize_clip(input_path, output_path, size=256, logger="bar", threads=6):
    clip = mp.VideoFileClip(input_path, audio=False)
    clip_resized = clip.resize(newsize=(size, size))
    clip_resized.write_videofile(output_path, logger=logger, threads=threads)


def makedirs(dir1, dir2):
    subdirs = [f.name for f in os.scandir(dir1) if f.is_dir()]
    for subdir in subdirs:
        Path(os.path.join(dir2, subdir)).mkdir(exist_ok=True)
    return subdirs


def resize_multiple_clips(input_dir, output_dir, size=256):
    subdirs = makedirs(input_dir, output_dir)

    print("\n=========== Starting resizing ===========\n")

    start = time()
    tot_dirs = len(subdirs)
    for subdir in subdirs:
        print("%d folders left\n" % (tot_dirs))
        tot_dirs -= 1

        print("===== Resizing clips in %s" %
              (os.path.join(input_dir, subdir)))
        clips = os.listdir(os.path.join(input_dir, subdir))
        for clip in tqdm(clips):
            input_path = os.path.join(input_dir, subdir, clip)
            output_path = os.path.join(output_dir, subdir, clip)
            if not os.path.exists(output_path):
                try:
                    resize_clip(input_path, output_path, size, None)
                except:
                    print("\nFailed: %s" % (output_path))
                    with open(os.path.join(output_dir, "failed_clips.txt"), "a") as txt:
                        txt.write(output_path + "\n")
        print("\n")

    end = time()
    tot_time = str(timedelta(seconds=round(end-start)))
    print("\n=========== Completed in %s ===========\n" % (tot_time))


if __name__ == "__main__":
    # input_path is the path to the folder containing the subfolders (applauding, jogging, ...) with the clips
    input_path = "/home/luca/Scrivania/Panasonic/optical_flow/RAFT/videos"
    # the output_path is the path to the output folder where the new subfolders will be stored
    # if the output folder doesn't exists it will be automatically created with all the parent folders
    output_path = "/home/luca/Scrivania/Panasonic/optical_flow/RAFT/videos_256"

    Path(output_path).mkdir(parents=True, exist_ok=True)

    resize_multiple_clips(input_path, output_path)

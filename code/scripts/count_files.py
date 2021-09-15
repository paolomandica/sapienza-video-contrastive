import os
import pdb
import pathlib

# root = "/data_volume/kinetics-downloader/dataset/train"
root_res = "/data_volume/sapienza-video-contrastive/kinetics/masks_slic_10_30"
root = "/data_volume/sapienza-video-contrastive/kinetics/train_256"


def get_files_set(root):
    dirs = os.listdir(root)
    files = set()
    for d in dirs:
        for f in os.listdir(os.path.join(root, d)):
            files.add(os.path.join(d, f))
            # files.add(f)
    return files


files_root = get_files_set(root)
files_root_res = get_files_set(root_res)

# diff = files_root_res.difference(files_root)
diff = files_root.difference(files_root_res)

# classes = set()
# for f in diff:
#     h, t = os.path.split(f)
#     classes.add(h)

print()
print("len(files_root) =", len(files_root))
print("len(files_root_res) =", len(files_root_res))
print("len(diff) =", len(diff))
# print("len(classes) =", len(classes))
print()

for f in diff:
    filepath = os.path.join(root, f)
    os.remove(filepath)

# pdb.set_trace()

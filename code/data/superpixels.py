import numpy as np
import torch

from fast_slic import Slic
from skimage.segmentation import felzenszwalb


def compute_sp_slic(img, num_components):
    slic = Slic(num_components=num_components, compactness=30)
    img = img.astype(dtype="uint8", order="C")
    seg = slic.iterate(img).astype(dtype="uint8")
    return seg


def compute_sp_FH(img):
    seg = felzenszwalb(img, scale=600, sigma=0.5, min_size=400)
    return seg


def compute_mask(video, sp_method, num_components, p):
    sp_tensor_time = []

    if sp_method == "random":
        # select random method for SP computation
        methods = ["slic", "fh"]
        method = np.random.choice(methods, 1, p=[p, 1 - p])
    else:
        method = sp_method

    for t in range(video.shape[0]):
        img = video[t, :, :, :]
        img = img.permute(1, 2, 0).cpu().numpy()
        if method == "slic":
            segments = compute_sp_slic(img, num_components)
        elif method == "fh":
            segments = compute_sp_FH(img)
        sp_tensor_time.append(torch.from_numpy(segments))

    mask = torch.stack(sp_tensor_time)
    mask = mask.unsqueeze(3).repeat(1, 1, 1, 3)
    mask = mask.permute(0, 3, 1, 2)

    return mask.numpy()
import numpy as np
import torch
import cv2

from fast_slic import Slic
from skimage.segmentation import felzenszwalb


def compute_sp_slic(img, num_components, compactness):
    slic = Slic(num_components=num_components, compactness=compactness)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img = img.astype(dtype="uint8", order="C")
    seg = slic.iterate(img).astype(dtype="uint8")
    return seg


def compute_sp_FH(img):
    seg = felzenszwalb(img, scale=600, sigma=0.5, min_size=400)
    return seg


def compute_mask(video, sp_method, num_components, p, randomise_superpixels, randomise_superpixels_range, compactness):
    sp_tensor_time = []

    if sp_method == "random":
        # select random method for SP computation
        methods = ["slic", "fh"]
        method = np.random.choice(methods, 1, p=[p, 1 - p])
    else:
        method = sp_method

    if randomise_superpixels:
        # Randomise the (max) number of segments in each frame over time
        for t in range(video.shape[0]):
            img = video[t, :, :, :]
            img = img.permute(1, 2, 0).cpu().numpy()
            if method == "slic":
                low, high = num_components - randomise_superpixels_range//2, num_components + \
                    randomise_superpixels_range//2
                segments = compute_sp_slic(img, torch.randint(
                    low=low, high=high, size=(1,)).item(), compactness)
            elif method == "fh":
                segments = compute_sp_FH(img)
            sp_tensor_time.append(torch.from_numpy(segments))
    else:
        for t in range(video.shape[0]):
            img = video[t, :, :, :]
            img = img.permute(1, 2, 0).cpu().numpy()
            if method == "slic":
                segments = compute_sp_slic(img, num_components, compactness)
            elif method == "fh":
                segments = compute_sp_FH(img)
            sp_tensor_time.append(torch.from_numpy(segments))

    mask = torch.stack(sp_tensor_time)
    mask = mask.unsqueeze(3).repeat(1, 1, 1, 3)
    mask = mask.permute(0, 3, 1, 2)

    return mask.numpy()

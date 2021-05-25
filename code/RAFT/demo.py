import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cpu'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()
    


    
def viz_2(flow_low, flow_up):
    print("Shape_1:", flow_low.shape, flow_up.shape)
    flo_d = flow_low[0].permute(1,2,0).cpu().numpy()
    flo_u = flow_up[0].permute(1,2,0).cpu().numpy()
    print("Shape_2:", flo_d.shape, flo_u.shape)
    flo_d = flow_viz.flow_to_image(flo_d)
    flo_u = flow_viz.flow_to_image(flo_u)
    print("Shape_3:", flo_d.shape, flo_u.shape)
    img_flo = np.concatenate([flo_d, flo_u], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()




def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=torch.device(DEVICE)))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            
            print("Pre-padding", image1.shape, image2.shape)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            
            print("Post-padding", image1.shape, image2.shape)
            print(type(image1), type(image2))

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            
            #viz(image1, flow_up)
            #viz_2(flow_low, flow_up)
            print(flow_up.shape)
            print(flow_low.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)

import numpy as np
import cv2
import skimage
import argparse
import scipy.spatial

from torchvision import transforms
from PIL import Image

import os

import sys
import numpy
np.set_printoptions(threshold=sys.maxsize)



def resize_fun(frame):

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256,256)),
        #transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    frame = transform(frame).permute(1, 2, 0).numpy()
    return (frame * 255).astype(np.uint8)



def compute_adjacency(p0, p1):

    dist_mat = scipy.spatial.distance.cdist(p0.squeeze(1), p1.squeeze(1))
    # print(p0.squeeze(1))
    # print(p1.squeeze(1))
    # print()
    edges = np.concatenate( (np.arange(p0.shape[0]).reshape(-1, 1), dist_mat.argmin(-1).reshape(-1, 1)),  1)
    adj_mat = np.zeros((49,49))
    for j in range(edges.shape[0]):
        ii,jj=edges[j,:]
        adj_mat[ii,jj]=1

    return adj_mat



def make_dir(path):

    try:
        os.mkdir(path)
    except OSError:
        pass




parser = argparse.ArgumentParser()
parser.add_argument('--video-path', default="./videos/video_prova/0hsa-fjf_Wc.mp4", type=str)
args = parser.parse_args()

save_path = './opt_flow/' + args.video_path.split('/')[-1].split('.')[0]
make_dir(save_path)
count = 0

cap = cv2.VideoCapture(args.video_path)


# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))




ret, old_frame = cap.read()
old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB)
old_frame_resized = resize_fun(old_frame)
old_gray = cv2.cvtColor(old_frame_resized, cv2.COLOR_RGB2GRAY).reshape((256, 256))

'''
im = Image.fromarray(old_frame_resized)
im.save("your_file.jpeg")
'''

step = 32
grid_y, grid_x = np.mgrid[step:old_gray.shape[0]:step, step:old_gray.shape[1]:step]
p0 = np.stack((grid_x.flatten(), grid_y.flatten()),axis=1).astype(np.float32)
p0 = np.expand_dims(p0, 1)


while(cap.isOpened()):

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame_resized)

    ret, new_frame = cap.read()
    # cv2 read and convert images to BGR
    new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
    # apply resize transformation
    new_frame_resize = resize_fun(new_frame)
    # for the visualization, beacause cv2 use BGR images
    new_frame_BGR = cv2.cvtColor(new_frame_resize, cv2.COLOR_BGR2RGB)
    # for the optical flow
    new_gray = cv2.cvtColor(new_frame_resize, cv2.COLOR_RGB2GRAY).reshape((256, 256))


    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]


    A = compute_adjacency(p0, p1)
    #print(A)

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()

        # Add dot to starting point
        print("S1:", new_frame_BGR.shape)
        new_frame_BGR = np.ascontiguousarray(new_frame_BGR, dtype = np.uint8)
        print("S2:", new_frame_BGR.shape)
        new_frame_BGR = cv2.circle(new_frame_BGR, (c,d), 2, [0,0,255], -1)
        print("S3:", new_frame_BGR.shape)
        # Add line between starting and final point
        print("M1:", mask.shape)
        mask = np.ascontiguousarray(mask, dtype = np.uint8)
        print("M2:", mask.shape)
        mask = cv2.line(mask, (a,b),(c,d), [0,0,255], 2)
        print("M3:", mask.shape)


    img = cv2.add(new_frame_BGR, mask)
    cv2.imshow('frame', img)

    #cv2.imwrite(save_path+'/'+str(count)+'.jpg', img)
    count += 1


    if cv2.waitKey(100) & 0xff == ord('q'):
        cv2.destroyAllWindows()
        cap.release()

    #break
    # Now update the previous frame and previous points
    old_gray = new_gray.copy()












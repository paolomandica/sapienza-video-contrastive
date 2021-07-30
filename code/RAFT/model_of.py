
#from geometric_median import geo_median

import sys
sys.path.append('core')

from raft import RAFT
import cv2

import argparse
import os
import numpy as np
# show full matrices on terminal
np.set_printoptions(threshold=sys.maxsize, linewidth=1000)
import torch

from utils import flow_viz
import skimage
import scipy

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F


DEVICE = 'cpu'



def viz(img, flo, img2):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo, img2], axis=1)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()



def compute_adjacency(p0, p1):

    dist_mat = scipy.spatial.distance.cdist(p0, p1)

    edges = np.concatenate( (np.arange(p0.shape[0]).reshape(-1, 1), dist_mat.argmin(-1).reshape(-1, 1)),  1)

    adj_mat = np.zeros((49,49))
    for j in range(edges.shape[0]):
        ii,jj=edges[j,:]
        adj_mat[ii,jj]=1

    return adj_mat



def compute_adjacency_2(p0, p1, temperature):

    dist_mat = scipy.spatial.distance.cdist(p0, p1) + 0.00001
    A = 1/dist_mat

    # for each row (= starting node) we select the closest 4 nodes of the next frame according to the movement 
    topK = 4
    ind = np.argpartition(A, -topK, axis=0)[-topK:, :].T

    row_ind = np.arange(A.shape[0]).repeat(topK).reshape(-1,)
    col_ind = ind.reshape(-1,)

    A_tensor = torch.Tensor(A[row_ind, col_ind].reshape(-1, topK))

    m = torch.nn.Softmax(dim=1)
    edges_weights = m(A_tensor/temperature).reshape(-1,).type(torch.float16)

    edges = np.concatenate((row_ind.reshape(-1,1), col_ind.reshape(-1,1)), 1)


    adj_mat = torch.zeros((49,49))

    for j in range(edges.shape[0]):
        ii,jj=edges[j]
        adj_mat[ii,jj] = edges_weights[j].item()

    return adj_mat, ind




def extract_displacement(flow, frame_num):

    of_patches = skimage.util.view_as_windows(flow.squeeze(0).permute(1, 2, 0).numpy(), (64, 64, 2), step=[32, 32, 2])

    displ_arr=np.empty((0,2))

    for x_c in range(of_patches.shape[0]):
        ll = []
        for y_c in range(of_patches.shape[1]):

            of_points = of_patches[x_c, y_c].squeeze(0).reshape(-1, 2)
            norm_arr = np.linalg.norm(of_points, axis=1)

            if args.threshold:
                norm_cond = norm_arr >= args.threshold
                of_points = of_points[norm_cond]
                if of_points.shape[0] == 0:
                    of_points = np.array([0,0]).reshape(1, 2)


            if args.aggreg == 'median':
                displ = np.median(of_points, axis=0)
            elif args.aggreg == 'mean':
                displ = np.mean(of_points, axis=0)
            elif args.aggreg == 'max':
                norm_arr = np.linalg.norm(of_points, axis=1)
                displ = of_patches[x_c, y_c].squeeze(0).reshape(-1, 2)[ np.argmax(norm_arr, axis=0), :]
            
            displ_arr = np.concatenate((displ_arr, np.round(displ, 0).astype(int).reshape(1,2)), 0)

    return displ_arr, of_patches, frame_num




def save_patches(of_patches, frame_num):

    fig, axs = plt.subplots(nrows=7, ncols=7, figsize=(7,7)) 
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for r, row_axs in enumerate(axs):
        for c, axx in enumerate(row_axs):
            flo = flow_viz.flow_to_image(of_patches[r,c].squeeze(0))

            axx.imshow(flo/255.0)
            axx.text(1, 0.01 , str(r*7+c), fontsize=6)
            axx.set_xticks([])
            axx.set_yticks([]) 

    fig.savefig('./pathces_frame'+str(frame_num)+'.jpg')







def demo(args):

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=torch.device(DEVICE)))

    model = model.module
    model.to(DEVICE)
    model.eval()


    cap = cv2.VideoCapture(args.video_path)
    ret, old_frame = cap.read()
    old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB)
    old_frame_tens = torch.Tensor(np.expand_dims(old_frame.transpose(2, 0, 1), axis=0))

    frame_num = 1

    step = 32
    grid_y, grid_x = np.mgrid[step:256:step, step:256:step]
    centroids = np.stack((grid_x.flatten(), grid_y.flatten()),axis=1).astype(np.float32)

    As = torch.empty((0,49,49))

    with torch.no_grad():

        while(cap.isOpened()):

            if frame_num%args.frame_skip != 0:
                ret, new_frame = cap.read()
                frame_num += 1
                continue

            ret, new_frame = cap.read()

            if type(new_frame) == type(None):
                return As

            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
            new_frame_tens = torch.Tensor(np.expand_dims(new_frame.transpose(2, 0, 1), axis=0))

            flow_low, flow_up = model(old_frame_tens, new_frame_tens, iters=20, test_mode=True)



            displ_arr, of_patches, frame_num = extract_displacement(flow_up, frame_num)


            A_mat, _ind = compute_adjacency_2(centroids, centroids+displ_arr, args.temperature)

            As = torch.cat((As, A_mat.unsqueeze(0)), dim=0)
            #print(As.shape)

            old_frame = new_frame
            old_frame_tens = new_frame_tens
            frame_num += 1

            
            

def stoch_mat(A, zero_diagonal=False, do_dropout=True, do_sinkhorn=False):
    ''' Affinity -> Stochastic Matrix '''

    edgedrop_rate = 0.1

    if do_dropout and edgedrop_rate > 0:
        A[torch.rand_like(A) < edgedrop_rate] = -1e20

    return F.softmax(A/args.temperature, dim=-1)



def xent_targets(A):
        B, N = A.shape[:2]
        key = '%s:%sx%s' % (str(A.device), B, N)

        if key not in _xent_targets:
            I = torch.arange(A.shape[-1])[None].repeat(B, 1)
            _xent_targets[key] = I.view(-1).to(A.device)

        return _xent_targets[key]



if __name__ == '__main__':

    # 1bFWbJLYvvs.mp4 ---> Potenzialità dell'oracolo
    # 4wX28uLBs-s.mp4 ---> esempio di come il background influenza troppo
    # 0DH97RkBp3M.mp4 ---> movimento della videocamera dominante
    # 0c1fjT6Jc54.mp4 ---> immagine noisy

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="models/raft-things.pth", help="restore checkpoint")
    parser.add_argument('--video-path', default="./videos_256/video_prova/0hsa-fjf_Wc.mp4", type=str)
    parser.add_argument('--frame-skip', default=8, type=int, help='distance among 2 frames to compute optical flow')
    parser.add_argument('--threshold', default=None, type=float, help='threshold for computation of optical flow')
    parser.add_argument('--temperature', default=0.07, type=float)
    parser.add_argument('--aggreg', default='median', type=str, help='how to aggregate displacement on a single patch')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()


    flip = False
    _xent_targets = dict()
    xent = nn.CrossEntropyLoss(reduction="none")
    device = 'cpu'
    EPS = 1e-20


    As = demo(args)
    As = As.unsqueeze(0) # simulate batch --> shape is:  B, T-1, N, N 
    B = As.shape[0]
    T = As.shape[1] + 1
    N = As.shape[2]
    H = 256

    print("B:", str(B), "T:", str(T), "N:", str(N))

    #################################################################
    # Compute walks 
    #################################################################
    walks = dict()
    A12s = [stoch_mat(As[:, i], do_dropout=True) for i in range(T-1)]

    #################################################### Palindromes
    
    A21s = [stoch_mat(As[:, i].transpose(-1, -2), do_dropout=True) for i in range(T-1)]
    AAs = []
    for i in list(range(1, len(A12s))):
        g = A12s[:i+1] + A21s[:i+1][::-1]
        aar = aal = g[0]
        for _a in g[1:]:
            aar, aal = aar @ _a, _a @ aal

        AAs.append((f"l{i}", aal) if flip else (f"r{i}", aar))

    for i, aa in AAs:
        walks[f"cyc {i}"] = [aa, xent_targets(aa)]

    #################################################################
    # Compute loss 
    #################################################################
    xents = [torch.tensor([0.]).to(device)]
    diags = dict()

    for name, (A, target) in walks.items():
        logits = torch.log(A+EPS).flatten(0, -2)
        loss = xent(logits, target).mean()
        acc = (torch.argmax(logits, dim=-1) == target).float().mean()
        diags.update({f"{H} xent {name}": loss.detach(),
                      f"{H} acc {name}": acc})
        xents += [loss]
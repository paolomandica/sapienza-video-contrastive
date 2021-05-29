
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


    adj_mat = np.zeros((49,49))

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



    with torch.no_grad():

        while(cap.isOpened()):

            if frame_num%args.frame_skip != 0:
                ret, new_frame = cap.read()
                frame_num += 1
                continue

            ret, new_frame = cap.read()
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
            new_frame_tens = torch.Tensor(np.expand_dims(new_frame.transpose(2, 0, 1), axis=0))

            flow_low, flow_up = model(old_frame_tens, new_frame_tens, iters=20, test_mode=True)

            # fig1 = plt.figure()
            # flo = flow_up[0].permute(1,2,0).cpu().numpy()
            # flo = flow_viz.flow_to_image(flo)
            # plt.imshow(flo/255.0)
            # fig1.savefig('./flow_frame'+str(frame_num)+'.jpg')


            # of_patches = skimage.util.view_as_windows(flow_up.squeeze(0).permute(1, 2, 0).numpy(), (64, 64, 2), step=[32, 32, 2])

            # print(" ------ Frame:", frame_num)
            # displ_arr=np.empty((0,2))

            # for x_c in range(of_patches.shape[0]):
            #     ll = []
            #     for y_c in range(of_patches.shape[1]):

            #         of_points = of_patches[x_c, y_c].squeeze(0).reshape(-1, 2)
            #         norm_arr = np.linalg.norm(of_points, axis=1)

            #         if args.threshold:
            #             norm_cond = norm_arr >= args.threshold
            #             of_points = of_points[norm_cond]
            #             if of_points.shape[0] == 0:
            #                 of_points = np.array([0,0]).reshape(1, 2)


            #         if args.aggreg == 'median':
            #             #displ = geo_median(of_points)
            #             displ = np.median(of_points, axis=0)
            #         elif args.aggreg == 'mean':
            #             displ = np.mean(of_points, axis=0)
            #         elif args.aggreg == 'max':
            #             norm_arr = np.linalg.norm(of_points, axis=1)
            #             displ = of_patches[x_c, y_c].squeeze(0).reshape(-1, 2)[ np.argmax(norm_arr, axis=0), :]
                    
            #         #ll.append(list(np.round(displ,0)))
            #         displ_arr = np.concatenate((displ_arr, np.round(displ, 0).astype(int).reshape(1,2)), 0)
                    
            #     print(ll)
            # _dist_mat = scipy.spatial.distance.cdist(centroids, centroids-displ_arr)
            # print(np.concatenate((np.arange(49).reshape(-1,1), centroids, displ_arr, 
            #   np.linalg.norm(displ_arr, axis=1).reshape(-1, 1), _dist_mat.argmin(-1).reshape(-1, 1)), 1))
            

            print(" ------ Frame:", frame_num)

            displ_arr, of_patches, frame_num = extract_displacement(flow_up, frame_num)


            A_mat, _ind = compute_adjacency_2(centroids, centroids+displ_arr, args.temperature)


            #print(np.concatenate((np.arange(49).reshape(-1,1), centroids, displ_arr, 
            #    np.linalg.norm(displ_arr, axis=1).reshape(-1, 1), _ind), 1))



            
            font = cv2.FONT_HERSHEY_SIMPLEX
            rep = 13

            image_adjacency = np.repeat(np.repeat(A_mat, rep, axis=0), rep, axis=1)

            for i in range(49):
                pos = rep//2+1 + i*rep
                cv2.putText(image_adjacency, str(i),(pos-5, pos), font, 0.25, (0,0,255), 1)

            cv2.imshow('adjacency matrix', image_adjacency)


            

            #print(np.concatenate((centroids, displ_arr, centroids+displ_arr), 1))

            try_line(old_frame, new_frame, centroids, centroids+displ_arr)


            #img = old_frame_tens[0].permute(1,2,0).cpu().numpy()
            #img = img[:, :, [2,1,0]]/255.0
            #mask = np.zeros((256, 256, 3))
            #img = cv2.line(img, (32,32,2),(230,230,2), [0,0,255], 2)
            #print(img.shape, mask.shape)
            #img = cv2.add(img, mask)
            #cv2.imshow('old frame', img)

            # plt.imshow(A_mat)
            # plt.savefig('./A_'+str(frame_num)+'.jpg')
            # print(A_mat)

            # old_image = old_frame_tens.squeeze(0).permute(1,2,0)
            # mask = np.zeros_like(old_frame_tens)
            # # Select good points
            # good_new = centroids
            # good_old = centroids+displ_arr

            # print("OLD_FRAME:", type(old_frame))
            # print("\t\t", old_frame.shape)

            # # draw the tracks
            # for i,(new,old) in enumerate(zip(good_new, good_old)):
            #     a,b = new.ravel()
            #     c,d = old.ravel()

            #     # Add dot to starting point
            #     # new_frame_BGR = np.ascontiguousarray(old_frame, dtype = np.uint8)
            #     # new_frame_BGR = cv2.circle(new_frame_BGR, (c,d), 2, [0,0,255], -1)

            #     new_frame_BGR = cv2.circle(old_frame, (c,d), 2, [0,0,255], -1)
            #     # Add line between starting and final point
            #     mask = np.ascontiguousarray(mask, dtype = np.uint8)
            #     mask = cv2.line(mask, (a,b),(c,d), [0,0,255], 2)


            # img = cv2.add(new_frame_BGR, mask)
            # cv2.imshow('frame', img)


            #save_patches(of_patches, frame_num)

            # fig, axs = plt.subplots(nrows=7, ncols=7, figsize=(7,7)) #, gridspec_kw={'hspace': 0.1, 'wspace': 0.1}
            # fig.subplots_adjust(hspace=0.1, wspace=0.1)
            # for r, row_axs in enumerate(axs):
            #     for c, axx in enumerate(row_axs):
            #         flo = flow_viz.flow_to_image(of_patches[r,c].squeeze(0))
            #         #cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
            #         axx.imshow(flo/255.0)
            #         axx.text(1, 0.01 , str(r*7+c), fontsize=6)
            #         axx.set_xticks([])
            #         axx.set_yticks([]) 

            # fig.savefig('./pathces_frame'+str(frame_num)+'.jpg')


            viz(old_frame_tens, flow_up, new_frame_tens)

            old_frame = new_frame
            old_frame_tens = new_frame_tens
            frame_num += 1

            
            


def try_line(image, image2, old_point, new_point):
    
    # cap = cv2.VideoCapture(args.video_path)
    # ret, image = cap.read()
    # image = cv2.imread('./pathces_frame56.jpg')
     
    # height = image.shape[0]
    # width = image.shape[1]

    # step = 32
    # grid_y, grid_x = np.mgrid[step:width:step, step:height:step]
    # centroids = np.stack((grid_x.flatten(), grid_y.flatten()),axis=1).astype(np.float32)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, (new, old) in enumerate(zip(new_point, old_point)):
        a,b = new.ravel().astype(int)
        c,d = old.ravel().astype(int)


        cv2.putText(image, str(i),(c,d+10), font, 0.25, (0,0,255), 1)

        cv2.circle(image, (c,d), 2, [0,0,255], -1)
        cv2.line(image, (a,b),(c,d), [0,255,255], 1)


    #print("TYPE", type(image))
    #print("SHAPE", image.shape)
    
    #cv2.line(image, (0,0), (width, height), (0,0,255), 10)
    #cv2.circle(image, (128,128), 15, [0,0,255], -1)

    img_flo = np.concatenate([image, image2], axis=1)
     
    cv2.imshow("Image", img_flo)
        
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()




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


    #try_line()
    demo(args)
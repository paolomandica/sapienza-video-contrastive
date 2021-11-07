import datetime
import os
import time
import sys
import numpy as np
import json

import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torchvision

import data
from data.kinetics import Kinetics400
from data.video import VideoList
from torchvision.datasets.samplers.clip_sampler import RandomClipSampler, UniformClipSampler

import utils
from model import CRW
import resnet

import pdb

from teacherstudent import CRWTeacherStudent

# Disable wandb syncing to the cloud
# os.environ['WANDB_MODE'] = 'offline'

####################################################################################################
# train_one_epoch function
####################################################################################################

def train_one_epoch(model, optimizer, lr_scheduler, data_loader, device, 
                    epoch, print_freq, vis=None, checkpoint_fn=None, prob=None, model_pretr=None):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = f'Epoch: [{epoch}]'

    # Initialise wandb
    if vis is not None:
        vis.wandb_init(model)

    for step, (video, orig) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        start_time = time.time()
        
        ################################################################################
        # Original Model
        ################################################################################

        # max_sp_num = len(torch.unique(sp_mask))
        # orig = orig.to(device)
        # sp_mask = sp_mask.to(device)

        # output, loss, diagnostics = model(orig, sp_mask, max_sp_num)
        
        # loss = loss.mean()

        # # if vis is not None and np.random.random() < 0.01:
        # if vis is not None:
        #     vis.log(dict(loss=loss.mean().item()))
        #     vis.log({k: v.mean().item() for k, v in diagnostics.items()})

        # # NOTE Stochastic checkpointing has been retained
        # if checkpoint_fn is not None and np.random.random() < 0.005:
        #     checkpoint_fn()

        # optimizer.zero_grad()
        # loss.backward()
        # # print(torch.nn.utils.clip_grad_norm_(model.parameters(), 1), 'grad norm')
        # optimizer.step()

        # metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        # metric_logger.meters['clips/s'].update(video.shape[0] / (time.time() - start_time))
        # lr_scheduler.step()

        ################################################################################
        # Parallel Model
        ################################################################################

        inp_video = torch.Tensor(orig).permute(0,2,1,3,4).to(device)
        feat_map_pretr = model_pretr(inp_video)
        sp_mask = segm_from_featmap(feat_map_pretr.squeeze(0).permute(0,2,3,4,1))

        # forward with patches
        video = video.to(device)
        output, loss, diagnostics = model(video, None, None)

        # forward with superpixels
        sp_mask = sp_mask.to(device)
        max_sp_num = len(torch.unique(sp_mask))
        output_sp, loss_sp, diagnostics_sp = model(orig, sp_mask, max_sp_num)

        loss = loss.mean()
        loss_sp = loss_sp.mean()
        a = prob
        loss_sum = a*loss + (1-a)*loss_sp

        diagnostics_dict = dict()
        items = list(diagnostics.items()) + list(diagnostics_sp.items())
        
        for k, v in items:
            if k not in diagnostics_dict:
                diagnostics_dict[k] = v
            else:
                diagnostics_dict[k] = torch.cat([diagnostics_dict[k], v])

        if vis is not None and np.random.random() < 0.1:
            vis.log(dict(loss=loss_sum.mean().item()))
            vis.log({k: v.mean().item() for k, v in diagnostics_dict.items()})

        if checkpoint_fn is not None and np.random.random() < 0.005:
            checkpoint_fn()

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

        metric_logger.update(loss=loss_sum.item(), 
                             lr=optimizer.param_groups[0]["lr"])

        metric_logger.meters['clips/s'].update(video.shape[0] / (time.time() - start_time))
        
        lr_scheduler.step()


    checkpoint_fn()

####################################################################################################
# Minor functions
# - _get_cache_path : get cache path for automatic caching of train dataset
# - collate_fn      : custom collate function for dataloader; removes audio from data samples
####################################################################################################

def segm_from_featmap(feat_map, n_centr=4):

    B, T, H, W, C = feat_map.shape
    step_row = int(H/n_centr)
    step_col = int(W/n_centr)

    row_idx = torch.tensor(range(0, H, step_row), device='cuda')
    col_idx = torch.tensor(range(0, W, step_col), device='cuda')

    centroids = torch.index_select(torch.index_select(feat_map, 2, row_idx), 3, col_idx)
    dist_mat = torch.cdist(feat_map.reshape(B, T, -1, C), centroids.reshape(B, T, -1, C))
    segments = torch.argmin(dist_mat, dim=3).reshape(B, T, H, W).to(torch.int16)
    
    return segments

def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision",
                              "datasets", "kinetics", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def collate_fn(batch):
    # remove audio and labels from the batch
    batch = [d[0] for d in batch]
    return default_collate(batch)

####################################################################################################
# Main
####################################################################################################

def main(args):

    # Eager Checks
    if args.teacher_student:
        assert args.prob == 1, "Teacher-Student training is not yet compatible with probabistic sp | patch sampling"

    print("Arguments", end="\n" + "-"*100 + "\n")
    for arg, value in vars(args).items():
        print(f"{arg} = {value}")
    print("-"*100)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    print("Preparing training dataloader", end="\n"+"-"*100+"\n")
    traindir = os.path.join(args.data_path, 'train_256' if not args.fast_test else 'val_256')
    valdir = os.path.join(args.data_path, 'val_256')

    st = time.time()
    cache_path = args.cache_path

    transform_train = utils.augs.get_train_transforms(args)

    
    # Load Pretrained model to extract superpixel from it
    # pretr_net = resnet.resnet18()
    model_pretr = CRW(args, vis=None).to('cuda')
    # pretr_net = utils.make_encoder(args).to(device)
    # checkpoint = torch.load('../pretrained.pth')

    checkpoint = torch.load('../pretrained.pth', map_location='cuda')
    utils.partial_load(checkpoint['model'], model_pretr)

    pretr_net = model_pretr.encoder

    # Dataset
    def make_dataset(is_train, cached=None):
        _transform = transform_train if is_train else transform_test

        if 'kinetics' in args.data_path.lower():
            return Kinetics400(
                traindir if is_train else valdir,
                frames_per_clip=args.clip_len,
                step_between_clips=1,
                transform=transform_train,
                extensions=('mp4'),
                frame_rate=args.frame_skip,
                # cached=cached,
                _precomputed_metadata=cached,
                sp_method=args.sp_method,
                num_components=args.num_sp,
                prob=args.prob
            )
        # HACK assume image dataset if data path is a directory
        elif os.path.isdir(args.data_path):
            return torchvision.datasets.ImageFolder(
                root=args.data_path,
                transform=_transform)
        else:
            return VideoList(
                filelist=args.data_path,
                clip_len=args.clip_len,
                is_train=is_train,
                frame_gap=args.frame_skip,
                transform=_transform,
                random_clip=True,
            )

    if args.cache_dataset and os.path.exists(cache_path):
        print(f"Loading dataset_train from {cache_path}", end="\n"+"-"*100+"\n")
        dataset, _ = torch.load(cache_path)
        cached = dict(video_paths=dataset.video_clips.video_paths,
                      video_fps=dataset.video_clips.video_fps,
                      video_pts=dataset.video_clips.video_pts)

        dataset = make_dataset(
            is_train=True, cached=cached)
        dataset.transform = transform_train
    else:
        dataset = make_dataset(is_train=True)
        if 'kinetics' in args.data_path.lower():  # args.cache_dataset and
            print(f"Saving dataset_train to {cache_path}", end="\n"+"-"*100+"\n")
            utils.mkdir(os.path.dirname(cache_path))
            dataset.transform = None
            torch.save((dataset, traindir), cache_path)
            dataset.transform = transform_train

    if hasattr(dataset, 'video_clips'):
        dataset.video_clips.compute_clips(args.clip_len, 1, frame_rate=args.frame_skip)

    print("Took", time.time() - st)

    # Data Loader
    def make_data_sampler(is_train, dataset):
        torch.manual_seed(0)
        if hasattr(dataset, 'video_clips'):
            _sampler = RandomClipSampler  # UniformClipSampler
            return _sampler(dataset.video_clips, args.clips_per_video)
        else:
            return torch.utils.data.sampler.RandomSampler(dataset) if is_train else None

    print("Creating data loaders", end="\n"+"-"*100+"\n")
    train_sampler = make_data_sampler(True, dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,  # shuffle=not args.fast_test,
        sampler=train_sampler, num_workers=args.workers//2, # 0 for the cpu
        pin_memory=True, collate_fn=collate_fn)

    # Visualisation
    vis = utils.visualize.Visualize(args) if args.visualize else None

    


    # Model
    print("Creating model", end="\n"+"-"*100+"\n")
    model = CRW(args, vis=vis).to(device)

    # print(model)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate schedule
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.lr_gamma)

    model_without_ddp = model

    accelerator = None

    # Parallelise model over GPUs
    if args.data_parallel:
        model = torch.nn.parallel.DataParallel(model)
        model_without_ddp = model.module

    # Partially load weights from model checkpoint
    if args.partial_reload:
        checkpoint = torch.load(args.partial_reload, map_location='cpu')
        utils.partial_load(checkpoint['model'], model_without_ddp)
        optimizer.param_groups[0]["lr"] = args.lr
        # args.start_epoch = checkpoint['epoch'] + 1

    # Resume from checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    def save_model_checkpoint():
        if args.output_dir:
            print("Saving at:", args.output_dir)
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            torch.save(
                checkpoint,
                os.path.join(args.output_dir, f'model_{epoch}.pth'))
            torch.save(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))
    
    # Start Training
    print("Start training", end="\n"+"-"*100+"\n")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, optimizer, lr_scheduler, data_loader,
                        device, epoch, args.print_freq,
                        vis=vis, checkpoint_fn=save_model_checkpoint,
                        prob=args.prob, model_pretr=pretr_net)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')

####################################################################################################
# Run as Script
####################################################################################################

if __name__ == "__main__":
    args = utils.arguments.train_args()
    main(args)

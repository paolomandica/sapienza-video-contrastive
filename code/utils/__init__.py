import resnet
from torchvision import transforms
from torch.nn import functional as F
from torch import nn
from collections import defaultdict, deque
import datetime
import time
import torch

import errno
import os
import sys
import numbers

from . import arguments
from . import visualize
from . import augs

#########################################################
# DEBUG
#########################################################


def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback
        import pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print
        # ...then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        pdb.post_mortem(tb)  # more "modern"


sys.excepthook = info

#########################################################
# Meters
#########################################################


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        import torch.distributed as dist
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


#################################################################################
# Network Utils
#################################################################################


def partial_load(pretrained_dict, model, skip_keys=[]):
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    filtered_dict = {k: v for k, v in pretrained_dict.items(
    ) if k in model_dict and not any([sk in k for sk in skip_keys])}
    skipped_keys = [k for k in pretrained_dict if k not in filtered_dict]

    # 2. overwrite entries in the existing state dict
    model_dict.update(filtered_dict)

    # 3. load the new state dict
    model.load_state_dict(model_dict)

    print('\nSkipped keys: ', skipped_keys)
    print('\nLoading keys: ', filtered_dict.keys())


def load_vince_model(path):
    checkpoint = torch.load(path, map_location={'cuda:0': 'cpu'})
    checkpoint = {k.replace('feature_extractor.module.model.', ''): checkpoint[k] for k in checkpoint if 'feature_extractor' in k}
    return checkpoint


def load_tc_model():
    path = 'tc_checkpoint.pth.tar'
    model_state = torch.load(path, map_location='cpu')['state_dict']

    net = resnet.resnet50()
    net_state = net.state_dict()

    for k in [k for k in model_state.keys() if 'encoderVideo' in k]:
        kk = k.replace('module.encoderVideo.', '')
        tmp = model_state[k]
        if net_state[kk].shape != model_state[k].shape and net_state[kk].dim() == 4 and model_state[k].dim() == 5:
            tmp = model_state[k].squeeze(2)
        net_state[kk][:] = tmp[:]

    net.load_state_dict(net_state)

    return net


def load_uvc_model():
    net = resnet.resnet18()
    net.avgpool, net.fc = None, None

    ckpt = torch.load('uvc_checkpoint.pth.tar', map_location='cpu')
    state_dict = {k.replace('module.gray_encoder.', ''): v for k,
                  v in ckpt['state_dict'].items() if 'gray_encoder' in k}
    net.load_state_dict(state_dict)

    return net


class From3D(nn.Module):
    ''' Use a 2D convnet as a 3D convnet '''

    def __init__(self, resnet):
        super(From3D, self).__init__()
        self.model = resnet

    def forward(self, x):
        N, C, T, h, w = x.shape
        xx = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, h, w)
        m = self.model(xx)

        return m.view(N, T, *m.shape[-3:]).permute(0, 2, 1, 3, 4)


def make_encoder(args):
    model_type = args.model_type
    if model_type == 'scratch':
        net = resnet.resnet18()
        net.modify(padding='reflect')

    elif model_type == 'scratch_zeropad':
        net = resnet.resnet18()

    elif model_type == 'scratch50':
        net = resnet.resnet50()
        net.modify(padding='reflect')

    elif model_type == 'imagenet18':
        net = resnet.resnet18(pretrained=True)

    elif model_type == 'imagenet50':
        net = resnet.resnet50(pretrained=True)

    elif model_type == 'moco50':
        net = resnet.resnet50(pretrained=False)
        net_ckpt = torch.load('moco_v2_800ep_pretrain.pth.tar')
        net_state = {k.replace('module.encoder_q.', ''): v for k, v in net_ckpt['state_dict'].items()
                     if 'module.encoder_q' in k}
        partial_load(net_state, net)

    elif model_type == 'timecycle':
        net = load_tc_model()

    elif model_type == 'uvc':
        net = load_uvc_model()

    elif model_type == 'r3d_18':
        net = resnet.resnet_3d_18(pretrained=True)
        net.modify(stride=0)

    elif model_type == 'r2plus1d_18':
        net = resnet.r2plus1d_18(pretrained=True)

    else:
        assert False, 'invalid args.model_type'

    if hasattr(net, 'modify') and model_type not in ['r3d_18', 'r2plus1d_18']:
        net.modify(remove_layers=args.remove_layers)

    if 'Conv2d' in str(net):
        print("2D ResNet")
        net = From3D(net)
    else:
        print("3D ResNet")

    return net


class MaskedAttention(nn.Module):
    '''
    A module that implements masked attention based on spatial locality
    TODO implement in a more efficient way (torch sparse or correlation filter)
    '''

    def __init__(self, radius, flat=True):
        super(MaskedAttention, self).__init__()
        self.radius = radius
        self.flat = flat
        self.masks = {}
        self.index = {}

    def mask(self, H, W):
        if not ('%s-%s' % (H, W) in self.masks):
            self.make(H, W)
        return self.masks['%s-%s' % (H, W)]

    def index(self, H, W):
        if not ('%s-%s' % (H, W) in self.index):
            self.make_index(H, W)
        return self.index['%s-%s' % (H, W)]

    def make(self, H, W):
        if self.flat:
            H = int(H**0.5)
            W = int(W**0.5)

        gx, gy = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        D = ((gx[None, None, :, :] - gx[:, :, None, None])**2 +
             (gy[None, None, :, :] - gy[:, :, None, None])**2).float() ** 0.5
        D = (D < self.radius)[None].float()

        if self.flat:
            D = self.flatten(D)
        self.masks['%s-%s' % (H, W)] = D

        return D

    def flatten(self, D):
        return torch.flatten(torch.flatten(D, 1, 2), -2, -1)

    def make_index(self, H, W, pad=False):
        mask = self.mask(H, W).view(1, -1).byte()
        idx = torch.arange(0, mask.numel())[mask[0]][None]

        self.index['%s-%s' % (H, W)] = idx

        return idx

    def forward(self, x):
        H, W = x.shape[-2:]
        sid = '%s-%s' % (H, W)
        if sid not in self.masks:
            self.masks[sid] = self.make(H, W).to(x.device)
        mask = self.masks[sid]

        return x * mask[0]


class ZeroSoftmax(nn.Module):
    def __init__(self):
        super(ZeroSoftmax, self).__init__()

    def forward(self, x, dim=0, eps=1e-5):
        x_exp = torch.pow(torch.exp(x) - 1, exponent=2)
        x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
        x = x_exp / (x_exp_sum + eps)
        return x

#########################################################
# Superpixels
#########################################################

# NB This code is adapted from the skimage.util.view_as_windows source code 
#    The script is located in the project at scikit-image/skimage/util/shape.py 
#    See:
#    https://github.com/scikit-image/scikit-image/blob/faf69a251fa15902e24cafcfe486cec047230b3a/skimage/util/shape.py

def view_as_windows(arr_in, window_shape, step=1):
    """Rolling window view of the input n-dimensional array.

    Windows are overlapping views of the input array, with adjacent windows
    shifted by a single row or column (or an index of a higher dimension).

    Parameters
    ----------
    arr_in : tensor
        N-d input array.
    window_shape : integer or tuple of length arr_in.ndim
        Defines the shape of the elementary n-dimensional orthotope
        (better know as hyperrectangle [1]_) of the rolling window view.
        If an integer is given, the shape will be a hypercube of
        sidelength given by its value.
    step : integer or tuple of length arr_in.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.

    Returns
    -------
    arr_out : tensor
        (rolling) window view of the input array.

    Notes
    -----
    One should be very careful with rolling views when it comes to
    memory usage.  Indeed, although a 'view' has the same memory
    footprint as its base array, the actual array that emerges when this
    'view' is used in a computation is generally a (much) larger array
    than the original, especially for 2-dimensional arrays and above.

    For example, let us consider a 3 dimensional array of size (100,
    100, 100) of ``float64``. This array takes about 8*100**3 Bytes for
    storage which is just 8 MB. If one decides to build a rolling view
    on this array with a window of (3, 3, 3) the hypothetical size of
    the rolling view (if one was to reshape the view for example) would
    be 8*(100-3+1)**3*3**3 which is about 203 MB! The scaling becomes
    even worse as the dimension of the input array becomes larger.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Hyperrectangle

    Examples
    --------
    >>> A = torch.arange(4*4).reshape(4,4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> window_shape = (2, 2)
    >>> B = view_as_windows(A, window_shape)
    >>> B[0, 0]
    array([[0, 1],
           [4, 5]])
    >>> B[0, 1]
    array([[1, 2],
           [5, 6]])

    >>> A = torch.arange(10)
    >>> A
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> window_shape = (3,)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (8, 3)
    >>> B
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6],
           [5, 6, 7],
           [6, 7, 8],
           [7, 8, 9]])

    >>> A = torch.arange(5*4).reshape(5, 4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19]])
    >>> window_shape = (4, 3)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (2, 2, 4, 3)
    >>> B  # doctest: +NORMALIZE_WHITESPACE
    array([[[[ 0,  1,  2],
             [ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14]],
            [[ 1,  2,  3],
             [ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15]]],
           [[[ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14],
             [16, 17, 18]],
            [[ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15],
             [17, 18, 19]]]])
    """

    # -- basic checks on arguments
    if not isinstance(arr_in, torch.Tensor):
        raise TypeError("`arr_in` must be a torch Tensor")

    ndim = arr_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = torch.tensor(arr_in.shape)
    window_shape = torch.tensor(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    window_stride = torch.tensor(arr_in.stride(), dtype=torch.long)

    indexing_stride = arr_in[slices].stride()

    win_indices_shape = (((torch.tensor(arr_in.shape) - torch.tensor(window_shape))
                          // torch.tensor(step)) + 1)

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    stride = tuple(list(indexing_stride) + list(window_stride))

    # import numpy as np
    # arr_out = np.lib.stride_tricks.as_strided(arr_in, shape=new_shape, strides=strides)
    
    arr_out = torch.as_strided(arr_in, size=new_shape, stride=stride)
    return arr_out

#########################################################
# Superpixel Mask Dilation: 2D Kernel creation functions
#########################################################

def make_dilation_kernel(args):
    kernel_size, kernel_shape = args.dilation_kernel_size, args.dilation_kernel_shape
    assert kernel_size % 2 != 0, "Use an odd kernel size"
    kernel = torch.zeros(kernel_size, kernel_size, device=args.device, dtype=torch.float16)
    center = kernel_size // 2
    if kernel_shape == 'L1':
        for i in range(kernel_size):
            for j in range(kernel_size):
                if (abs(center-i) + abs(center-j)) <= center:
                    kernel[i, j] = 1
    elif kernel_shape == 'cross':
        kernel[:, center] = 1
        kernel[center, :] = 1
    elif kernel_shape == 'circle':
        for i in range(kernel_size):
            for j in range(kernel_size):
                if ((center-i)**2 + (center-j)**2) <= center**2:
                    kernel[i, j] = 1
    return kernel

#########################################################
# Misc
#########################################################


def sinkhorn_knopp(A, tol=0.01, max_iter=1000, verbose=False):
    _iter = 0

    if A.ndim > 2:
        A = A / A.sum(-1).sum(-1)[:, None, None]
    else:
        A = A / A.sum(-1).sum(-1)[None, None]

    A1 = A2 = A

    while (A2.sum(-2).std() > tol and _iter < max_iter) or _iter == 0:
        A1 = F.normalize(A2, p=1, dim=-2)
        A2 = F.normalize(A1, p=1, dim=-1)

        _iter += 1
        if verbose:
            print(A2.max(), A2.min())
            print('row/col sums', A2.sum(-1).std().item(),
                  A2.sum(-2).std().item())

    if verbose:
        print('------------row/col sums aft',
              A2.sum(-1).std().item(), A2.sum(-2).std().item())

    return A2


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img

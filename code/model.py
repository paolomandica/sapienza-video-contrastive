import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils

import skimage
from skimage.util import img_as_float

from utils import ZeroSoftmax, view_as_windows

EPS = 1e-20

class CRW(nn.Module):
    def __init__(self, args, vis=None):
        super(CRW, self).__init__()
        self.args = args

        self.edgedrop_rate = getattr(args, "dropout", 0)
        self.featdrop_rate = getattr(args, "featdrop", 0)
        self.temperature = getattr(args, "temp", getattr(args, "temperature", 0.07))

        self.encoder = utils.make_encoder(args).to(self.args.device)
        self.infer_dims()
        self.selfsim_fc = self.make_head(depth=getattr(args, "head_depth", 0))
        self.zero_softmax = ZeroSoftmax()

        self.xent = nn.CrossEntropyLoss(reduction="none")
        self._xent_targets = dict()

        self.dropout = nn.Dropout(p=self.edgedrop_rate, inplace=False)
        self.featdrop = nn.Dropout(p=self.featdrop_rate, inplace=False)

        self.flip = getattr(args, "flip", False)
        self.sk_targets = getattr(args, "sk_targets", False)
        self.vis = vis

        self.dilation_kernel = utils.make_dilation_kernel(args) if args.dilate_superpixels else None

    def infer_dims(self):
        in_sz = 256
        dummy = torch.zeros(1, 3, 1, in_sz, in_sz).to(next(self.encoder.parameters()).device)
        dummy_out = self.encoder(dummy)
        self.enc_hid_dim = dummy_out.shape[1]
        self.map_scale = in_sz // dummy_out.shape[-1]

    def make_head(self, depth=1):
        head = []
        if depth >= 0:
            dims = [self.enc_hid_dim] + [self.enc_hid_dim] * depth + [128]
            for d1, d2 in zip(dims, dims[1:]):
                h = nn.Linear(d1, d2, bias=False)
                head += [h, nn.ReLU()]
            head = head[:-1]

        return nn.Sequential(*head)

    def zeroout_diag(self, A, zero=0):
        mask = (torch.eye(A.shape[-1]).unsqueeze(0).repeat(A.shape[0], 1, 1).bool() < 1)
        mask = mask.float().cuda()
        return A * mask

    def affinity(self, x1, x2):
        in_t_dim = x1.ndim
        if in_t_dim < 4:  # add in time dimension if not there
            x1, x2 = x1.unsqueeze(-2), x2.unsqueeze(-2)

        A = torch.einsum("bctn,bctm->btnm", x1, x2)
        # if self.restrict is not None:
        #     A = self.restrict(A)

        return A.squeeze(1) if in_t_dim < 4 else A

    def stoch_mat(self, A, zero_diagonal=False, do_dropout=True, do_sinkhorn=False):
        """Affinity -> Stochastic Matrix"""

        if zero_diagonal:
            A = self.zeroout_diag(A)

        if do_dropout and self.edgedrop_rate > 0:
            A[torch.rand_like(A) < self.edgedrop_rate] = -1e20

        if do_sinkhorn:
            return utils.sinkhorn_knopp((A / self.temperature).exp(),
                                        tol=0.01,
                                        max_iter=100,
                                        verbose=False)

        # return F.softmax(A / self.temperature, dim=-1)
        return self.zero_softmax(A / self.temperature, dim=-1)

    def pixels_to_nodes(self, x):
        """
        pixel maps -> node embeddings
        Handles cases where input is a list of patches of images (N>1), or list of whole images (N=1)

        Inputs:
            -- 'x' (B x N x C x T x h x w), batch of images
        Outputs:
            -- 'feats' (B x C x T x N), node embeddings
            -- 'maps'  (B x N x C x T x H x W), node feature maps
        """
        B, N, C, T, h, w = x.shape
        maps = self.encoder(x.flatten(0, 1))
        H, W = maps.shape[-2:]

        if self.featdrop_rate > 0:
            maps = self.featdrop(maps)

        if N == 1:  # flatten single image's feature map to get node feature 'maps'
            maps = maps.permute(0, -2, -1, 1, 2).contiguous()
            maps = maps.view(-1, *maps.shape[3:])[..., None, None]
            N, H, W = maps.shape[0] // B, 1, 1

        # compute node embeddings by spatially pooling node feature maps
        feats = maps.sum(-1).sum(-1) / (H * W)
        feats = self.selfsim_fc(feats.transpose(-1, -2)).transpose(-1, -2)
        feats = F.normalize(feats, p=2, dim=1)

        feats = feats.view(B, N, feats.shape[1], T).permute(0, 2, 3, 1)
        maps = maps.view(B, N, *maps.shape[1:])

        return feats, maps

    ############################################################
    # CPU Superpixel functions - Retained for reference
    ############################################################

    def extract_sp_feat_cpu(self, img, img_maps, sp_mask):
        """
        img has shape of c, T, h, w
        img_maps has shape of C, T, H, W
        sp_mask has shape of T, h, w
        """

        c, T, h, w = img.shape
        C, T, H, W = img_maps.shape

        final_feats = []
        final_segment = []

        for t in range(T):

            device = "cuda" if torch.cuda.is_available() else "cpu"

            img_map = img_maps[:, t, :, :].permute(1, 2, 0)
            # New shape is: (H, W, C) = (32, 32, 512)

            segments = sp_mask[t, :, :]  # Shape is: (h, w) = (256, 256)

            # Compute mask for each superpixel
            sp_tensor = []

            for sp in torch.unique(segments):
                # Select specific SP
                single_sp = (segments == sp) * 1
                sp_tensor.append(single_sp)

            # This has shape: (num_sp, h, w) = (~50, 256, 256)
            sp_tensor = torch.stack(sp_tensor).cpu().numpy()

            # Compute receptive fields relative to each superpixel mask
            out = skimage.util.view_as_windows(
                sp_tensor, (sp_tensor.shape[0], int(h / H), int(w / W)), step=int(h / H)
            ).squeeze(0)
            # This should have as shape (num_windows, num_windows, num_sp, window_size, window_size) = (32,32,~50,8,8)

            # Extract features weight as normalized interesction of sp mask and receptive field of each features
            # size of superpixels for each receptive field - shape is (num_windows, num_windows, num_sp) = (32,32,~50)
            ww_not_norm = torch.sum(
                torch.sum(torch.from_numpy(out).to(device), dim=-1), dim=-1
            )
            # Size of each superpixel - shape is num_sp = = (~50)
            sp_size = torch.sum(
                torch.sum(torch.from_numpy(sp_tensor).to(device), dim=-1), dim=-1
            )
            ww_norm = ww_not_norm / sp_size

            # Expand correctly weights and features map to use tensor instead of for loop
            # Shape is: (num_windows, num_windows, C, num_sp) = (32,32,512,~50)
            ww_norm_expand = ww_norm.unsqueeze(2).repeat(1, 1, C, 1)
            # Shape is: (num_feat, num_feat, C, num_sp) = (32,32,512,~50)
            img_map_expand = img_map.unsqueeze(-1).repeat(
                1, 1, 1, ww_norm_expand.shape[-1]
            )
            # Please note num_windows and num_feat are the same.
            # So we repeat weights for each feature channels and feat for each superpixels, because they are independent

            # Weighted mean of the features
            oo = ww_norm_expand * img_map_expand
            feats = torch.sum(torch.sum(oo, 0), 0).permute(
                1, 0)  # Shape is: (~50, 512)

            final_feats.append(feats)

        return final_feats, final_segment

    def image_to_nodes_cpu(self, x, sp_mask, max_sp_num):
        """Inputs:
            -- 'x' (B x C x T x h x w), batch of images
        Outputs:
            -- 'feats' (B x C x T x N), node embeddings
            -- 'maps'  (B x C x T x H x W), node feature maps
        """

        B, T, c, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # New shape B, c, T, h, w
        maps = self.encoder(x)
        B, C, T, H, W = maps.shape
        N = max_sp_num

        if self.featdrop_rate > 0:
            maps = self.featdrop(maps)

        ff_list = []
        seg_list = []

        for b in range(B):
            ff, seg = self.extract_sp_feat_cpu(
                x[b], maps[b], sp_mask[b, :, 0, :, :])

            ff_list.append(ff)
            seg_list.append(seg)

        ff_tensor = torch.empty(
            (0, T, max_sp_num, C), requires_grad=True, device="cuda"
        )

        for ff in ff_list:
            ff_time_tensor = torch.empty(
                (0, max_sp_num, C), requires_grad=True, device="cuda"
            )

            for sp_feats in ff:
                temp_sp_feats = nn.functional.pad(
                    sp_feats,
                    pad=(0, 0, 0, max_sp_num - sp_feats.shape[0]),
                    mode="constant",
                ).unsqueeze(0)
                ff_time_tensor = torch.cat(
                    (ff_time_tensor, temp_sp_feats), dim=0)

            ff_tensor = torch.cat(
                (ff_tensor, ff_time_tensor.unsqueeze(0)), dim=0)

        # compute frame embeddings by spatially pooling frame feature maps
        # shape (B,T,SP,C) -> (B,SP,C,T)
        ff_tensor = ff_tensor.permute(0, 2, 3, 1)
        ff_tensor = self.selfsim_fc(
            ff_tensor.transpose(-1, -2)).transpose(-1, -2)
        ff_tensor = F.normalize(ff_tensor, p=2, dim=2)
        ff_tensor = ff_tensor.permute(0, 2, 3, 1)  # B, C, T, SP

        return ff_tensor, seg_list

    ############################################################
    # Parallelised (GPU) Superpixels
    ############################################################

    def image_to_nodes(self, x, sp_mask, max_sp_num):
        """ 
        Compute superpixel node representations by spatially average pooling feature maps within 
        superpixels.

        In summary: video + superpixel mask -> superpixel node embeddings

        Inputs:
            -- 'x' (B x T x c x h x w), video (frame image sequence)
            -- 'sp_mask' (B x T x c x h x w), dense superpixel maks; integers 0, ..., (max_sp_num-1)
            -- 'max_sp_num' (int), maximum number of superpixels used (value passed to segmentation algo)
        Outputs:
            # -- 'sp_feats' (B x C_reduced x T x N), superpixel node embeddings
            # -- 'maps'  (B x C x T x H x W), video (frame) feature maps

        Notes
        -----

        C_reduced above refers to the reduced latent-space dimensionality of the
        superpixel node representations (sp_feats) after being passed through the
        projection head (selfsim_fc). 
        """

        # Compute frame feature maps using encoder
        B, T, c, h, w = x.shape
        x = x.transpose(1, 2)  # swap T(ime) and c(hannel) dimensions
        maps = self.encoder(x)
        _B, C, _T, H, W = maps.shape

        # Regularise by dropping frame features (optional)
        if self.featdrop_rate > 0:
            maps = self.featdrop(maps)

        # Number of superpixels present in each mask; (B, T); (8, 4). Useful for downstream checks
        # n_superpixels = torch.max(sp_mask.flatten(2,3), dim=2)[0]

        # Make sp_mask "one-hot" from dense, with a new SP dimension;  B, T, SP, h, w
        # NOTE Segmentation (e.g. SLIC) returns a 3-channel mask
        sp_mask = sp_mask[:, :, 0, :, :]
        idxs_sp = torch.arange(max_sp_num, device=self.args.device)[None, :, None, None]
        # NOTE sp_mask broadcasts over SP dimension
        sp_mask = (sp_mask.unsqueeze(2) == idxs_sp).int()

        if self.dilation_kernel is not None:
            B, T, SP, h, w = sp_mask.shape
            padding = self.args.dilation_kernel_size // 2
            sp_mask = sp_mask.flatten(1, 2).to(torch.float16)
            kernel = self.dilation_kernel.repeat(T*SP, 1, 1).unsqueeze(1) # out_chan, in_chan/groups 
            sp_mask = (F.conv2d(sp_mask, weight = kernel, padding=padding, groups=T*SP) > 0).int()
            sp_mask = sp_mask.view(B, T, SP, h, w)

        # Create a weighted superpixel mask to apply to feature maps
        # NOTE window_shape is (B, T, SP, h//H, w//W)
        window_shape, window_step = (*sp_mask.shape[:3], h//H, w//W), h // H
        sp_mask_wndws = view_as_windows(sp_mask, window_shape, step=window_step)[0, 0, 0]  # drop singleton dims
        # sum over windows h//H and w//W
        sp_mask_wndws = sp_mask_wndws.sum(-1).sum(-1)
        # (H, W, B, T, SP) -> (B, T, H, W, SP)
        sp_mask_wndws = sp_mask_wndws.permute(2, 3, 0, 1, 4)
        sp_mask_wndws = sp_mask_wndws / (sp_mask.sum(-1).sum(-1) + EPS)[:, :, None, None, :]  # normalise mask by SP sizes
        sp_mask_wndws = sp_mask_wndws.unsqueeze(4)

        # Compute superpixel node embeddings by multiplying feature maps by weighted mask
        img_map_expanded = maps.permute(0, 2, 3, 4, 1).unsqueeze(-1)
        sp_feats = sp_mask_wndws * img_map_expanded
        sp_feats = sp_feats.sum(2).sum(2).permute(0, 3, 1, 2)

        # Reduce latent dimensionality of superpixel nodes via projection head
        sp_feats = self.selfsim_fc(sp_feats)
        sp_feats = F.normalize(sp_feats, p=2, dim=3)
        sp_feats = sp_feats.permute(0, 3, 2, 1)  # B, C, T, SP

        return sp_feats, maps

    def forward(self, x, sp_mask, max_sp_num, just_feats=False, orig_unnorm=None):
        """
        Input is B x T x N*C x H x W, where either
           N>1 -> list of patches of images
           N=1 -> list of images
        """
        B, T, C, H, W = x.shape

        #################################################################
        # Image/Pixels to Nodes
        #################################################################

        if sp_mask is None:
            # Patches
            _N, C = C // 3, 3
            x = x.transpose(1, 2).view(B, _N, C, T, H, W)
            q, mm = self.pixels_to_nodes(x)
        else:
            # Superpixels
            q, mm = self.image_to_nodes(x, sp_mask, max_sp_num)

        B, C, T, N = q.shape

        if just_feats:
            h, w = np.ceil(
                np.array(x.shape[-2:]) / self.map_scale).astype(np.int)
            return (q, mm) if _N > 1 else (q, q.view(*q.shape[:-1], h, w))

        #################################################################
        # Compute walks
        #################################################################

        walks = dict()

        As = self.affinity(q[:, :, :-1], q[:, :, 1:])
        A12s = [self.stoch_mat(As[:, i], do_dropout=True)
                for i in range(T - 1)]

        # Palindromes
        A21s = [self.stoch_mat(As[:, i].transpose(-1, -2),
                               do_dropout=True) for i in range(T - 1)]
        AAs = []
        for i in list(range(1, len(A12s))):
            g = A12s[: i + 1] + A21s[: i + 1][::-1]
            aar = aal = g[0]
            for _a in g[1:]:
                aar, aal = aar @ _a, _a @ aal

            AAs.append((f"l{i}", aal) if self.flip else (f"r{i}", aar))

        for i, aa in AAs:
            walks[f"cyc {i}"] = [aa, self.xent_targets(aa)]

        #################################################################
        # Compute loss
        #################################################################

        xents = [torch.tensor([0.0]).to(self.args.device)]
        diags = dict()

        for name, (A, target) in walks.items():
            logits = torch.log(A + EPS).flatten(0, -2)
            loss = self.xent(logits, target).mean()
            acc = (torch.argmax(logits, dim=-1) == target).float().mean()
            diags.update({f"{H} xent {name}": loss.detach(),
                          f"{H} acc {name}": acc})
            xents += [loss]

        #################################################################
        # Visualizations
        #################################################################
        if (np.random.random() < 1) and (self.vis is not None) and False:
            with torch.no_grad():
                vid = x[0].cpu().detach().numpy()
                mask = sp_mask[0].cpu().detach().numpy()
                A12s_vis = torch.stack(A12s)[:, 0, :, :].cpu().detach().numpy()
                utils.visualize.vis_adj(
                    vid, mask, A12s_vis, self.vis.vis, orig_unnorm[0].cpu().detach().numpy())

        loss = sum(xents) / max(1, len(xents) - 1)

        return q, loss, diags

    def xent_targets(self, A):
        B, N = A.shape[:2]
        key = "%s:%sx%s" % (str(A.device), B, N)

        if key not in self._xent_targets:
            I = torch.arange(A.shape[-1])[None].repeat(B, 1)
            self._xent_targets[key] = I.view(-1).to(A.device)

        return self._xent_targets[key]

    def visualize_patches(self, x, q):
        # all patches
        all_x = x.permute(0, 3, 1, 2, 4, 5)
        all_x = all_x.reshape(-1, *all_x.shape[-3:])
        all_f = q.permute(0, 2, 3, 1).reshape(-1, q.shape[1])
        all_f = all_f.reshape(-1, *all_f.shape[-1:])
        all_A = torch.einsum("ij,kj->ik", all_f, all_f)
        utils.visualize.nn_patches(self.vis.vis, all_x, all_A[None])

    def visualize_frame_pair(self, x, q, mm):
        t1, t2 = np.random.randint(0, q.shape[-2], (2))
        f1, f2 = q[:, :, t1], q[:, :, t2]

        A = self.affinity(f1, f2)
        A1, A2 = self.stoch_mat(A, False, False), self.stoch_mat(
            A.transpose(-1, -2), False, False)
        AA = A1 @ A2
        xent_loss = self.xent(
            torch.log(AA + EPS).flatten(0, -2), self.xent_targets(AA))

        utils.visualize.frame_pair(
            x, q, mm, t1, t2, A, AA, xent_loss, self.vis.vis)

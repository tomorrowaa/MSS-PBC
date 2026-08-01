# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init
from mmcv.runner import _load_checkpoint, load_state_dict

from mmedit.models.common import make_layer
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger

import pdb

import io
import logging
import os
import os.path as osp
import pkgutil
import re
import time
import warnings
from collections import OrderedDict
from importlib import import_module
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, Optional, Tuple, Union

from .DABC import BinaryConv2dSkip1x1, BNNDownSample, BNNUpSample, BNNSkipUpSample
from .DABC import DABCConv2d as BinaryConv2d
import math
from mmcv.runner import HOOKS, Hook
from mmedit.models.builder import LOSSES


@HOOKS.register_module(force=True)
class GFLBandSchedulerHook(Hook):
    def __init__(self, attr_paths=None):
        self.attr_paths = attr_paths or [
            "module.backbone.loss_gfl", 
            "module.loss_gfl",  
        ]

    def _resolve_attr(self, root, path):
        node = root
        for name in path.split('.'):
            node = getattr(node, name, None)
            if node is None:
                return None
        return node

    def before_train_epoch(self, runner):
        model = runner.model
        for path in self.attr_paths:
            loss_gfl = self._resolve_attr(model, path)
            if (loss_gfl is not None) and hasattr(loss_gfl, "set_epoch"):
                loss_gfl.set_epoch(epoch_idx=runner.epoch)


@HOOKS.register_module(force=True)
class GFLIterSchedulerHook(Hook):
    def __init__(self, total_iters: int,
                 omega0: float = 0.50, omegaF: float = 0.80,
                 attr_paths=None, log: bool = True):
        self.total_iters = int(total_iters)
        self.omega0, self.omegaF = float(omega0), float(omegaF)
        self.attr_paths = attr_paths or [
            "module.pixel_loss", "pixel_loss",
            "module.loss_gfl", "loss_gfl",
            "module.generator.pixel_loss", "generator.pixel_loss",
            "module.backbone.loss_gfl", "backbone.loss_gfl",
        ]
        self.log = log

    def _resolve_attr(self, root, path):
        node = root
        for name in path.split('.'):
            node = getattr(node, name, None)
            if node is None:
                return None
        return node

    def before_train_iter(self, runner):
        cur = runner.iter + 1
        ratio = min(1.0, cur / max(1, self.total_iters))
        omega = self.omega0 + (self.omegaF - self.omega0) * ratio
        model = runner.model
        for path in self.attr_paths:
            loss_obj = self._resolve_attr(model, path)
            if (loss_obj is not None) and hasattr(loss_obj, "set_threshold"):
                loss_obj.set_threshold(omega)
                if self.log and (cur % 1000 == 0):
                    runner.logger.info(f"[GFL] iter={cur}/{self.total_iters}  omega={omega:.3f} (path={path})")
                break




def _as_bchw(x: torch.Tensor):
    if x.dim() == 5:  # (B,T,C,H,W) -> (B*T,C,H,W)
        B, T, C, H, W = x.shape
        return x.reshape(B * T, C, H, W), (B, T, C, H, W)
    elif x.dim() == 4:
        return x, None
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {x.shape}")

def _gaussian_kernel_5x5(device, dtype=torch.float32):
    base = torch.tensor([1., 4., 6., 4., 1.], device=device, dtype=dtype)
    k = torch.outer(base, base); k = k / k.sum()
    return k[None, None, :, :]  # (1,1,5,5)

def _gaussian_blur(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    C = x.shape[1]
    k = kernel.to(x.dtype).expand(C, 1, 5, 5).contiguous()
    return F.conv2d(x, k, padding=2, groups=C)

def _downsample_octave(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    x_blur = _gaussian_blur(x, kernel)
    return x_blur[:, :, ::2, ::2]  # decimate ×2

def _upsample_octave(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)

def laplacian_pyramid_depth1(x: torch.Tensor) -> torch.Tensor:
    x_bchw, _ = _as_bchw(x)
    k = _gaussian_kernel_5x5(x_bchw.device, x_bchw.dtype)
    G0 = x_bchw
    G1 = _downsample_octave(G0, k)
    uG1 = _upsample_octave(G1)
    if uG1.shape[-2:] != G0.shape[-2:]:
        uG1 = F.interpolate(uG1, size=G0.shape[-2:], mode="bilinear", align_corners=False)
    uG1 = _gaussian_blur(uG1, k)
    L0 = G0 - uG1
    return L0

def radial_highpass_mask(H: int, W: int, thr: float, device, dtype):
    yy = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype).unsqueeze(1).repeat(1, W)
    xx = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype).unsqueeze(0).repeat(H, 1)
    rr = torch.sqrt(xx * xx + yy * yy); rr = rr / rr.max()
    return (rr >= thr).to(dtype)  # (H,W)

def highpass_filter_fft(x: torch.Tensor, thr: float) -> torch.Tensor:
    x_bchw, _ = _as_bchw(x)
    X = torch.fft.fftshift(torch.fft.fft2(x_bchw, norm="ortho"))
    M = radial_highpass_mask(x_bchw.shape[-2], x_bchw.shape[-1], thr, x_bchw.device, x_bchw.dtype)
    M = M.view(1, 1, *M.shape)
    Y = X * M
    y = torch.fft.ifft2(torch.fft.ifftshift(Y), norm="ortho").real
    return y





class BandAllocatorState:
    def __init__(self, omega0: float, omegaF: float, num_epochs: int, num_stages: int,
                 mode: str = "static", loss_threshold: float = 0.05):
        assert 0.0 <= omega0 < omegaF <= 1.0
        assert num_stages >= 1 and mode in ("static", "dynamic")
        self.omega0, self.omegaF = float(omega0), float(omegaF)
        self.N, self.S = int(num_epochs), int(num_stages)
        self.mode, self.loss_threshold = mode, float(loss_threshold)
        self.stage, self.omega_cur = 0, self.omega0

    def _omega_for_stage(self, s: int) -> float:
        s = max(0, min(s, self.S))
        return self.omega0 + (self.omegaF - self.omega0) * (s / self.S)

    def step(self, epoch_idx: int, last_gfl_loss: float = None) -> float:
        if self.mode == "static":
            new_stage = min(self.S, int(math.floor((epoch_idx + 1) * self.S / self.N)))
            if new_stage != self.stage:
                self.stage = new_stage
                self.omega_cur = self._omega_for_stage(self.stage)
        else:
            if (last_gfl_loss is not None) and (last_gfl_loss < self.loss_threshold) and (self.stage < self.S):
                self.stage += 1
                self.omega_cur = self._omega_for_stage(self.stage)
        if self.omega_cur > self.omegaF:
            self.omega_cur = self.omegaF
        return self.omega_cur
@LOSSES.register_module(force=True)
class GuidedFrequencyLoss(nn.Module):
   
    def __init__(self,
                 epsilon: float = 1e-3,
                 omega0: float = 0.50,
                 omegaF: float = 0.80,
                 num_epochs: int = 100,
                 num_stages: int = 5,
                 mode: str = "static",
                 loss_threshold: float = 0.05,
                 reduction: str = "mean"):
        super().__init__()
        self.epsilon, self.reduction = float(epsilon), reduction
        self._allocator = BandAllocatorState(
            omega0=omega0, omegaF=omegaF, num_epochs=num_epochs, num_stages=num_stages,
            mode=mode, loss_threshold=loss_threshold
        )
        self.register_buffer('_omega_cur_buf', torch.tensor(omega0, dtype=torch.float32), persistent=False)

    @torch.no_grad()
    def set_epoch(self, epoch_idx: int, last_gfl_loss: float = None) -> float:
        omega = self._allocator.step(epoch_idx=epoch_idx, last_gfl_loss=last_gfl_loss)
        self._omega_cur_buf.fill_(float(omega))
        return float(omega)

    @torch.no_grad()
    def current_threshold(self) -> float:
        return float(self._omega_cur_buf.item())

    @torch.no_grad()
    def set_threshold(self, omega: float) -> float:
        omega = float(max(0.0, min(1.0, omega)))
        self._omega_cur_buf.fill_(omega)
        return omega

    def _component_charbonnier_sq(self, pred, target):
        x, _ = _as_bchw(pred); y, _ = _as_bchw(target)
        diff = x - y
        ch = (diff * diff).flatten(1).sum(dim=1) + (self.epsilon ** 2)
        return ch

    def _component_laplacian_sq(self, pred, target):
        Lp = laplacian_pyramid_depth1(pred); Lt = laplacian_pyramid_depth1(target)
        diff = Lp - Lt
        return (diff * diff).flatten(1).sum(dim=1)

    def _component_theta_sq(self, pred, target, thr: float):
        Tp = highpass_filter_fft(pred, thr); Tt = highpass_filter_fft(target, thr)
        diff = Tp - Tt
        return (diff * diff).flatten(1).sum(dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ch = self._component_charbonnier_sq(pred, target)
        pi = self._component_laplacian_sq(pred, target)
        th = self._component_theta_sq(pred, target, float(self._omega_cur_buf.item()))
        gfl_sample = torch.sqrt(ch + pi + th + 1e-12)
        if self.reduction == "mean":
            return gfl_sample.mean()
        elif self.reduction == "sum":
            return gfl_sample.sum()
        else:
            return gfl_sample



class MWCNNHaarDWT(nn.Module):
  
    def __init__(self, channels: int):
        super().__init__()
        self.C = channels
        T = torch.tensor([
            [ 0.5,  0.5,  0.5,  0.5],  # LL
            [ 0.5, -0.5,  0.5, -0.5],  # LH
            [ 0.5,  0.5, -0.5, -0.5],  # HL
            [ 0.5, -0.5, -0.5,  0.5],  # HH
        ], dtype=torch.float32)  # (4,4)
        # 深度可分组 1×1：每组 (4->4)
        w = torch.zeros(4 * channels, 4, 1, 1, dtype=torch.float32)
        for c in range(channels):
            w[4*c:4*c+4, :, 0, 0] = T
        self.register_buffer('w', w, persistent=False)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        assert C == self.C, f'channels mismatch: {C} vs {self.C}'
        assert (H % 2 == 0) and (W % 2 == 0), 'H,W must be divisible by 2 for Haar-DWT (MWCNN)'

        # 子像素打包：[B,C,H,W] -> [B,4C,H/2,W/2]
        x_un = F.pixel_unshuffle(x, 2)
        # 每通道组 4->4 的固定线性变换（无训练参数）
        y = F.conv2d(x_un, self.w, bias=None, stride=1, padding=0, groups=self.C)
        LL, LH, HL, HH = torch.chunk(y, 4, dim=1)
        return LL, LH, HL, HH


class MWCNNHaarIWT(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__()
        self.C = channels
        T = torch.tensor([
            [ 0.5,  0.5,  0.5,  0.5],
            [ 0.5, -0.5,  0.5, -0.5],
            [ 0.5,  0.5, -0.5, -0.5],
            [ 0.5, -0.5, -0.5,  0.5],
        ], dtype=torch.float32)
        w_inv = torch.zeros(4 * channels, 4, 1, 1, dtype=torch.float32)
        for c in range(channels):
            w_inv[4*c:4*c+4, :, 0, 0] = T
        self.register_buffer('w_inv', w_inv, persistent=False)

    def forward(self, LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor):
        B, C, h, w = LL.shape
        assert C == self.C
        y = torch.cat([LL, LH, HL, HH], dim=1)         # [B,4C,h,w]
        z = F.conv2d(y, self.w_inv, bias=None, stride=1, padding=0, groups=self.C)
        out = F.pixel_shuffle(z, 2)                    # [B,C,2h,2w]
        return out


class MBFusionBlock(nn.Module):

    def __init__(self,
                 n_feat: int,
                 kernel_size: int = 3,  
                 num_frame: int = 3,
                 conv_kernel: int = 3,  
                 gate_kernel: int = 1,  
                 gate_channelwise: bool = True,  
                 neighbor_mode: str = "tsm", 
                 temporal_shift_ratio: float = 0.25,  
                 use_spatial_shift: bool = True, 
                 spatial_shift_px: int = 1, 
                 use_hard_sigmoid: bool = True,  
                 gain: float = 1.0): 
        super().__init__()
        assert num_frame == 3, 
        assert neighbor_mode in ("avg", "tsm")
        self.T = num_frame
        self.C = n_feat
        self.neighbor_mode = neighbor_mode
        self.temporal_shift_ratio = float(temporal_shift_ratio)
        self.use_spatial_shift = bool(use_spatial_shift)
        self.spx = int(spatial_shift_px)
        self.gate_channelwise = bool(gate_channelwise)
        self.use_hard_sigmoid = use_hard_sigmoid
        self.gain = gain

        
        self.conv_branch = BinaryConv2dSkip1x1(3 * n_feat, n_feat, conv_kernel)

      
        gate_out = 2 * (n_feat if gate_channelwise else 1)
        self.gate = BinaryConv2dSkip1x1(3 * n_feat, gate_out, gate_kernel)

      
        self.body = BinaryConv2d(n_feat, n_feat, kernel_size)

        self.alpha_min: float = 0.50
        self.alpha_max: float = 0.85
        self.motion_momentum: float = 0.9
        self.register_buffer('motion_ema', torch.tensor(0.0))

      
        self.mw_dwt = MWCNNHaarDWT(n_feat)  
        self.mw_iwt = MWCNNHaarIWT(n_feat) 
       
        self.hf_refine = BinaryConv2d(3 * n_feat, 3 * n_feat, kernel_size=3)
        
        self.hf_scale = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _shift2d(x, dx, dy):  
        B, C, H, W = x.shape
        left, right = max(dx, 0), max(-dx, 0)
        top, bottom = max(dy, 0), max(-dy, 0)
        x_pad = F.pad(x, (left, right, top, bottom))
        return x_pad[:, :, bottom:bottom + H, left:left + W]

    def _spatial_shift_pack(self, f):
    
        B, C, H, W = f.shape
        g = max(1, C // 8) 
        c0 = C - 4 * g
        x0 = f[:, :c0]
        x1 = self._shift2d(f[:, c0:c0 + g], self.spx, 0) 
        x2 = self._shift2d(f[:, c0 + g:c0 + 2 * g], -self.spx, 0) 
        x3 = self._shift2d(f[:, c0 + 2 * g:c0 + 3 * g], 0, self.spx) 
        x4 = self._shift2d(f[:, c0 + 3 * g:c0 + 4 * g], 0, -self.spx) 
        return torch.cat([x0, x1, x2, x3, x4], dim=1)

    def _motion_intensity(self, f_prev, f_next):
       
        motion_raw = (f_next - f_prev).abs().mean(dim=(1, 2, 3), keepdim=True)  # [B,1,1,1]
        batch_mean = motion_raw.detach().mean()
        if self.motion_ema.item() == 0.0:
            self.motion_ema = batch_mean
        else:
            self.motion_ema = self.motion_momentum * self.motion_ema + (1.0 - self.motion_momentum) * batch_mean
        m = motion_raw / (self.motion_ema + 1e-6)
        return torch.clamp(m, 0.0, 2.0)  # [B,1,1,1]

    def _shift_branch(self, f_prev, f_cur,
                      f_next): 

        if self.neighbor_mode == "avg":
            y = f_cur + 0.5 * (f_prev + f_next)
        else:  # "tsm"
            C = f_cur.shape[1]
            k = max(1, int(C * self.temporal_shift_ratio))


        
            m = self._motion_intensity(f_prev, f_next)  # [B,1,1,1]
            alpha = torch.clamp(0.5 - 0.35 * m.mean(), self.alpha_min, self.alpha_max)

            y = f_cur.clone()
            y[:, :k] = alpha * f_prev[:, :k] + (1.0 - alpha) * f_cur[:, :k] 
            y[:, -k:] = alpha * f_next[:, -k:] + (1.0 - alpha) * f_cur[:, -k:] 

        if self.use_spatial_shift:
            y = self._spatial_shift_pack(y)
        return y

    def forward(self, x, reverse: bool = False):
        
        Bt, C, H, W = x.shape
        assert C == self.C and Bt % self.T == 0
        B = Bt // self.T

        xt = x.view(B, self.T, C, H, W)
        if reverse:
            xt = torch.flip(xt, dims=[1])

        outs = []
        for ti in range(self.T):
            f_cur = xt[:, ti]

            #f_prev = xt[:, ti - 1] if ti - 1 >= 0 else torch.zeros_like(f_cur)
            #f_next = xt[:, ti + 1] if ti + 1 < self.T else torch.zeros_like(f_cur)
            f_prev = xt[:, ti - 1] if ti - 1 >= 0 else f_cur
            f_next = xt[:, ti + 1] if ti + 1 < self.T else f_cur

         
            f_shift = self._shift_branch(f_prev, f_cur, f_next)

          
            cat3 = torch.cat([f_prev, f_cur, f_next], dim=1)
            f_conv = self.conv_branch(cat3)

            
            LL, LH, HL, HH = self.mw_dwt(f_conv) 
            HF = torch.cat([LH, HL, HH], dim=1)  
            HF = self.hf_refine(HF)  
            LH, HL, HH = torch.chunk(HF, 3, dim=1)  
            f_hf = self.mw_iwt(LL, LH, HL, HH) 
            f_conv = f_conv + self.hf_scale * f_hf  

         
            gate_raw = self.gate(cat3)
            if self.gate_channelwise:
                w_shift = torch.sigmoid(gate_raw[:, :C, :, :])  # [B,C,H,W]
                w_conv = torch.sigmoid(gate_raw[:, C:, :, :])
            else:
              
                w_shift = torch.sigmoid(gate_raw[:, 0:1, :, :]).expand(-1, C, -1, -1)
                w_conv = torch.sigmoid(gate_raw[:, 1:2, :, :]).expand(-1, C, -1, -1)

          
            fused = w_shift * f_shift + w_conv * f_conv  # [B,C,H,W]
            out_t = self.body(fused)  # [B,C,H,W]
            outs.append(out_t.unsqueeze(1))

        y = torch.cat(outs, dim=1)  # [B,3,C,H,W]
        if reverse:
            y = torch.flip(y, dims=[1])
        return y.view(Bt, C, H, W)  # [B*T,C,H,W]


"""
def phase_correlation_align(ref_frame, moving_frame):

    # 转灰度
    if ref_frame.ndim == 3 and ref_frame.shape[2] > 1:
        ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        mov_gray = cv2.cvtColor(moving_frame, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray, mov_gray = ref_frame, moving_frame
    ref_gray = np.float32(ref_gray)
    mov_gray = np.float32(mov_gray)

    # 相位相关
    shift, _ = cv2.phaseCorrelate(mov_gray, ref_gray)
    dx, dy = shift  # dx: 列方向, dy: 行方向

    # 平移矩阵
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    aligned = cv2.warpAffine(moving_frame, M, (moving_frame.shape[1], moving_frame.shape[0]),
                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REFLECT101)
    return aligned


"""


def load_checkpoint(
        model: torch.nn.Module,
        filename: str,
        map_location: Union[str, Callable, None] = None,
        strict: bool = False,
        logger: Optional[logging.Logger] = None,
        revise_keys: list = [(r'^module\.', '')]) -> Union[dict, OrderedDict]:
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Default: strip
            the prefix 'module.' by [(r'^module\\.', '')].

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')

    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
             for k, v in state_dict.items()})

    state_dict.pop('step_counter')

    # Keep metadata in state_dict
    state_dict._metadata = metadata

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class DABCWithInputConv(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                BinaryConv2d, num_blocks,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                bias=False))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)


class BBCUWithBinarizedInputConv(nn.Module):
    def __init__(self, in_channels, in_groups, out_channels=64, num_blocks=30, kernel_size=3):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(BinaryConv2dSkip1x1(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            groups=in_groups,
            bias=False
        ))

        # residual blocks
        main.append(
            make_layer(
                BinaryConv2d, num_blocks,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=False))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)


class BNNUnet(nn.Module):
    def __init__(self, n_feat=[24, 36, 48], n_block=[1, 3, 3]):
        super(BNNUnet, self).__init__()
        # Encoder
        self.encoder_level1 = make_layer(
            BinaryConv2d, n_block[0],
            in_channels=n_feat[0],
            out_channels=n_feat[0],
            kernel_size=3,
            bias=False)
        self.down12 = BNNDownSample(n_feat[0], n_feat[1])

        self.encoder_level2 = make_layer(
            BinaryConv2d, n_block[1],
            in_channels=n_feat[1],
            out_channels=n_feat[1],
            kernel_size=3,
            bias=False)
        self.down23 = BNNDownSample(n_feat[1], n_feat[2])

        self.encoder_level3 = make_layer(
            BinaryConv2d, n_block[2],
            in_channels=n_feat[2],
            out_channels=n_feat[2],
            kernel_size=3,
            bias=False)

        # Decoder
        self.decoder_level3 = make_layer(
            BinaryConv2d, n_block[2],
            in_channels=n_feat[2],
            out_channels=n_feat[2],
            kernel_size=3,
            bias=False)

        self.skip_conv2 = BinaryConv2d(n_feat[1], n_feat[1], 3)
        self.up32 = BNNSkipUpSample(n_feat[2], n_feat[1])
        self.decoder_level2 = make_layer(
            BinaryConv2d, n_block[1],
            in_channels=n_feat[1],
            out_channels=n_feat[1],
            kernel_size=3,
            bias=False)

        self.skip_conv1 = BinaryConv2d(n_feat[0], n_feat[0], 3)
        self.up21 = BNNSkipUpSample(n_feat[1], n_feat[0])
        self.decoder_level1 = make_layer(
            BinaryConv2d, n_block[0],
            in_channels=n_feat[0],
            out_channels=n_feat[0],
            kernel_size=3,
            bias=False)

    def forward(self, x):
        shortcut = x
        enc1 = self.encoder_level1(x)
        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)

        dec3 = self.decoder_level3(enc3)
        x = self.up32(dec3, self.skip_conv2(enc2))
        dec2 = self.decoder_level2(x)
        x = self.up21(dec2, self.skip_conv1(enc1))
        dec1 = self.decoder_level1(x)
        return dec1 + shortcut




class ShiftEncoder(nn.Module):
    def __init__(self, n_feat=[24, 80, 80, 80], num_frame=3, gate_kernel=1, gate_channelwise=True,
                 temporal_shift_ratio=0.25, use_spatial_shift=True, spatial_shift_px=1, gain=1.0):
        super(ShiftEncoder, self).__init__()

        self.conv_in = BinaryConv2d(n_feat[0], n_feat[0], kernel_size=3)

      
        self.encoder_level0 = MBFusionBlock(
            n_feat[0], kernel_size=3, num_frame=num_frame, conv_kernel=3, gate_kernel=gate_kernel,
            gate_channelwise=gate_channelwise, neighbor_mode="tsm", temporal_shift_ratio=temporal_shift_ratio,
            use_spatial_shift=use_spatial_shift, spatial_shift_px=spatial_shift_px, gain=gain
        )
        self.encoder_level0_1 = MBFusionBlock(
            n_feat[0], kernel_size=3, num_frame=num_frame, conv_kernel=3, gate_kernel=gate_kernel,
            gate_channelwise=gate_channelwise, neighbor_mode="tsm", temporal_shift_ratio=temporal_shift_ratio,
            use_spatial_shift=use_spatial_shift, spatial_shift_px=spatial_shift_px, gain=gain
        )
        self.down01 = BNNDownSample(n_feat[0], n_feat[1])

        self.encoder_level1 = MBFusionBlock(
            n_feat[1], kernel_size=3, num_frame=num_frame, conv_kernel=3, gate_kernel=gate_kernel,
            gate_channelwise=gate_channelwise, neighbor_mode="tsm", temporal_shift_ratio=temporal_shift_ratio,
            use_spatial_shift=use_spatial_shift, spatial_shift_px=spatial_shift_px, gain=gain
        )
        self.encoder_level1_1 = MBFusionBlock(
            n_feat[1], kernel_size=3, num_frame=num_frame, conv_kernel=3, gate_kernel=gate_kernel,
            gate_channelwise=gate_channelwise, neighbor_mode="tsm", temporal_shift_ratio=temporal_shift_ratio,
            use_spatial_shift=use_spatial_shift, spatial_shift_px=spatial_shift_px, gain=gain
        )
        self.down12 = BNNDownSample(n_feat[1], n_feat[2])

        self.encoder_level2 = BinaryConv2d(n_feat[2], n_feat[2], kernel_size=3)
        self.encoder_level2_1 = BinaryConv2d(n_feat[2], n_feat[2], kernel_size=3)
        self.down23 = BNNDownSample(n_feat[2], n_feat[3])

        self.encoder_level3 = BinaryConv2d(n_feat[3], n_feat[3], kernel_size=3)
        self.encoder_level3_1 = BinaryConv2d(n_feat[3], n_feat[3], kernel_size=3)

        self.skip_conv0 = BinaryConv2d(n_feat[0], n_feat[0], kernel_size=3)
        self.skip_conv1 = BinaryConv2d(n_feat[1], n_feat[1], kernel_size=3)
        self.skip_conv2 = BinaryConv2d(n_feat[2], n_feat[2], kernel_size=3)

  
        self.decoder_level3 = MBFusionBlock(
            n_feat[3], kernel_size=5, num_frame=num_frame, conv_kernel=3, gate_kernel=gate_kernel,
            gate_channelwise=gate_channelwise, neighbor_mode="tsm", temporal_shift_ratio=temporal_shift_ratio,
            use_spatial_shift=use_spatial_shift, spatial_shift_px=spatial_shift_px, gain=gain
        )
        self.decoder_level3_1 = MBFusionBlock(
            n_feat[3], kernel_size=5, num_frame=num_frame, conv_kernel=3, gate_kernel=gate_kernel,
            gate_channelwise=gate_channelwise, neighbor_mode="tsm", temporal_shift_ratio=temporal_shift_ratio,
            use_spatial_shift=use_spatial_shift, spatial_shift_px=spatial_shift_px, gain=gain
        )
        self.up32 = BNNSkipUpSample(n_feat[3], n_feat[2])

        self.decoder_level2 = MBFusionBlock(
            n_feat[2], kernel_size=5, num_frame=num_frame, conv_kernel=3, gate_kernel=gate_kernel,
            gate_channelwise=gate_channelwise, neighbor_mode="tsm", temporal_shift_ratio=temporal_shift_ratio,
            use_spatial_shift=use_spatial_shift, spatial_shift_px=spatial_shift_px, gain=gain
        )
        self.decoder_level2_1 = MBFusionBlock(
            n_feat[2], kernel_size=5, num_frame=num_frame, conv_kernel=3, gate_kernel=gate_kernel,
            gate_channelwise=gate_channelwise, neighbor_mode="tsm", temporal_shift_ratio=temporal_shift_ratio,
            use_spatial_shift=use_spatial_shift, spatial_shift_px=spatial_shift_px, gain=gain
        )
        self.up21 = BNNSkipUpSample(n_feat[2], n_feat[1])

        self.decoder_level1 = MBFusionBlock(
            n_feat[1], kernel_size=5, num_frame=num_frame, conv_kernel=3, gate_kernel=gate_kernel,
            gate_channelwise=gate_channelwise, neighbor_mode="tsm", temporal_shift_ratio=temporal_shift_ratio,
            use_spatial_shift=use_spatial_shift, spatial_shift_px=spatial_shift_px, gain=gain
        )
        self.decoder_level1_1 = MBFusionBlock(
            n_feat[1], kernel_size=5, num_frame=num_frame, conv_kernel=3, gate_kernel=gate_kernel,
            gate_channelwise=gate_channelwise, neighbor_mode="tsm", temporal_shift_ratio=temporal_shift_ratio,
            use_spatial_shift=use_spatial_shift, spatial_shift_px=spatial_shift_px, gain=gain
        )
        self.decoder_level1_2 = MBFusionBlock(
            n_feat[1], kernel_size=5, num_frame=num_frame, conv_kernel=3, gate_kernel=gate_kernel,
            gate_channelwise=gate_channelwise, neighbor_mode="tsm", temporal_shift_ratio=temporal_shift_ratio,
            use_spatial_shift=use_spatial_shift, spatial_shift_px=spatial_shift_px, gain=gain
        )
        self.up10 = BNNUpSample(n_feat[1], n_feat[0])

   
        self.conv_hr = BinaryConv2dSkip1x1(2 * n_feat[0], n_feat[0], 3)
        self.conv_out = BinaryConv2d(n_feat[0], n_feat[0], 3)

    def forward(self, x):
        n, t, c, h, w = x.shape
        x = self.conv_in(x.view(n * t, c, h, w))

        # Encoder 0
        enc0 = self.encoder_level0(x, reverse=False)
        enc0 = self.encoder_level0_1(enc0, reverse=True)
        enc0_down = self.down01(enc0)

        # Encoder 1
        enc1 = self.encoder_level1(enc0_down, reverse=False)
        enc1 = self.encoder_level1_1(enc1, reverse=True)
        enc1_down = self.down12(enc1)

        # Encoder 2
        enc2 = self.encoder_level2(enc1_down)
        enc2 = self.encoder_level2_1(enc2)
        enc2_down = self.down23(enc2)

        # Encoder 3
        enc3 = self.encoder_level3(enc2_down)
        enc3 = self.encoder_level3_1(enc3)

        # Decoder 3
        dec3 = self.decoder_level3(enc3)
        dec3 = self.decoder_level3_1(dec3)
        dec3_up = self.up32(dec3, self.skip_conv2(enc2))

        # Decoder 2
        dec2 = self.decoder_level2(dec3_up)
        dec2 = self.decoder_level2_1(dec2)
        dec2_up = self.up21(dec2, self.skip_conv1(enc1))

        # Decoder 1
        dec1 = self.decoder_level1(dec2_up)
        dec1 = self.decoder_level1_1(dec1)
        dec1 = self.decoder_level1_2(dec1)
        out = self.conv_out(self.conv_hr(torch.cat((self.up10(dec1), self.skip_conv0(x)), dim=1))).view(n, t, -1, h, w)

        return out[:, 0, :, :, :], out[:, 1, :, :, :], out[:, 2, :, :, :]



@BACKBONES.register_module()
class BNNVD(nn.Module):

    def __init__(self,
                 in_channels=4,
                 mid_channels=24,
                 feat_extract_blocks=3,
                 num_unets=1,
                 unet_n_feat=[24, 48, 96],
                 unet_n_block=[1, 3, 3],
                 stage1_n_feat=[24, 48, 48, 48],
                 task='Raw2Raw'):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.task = task

       
        self.k = self.k = (130 * mid_channels) / 64.0
        self.feat_extract = DABCWithInputConv(in_channels, mid_channels, feat_extract_blocks)

        self.stage0 = make_layer(
            BNNUnet, num_unets,
            n_feat=unet_n_feat,
            n_block=unet_n_block
        )

        self.stage1 = ShiftEncoder(stage1_n_feat)

        self.fusion = BinaryConv2dSkip1x1(3 * mid_channels, mid_channels, 3)
        self.stage2 = make_layer(
            BNNUnet, num_unets,
            n_feat=unet_n_feat,
            n_block=unet_n_block
        )

        self.conv_out = nn.Conv2d(mid_channels, 4, 3, 1, 1, bias=True)

    

    def forward_test(self, lqs):
      
        n, t, c, h, w = lqs.size()

        lqs = lqs * self.k

  
        feat_in_l = self.feat_extract(lqs[:, 0, :, :, :])
        feat_l = self.stage0(feat_in_l)

   
        feat_in_m = self.feat_extract(lqs[:, 1, :, :, :])
        feat_m = self.stage0(feat_in_m)

        outputs = []

   
        for i in range(2, t):
          
            feat_in_r = self.feat_extract(lqs[:, i, :, :, :])
            feat_r = self.stage0(feat_in_r)

          
            feat_in = feat_in_l.clone()
            feat_stage0 = feat_l.clone()

    

            feat_stage1, feat_l, feat_m = self.stage1(torch.stack([feat_l, feat_m, feat_r], dim=1))

     
            feat_in_l = feat_in_m
            feat_in_m = feat_in_r

           
            fusion_in = torch.cat([feat_in, feat_stage0, feat_stage1], dim=1)

            
            out = lqs + self.conv_out(self.stage2(self.fusion(fusion_in)))

          
            outputs.append(out)

       
        outputs = torch.stack(outputs, dim=1)

     
        return outputs / self.k

    def forward(self, lqs):

        n, t, c, h, w = lqs.size()
        lqs = lqs * self.k

      
        feat_in = self.feat_extract(lqs.view(-1, c, h, w))
        stage0_feat = self.stage0(feat_in)
        stage0_feat = stage0_feat.view(n, t, -1, h, w)

       
        stage1_feat = []
       
        feat_l = stage0_feat[:, 0, :, :, :]
        feat_m = stage0_feat[:, 1, :, :, :]
       
        for i in range(2, t):
            feat_r = stage0_feat[:, i, :, :, :]
            out, feat_l, feat_m = self.stage1(torch.stack([feat_l, feat_m, feat_r], dim=1))
            stage1_feat.append(out)

      

        stage1_feat.append(feat_l)
        stage1_feat.append(feat_m)
        stage1_feat = torch.stack(stage1_feat, dim=1)

    
        fusion_in = torch.cat([feat_in, stage0_feat.view(n * t, -1, h, w), stage1_feat.view(n * t, -1, h, w)], dim=1)
        out = lqs[:, :, :4, :, :] + self.conv_out(self.stage2(self.fusion(fusion_in))).view(n, t, -1, h, w)

        return out / self.k

    def init_weights(self, pretrained=None, strict=True):

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger, revise_keys=[(r'^generator\.', '')])
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


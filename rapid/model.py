"""RaPID CNN: residual U-Net with a self-attention bottleneck and multi-head output
(Section III-B of the paper).

The network receives pattern-shaped input channels (181 x 360) derived from the
physics-based composition and predicts a correction, in dB, that is added to the
composed pattern. Four output heads are summed with learned weights:

  bulk         overall correction, suppressed inside the main beam
  deep-null    refinement that is active in the null region only
  elevation    1-D sidelobe correction along theta through the beam peak
  azimuth      1-D sidelobe correction along phi through the beam peak

An auxiliary head predicts the depths of the deepest nulls; it is used for
supervision during training only.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rapid.config import MAINBEAM_DB, NPH, NTH

BASE_CHANNELS = 48
ATTN_HEADS = 8
TOP_K_NULLS = 10

# Input channels of the released model, in order.
CHANNELS = ("composition", "coupling_gap")

NULL_MASK_SLOPE = 0.85
NULL_MASK_MARGIN = 6.0
SIDELOBE_SLOPE = 0.70
MAIN_SHIELD_SLOPE = 1.25
MAIN_SHIELD_MARGIN = 2.5

# Bounds of the learned head weights
BULK_WEIGHT_MAX = 0.35
NULL_WEIGHT_MIN = 0.55
CUT_WEIGHT_MAX = 0.45


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.skip = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        s = self.skip(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + s)


class DownRes(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f = self.res(x)
        return f, self.pool(f)


class UpRes(nn.Module):
    """Decoder stage. With skip_ch = 0 the encoder features are not concatenated
    (the "w/o skip connections" ablation)."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.use_skip = skip_ch > 0
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.res = ResidualBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if self.use_skip:
            x = torch.cat([x, skip], dim=1)
        return self.res(x)


class SelfAttention2D(nn.Module):
    """Multi-head self-attention over the flattened spatial dimensions."""

    def __init__(self, channels, num_heads=ATTN_HEADS):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        h_in = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, self.head_dim, n)
        q, k, v = (t.transpose(-1, -2) for t in (qkv[:, 0], qkv[:, 1], qkv[:, 2]))
        attn = torch.softmax((q @ k.transpose(-1, -2)) * self.scale, dim=-1)
        out = (attn @ v).transpose(-1, -2).reshape(b, c, h, w)
        return h_in + self.proj(out)


class RaPIDNet(nn.Module):
    """Four-level residual U-Net (48 base channels, 21.1 M parameters).

    Args:
      in_ch       number of input channels
      base_idx    index of the input channel that holds the base pattern the
                  correction is added to (the physics-based composition)
      base_mean, base_std, res_mean, res_std
                  per-pixel training statistics of the base channel and of the
                  correction target; stored as buffers
      use_skips, use_heads
                  architecture ablations (Table IV)
    """

    def __init__(self, in_ch: int = len(CHANNELS), base_idx: int = 0,
                 base: int = BASE_CHANNELS, attn_heads: int = ATTN_HEADS,
                 top_k: int = TOP_K_NULLS,
                 base_mean: np.ndarray | None = None, base_std: np.ndarray | None = None,
                 res_mean: np.ndarray | None = None, res_std: np.ndarray | None = None,
                 use_skips: bool = True, use_heads: bool = True):
        super().__init__()
        self.base_idx = base_idx
        self.use_heads = use_heads

        self.d1 = DownRes(in_ch, base)
        self.d2 = DownRes(base, base * 2)
        self.d3 = DownRes(base * 2, base * 4)
        self.d4 = DownRes(base * 4, base * 8)
        self.bottleneck = ResidualBlock(base * 8, base * 16)
        self.attn = SelfAttention2D(base * 16, num_heads=attn_heads)
        k = 1 if use_skips else 0
        self.u1 = UpRes(base * 16, k * base * 8, base * 8)
        self.u2 = UpRes(base * 8, k * base * 4, base * 4)
        self.u3 = UpRes(base * 4, k * base * 2, base * 2)
        self.u4 = UpRes(base * 2, k * base, base)

        self.base_head = nn.Conv2d(base, 1, 1)
        self.null_refine = nn.Conv2d(base, 1, 1)
        nn.init.zeros_(self.null_refine.weight)
        nn.init.zeros_(self.null_refine.bias)

        b_dim = base * 16
        self.e_cut_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(b_dim, NTH))
        self.h_cut_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(b_dim, NPH))
        for head in (self.e_cut_head, self.h_cut_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

        self.null_depth_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(b_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, top_k))

        # learned weights of the four output heads
        self._w_bulk_logit = nn.Parameter(torch.tensor(0.0))
        self._w_null_logit = nn.Parameter(torch.tensor(1.0))
        self._w_e_logit = nn.Parameter(torch.tensor(-0.5))
        self._w_h_logit = nn.Parameter(torch.tensor(-0.5))

        def stat(a):
            if a is None:
                return torch.zeros(1, 1, NTH, NPH)
            return torch.from_numpy(np.asarray(a, dtype=np.float32))[None, None]

        self.register_buffer("base_mean", stat(base_mean))
        self.register_buffer("base_std", stat(base_std) if base_std is not None
                             else torch.ones(1, 1, NTH, NPH))
        self.register_buffer("res_mean", stat(res_mean))
        self.register_buffer("res_std", stat(res_std) if res_std is not None
                             else torch.ones(1, 1, NTH, NPH))

    # head weights
    @property
    def w_bulk(self):
        return torch.sigmoid(self._w_bulk_logit) * BULK_WEIGHT_MAX

    @property
    def w_null(self):
        return NULL_WEIGHT_MIN + (1.0 - NULL_WEIGHT_MIN) * torch.sigmoid(self._w_null_logit)

    @property
    def w_e(self):
        return torch.sigmoid(self._w_e_logit) * CUT_WEIGHT_MAX

    @property
    def w_h(self):
        return torch.sigmoid(self._w_h_logit) * CUT_WEIGHT_MAX

    def _encode_decode(self, x):
        s1, p1 = self.d1(x)
        s2, p2 = self.d2(p1)
        s3, p3 = self.d3(p2)
        s4, p4 = self.d4(p3)
        b = self.attn(self.bottleneck(p4))
        dec = self.u1(b, s4)
        dec = self.u2(dec, s3)
        dec = self.u3(dec, s2)
        dec = self.u4(dec, s1)
        if dec.shape[-2:] != x.shape[-2:]:
            dec = F.interpolate(dec, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return b, dec

    @staticmethod
    def _monotonic_null_depths(raw):
        d0 = -F.softplus(raw[:, :1])
        if raw.shape[1] == 1:
            return d0
        return torch.cat([d0, d0 + torch.cumsum(F.softplus(raw[:, 1:]), dim=1)], dim=1)

    @staticmethod
    def _cut_map(corr, base_db, sl_mask, plane: str):
        """Place a 1-D correction on the row/column through the beam peak."""
        b, _, h, w = base_db.shape
        flat_idx = base_db.view(b, -1).argmax(dim=1)
        bi = torch.arange(b, device=base_db.device)
        out = torch.zeros(b, 1, h, w, device=base_db.device, dtype=corr.dtype)
        if plane == "e":                      # along theta, at the peak phi
            phi_idx = flat_idx % w
            out[bi, 0, :, phi_idx] = corr * sl_mask[bi, 0, :, phi_idx]
        else:                                 # along phi, at the peak theta
            theta_idx = flat_idx // w
            out[bi, 0, theta_idx, :] = corr * sl_mask[bi, 0, theta_idx, :]
        return out

    def forward(self, x, return_aux: bool = False):
        """x: (B, in_ch, 181, 360) standardised inputs -> standardised correction (B, 1, 181, 360)."""
        base_n = x[:, self.base_idx:self.base_idx + 1]
        b, dec = self._encode_decode(x)

        base_db = base_n * self.base_std + self.base_mean
        rel = base_db - base_db.amax(dim=(-2, -1), keepdim=True)
        shield = torch.sigmoid(MAIN_SHIELD_SLOPE * (-rel - MAIN_SHIELD_MARGIN))
        null_w = torch.sigmoid(NULL_MASK_SLOPE * (-rel - NULL_MASK_MARGIN))
        sl_w = (torch.sigmoid(SIDELOBE_SLOPE * (-rel - MAINBEAM_DB))
                * torch.sigmoid(SIDELOBE_SLOPE * (rel + 20.0)))

        bulk = self.base_head(dec)
        residual = self.w_bulk * shield * bulk
        if self.use_heads:
            null_part = self.w_null * null_w * self.null_refine(dec)
            e_map = self.w_e * self._cut_map(self.e_cut_head(b), base_db, sl_w, "e")
            h_map = self.w_h * self._cut_map(self.h_cut_head(b), base_db, sl_w, "h")
            residual = residual + null_part + e_map + h_map

        if not return_aux:
            return residual
        return {"residual": residual,
                "null_depths": self._monotonic_null_depths(self.null_depth_head(b))}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

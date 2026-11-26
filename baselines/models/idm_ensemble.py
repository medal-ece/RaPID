"""Onat et al. (IEEE TAP vol. 72 no. 9, 2024) IDM-assisted ensemble EEP predictor,
re-implemented as a baseline trained on patterns of the target array size.

Reference: N. B. Onat, I. Roldan, F. Fioranelli, A. Yarovoy, Y. Aslan,
"Constrained Infinitesimal Dipole Modeling-Assisted Ensemble Prediction of
Embedded Element Patterns via Machine Learning," DOI 10.1109/TAP.2024.3433515.

Two branches, both fed the SAME 4-channel element tensor R = [x, y, Re w, Im w]
of shape (B, 4, n_elem) as the BO-CNN / CV-GNN baselines:

  DirectEEPNet  (paper Sec. III-A): Conv1D(16 kernels) over the element rows ->
      seven FC layers widening to a low-resolution pattern (paper 36x36; here
      37x72) -> ESPCN sub-pixel upscaler x5 (two 3x3 convs with 32 filters, then
      a 25-filter conv + PixelShuffle) -> 185x360, cropped to 181x360. Trained
      with the SSIM loss (11x11 Gaussian window, sigma 1.5).
  IDMCoefNet    (paper Sec. III-B): same Conv1D + seven FC layers -> 2*Nd outputs
      (Re/Im excitation of Nd magnetic infinitesimal dipoles). Trained with MSE
      on the coefficients; the pattern is the LINEAR dipole forward operator.
  Ensemble      (paper Sec. III-C): pixel-wise weighted sum in dB,
      P = beta*P_direct + (1-beta)*P_idm, beta chosen on validation per data size.

Conventions shared with the other baselines: Sigmoid/[0,1] targets
via PatternScaler, per-pixel mean-output bias initialisation (here a learnable
per-pixel output bias, since the ESPCN head has no per-pixel bias of its own).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NTH, NPH = 181, 360


def he_init_(module: nn.Module) -> nn.Module:
    """Kaiming-normal (He) init for every Linear/Conv layer. Required here: with
    PyTorch's default Linear init the seven-layer FC stack shrinks activation
    variance ~6x per ReLU layer, the low-res map reaches the sub-pixel stage at
    ~1e-2 scale and the head cannot fit even its training set (training MAE stuck
    at ~40 dB). The two-FC-layer BO-CNN is unaffected."""
    for mod in module.modules():
        if isinstance(mod, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
    return module


def _fc_stack(in_d: int, dims: tuple[int, ...], out_d: int, final_act: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_d
    for h in dims:
        layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
        d = h
    layers += [nn.Linear(d, out_d)]
    if final_act:
        layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class DirectEEPNet(nn.Module):
    """Direct pattern branch: Conv1D -> 7 FC -> low-res map -> ESPCN x5 -> Sigmoid."""

    def __init__(self, n_elem: int, n_c: int = 16, f_c: int = 3,
                 fc_dims: tuple[int, ...] = (64, 128, 256, 512, 768, 1024),
                 low_hw: tuple[int, int] = (37, 72), up: int = 5, low_ch: int = 1,
                 out_hw: tuple[int, int] = (NTH, NPH), espcn_ch: int = 32):
        """low_ch: channels of the low-resolution map the FC stack emits. The paper
        uses 1 (a 36x36 'low-resolution EEP'), adequate for a smooth element
        pattern; a whole-array pattern has sidelobe structure at the pixel scale,
        which one scalar per 5x5 output patch cannot encode (training MAE stuck at
        ~40 dB). low_ch>1 is the standard ESPCN
        feature-map usage and is the minimal deviation that lets the head fit."""
        super().__init__()
        self.low_hw, self.up, self.out_hw, self.low_ch = low_hw, up, out_hw, low_ch
        self.conv = nn.Sequential(nn.Conv1d(4, n_c, f_c, padding=f_c // 2), nn.ReLU(inplace=True))
        self.fc = _fc_stack(n_c * n_elem, fc_dims, low_hw[0] * low_hw[1] * low_ch, final_act=True)
        self.espcn = nn.Sequential(
            nn.Conv2d(low_ch, espcn_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(espcn_ch, espcn_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(espcn_ch, up * up, 3, padding=1),
            nn.PixelShuffle(up),
        )
        self.out_bias = nn.Parameter(torch.zeros(1, 1, out_hw[0], out_hw[1]))
        he_init_(self)

    def forward_logits(self, R: torch.Tensor) -> torch.Tensor:
        h = self.conv(R).flatten(1)
        low = self.fc(h).view(-1, self.low_ch, self.low_hw[0], self.low_hw[1])
        hi = self.espcn(low)[:, :, : self.out_hw[0], : self.out_hw[1]]
        return hi + self.out_bias

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(R)).flatten(1)      # (B, NTH*NPH) in [0,1]

    def set_output_bias(self, mean01: np.ndarray):
        with torch.no_grad():
            m = np.clip(mean01.reshape(self.out_hw), 1e-3, 1 - 1e-3).astype(np.float32)
            self.out_bias.copy_(torch.from_numpy(np.log(m / (1 - m)))[None, None].to(self.out_bias.device))
            self.espcn[-2].weight.mul_(0.1)   # small so the mean-init dominates at start

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class IDMCoefNet(nn.Module):
    """Physics branch: Conv1D -> 7 FC -> 2*Nd (Re, Im) dipole coefficients."""

    def __init__(self, n_elem: int, n_dip: int, n_c: int = 16, f_c: int = 3,
                 fc_dims: tuple[int, ...] = (64, 128, 256, 512, 768, 1024)):
        super().__init__()
        self.n_dip = n_dip
        self.conv = nn.Sequential(nn.Conv1d(4, n_c, f_c, padding=f_c // 2), nn.ReLU(inplace=True))
        self.fc = _fc_stack(n_c * n_elem, fc_dims, 2 * n_dip, final_act=False)
        he_init_(self)

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(R).flatten(1))                     # (B, 2*Nd)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class IDMForward:
    """Linear dipole forward operator: coefficients -> peak-normalised dB pattern."""

    def __init__(self, A_th: torch.Tensor, A_ph: torch.Tensor):
        self.A_th_T = A_th.T.contiguous()       # (Nd, M) complex64
        self.A_ph_T = A_ph.T.contiguous()

    def coef_from_output(self, out: torch.Tensor) -> torch.Tensor:
        nd = out.shape[1] // 2
        return torch.complex(out[:, :nd], out[:, nd:])

    def pattern_db(self, c: torch.Tensor, floor_db: float | None = None) -> torch.Tensor:
        """c: (B, Nd) complex -> (B, NTH*NPH) peak-normalised dB (optionally clipped)."""
        e_th = c @ self.A_th_T
        e_ph = c @ self.A_ph_T
        mag = torch.sqrt(e_th.abs() ** 2 + e_ph.abs() ** 2).clamp_min(1e-12)
        db = 20.0 * torch.log10(mag)
        db = db - db.max(dim=1, keepdim=True).values
        if floor_db is not None:
            db = db.clamp_min(floor_db)
        return db


# ── SSIM loss (paper Eq. 2-3) ──

def _gaussian_window(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w_2d = g[:, None] * g[None, :]
    return w_2d.unsqueeze(0).unsqueeze(0)


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0,
         window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Mean SSIM over a batch of (B,1,H,W) images."""
    win = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    pad = window_size // 2
    mu_p = F.conv2d(pred, win, padding=pad)
    mu_t = F.conv2d(target, win, padding=pad)
    mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
    sig_p = F.conv2d(pred * pred, win, padding=pad) - mu_p2
    sig_t = F.conv2d(target * target, win, padding=pad) - mu_t2
    sig_pt = F.conv2d(pred * target, win, padding=pad) - mu_pt
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_pt + c1) * (2 * sig_pt + c2)) / ((mu_p2 + mu_t2 + c1) * (sig_p + sig_t + c2))
    return ssim_map.mean()


def ssim_loss(pred01_flat: torch.Tensor, target01_flat: torch.Tensor) -> torch.Tensor:
    p = pred01_flat.view(-1, 1, NTH, NPH)
    t = target01_flat.view(-1, 1, NTH, NPH)
    return 1.0 - ssim(p, t, data_range=1.0)

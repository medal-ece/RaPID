"""BO-CNN baseline for array-pattern prediction (Zhang et al., EuCAP 2025).

Reference
---------
Y. Zhang, H. Jiang, Y. Bao, S. Liao, Q. Xue, "Efficient Array Pattern Prediction
Using a Deep Learning Algorithm." The array geometry and per-element complex
excitation are encoded as a 4-channel vector R = [x, y, Re(w), Im(w)] (one row
per element); a stack of 1-D convolution layers feeds fully-connected layers with
a final Sigmoid that regresses the min-max-normalised far-field pattern, trained
with MSE + Adam.

This module implements the network of their Section III-A. Its hyper-parameters
are chosen by the Bayesian optimisation of Section III-B, which is implemented
in baselines/bocnn_bo.py.

For a uniform-amplitude N x N planar phased array steered by per-step phases
(dphase_x, dphase_y) in degrees, element (ix, iy) has excitation
    w = exp(j * (ix * dpx_rad + iy * dpy_rad)),  |w| = 1,
so R[:, k] = [x_k, y_k, cos(phase_k), sin(phase_k)] for the k-th element.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ── Input encoding ────────────────────────────────────────────────────────────

def element_positions(n: int) -> np.ndarray:
    """Centred integer element positions (in spacing units) for an n x n grid.

    Returns (2, n*n): row 0 = x, row 1 = y, raster order (ix outer, iy inner).
    """
    ix, iy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x = (ix - (n - 1) / 2.0).ravel()
    y = (iy - (n - 1) / 2.0).ravel()
    return np.stack([x, y]).astype(np.float32)


def encode_R(dpx_deg: np.ndarray, dpy_deg: np.ndarray, n: int) -> np.ndarray:
    """Paper's 4-channel R encoding for a batch of steerings.

    Parameters
    ----------
    dpx_deg, dpy_deg : (B,) per-step steering phases in degrees.
    n : array side length (n x n elements).

    Returns
    -------
    (B, 4, n*n) float32 — channels [x, y, Re(w), Im(w)].
    """
    dpx_deg = np.asarray(dpx_deg, dtype=np.float64).ravel()
    dpy_deg = np.asarray(dpy_deg, dtype=np.float64).ravel()
    ix, iy = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    ix = ix.ravel().astype(np.float64)
    iy = iy.ravel().astype(np.float64)
    pos = element_positions(n)                       # (2, n*n)
    B = dpx_deg.shape[0]
    out = np.empty((B, 4, n * n), dtype=np.float32)
    dpx_r = np.deg2rad(dpx_deg)[:, None]
    dpy_r = np.deg2rad(dpy_deg)[:, None]
    phase = ix[None, :] * dpx_r + iy[None, :] * dpy_r   # (B, n*n)
    out[:, 0] = pos[0][None, :]
    out[:, 1] = pos[1][None, :]
    out[:, 2] = np.cos(phase).astype(np.float32)
    out[:, 3] = np.sin(phase).astype(np.float32)
    return out


def fit_R_minmax(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel min/max over a training batch of R (paper: min-max to [0,1])."""
    lo = R.min(axis=(0, 2)).astype(np.float32)       # (4,)
    hi = R.max(axis=(0, 2)).astype(np.float32)
    span = np.maximum(hi - lo, 1e-8).astype(np.float32)
    return lo, span


def apply_R_minmax(R: np.ndarray, lo: np.ndarray, span: np.ndarray) -> np.ndarray:
    return ((R - lo[None, :, None]) / span[None, :, None]).astype(np.float32)


# ── Target (pattern) normalisation ────────────────────────────────────────────

def peaknorm(x: np.ndarray) -> np.ndarray:
    """Shift each pattern so its peak is 0 dB (matches the eval metric protocol)."""
    return (x - x.max(axis=(1, 2), keepdims=True)).astype(np.float32)


class PatternScaler:
    """Min-max maps peak-normalised dB patterns to [0,1] for the Sigmoid head.

    vmax is fixed at 0 dB (peak); vmin is taken from the training floor so the
    map is invertible and covers the true dynamic range.
    """

    def __init__(self, vmin: float):
        self.vmin = float(vmin)
        self.vmax = 0.0

    @classmethod
    def from_train(cls, patterns_pn: np.ndarray, floor_percentile: float = 0.0):
        if floor_percentile <= 0:
            vmin = float(patterns_pn.min())
        else:
            vmin = float(np.percentile(patterns_pn, floor_percentile))
        return cls(vmin)

    def to01(self, x_db: np.ndarray) -> np.ndarray:
        z = (np.clip(x_db, self.vmin, self.vmax) - self.vmin) / (self.vmax - self.vmin)
        return z.astype(np.float32)

    def to_db(self, z01: np.ndarray) -> np.ndarray:
        return (z01 * (self.vmax - self.vmin) + self.vmin).astype(np.float32)


# ── Model ─────────────────────────────────────────────────────────────────────

class BOCNNBaseline(nn.Module):
    """BO-CNN: 1-D conv stack + FC head + Sigmoid.

    Parameters mirror the paper's hyper-parameter vector [n_c, f_c, L, Q]:
      n_c    : conv channel width
      f_c    : conv kernel size
      n_conv : number of convolution layers (L)
      fc_dims: hidden FC widths (Q = len(fc_dims) + 1 incl. output layer)
    """

    def __init__(
        self,
        n_elem: int,
        out_dim: int,
        n_c: int = 32,
        f_c: int = 3,
        n_conv: int = 3,
        fc_dims: tuple[int, ...] = (1024,),
        pool: int = 1,
    ):
        super().__init__()
        pad = f_c // 2
        convs: list[nn.Module] = []
        in_ch = 4
        length = n_elem
        for _ in range(n_conv):
            convs += [nn.Conv1d(in_ch, n_c, f_c, padding=pad), nn.ReLU(inplace=True)]
            if pool > 1 and length >= 2 * pool:
                convs += [nn.MaxPool1d(pool)]     # paper's xi: pooling factor (1 = none)
                length //= pool
            in_ch = n_c
        self.conv = nn.Sequential(*convs)

        flat = n_c * length
        fc: list[nn.Module] = []
        d = flat
        for h in fc_dims:
            fc += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            d = h
        fc += [nn.Linear(d, out_dim), nn.Sigmoid()]
        self.fc = nn.Sequential(*fc)

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        h = self.conv(R)                 # (B, n_c, n_elem)
        h = h.flatten(1)                 # (B, n_c*n_elem)
        return self.fc(h)                # (B, out_dim) in [0,1]

    def set_output_bias(self, mean01: np.ndarray):
        """Init the Sigmoid head to predict the per-pixel training mean (escapes
        the flat 'predict-a-constant' plateau the huge FC output gets stuck in)."""
        with torch.no_grad():
            m = np.clip(mean01, 1e-3, 1 - 1e-3).astype(np.float32)
            final = self.fc[-2]          # last Linear (fc[-1] is Sigmoid)
            final.bias.copy_(torch.from_numpy(np.log(m / (1 - m))).to(final.bias.device))
            final.weight.mul_(0.1)       # small (not zero) so mean-init dominates at start

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

"""Training loss, null-depth targets and weight EMA for the RaPID CNN."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import minimum_filter

from rapid.config import MAINBEAM_DB, NULL_THRESHOLD_DB
from rapid.model import TOP_K_NULLS

LAMBDA_L1 = 150.0
LAMBDA_SSIM = 0.5
LAMBDA_NULL_DEPTH = 8.0
LAMBDA_MAINBEAM = 55.0
LAMBDA_NULL_REGION = 24.0
LAMBDA_EPLANE = 8.0
LAMBDA_HPLANE = 8.0
NULL_PIXEL_WEIGHT = 3.0
SIDELOBE_PIXEL_WEIGHT = 2.5
EMA_DECAY = 0.999


def _gaussian_window(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)


def ssim_db(pred_db, target_db, data_range=60.0, window_size=11, sigma=1.5):
    """SSIM averaged over a batch of single-channel patterns in dB."""
    win = _gaussian_window(window_size, sigma, pred_db.device, pred_db.dtype)
    pad = window_size // 2
    mu_p = F.conv2d(pred_db, win, padding=pad)
    mu_t = F.conv2d(target_db, win, padding=pad)
    sig_p = F.conv2d(pred_db * pred_db, win, padding=pad) - mu_p * mu_p
    sig_t = F.conv2d(target_db * target_db, win, padding=pad) - mu_t * mu_t
    sig_pt = F.conv2d(pred_db * target_db, win, padding=pad) - mu_p * mu_t
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_p * mu_t + c1) * (2 * sig_pt + c2)) / (
        (mu_p * mu_p + mu_t * mu_t + c1) * (sig_p + sig_t + c2))
    return ssim_map.mean()


def topk_null_depths(pattern_db: np.ndarray, k: int = TOP_K_NULLS,
                     null_threshold_db: float = NULL_THRESHOLD_DB) -> np.ndarray:
    """Depths (dB below the peak) of the k deepest local minima in the null region."""
    peak = float(pattern_db.max())
    null_mask = pattern_db < (peak + null_threshold_db)
    is_min = (pattern_db == minimum_filter(pattern_db, size=5)) & null_mask
    if not is_min.any():
        return np.full(k, null_threshold_db, dtype=np.float32)
    depths = np.sort(pattern_db[is_min].astype(np.float32))[:k] - peak
    if len(depths) < k:
        depths = np.pad(depths, (0, k - len(depths)), constant_values=depths[-1])
    return depths.astype(np.float32)


class RaPIDLoss(nn.Module):
    """Region-weighted L1 on the correction, SSIM on the corrected pattern,
    region terms (main beam, null region, E/H-plane sidelobe cuts) and the
    auxiliary null-depth term.

    `anchor_main_beam`: inside the main beam the corrected pattern is pulled to
    the base pattern (the composition already reconstructs the main beam). When
    the base is the ideal array factor instead ("w/o composition" ablation) the
    main beam is pulled to the ground truth.
    """

    def __init__(self, res_mean, res_std, base_mean, base_std, anchor_main_beam: bool = True):
        super().__init__()
        self.anchor_main_beam = anchor_main_beam
        for name, a in (("res_mean", res_mean), ("res_std", res_std),
                        ("base_mean", base_mean), ("base_std", base_std)):
            self.register_buffer(name, torch.from_numpy(np.asarray(a, np.float32))[None, None])

    @staticmethod
    def _plane_cut_loss(pred, true, base_db, plane):
        b, _, h, w = pred.shape
        rel = base_db - base_db.amax(dim=(-2, -1), keepdim=True)
        flat_idx = base_db.view(b, -1).argmax(dim=1)
        bi = torch.arange(b, device=pred.device)
        if plane == "e":
            idx = flat_idx % w
            cp, ct, rc = pred[bi, 0, :, idx], true[bi, 0, :, idx], rel[bi, 0, :, idx]
        else:
            idx = flat_idx // w
            cp, ct, rc = pred[bi, 0, idx, :], true[bi, 0, idx, :], rel[bi, 0, idx, :]
        sl = ((rc < -MAINBEAM_DB) & (rc >= -20.0)).float()
        if sl.sum() < 1:
            return torch.zeros((), device=pred.device)
        return (torch.abs(cp - ct) * sl).sum() / sl.sum().clamp(min=1.0)

    def forward(self, out, target_res_n, null_mask, target_null_depths, base_n, aux_scale=1.0):
        pred_db = out["residual"] * self.res_std + self.res_mean
        target_db = target_res_n * self.res_std + self.res_mean
        base_db = base_n * self.base_std + self.base_mean
        pattern_pred = base_db + pred_db
        pattern_true = base_db + target_db

        rel = base_db - base_db.amax(dim=(-2, -1), keepdim=True)
        sidelobe = ((rel < -MAINBEAM_DB) & (rel >= -20.0)).float()
        weight = 1.0 + (NULL_PIXEL_WEIGHT - 1.0) * null_mask + (SIDELOBE_PIXEL_WEIGHT - 1.0) * sidelobe
        l1 = (torch.abs(pred_db - target_db) * weight).mean()
        recon = LAMBDA_L1 * l1 + LAMBDA_SSIM * (1.0 - ssim_db(pattern_pred, pattern_true))

        null_depth = F.l1_loss(out["null_depths"], target_null_depths)

        main_mask = (rel >= -MAINBEAM_DB).float()
        main_ref = base_db if self.anchor_main_beam else pattern_true
        mainbeam = (torch.abs(pattern_pred - main_ref) * main_mask).sum() / main_mask.sum().clamp(min=1.0)

        null_region = ((torch.abs(pattern_pred - pattern_true) * null_mask).sum()
                       / null_mask.sum().clamp(min=1.0))
        eplane = self._plane_cut_loss(pattern_pred, pattern_true, base_db, "e")
        hplane = self._plane_cut_loss(pattern_pred, pattern_true, base_db, "h")

        return recon + aux_scale * (LAMBDA_NULL_DEPTH * null_depth + LAMBDA_MAINBEAM * mainbeam
                                    + LAMBDA_NULL_REGION * null_region
                                    + LAMBDA_EPLANE * eplane + LAMBDA_HPLANE * hplane)


class EMA:
    """Exponential moving average of the model weights (the saved checkpoint)."""

    def __init__(self, module: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow = copy.deepcopy(module.state_dict())

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for k, v in module.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_((1.0 - self.decay) * v.detach())
            else:
                self.shadow[k].copy_(v.detach())

    def state_dict(self):
        return self.shadow

    def swap_into(self, module: nn.Module):
        live = copy.deepcopy(module.state_dict())
        module.load_state_dict(self.shadow)
        return live

    def restore(self, module: nn.Module, live) -> None:
        module.load_state_dict(live)

"""Inference with a trained RaPID model: composed pattern in, predicted pattern out.

    prediction = peak-normalise( composition + CNN correction )

The CNN correction is used exactly as the network outputs it. The only
post-processing is that it is switched off inside the 3 dB main-beam core, where
the physics-based composition is kept unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rapid.composition import peaknorm
from rapid.config import (DEVICE, MAINBEAM_DB, NPH, NTH, RAPID_CKPT, RAPID_CKPT_DIR,
                          SPLITS_JSON, compose_npz)
from rapid.model import CHANNELS, RaPIDNet

INFER_BATCH = 8


# ── splits ───────────────────────────────────────────────────────────────────
def load_splits() -> dict:
    """Sample ids of every split (see data/README.md)."""
    with open(SPLITS_JSON) as f:
        return json.load(f)


def split_indices(ids: np.ndarray, split_ids) -> np.ndarray:
    """Row indices of `split_ids` inside an array ordered like `ids`."""
    pos = {int(v): i for i, v in enumerate(ids)}
    return np.array([pos[int(v)] for v in split_ids if int(v) in pos], dtype=np.int64)


# ── model ────────────────────────────────────────────────────────────────────
def load_model(ckpt: Path | str = RAPID_CKPT, device=DEVICE) -> tuple[RaPIDNet, dict]:
    """Load a checkpoint written by scripts/train.py. Returns (model, config)."""
    blob = torch.load(ckpt, map_location=device, weights_only=True)
    cfg = blob["config"]
    model = RaPIDNet(in_ch=len(cfg["channels"]), base_idx=cfg["base_idx"],
                     use_skips=cfg.get("use_skips", True),
                     use_heads=cfg.get("use_heads", True)).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, cfg


# ── input channels ───────────────────────────────────────────────────────────
def pixel_stats(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = a.astype(np.float32)
    return a.mean(0).astype(np.float32), np.maximum(a.std(0), 1e-6).astype(np.float32)


def input_stats_from_patterns(composition: np.ndarray, af: np.ndarray) -> dict:
    """Per-pixel input statistics of one array size.

    Computed from composed patterns and array factors only (no ground truth), on
    the steering directions of the training split.
    """
    comp_mean, comp_std = pixel_stats(composition)
    gap_mean, gap_std = pixel_stats(composition.astype(np.float32) - af.astype(np.float32))
    af_mean, af_std = pixel_stats(af)
    return {"composition_mean": comp_mean, "composition_std": comp_std,
            "coupling_gap_mean": gap_mean, "coupling_gap_std": gap_std,
            "af_mean": af_mean, "af_std": af_std}


def shipped_input_stats(n: int) -> dict:
    """Input statistics shipped with the released checkpoint for array size n."""
    path = RAPID_CKPT_DIR / f"input_stats_{n}x{n}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"no input statistics for {n}x{n} ({path.name}). Build the composition for "
            f"this size with scripts/build_compose.py; it writes them.")
    return dict(np.load(path))


def build_inputs(composition: np.ndarray, af: np.ndarray, stats: dict,
                 channels=CHANNELS) -> np.ndarray:
    """Standardised (N, C, 181, 360) network input.

    composition   physics-composed pattern, peak-normalised dB
    coupling_gap  composition minus the ideal array factor: where and by how
                  much the measured sub-blocks deviate from isolated elements
    af            ideal array factor (only used by the "w/o composition" ablation)
    """
    maps = {
        "composition": lambda: (composition - stats["composition_mean"]) / stats["composition_std"],
        "coupling_gap": lambda: (composition - af - stats["coupling_gap_mean"]) / stats["coupling_gap_std"],
        "af": lambda: (af - stats["af_mean"]) / stats["af_std"],
    }
    return np.stack([maps[c]() for c in channels], axis=1).astype(np.float32)


# ── prediction ───────────────────────────────────────────────────────────────
def shield_main_beam(correction_db: np.ndarray, base_db: np.ndarray) -> np.ndarray:
    """Zero the correction inside the 3 dB main-beam core of the base pattern."""
    out = correction_db.copy()
    rel = base_db - base_db.max(axis=(-2, -1), keepdims=True)
    out[rel >= -MAINBEAM_DB] = 0.0
    return out


@torch.no_grad()
def predict_correction(model: RaPIDNet, x: np.ndarray, device=DEVICE) -> np.ndarray:
    """Network correction in dB for standardised inputs x (N, C, 181, 360)."""
    res_mean = model.res_mean[0, 0].cpu().numpy()
    res_std = model.res_std[0, 0].cpu().numpy()
    out = np.empty((x.shape[0], NTH, NPH), dtype=np.float32)
    for i0 in range(0, x.shape[0], INFER_BATCH):
        xb = torch.from_numpy(x[i0:i0 + INFER_BATCH]).to(device)
        out[i0:i0 + INFER_BATCH] = model(xb).squeeze(1).float().cpu().numpy() * res_std + res_mean
    return out


def predict_patterns(model: RaPIDNet, cfg: dict, composition: np.ndarray, af: np.ndarray,
                     stats: dict, device=DEVICE) -> np.ndarray:
    """Predicted far-field patterns, peak-normalised dB, (N, 181, 360)."""
    x = build_inputs(composition, af, stats, cfg["channels"])
    base = composition if cfg["channels"][cfg["base_idx"]] == "composition" else af
    correction = shield_main_beam(predict_correction(model, x, device), base)
    return peaknorm(base + correction)


# ── processed compositions ───────────────────────────────────────────────────
def load_composition(n: int) -> dict:
    """Arrays written by scripts/build_compose.py for array size n."""
    path = compose_npz(n)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/build_compose.py --size {n}")
    d = np.load(path)
    out = {"composition": d["composed_dB"].astype(np.float32), "ids": d["ids"],
           "dpx": d["dpx"].astype(np.float64), "dpy": d["dpy"].astype(np.float64)}
    if "truth_dB" in d.files:
        out["truth"] = d["truth_dB"].astype(np.float32)
    return out

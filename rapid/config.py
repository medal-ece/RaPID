"""Paths and constants shared by the RaPID code base."""
from __future__ import annotations

import os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw HFSS exports live outside git. Point RAPID_DATA_DIR somewhere else if needed.
DATA_DIR = Path(os.environ.get("RAPID_DATA_DIR", PROJECT_ROOT / "data"))
PROCESSED_DIR = PROJECT_ROOT / "processed"      # derived arrays (not tracked)
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
EXAMPLES_DIR = PROJECT_ROOT / "examples"

SPLITS_JSON = PROJECT_ROOT / "data" / "splits.json"
RAPID_CKPT_DIR = CHECKPOINTS_DIR / "rapid"
RAPID_CKPT = RAPID_CKPT_DIR / "rapid.pt"

# Raw dataset folders (see data/README.md for the CSV layouts)
SUBBLOCK_FIELD_DIR = DATA_DIR / "6x6_subblock_fields"        # complex rE of the nine 2x2 sub-blocks
SUBBLOCK_MAG_DIR = DATA_DIR / "6x6_subblock_magnitude"       # dB magnitude of the nine 2x2 sub-blocks
ARRAY_DIR_FMT = "{n}x{n}_array"                             # whole-array patterns (dB, peak-normalised)


def array_dir(n: int) -> Path:
    return DATA_DIR / ARRAY_DIR_FMT.format(n=n)


def compose_npz(n: int) -> Path:
    return PROCESSED_DIR / f"compose_{n}x{n}.npz"


# Far-field sampling used throughout: theta 0..180 deg (1 deg), phi -179.5..179.5 deg (1 deg)
NTH, NPH = 181, 360

NULL_THRESHOLD_DB = -20.0       # below this (relative to the peak) is the null region
MAINBEAM_DB = 3.0               # main-beam core: within 3 dB of the peak

BATCH_SIZE = 16
RANDOM_SEED = 42

# Set RAPID_DEVICE=cpu to force CPU execution.
DEVICE = torch.device(os.environ.get("RAPID_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

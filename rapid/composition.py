"""Physics-based composition (Section III-A of the paper).

A 6x6 array is simulated once per steering direction with each of its nine 2x2
sub-blocks excited in turn while the other elements stay in place. Every
sub-block pattern therefore already contains the element patterns and the mutual
coupling of its neighbourhood (corner, edge or interior). An arbitrarily large
NxN array is then built by tiling 2x2 sub-blocks: each tile reuses the measured
sub-block of the same position class and is moved to its new location with a
phase shift (Eq. (3)-(4)). No full-wave simulation of the large array is needed.

This module holds the CSV parsers for the HFSS exports, the composition itself
and the ideal array factor (the analytical pattern that ignores mutual coupling).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from rapid.config import NPH, NTH

N_SUB = 9          # 2x2 sub-blocks in a 6x6 array (row-major, 3x3)
N_QUANT = 4        # re/im of E_phi, re/im of E_theta
MV_THRESHOLD = 10.0
KD = np.pi         # half-wavelength element spacing

PHI_DEG = np.linspace(-179.5, 179.5, NPH, dtype=np.float64)
THETA_DEG = np.arange(NTH, dtype=np.float64)
TH = np.deg2rad(THETA_DEG)[:, None]             # (NTH, 1)
PH = np.deg2rad(PHI_DEG)[None, :]               # (1, NPH)
SIN_TH = np.sin(TH)
COS_PH = np.cos(PH)
SIN_PH = np.sin(PH)


# ── sub-block bookkeeping ────────────────────────────────────────────────────
def subblock_for_tile(m: int, n: int, grid_n: int) -> int:
    """6x6 sub-block (0..8, row-major) reused at tile (m, n) of a grid_n x grid_n tiling.

    Corner tiles take the matching corner sub-block, edge tiles the matching edge
    sub-block and every interior tile takes the centre sub-block.
    """
    row = 0 if m == 0 else (2 if m == grid_n - 1 else 1)
    col = 0 if n == 0 else (2 if n == grid_n - 1 else 1)
    return 3 * row + col


def src_pos_6x6(b_idx: int) -> tuple[int, int]:
    """Position of sub-block b_idx inside the 6x6 array, in element units."""
    m6, n6 = divmod(b_idx, 3)
    return 2 * m6 - 2, 2 * n6 - 2


def tile_class_map(grid_n: int) -> np.ndarray:
    """(grid_n, grid_n) map of tile classes: 0 = corner, 1 = edge, 2 = interior."""
    cm = np.zeros((grid_n, grid_n), dtype=np.int64)
    for m in range(grid_n):
        for n in range(grid_n):
            b = subblock_for_tile(m, n, grid_n)
            cm[m, n] = 0 if b in (0, 2, 6, 8) else (2 if b == 4 else 1)
    return cm


# ── CSV parsers ──────────────────────────────────────────────────────────────
def _check_rows(path: Path, n_rows: int) -> None:
    if n_rows != NTH * NPH:
        raise ValueError(f"{path.name}: got {n_rows} data rows, expected {NTH * NPH} "
                         f"(file may be truncated)")


def parse_subblock_field_csv(path: Path):
    """Complex far fields of the nine 2x2 sub-blocks of a 6x6 array.

    Layout: row 0 header (theta_deg, phi_deg, then per sample and sub-block
    sXXXXX_bK_reEPhi, _imEPhi, _reETheta, _imETheta), row 1 dphase_x (deg),
    row 2 dphase_y (deg), then NPH blocks of NTH rows.

    Returns E_theta, E_phi : (Ns, 9, NTH, NPH) complex64; dpx, dpy : (Ns,) deg;
    ids : (Ns,) int.
    """
    with open(path) as f:
        rows = list(csv.reader(f))
    header, dpx_row, dpy_row, data = rows[0], rows[1], rows[2], rows[3:]
    _check_rows(path, len(data))

    n_field = len(header) - 2
    stride = N_SUB * N_QUANT
    if n_field % stride != 0:
        raise ValueError(f"{path.name}: bad column count {n_field}")
    n_samp = n_field // stride

    field = np.array(data, dtype=np.float64)[:, 2:2 + n_field]
    # HFSS writes some columns in mV; the unit suffix is not in the header, so it
    # is detected from the value range.
    is_mv = np.abs(field).max(axis=0) > MV_THRESHOLD
    if is_mv.any():
        field = field.copy()
        field[:, is_mv] *= 1e-3

    dpx = np.array([float(dpx_row[2 + s * stride]) for s in range(n_samp)])
    dpy = np.array([float(dpy_row[2 + s * stride]) for s in range(n_samp)])
    ids = np.array([int(header[2 + s * stride].split("_")[0][1:]) for s in range(n_samp)])

    body = field.reshape(NPH, NTH, n_samp, N_SUB, N_QUANT)
    body = np.transpose(body, (2, 3, 1, 0, 4))           # (Ns, 9, NTH, NPH, 4)
    e_phi = body[..., 0] + 1j * body[..., 1]
    e_theta = body[..., 2] + 1j * body[..., 3]
    return e_theta.astype(np.complex64), e_phi.astype(np.complex64), dpx, dpy, ids


def read_sample_ids(path: Path, cols_per_sample: int = 1) -> np.ndarray:
    """Sample ids of a CSV from its header line only (cheap)."""
    with open(path) as f:
        header = next(csv.reader(f))
    n_samp = (len(header) - 2) // cols_per_sample
    return np.array([int(header[2 + s * cols_per_sample].split("_")[0][1:])
                     for s in range(n_samp)])


def parse_magnitude_csv(path: Path, cols_per_sample: int = 1):
    """Magnitude patterns in dB, `cols_per_sample` columns per sample.

    Used for whole-array patterns (1 column per sample) and for the magnitude
    export of the nine 6x6 sub-blocks (9 columns per sample). Layout: row 0
    header, row 1 dphase_x, row 2 dphase_y, rows 3-4 peak direction, data from
    row 5.

    Returns pat : (Ns, cols_per_sample, NTH, NPH) float32 (the second axis is
    dropped when cols_per_sample == 1); dpx, dpy : (Ns,) deg; ids : (Ns,) int.
    """
    with open(path) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    nb = cols_per_sample
    n_samp = (len(header) - 2) // nb
    blk = np.array(rows[5:], dtype=np.float32)
    _check_rows(path, blk.shape[0])
    pat = blk[:, 2:2 + n_samp * nb].T.reshape(n_samp, nb, NPH, NTH)
    pat = np.transpose(pat, (0, 1, 3, 2))
    dpx = np.array([float(rows[1][2 + nb * s]) for s in range(n_samp)])
    dpy = np.array([float(rows[2][2 + nb * s]) for s in range(n_samp)])
    ids = np.array([int(header[2 + nb * s].split("_")[0][1:]) for s in range(n_samp)])
    if nb == 1:
        pat = pat[:, 0]
    return pat, dpx, dpy, ids


# ── composition (Eq. (3)-(4)) ────────────────────────────────────────────────
def compose(e_theta_sub, e_phi_sub, dpx_deg, dpy_deg, n: int, return_complex: bool = False):
    """Compose the pattern of an n x n array from 6x6 sub-block fields.

    Each sub-block field already carries the phase of its position in the 6x6
    array, so only the phase of the displacement (dx, dy) to the new tile
    position is applied:

        psi = pi sin(theta) (dx cos(phi) + dy sin(phi)) + dx dpx + dy dpy

    Args:
      e_theta_sub, e_phi_sub : (Ns, 9, NTH, NPH) complex sub-block fields
      dpx_deg, dpy_deg       : (Ns,) per-element steering phases, degrees
      n                      : array side (even, >= 4)

    Returns |E| as (Ns, NTH, NPH) float32, or (E_theta, E_phi) complex64 when
    return_complex is True.
    """
    if n % 2 or n < 4:
        raise ValueError("array side must be even and >= 4")
    grid_n, offset = n // 2, n // 2 - 1
    ns = e_theta_sub.shape[0]
    dpx_r = np.deg2rad(np.asarray(dpx_deg, dtype=np.float64))
    dpy_r = np.deg2rad(np.asarray(dpy_deg, dtype=np.float64))

    e_th = np.zeros((ns, NTH, NPH), dtype=np.complex64)
    e_ph = np.zeros((ns, NTH, NPH), dtype=np.complex64)
    for m in range(grid_n):
        for k in range(grid_n):
            b = subblock_for_tile(m, k, grid_n)
            x6, y6 = src_pos_6x6(b)
            dx, dy = 2 * m - offset - x6, 2 * k - offset - y6
            if dx == 0 and dy == 0:
                e_th += e_theta_sub[:, b]
                e_ph += e_phi_sub[:, b]
                continue
            psi_geom = KD * SIN_TH * (dx * COS_PH + dy * SIN_PH)
            psi_steer = (dx * dpx_r + dy * dpy_r).reshape(ns, 1, 1)
            phase = np.exp(1j * (psi_geom + psi_steer)).astype(np.complex64)
            e_th += e_theta_sub[:, b] * phase
            e_ph += e_phi_sub[:, b] * phase

    if return_complex:
        return e_th, e_ph
    return np.sqrt(np.abs(e_th) ** 2 + np.abs(e_ph) ** 2).astype(np.float32)


def magnitude_compose_6x6(sub_db, dpx_deg, dpy_deg):
    """6x6 composition from sub-block magnitudes only (no measured phase).

    The nine sub-block magnitudes are placed with the analytical position and
    steering phase. Unlike the complex composition this does not reproduce the
    6x6 pattern exactly, which makes it a usable training task at the 6x6 scale.

    sub_db : (Ns, 9, NTH, NPH) dB.  Returns (Ns, NTH, NPH) float32, peak-normalised dB.
    """
    ns = sub_db.shape[0]
    out = np.empty((ns, NTH, NPH), dtype=np.float32)
    for s in range(ns):
        dpx_r, dpy_r = np.deg2rad(dpx_deg[s]), np.deg2rad(dpy_deg[s])
        lin = 10.0 ** (sub_db[s].astype(np.float64) / 20.0)
        tot = np.zeros((NTH, NPH), dtype=np.complex128)
        for b in range(N_SUB):
            x, y = src_pos_6x6(b)
            psi = KD * SIN_TH * (x * COS_PH + y * SIN_PH) + x * dpx_r + y * dpy_r
            tot += lin[b] * np.exp(1j * psi)
        db = 20.0 * np.log10(np.maximum(np.abs(tot), 1e-12))
        out[s] = (db - db.max()).astype(np.float32)
    return out


# ── analytical pattern (ideal array factor, no mutual coupling) ──────────────
def array_factor_db(n: int, dpx_deg, dpy_deg) -> np.ndarray:
    """Ideal n x n array factor for progressive steering phases, peak-normalised dB.

    Isotropic, mutually isolated elements on a half-wavelength square lattice.
    Returns (Ns, NTH, NPH) float32.
    """
    dpx_deg = np.atleast_1d(np.asarray(dpx_deg, dtype=np.float64))
    dpy_deg = np.atleast_1d(np.asarray(dpy_deg, dtype=np.float64))
    kx = KD * SIN_TH * COS_PH
    ky = KD * SIN_TH * SIN_PH
    idx = (np.arange(n) - (n - 1) / 2.0)[:, None, None]
    out = np.empty((len(dpx_deg), NTH, NPH), dtype=np.float32)
    for s in range(len(dpx_deg)):
        af_x = np.exp(1j * idx * (kx[None] + np.deg2rad(dpx_deg[s]))).sum(0)
        af_y = np.exp(1j * idx * (ky[None] + np.deg2rad(dpy_deg[s]))).sum(0)
        db = 20.0 * np.log10(np.maximum(np.abs(af_x * af_y), 1e-12))
        out[s] = (db - db.max()).astype(np.float32)
    return out


# ── small helpers ────────────────────────────────────────────────────────────
def to_db_peaknorm(amp: np.ndarray) -> np.ndarray:
    """Linear magnitude -> dB, normalised to the per-sample peak."""
    db = 20.0 * np.log10(np.maximum(amp, 1e-12))
    return (db - db.max(axis=(-2, -1), keepdims=True)).astype(np.float32)


def peaknorm(db: np.ndarray) -> np.ndarray:
    """Re-reference dB patterns (..., NTH, NPH) to their own peak."""
    return (db - db.max(axis=(-2, -1), keepdims=True)).astype(np.float32)

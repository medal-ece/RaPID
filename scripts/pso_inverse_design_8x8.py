"""Antenna array design with RaPID as the surrogate model (Section IV-E).

A particle swarm optimiser (PSO) searches the element excitation phases of an
8x8 array so that every sidelobe stays below a ceiling while the main beam keeps
pointing at the target direction. Every candidate is scored with RaPID instead
of a full-wave simulation.

Design variables. The amplitude is uniform. On top of the steering phase a phase
offset is applied per element class (corner, edge and interior 2x2 sub-blocks),
so the search has three variables. A common phase on a 2x2 sub-block multiplies
that sub-block's field by exp(j*phase), which the physics-based composition
represents exactly.

The steering direction and the sub-block fields come from the example shipped in
examples/ (main beam at theta = 39 deg, phi = -172.5 deg).

Usage:
    python -m scripts.pso_inverse_design_8x8
    python -m scripts.pso_inverse_design_8x8 --quick      # short run

Outputs (results/pso_inverse_8x8/): the optimised phases of all 64 elements
(CSV, ready for a full-wave check), the predicted patterns and a figure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rapid.composition import (COS_PH, KD, PHI_DEG, SIN_PH, SIN_TH, THETA_DEG, src_pos_6x6,
                               subblock_for_tile, tile_class_map)
from rapid.config import EXAMPLES_DIR, NPH, NTH, RAPID_CKPT, RESULTS_DIR
from rapid.inference import load_model, predict_patterns, shipped_input_stats

N = 8                         # array side
GRID = N // 2                 # 4x4 tiles of 2x2 sub-blocks
OUT_DIR = RESULTS_DIR / "pso_inverse_8x8"

# Design specification
TARGET_THETA, TARGET_PHI = 39.0, -172.5   # deg, main-beam direction of the example
SIDELOBE_CEILING_DB = -20.0               # every sidelobe should stay below this
BEAM_CORE_DEG = 8.0                       # cone the main beam must keep filled
SIDELOBE_START_DEG = 20.0                 # sidelobe region starts beyond this angle
BEAM_FLOOR_DB = -3.0
W_BEAM, W_SIDE, W_POINT = 1.0, 1.0, 0.5

# PSO
SWARM, ITERS = 40, 80
INERTIA_HI, INERTIA_LO = 0.72, 0.40
C1, C2 = 1.6, 1.6
PHASE_MIN, PHASE_MAX = -np.pi, np.pi
V_CLAMP = 1.0

CLASS_MAP = tile_class_map(GRID)          # 0 corner, 1 edge, 2 interior
CLASS_NAMES = ["corner", "edge", "interior"]


class Surrogate:
    """Sub-block excitation weights (P, 4, 4) complex -> predicted pattern (P, 181, 360) dB."""

    def __init__(self, model, cfg, e_theta, e_phi, dpx_deg, dpy_deg):
        self.model, self.cfg = model, cfg
        self.e_theta, self.e_phi = e_theta, e_phi           # (9, NTH, NPH) complex
        self.stats = shipped_input_stats(N)
        dpx_r, dpy_r = np.deg2rad(dpx_deg), np.deg2rad(dpy_deg)
        # per-element phase factors of the ideal array factor
        xs = np.arange(N) - (N - 1) / 2.0
        self.ex = np.stack([np.exp(1j * x * (KD * SIN_TH * COS_PH + dpx_r)) for x in xs]).astype(np.complex64)
        self.ey = np.stack([np.exp(1j * y * (KD * SIN_TH * SIN_PH + dpy_r)) for y in xs]).astype(np.complex64)
        # per-tile sub-block fields already moved to their 8x8 position (Eq. (3)-(4))
        off = GRID - 1
        self.tile_th = np.empty((GRID, GRID, NTH, NPH), np.complex64)
        self.tile_ph = np.empty((GRID, GRID, NTH, NPH), np.complex64)
        for m in range(GRID):
            for k in range(GRID):
                b = subblock_for_tile(m, k, GRID)
                x6, y6 = src_pos_6x6(b)
                dx, dy = 2 * m - off - x6, 2 * k - off - y6
                phase = np.exp(1j * (KD * SIN_TH * (dx * COS_PH + dy * SIN_PH)
                                     + dx * dpx_r + dy * dpy_r)).astype(np.complex64)
                self.tile_th[m, k] = e_theta[b] * phase
                self.tile_ph[m, k] = e_phi[b] * phase

    @staticmethod
    def _db(mag):
        db = 20.0 * np.log10(np.maximum(mag, 1e-12))
        return (db - db.max(axis=(1, 2), keepdims=True)).astype(np.float32)

    def array_factor(self, w):
        af = np.zeros((w.shape[0], NTH, NPH), np.complex64)
        for i in range(N):
            for j in range(N):
                af += w[:, i // 2, j // 2][:, None, None] * (self.ex[i] * self.ey[j])[None]
        return self._db(np.abs(af))

    def composition(self, w):
        e_th = np.einsum("pmk,mkij->pij", w, self.tile_th)
        e_ph = np.einsum("pmk,mkij->pij", w, self.tile_ph)
        return self._db(np.sqrt(np.abs(e_th) ** 2 + np.abs(e_ph) ** 2))

    def __call__(self, w):
        return predict_patterns(self.model, self.cfg, self.composition(w),
                                self.array_factor(w), self.stats)


def decode(design):
    """(P, 3) class phases in rad -> (P, 4, 4) complex unit weights."""
    return np.exp(1j * design[:, CLASS_MAP]).astype(np.complex64)


def element_phases_deg(class_phase_rad, dpx_deg, dpy_deg):
    """Excitation phase of each of the 64 elements (steering + class offset), degrees."""
    xs = np.arange(N) - (N - 1) / 2.0
    steer = xs[:, None] * dpx_deg + xs[None, :] * dpy_deg
    offs = np.rad2deg(class_phase_rad)[CLASS_MAP]
    total = steer + np.kron(offs, np.ones((2, 2)))
    return (total + 180.0) % 360.0 - 180.0


def build_masks():
    t0, p0 = np.deg2rad(TARGET_THETA), np.deg2rad(TARGET_PHI)
    tt, pp = np.meshgrid(np.deg2rad(THETA_DEG), np.deg2rad(PHI_DEG), indexing="ij")
    cosd = np.sin(tt) * np.sin(t0) * np.cos(pp - p0) + np.cos(tt) * np.cos(t0)
    ang = np.degrees(np.arccos(np.clip(cosd, -1, 1)))
    ti = int(np.abs(THETA_DEG - TARGET_THETA).argmin())
    pj = int(np.abs(PHI_DEG - TARGET_PHI).argmin())
    return ang <= BEAM_CORE_DEG, ang > SIDELOBE_START_DEG, (ti, pj)


def cost(pred, beam, side, target_idx):
    ti, pj = target_idx
    over = np.maximum(pred[:, side] - SIDELOBE_CEILING_DB, 0.0)
    side_term = over.max(axis=1) ** 2 + 0.2 * (over ** 2).mean(axis=1)   # worst sidelobe first
    beam_term = (np.maximum(BEAM_FLOOR_DB - pred[:, beam], 0.0) ** 2).mean(axis=1)
    point_term = pred[:, ti, pj] ** 2                                   # peak stays on target
    return W_SIDE * side_term + W_BEAM * beam_term + W_POINT * point_term


def run_pso(surrogate, beam, side, target_idx, swarm, iters, rng):
    dim = 3
    x = rng.uniform(PHASE_MIN, PHASE_MAX, size=(swarm, dim)).astype(np.float32)
    x[0] = 0.0                                             # one particle starts at the plain steering
    v = rng.uniform(-0.2, 0.2, size=(swarm, dim)).astype(np.float32)
    fit = cost(surrogate(decode(x)), beam, side, target_idx)
    pbest, pbest_fit = x.copy(), fit.copy()
    g = int(fit.argmin())
    gbest, gbest_fit = x[g].copy(), float(fit[g])
    for it in range(iters):
        w = INERTIA_HI + (INERTIA_LO - INERTIA_HI) * it / max(iters - 1, 1)
        r1 = rng.random((swarm, dim), dtype=np.float32)
        r2 = rng.random((swarm, dim), dtype=np.float32)
        v = np.clip(w * v + C1 * r1 * (pbest - x) + C2 * r2 * (gbest[None] - x), -V_CLAMP, V_CLAMP)
        x = np.clip(x + v, PHASE_MIN, PHASE_MAX).astype(np.float32)
        fit = cost(surrogate(decode(x)), beam, side, target_idx)
        better = fit < pbest_fit
        pbest[better], pbest_fit[better] = x[better], fit[better]
        g = int(pbest_fit.argmin())
        if pbest_fit[g] < gbest_fit:
            gbest, gbest_fit = pbest[g].copy(), float(pbest_fit[g])
        if it % 10 == 0 or it == iters - 1:
            print(f"  iteration {it:03d}/{iters}  best cost = {gbest_fit:.5f}", flush=True)
    return gbest, gbest_fit


def peak_sidelobe(pred, side):
    return float(pred[side].max())


def make_plot(pred_u, pred_o, side, target_idx, out_png):
    ti, pj = target_idx
    ext = [PHI_DEG[0], PHI_DEG[-1], THETA_DEG[-1], THETA_DEG[0]]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for a, (name, img) in zip(ax[:2], [("steering only", pred_u), ("optimised phases", pred_o)]):
        im = a.imshow(img, extent=ext, aspect="auto", cmap="viridis", vmin=-40, vmax=0)
        a.plot(TARGET_PHI, TARGET_THETA, "r+", ms=10)
        a.set_title(f"{name} (peak sidelobe {peak_sidelobe(img, side):.1f} dB)")
        a.set_xlabel("phi (deg)"); a.set_ylabel("theta (deg)")
        plt.colorbar(im, ax=a, shrink=0.85, label="dB")
    ax[2].plot(THETA_DEG, pred_u[:, pj], color="0.5", label="steering only")
    ax[2].plot(THETA_DEG, pred_o[:, pj], color="C3", label="optimised phases")
    ax[2].axhline(SIDELOBE_CEILING_DB, color="k", ls="--", lw=0.8, label="sidelobe ceiling")
    ax[2].set_xlim(0, 90); ax[2].set_ylim(-45, 2); ax[2].grid(color="0.9")
    ax[2].set_xlabel(f"theta (deg), cut at phi = {PHI_DEG[pj]:.1f} deg")
    ax[2].set_ylabel("normalized gain (dB)"); ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fields", default=str(EXAMPLES_DIR / "steering_0001.npz"))
    ap.add_argument("--ckpt", default=str(RAPID_CKPT))
    ap.add_argument("--swarm", type=int, default=SWARM)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="short run (12 particles, 12 iterations)")
    args = ap.parse_args()
    if args.quick:
        args.swarm, args.iters = 12, 12

    d = np.load(args.fields)
    dpx, dpy = float(d["dpx"]), float(d["dpy"])
    model, cfg = load_model(args.ckpt)
    surrogate = Surrogate(model, cfg, d["e_theta"], d["e_phi"], dpx, dpy)
    beam, side, target_idx = build_masks()

    pred_u = surrogate(decode(np.zeros((1, 3), np.float32)))[0]
    print(f"target beam (theta, phi) = ({TARGET_THETA}, {TARGET_PHI}) deg, "
          f"sidelobe ceiling {SIDELOBE_CEILING_DB} dB")
    print(f"steering only: peak sidelobe = {peak_sidelobe(pred_u, side):.2f} dB")

    rng = np.random.default_rng(args.seed)
    print(f"running PSO (swarm {args.swarm}, {args.iters} iterations) ...", flush=True)
    gbest, gbest_fit = run_pso(surrogate, beam, side, target_idx, args.swarm, args.iters, rng)
    pred_o = surrogate(decode(gbest[None]))[0]

    deg = np.rad2deg(gbest)
    print(f"optimised:     peak sidelobe = {peak_sidelobe(pred_o, side):.2f} dB")
    print("class phase offsets (deg): " + ", ".join(f"{n} = {v:+.1f}" for n, v in zip(CLASS_NAMES, deg)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phases = element_phases_deg(gbest, dpx, dpy)
    np.savetxt(OUT_DIR / "element_phases_deg.csv", phases, fmt="%.1f", delimiter=",",
               header="excitation phase (deg) of the 8x8 elements, rows = x index, columns = y index")
    np.savez_compressed(OUT_DIR / "result.npz", theta_deg=THETA_DEG, phi_deg=PHI_DEG,
                        pattern_steering_only_dB=pred_u, pattern_optimised_dB=pred_o,
                        class_phase_deg=deg, element_phases_deg=phases, cost=gbest_fit)
    make_plot(pred_u, pred_o, side, target_idx, OUT_DIR / "pso_inverse_8x8.png")
    print(f"saved results to {OUT_DIR}")


if __name__ == "__main__":
    main()

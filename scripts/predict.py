"""Predict the radiation pattern of an NxN array from 6x6 sub-block fields.

Runs the full RaPID pipeline for one steering direction: physics-based
composition -> input channels -> CNN correction -> predicted pattern. The
default input is the example shipped with the repository, so this runs right
after cloning:

    python -m scripts.predict --size 16
    python -m scripts.predict --size 8 12 16 20 24 28

Your own HFSS export can be used instead:

    python -m scripts.predict --size 28 --csv data/6x6_subblock_fields/patterns_global_0001.csv --sample 3

Outputs (results/predict/): pattern_NxN.npz with the RaPID prediction, the
composition without the CNN and the analytical pattern, plus a figure.
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

from rapid.composition import (PHI_DEG, THETA_DEG, array_factor_db, compose,
                               parse_subblock_field_csv, to_db_peaknorm)
from rapid.config import EXAMPLES_DIR, RAPID_CKPT, RESULTS_DIR
from rapid.inference import load_model, predict_patterns, shipped_input_stats

EXAMPLE = EXAMPLES_DIR / "steering_0001.npz"


def load_fields(args):
    """Returns e_theta, e_phi (1, 9, 181, 360), dpx, dpy (1,)."""
    if args.csv:
        e_th, e_ph, dpx, dpy, ids = parse_subblock_field_csv(Path(args.csv))
        s = args.sample
        print(f"sample id {ids[s]} from {args.csv}")
        return e_th[s:s + 1], e_ph[s:s + 1], dpx[s:s + 1], dpy[s:s + 1]
    d = np.load(args.fields)
    return (d["e_theta"][None], d["e_phi"][None],
            np.array([float(d["dpx"])]), np.array([float(d["dpy"])]))


def plot(n, rapid, comp, af, out_png):
    _, pj = np.unravel_index(int(comp.argmax()), comp.shape)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    im = ax[0].imshow(rapid, extent=[PHI_DEG[0], PHI_DEG[-1], THETA_DEG[-1], THETA_DEG[0]],
                      aspect="auto", cmap="viridis", vmin=-50, vmax=0)
    ax[0].set_xlabel("phi (deg)"); ax[0].set_ylabel("theta (deg)")
    ax[0].set_title(f"RaPID prediction, {n}x{n}")
    plt.colorbar(im, ax=ax[0], label="normalized magnitude (dB)")
    # elevation cut through the beam peak
    ax[1].plot(THETA_DEG, af[:, pj], color="0.6", lw=1.0, label="analytical (array factor)")
    ax[1].plot(THETA_DEG, comp[:, pj], color="C0", lw=1.0, ls="--", label="physics-based composition")
    ax[1].plot(THETA_DEG, rapid[:, pj], color="C3", lw=1.4, label="RaPID")
    ax[1].set_ylim(-50, 2); ax[1].set_xlim(0, 180)
    ax[1].set_xlabel(f"theta (deg), cut at phi = {PHI_DEG[pj]:.1f} deg")
    ax[1].set_ylabel("normalized magnitude (dB)")
    ax[1].grid(color="0.9"); ax[1].legend(fontsize=8, loc="lower center")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--size", type=int, nargs="+", default=[16], help="array side(s) N")
    ap.add_argument("--fields", default=str(EXAMPLE),
                    help="npz with e_theta, e_phi (9,181,360 complex), dpx, dpy (deg)")
    ap.add_argument("--csv", default=None, help="HFSS sub-block field CSV instead of --fields")
    ap.add_argument("--sample", type=int, default=0, help="column index inside --csv")
    ap.add_argument("--ckpt", default=str(RAPID_CKPT))
    ap.add_argument("--out-dir", default=str(RESULTS_DIR / "predict"))
    args = ap.parse_args()

    e_th, e_ph, dpx, dpy = load_fields(args)
    model, cfg = load_model(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"steering phases: dpx = {dpx[0]:.2f} deg, dpy = {dpy[0]:.2f} deg per element")

    for n in args.size:
        comp = to_db_peaknorm(compose(e_th, e_ph, dpx, dpy, n))
        af = array_factor_db(n, dpx, dpy)
        rapid = predict_patterns(model, cfg, comp, af, shipped_input_stats(n))
        ti, pj = np.unravel_index(int(rapid[0].argmax()), rapid[0].shape)
        np.savez_compressed(out_dir / f"pattern_{n}x{n}.npz", theta_deg=THETA_DEG, phi_deg=PHI_DEG,
                            rapid_dB=rapid[0], composition_dB=comp[0], analytical_dB=af[0],
                            dpx=dpx[0], dpy=dpy[0])
        plot(n, rapid[0], comp[0], af[0], out_dir / f"pattern_{n}x{n}.png")
        print(f"{n}x{n}: beam peak at theta = {THETA_DEG[ti]:.0f} deg, phi = {PHI_DEG[pj]:.1f} deg  "
              f"-> {out_dir / f'pattern_{n}x{n}.png'}")


if __name__ == "__main__":
    main()

"""Zero-shot evaluation of RaPID against full-wave ground truth.

For every requested array size the script loads the physics-based composition
(processed/compose_NxN.npz, built by scripts/build_compose.py with --truth-dir),
runs the CNN and reports the mean absolute error in dB on the held-out test
steering directions, over the full pattern, the null region (< -20 dB) and the
main-beam region (>= -10 dB). The analytical pattern (ideal array factor) and the
composition without the CNN are reported alongside.

No information from the evaluated array size enters the prediction: the CNN is
trained on <= 6x6 data, its output is used as is, and the input statistics are
computed from composed patterns only.

Usage:
    python -m scripts.evaluate --size 8 16 20 28
    python -m scripts.evaluate --size 28 --ckpt checkpoints/my_run/rapid.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rapid.composition import array_factor_db, peaknorm
from rapid.config import NULL_THRESHOLD_DB, RAPID_CKPT, RESULTS_DIR
from rapid.inference import (input_stats_from_patterns, load_composition, load_model,
                             load_splits, predict_patterns, split_indices)


def region_mae(pred_pn: np.ndarray, true_pn: np.ndarray) -> dict:
    diff = np.abs(pred_pn - true_pn)
    return {"full": float(diff.mean()),
            "null": float(diff[true_pn < NULL_THRESHOLD_DB].mean()),
            "main": float(diff[true_pn >= -10.0].mean())}


def evaluate_size(model, cfg, n: int, splits: dict) -> dict:
    data = load_composition(n)
    if "truth" not in data:
        raise SystemExit(f"compose_{n}x{n}.npz has no ground truth; rebuild with --truth-dir")
    tr = split_indices(data["ids"], splits["train"])
    te = split_indices(data["ids"], splits["test"])

    # Input statistics: composed patterns + array factors of the training
    # steering directions at this size. No ground truth involved.
    af_tr = array_factor_db(n, data["dpx"][tr], data["dpy"][tr])
    stats = input_stats_from_patterns(data["composition"][tr], af_tr)
    del af_tr

    af_te = array_factor_db(n, data["dpx"][te], data["dpy"][te])
    comp_te = data["composition"][te]
    true_pn = peaknorm(data["truth"][te])
    pred_pn = predict_patterns(model, cfg, comp_te, af_te, stats)

    return {"n_test": int(len(te)),
            "rapid": region_mae(pred_pn, true_pn),
            "composition_only": region_mae(peaknorm(comp_te), true_pn),
            "analytical": region_mae(af_te, true_pn)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--size", type=int, nargs="+", default=[8, 12, 16, 20, 24, 28],
                    help="array side(s) N for NxN")
    ap.add_argument("--ckpt", default=str(RAPID_CKPT))
    ap.add_argument("--out", default=str(RESULTS_DIR / "rapid_eval.json"))
    args = ap.parse_args()

    model, cfg = load_model(args.ckpt)
    splits = load_splits()
    results = {}
    print(f"{'size':>7s} | {'RaPID full':>10s} {'null':>6s} {'main':>6s} | "
          f"{'compose':>7s} | {'analytical':>10s}   (MAE, dB)")
    for n in args.size:
        r = evaluate_size(model, cfg, n, splits)
        results[f"{n}x{n}"] = r
        print(f"{n:>4d}x{n:<2d} | {r['rapid']['full']:10.3f} {r['rapid']['null']:6.3f} "
              f"{r['rapid']['main']:6.3f} | {r['composition_only']['full']:7.3f} | "
              f"{r['analytical']['full']:10.3f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": str(args.ckpt), "results": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

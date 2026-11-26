"""Physics-based composition of an NxN array from the 6x6 sub-block fields.

Reads the complex far fields of the nine 2x2 sub-blocks of a 6x6 array
(data/6x6_subblock_fields/) and composes the pattern of an NxN array for every
steering direction. No full-wave data of the NxN array is needed. If a folder
with NxN ground-truth patterns is given (--truth-dir), they are stored alongside
for evaluation or for training the baselines.

Outputs
    processed/compose_NxN.npz               composed_dB, ids, dpx, dpy [, truth_dB]
    checkpoints/rapid/input_stats_NxN.npz   per-pixel input statistics (with --write-stats)

Usage:
    python -m scripts.build_compose --size 28
    python -m scripts.build_compose --size 28 --truth-dir data/28x28_array
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rapid.composition import (array_factor_db, compose, parse_magnitude_csv,
                               parse_subblock_field_csv, to_db_peaknorm)
from rapid.config import RAPID_CKPT_DIR, SUBBLOCK_FIELD_DIR, compose_npz
from rapid.inference import input_stats_from_patterns, load_splits, split_indices


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--size", type=int, required=True, help="array side N (even, >= 4)")
    ap.add_argument("--field-dir", default=str(SUBBLOCK_FIELD_DIR))
    ap.add_argument("--truth-dir", default=None,
                    help="folder with whole-array NxN patterns (same file names)")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--write-stats", action="store_true",
                    help="also write the input statistics next to the checkpoint")
    args = ap.parse_args()

    n = args.size
    files = sorted(Path(args.field_dir).glob("*.csv"))
    if not files:
        raise SystemExit(f"no CSV files in {args.field_dir} (see data/README.md)")
    if args.max_files:
        files = files[:args.max_files]
    truth_dir = Path(args.truth_dir) if args.truth_dir else None

    comp, truth, ids, dpx, dpy = [], [], [], [], []
    t0 = time.time()
    for k, f in enumerate(files, 1):
        try:
            e_th, e_ph, dx, dy, ii = parse_subblock_field_csv(f)
        except ValueError as err:
            print(f"[{k:3d}/{len(files)}] {f.name}  skipped ({err})", flush=True)
            continue
        c_db = to_db_peaknorm(compose(e_th, e_ph, dx, dy, n))
        del e_th, e_ph
        msg = ""
        if truth_dir is not None:
            t_db, tdx, tdy, tid = parse_magnitude_csv(truth_dir / f.name)
            assert np.array_equal(ii, tid), f"sample ids differ in {f.name}"
            assert np.allclose(dx, tdx, atol=1e-3) and np.allclose(dy, tdy, atol=1e-3)
            truth.append(t_db)
            msg = f"  composition-only MAE {np.abs(c_db - t_db).mean():.3f} dB"
        comp.append(c_db); ids.append(ii); dpx.append(dx); dpy.append(dy)
        print(f"[{k:3d}/{len(files)}] {f.name}{msg}  ({time.time() - t0:.0f}s)", flush=True)

    out = {"composed_dB": np.concatenate(comp), "ids": np.concatenate(ids),
           "dpx": np.concatenate(dpx), "dpy": np.concatenate(dpy)}
    if truth:
        out["truth_dB"] = np.concatenate(truth)
    path = Path(args.out) if args.out else compose_npz(n)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    print(f"saved {path}  ({len(out['ids'])} steering directions)")

    if args.write_stats:
        tr = split_indices(out["ids"], load_splits()["train"])
        stats = input_stats_from_patterns(out["composed_dB"][tr],
                                          array_factor_db(n, out["dpx"][tr], out["dpy"][tr]))
        sp = RAPID_CKPT_DIR / f"input_stats_{n}x{n}.npz"
        np.savez_compressed(sp, **stats)
        print(f"saved {sp}")


if __name__ == "__main__":
    main()

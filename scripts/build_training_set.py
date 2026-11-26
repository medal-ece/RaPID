"""Build the RaPID training set. Only arrays up to 6x6 are used.

The complex composition reproduces the 6x6 pattern exactly, so there is nothing
left to learn from it at the 6x6 scale itself. The network is therefore trained
on two tasks whose error has to be corrected by the CNN:

  6x6 task          base = 6x6 composed from the sub-block MAGNITUDES only
                    (measured phase discarded), target = full-wave 6x6 pattern.
  cross-scale task  base = 4x4 composed from the 6x6 sub-block fields (corner
                    sub-blocks moved inwards), target = full-wave 4x4 pattern.
                    Sub-blocks are reused outside the neighbourhood they were
                    simulated in, which is the same error mechanism as composing
                    a larger array.

Steering directions that belong to the validation or test split of the
evaluation are not used by the 6x6 task (see data/splits.json).

Inputs (see data/README.md)
    data/6x6_subblock_magnitude/   data/6x6_array/   data/4x4_array/
    processed/compose_4x4.npz      (python -m scripts.build_compose --size 4 --truth-dir data/4x4_array)

Outputs
    processed/train_set.npz        composition, af, truth, dpx, dpy, ids, task
    processed/train_split.npz      train, val (row indices)

Usage:
    python -m scripts.build_training_set
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rapid.composition import (array_factor_db, magnitude_compose_6x6, parse_magnitude_csv,
                               read_sample_ids)
from rapid.config import PROCESSED_DIR, SUBBLOCK_MAG_DIR, array_dir
from rapid.inference import load_composition, load_splits, split_indices


def build_6x6_task(wanted: set[int], mag_dir: Path, whole_dir: Path):
    comp, truth, ids, dpx, dpy = [], [], [], [], []
    files = sorted(mag_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"no CSV files in {mag_dir} (see data/README.md)")
    for k, f in enumerate(files, 1):
        file_ids = read_sample_ids(f, cols_per_sample=9)
        keep = np.array([int(i) in wanted for i in file_ids])
        if not keep.any():
            continue
        sub_db, dx, dy, ii = parse_magnitude_csv(f, cols_per_sample=9)
        whole_db, _, _, wid = parse_magnitude_csv(whole_dir / f.name)
        assert np.array_equal(ii, wid), f"sample ids differ in {f.name}"
        comp.append(magnitude_compose_6x6(sub_db[keep], dx[keep], dy[keep]))
        truth.append(whole_db[keep]); ids.append(ii[keep]); dpx.append(dx[keep]); dpy.append(dy[keep])
        print(f"  6x6 task: {f.name}  ({int(keep.sum())} samples)", flush=True)
    cat = np.concatenate
    return cat(comp), cat(truth), cat(ids), cat(dpx), cat(dpy)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mag-dir", default=str(SUBBLOCK_MAG_DIR))
    ap.add_argument("--array6-dir", default=str(array_dir(6)))
    ap.add_argument("--out", default=str(PROCESSED_DIR / "train_set.npz"))
    ap.add_argument("--split-out", default=str(PROCESSED_DIR / "train_split.npz"))
    args = ap.parse_args()
    splits = load_splits()

    # ── 6x6 task ─────────────────────────────────────────────────────────────
    wanted = set(splits["train_6x6"]) | set(splits["val_6x6"])
    comp6, truth6, ids6, dpx6, dpy6 = build_6x6_task(wanted, Path(args.mag_dir), Path(args.array6_dir))
    af6 = array_factor_db(6, dpx6, dpy6)

    # ── cross-scale 6 -> 4 task ──────────────────────────────────────────────
    c4 = load_composition(4)
    if "truth" not in c4:
        raise SystemExit("compose_4x4.npz has no ground truth; rebuild it with --truth-dir")
    keep4 = np.concatenate([split_indices(c4["ids"], splits["train"]),
                            split_indices(c4["ids"], splits["val"])])
    comp4, truth4 = c4["composition"][keep4], c4["truth"][keep4]
    ids4, dpx4, dpy4 = c4["ids"][keep4], c4["dpx"][keep4], c4["dpy"][keep4]
    af4 = array_factor_db(4, dpx4, dpy4)
    print(f"  cross-scale task: mean |truth - composition| = {np.abs(truth4 - comp4).mean():.3f} dB")

    f16 = lambda a, b: np.concatenate([a, b]).astype(np.float16)
    ids = np.concatenate([ids6, ids4])
    task = np.concatenate([np.full(len(ids6), 6), np.full(len(ids4), 4)]).astype(np.int8)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, composition=f16(comp6, comp4), af=f16(af6, af4),
                        truth=f16(truth6, truth4), dpx=np.concatenate([dpx6, dpx4]),
                        dpy=np.concatenate([dpy6, dpy4]), ids=ids, task=task)

    rows6, rows4 = np.where(task == 6)[0], np.where(task == 4)[0]
    pick = lambda rows, wanted_ids: rows[np.isin(ids[rows], list(wanted_ids))]
    train = np.concatenate([pick(rows6, splits["train_6x6"]), pick(rows4, splits["train"])])
    val = np.concatenate([pick(rows6, splits["val_6x6"]), pick(rows4, splits["val"])])
    np.savez_compressed(args.split_out, train=train, val=val)
    print(f"saved {out} ({len(ids)} samples) and {args.split_out} "
          f"(train {len(train)}, val {len(val)})")


if __name__ == "__main__":
    main()

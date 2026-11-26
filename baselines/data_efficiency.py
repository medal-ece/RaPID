"""Data-budget sweep of the BO-CNN and CV-GNN baselines at one array size.

Both baselines learn the pattern of a fixed array size directly from full-wave
patterns of that size. This script trains them on growing subsets of the
target-size training pool and reports the test error at each budget (Fig. 4 and
Table III of the paper).

Protocol
  * Test set: the held-out test steering directions (data/splits.json), the same
    ones RaPID is evaluated on, scored with the same full-pattern MAE in dB on
    peak-normalised 181x360 patterns.
  * Training pool: the remaining 1,575 patterns (train + val). For every budget
    N and seed a random subset of N patterns is drawn; 12 % of it is held out
    for early stopping.

Input: processed/compose_NxN.npz built with ground truth
    python -m scripts.build_compose --size 28 --truth-dir data/28x28_array

Usage:
    # CV-GNN (Jin et al.), complex-valued head as published
    python -m baselines.data_efficiency --size 28 --model cvgnn_cx --lr 1e-3 --tag cvgnn_28
    # BO-CNN (Zhang et al.) with the architecture found by baselines/bocnn_bo.py
    python -m baselines.data_efficiency --size 28 --model bocnn --nc 18 --fc 1 --nconv 2 \
        --fc-hidden 512 --tag bocnn_28
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baselines.models.bocnn import (BOCNNBaseline, PatternScaler, apply_R_minmax, encode_R,
                                    fit_R_minmax, peaknorm)
from baselines.models.cvgnn import CVGNNBaseline
from rapid.config import DEVICE, NPH, NTH, RESULTS_DIR
from rapid.inference import load_composition, load_splits, split_indices

OUT_DIR = RESULTS_DIR / "baselines"
OUT_DIM = NTH * NPH
RAPID_EVAL_JSON = RESULTS_DIR / "rapid_eval.json"


def build_model(model_name: str, n_elem: int, arch: dict):
    """Both models map R = (B, 4, n_elem) to (B, OUT_DIM)."""
    if model_name == "bocnn":
        return BOCNNBaseline(n_elem=n_elem, out_dim=OUT_DIM, n_c=arch["n_c"],
                             f_c=arch["f_c"], n_conv=arch["n_conv"], fc_dims=arch["fc_dims"],
                             pool=arch.get("pool", 1))
    if model_name in ("cvgnn", "cvgnn_cx"):
        return CVGNNBaseline(n_elem=n_elem, out_dim=OUT_DIM,
                             gin_hidden=arch.get("gin_hidden", 32),
                             gin_layers=arch.get("gin_layers", 3),
                             fc_hidden=arch.get("fc_hidden", 256),
                             head="complex" if model_name == "cvgnn_cx" else "mlp",
                             head_layers=arch.get("head_layers", 4))
    raise ValueError(f"unknown model {model_name}")


def load_pool_data(size: int) -> dict:
    """Ground-truth patterns and steering phases of one array size, plus the
    training pool / test row indices."""
    data = load_composition(size)
    if "truth" not in data:
        raise SystemExit(f"compose_{size}x{size}.npz has no ground truth; rebuild with --truth-dir")
    sp = load_splits()
    tr = split_indices(data["ids"], sp["train"])
    va = split_indices(data["ids"], sp["val"])
    te = split_indices(data["ids"], sp["test"])
    return {"hfss": data["truth"], "dpx": data["dpx"].astype(np.float32),
            "dpy": data["dpy"].astype(np.float32), "pool": np.concatenate([tr, va]), "test": te}


def rapid_reference(size: int) -> float | None:
    """RaPID zero-shot test MAE at this size, if scripts/evaluate.py has been run."""
    if not RAPID_EVAL_JSON.exists():
        return None
    r = json.loads(RAPID_EVAL_JSON.read_text())["results"].get(f"{size}x{size}")
    return None if r is None else float(r["rapid"]["full"])


def train_subset(
    R_all: np.ndarray, y01_all: np.ndarray, true_pn_te: np.ndarray,
    scaler: PatternScaler, train_idx: np.ndarray, val_idx: np.ndarray, te_local: np.ndarray,
    n_elem: int, arch: dict, *, model_name: str, epochs: int, batch: int, lr: float,
    patience: int, seed: int, min_epochs: int = 150, loss: str = "mse",
) -> dict:
    """Train on R_all[train_idx]; early-stop on validation MAE (dB); report test MAE (dB)."""
    torch.manual_seed(seed)
    # Min-max R using the training subset only (inputs normalised to [0, 1]).
    lo, span = fit_R_minmax(R_all[train_idx])
    Rn = apply_R_minmax(R_all, lo, span)

    Rtr = torch.from_numpy(Rn[train_idx]); ytr = torch.from_numpy(y01_all[train_idx])
    Rva = torch.from_numpy(Rn[val_idx]).to(DEVICE)
    Rte = torch.from_numpy(Rn[te_local]).to(DEVICE)
    val_true_db = scaler.to_db(y01_all[val_idx].reshape(-1, NTH, NPH))

    model = build_model(model_name, n_elem, arch).to(DEVICE)
    # Start from the per-pixel training mean; without it the very wide output
    # layer stays on a constant-prediction plateau.
    model.set_output_bias(y01_all[train_idx].mean(0))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(8, patience // 3))
    loss_fn = nn.MSELoss() if loss == "mse" else nn.L1Loss()
    loader = DataLoader(TensorDataset(Rtr, ytr), batch_size=batch, shuffle=True, drop_last=False)

    best_val = float("inf"); best_state = None; bad = 0
    for ep in range(epochs):
        model.train()
        for Rb, yb in loader:
            Rb = Rb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            loss_val = loss_fn(model(Rb), yb)
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Rva).clamp(0.0, 1.0).cpu().numpy()
        pv_db = peaknorm(scaler.to_db(pv).reshape(-1, NTH, NPH))
        vmae = float(np.mean(np.abs(pv_db - peaknorm(val_true_db))))
        sched.step(vmae)
        if vmae < best_val - 1e-4:
            best_val = vmae; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if ep + 1 >= min_epochs and bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pt = model(Rte).clamp(0.0, 1.0).cpu().numpy()
    pt_db = peaknorm(scaler.to_db(pt).reshape(-1, NTH, NPH))
    return {"test_mae": float(np.mean(np.abs(pt_db - true_pn_te))), "val_mae": best_val,
            "epochs": ep + 1, "n_params": model.count_params()}


def prepare_tensors(size: int, sd: dict):
    """Element-excitation inputs R and [0, 1]-scaled targets for pool + test."""
    pool, te = sd["pool"], sd["test"]
    all_idx = np.concatenate([pool, te])
    R_all = encode_R(sd["dpx"][all_idx], sd["dpy"][all_idx], size)
    hfss_pn = peaknorm(sd["hfss"][all_idx])
    pool_local = np.arange(len(pool))
    te_local = len(pool) + np.arange(len(te))
    scaler = PatternScaler.from_train(hfss_pn[pool_local])
    y01 = scaler.to01(hfss_pn).reshape(len(all_idx), -1)
    return R_all, y01, hfss_pn, scaler, pool_local, te_local


def subset_split(pool_local: np.ndarray, n: int, split_seed: int):
    """Random subset of n pool patterns, 12 % of it held out for early stopping."""
    rng = np.random.default_rng(1000 * split_seed + n)
    ps = pool_local.copy(); rng.shuffle(ps)
    sub = ps[:n]
    n_val = max(8, int(round(0.12 * n)))
    n_val = max(1, min(n_val, n - max(8, n // 2)))
    return sub[n_val:], sub[:n_val]


def run_size(size: int, args) -> dict:
    n_elem = size * size
    sd = load_pool_data(size)
    R_all, y01, hfss_pn, scaler, pool_local, te_local = prepare_tensors(size, sd)
    true_pn_te = hfss_pn[te_local]
    reference = rapid_reference(size)
    print(f"\n[{size}x{size}] training pool {len(pool_local)}, test {len(te_local)}"
          + (f", RaPID zero-shot reference {reference:.3f} dB" if reference else ""), flush=True)

    rows = []
    for n in args.sizes:
        if n > len(pool_local):
            continue
        maes = []
        for seed in args.seeds:
            train_l, val_l = subset_split(pool_local, n, seed)
            t1 = time.time()
            res = train_subset(R_all, y01, true_pn_te, scaler, train_idx=train_l, val_idx=val_l,
                               te_local=te_local, n_elem=n_elem, arch=args.arch,
                               model_name=args.model, epochs=args.epochs, batch=args.batch,
                               lr=args.lr, patience=args.patience, seed=seed, loss=args.loss,
                               min_epochs=args.min_epochs)
            maes.append(res["test_mae"])
            print(f"  N={n:5d} seed={seed} (train {len(train_l):4d} / val {len(val_l):3d})  "
                  f"test MAE = {res['test_mae']:.4f} dB  ({res['epochs']} epochs, "
                  f"{time.time() - t1:.0f}s)", flush=True)
        arr = np.array(maes, float)
        rows.append({"N": int(n), "test_mae_per_seed": [float(x) for x in maes],
                     "test_mae_mean": float(arr.mean()), "test_mae_std": float(arr.std())})
        print(f"  -> N={n:5d}  {arr.mean():.4f} +/- {arr.std():.4f} dB", flush=True)

    return {"size": size, "model": args.model, "rapid_zero_shot_mae": reference, "rows": rows,
            "arch": {k: (list(v) if k == "fc_dims" else v) for k, v in args.arch.items()},
            "n_params": build_model(args.model, n_elem, args.arch).count_params()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--size", type=int, nargs="+", default=[28], help="array side(s) N")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[50, 100, 200, 400, 800, 1200, 1575],
                    help="training budgets (number of target-size patterns)")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--min-epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--model", choices=["bocnn", "cvgnn", "cvgnn_cx"], default="bocnn",
                    help="bocnn (Zhang et al.) | cvgnn (Jin et al. GIN + real MLP head) | "
                         "cvgnn_cx (Jin et al. GIN + complex-valued residual head, as published)")
    ap.add_argument("--loss", choices=["mse", "l1"], default=None,
                    help="default: mse; l1 for cvgnn_cx as in Jin et al.")
    # BO-CNN architecture [n_c, f_c, xi, L, Q]
    ap.add_argument("--nc", type=int, default=18, help="conv channel width n_c")
    ap.add_argument("--fc", type=int, default=1, help="conv kernel size f_c")
    ap.add_argument("--pool", type=int, default=1, help="pooling factor xi (1 = none)")
    ap.add_argument("--nconv", type=int, default=2, help="number of conv layers L")
    ap.add_argument("--fc-hidden", type=int, nargs="*", default=[512],
                    help="hidden FC widths (Q = len + 1 including the output layer)")
    # CV-GNN architecture (3-layer GIN backbone + 4-layer complex FCN)
    ap.add_argument("--gin-hidden", type=int, default=32)
    ap.add_argument("--gin-layers", type=int, default=3)
    ap.add_argument("--cvgnn-fc-hidden", type=int, default=256)
    ap.add_argument("--head-layers", type=int, default=4)
    ap.add_argument("--tag", type=str, default=None, help="output file tag")
    args = ap.parse_args()
    if args.tag is None:
        args.tag = args.model
    if args.loss is None:
        args.loss = "l1" if args.model == "cvgnn_cx" else "mse"
    args.arch = {"n_c": args.nc, "f_c": args.fc, "n_conv": args.nconv,
                 "fc_dims": tuple(args.fc_hidden), "pool": args.pool,
                 "gin_hidden": args.gin_hidden, "gin_layers": args.gin_layers,
                 "fc_hidden": args.cvgnn_fc_hidden, "head_layers": args.head_layers}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {str(s): run_size(s, args) for s in args.size}
    out_json = OUT_DIR / f"results_{args.tag}.json"
    out_json.write_text(json.dumps(results, indent=2))

    for s, r in results.items():
        print(f"\n[{s}x{s}] {args.model}  ({r['n_params'] / 1e6:.1f} M parameters)")
        print(f"  {'N':>6s} {'test MAE (dB)':>14s} {'std':>7s}")
        for row in r["rows"]:
            print(f"  {row['N']:6d} {row['test_mae_mean']:14.4f} {row['test_mae_std']:7.4f}")
        if r["rapid_zero_shot_mae"] is not None:
            print(f"  RaPID, zero-shot (0 patterns of this size): {r['rapid_zero_shot_mae']:.4f} dB")
    print(f"\nresults: {out_json}")


if __name__ == "__main__":
    main()

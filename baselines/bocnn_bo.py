"""Bayesian optimisation of the BO-CNN hyper-parameters (Zhang et al., Section III-B).

The published method wraps the CNN of Sec. III-A in a Bayesian optimisation of
its hyper-parameter vector [n_c, f_c, xi, L, Q] (+ learning rate). The optimum
reported in that paper ([18, 1, 1, 2, 2]) was found on an 11x11 array; this
script re-runs the search on the array size at hand so that the baseline gets
the architecture search it was published with.

Implementation: Gaussian-process surrogate (ARD Matern-5/2, scikit-learn) with
the Expected-Improvement acquisition over a discrete candidate grid.

Objective: validation MAE (dB) on a held-out slice of the 1,575-pattern training
pool (the test set is never used by the search). Each trial trains one CNN on
the full pool with a reduced epoch cap.

After the search, run the data-budget sweep with the printed optimum:
    python -m baselines.data_efficiency --size 28 --model bocnn --nc <n_c> --fc <f_c> ...

Usage:
    python -m baselines.bocnn_bo --size 28 --trials 30 --init 6 --epochs 120
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from baselines.data_efficiency import OUT_DIR, load_pool_data, prepare_tensors, train_subset
from rapid.config import NPH, NTH

# ── Search space (paper's hyper-parameter vector + learning rate) ─────────────
SPACE = {
    "n_c":   [8, 16, 18, 24, 32, 48, 64],       # conv channel width
    "f_c":   [1, 3, 5, 7],                      # conv kernel size
    "xi":    [1, 2],                            # pooling factor (1 = none)
    "L":     [1, 2, 3, 4],                      # number of conv layers
    "Q":     [1, 2, 3],                         # FC layers incl. output (Q-1 hidden)
    "width": [256, 512, 1024, 2048],            # hidden FC width
    "lr":    [1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3],  # Adam learning rate
}
KEYS = list(SPACE)


def cand_grid() -> np.ndarray:
    """All candidate configurations as an (M, 7) array of raw values."""
    rows = list(itertools.product(*(SPACE[k] for k in KEYS)))
    return np.array(rows, dtype=np.float64)


def encode(X: np.ndarray) -> np.ndarray:
    """Map raw config values to a [0,1]^7 box (log scale for widths / lr)."""
    Z = np.empty_like(X)
    for j, k in enumerate(KEYS):
        v = np.array(SPACE[k], dtype=np.float64)
        col = X[:, j]
        if k in ("n_c", "width", "lr"):
            lo, hi = np.log(v.min()), np.log(v.max())
            Z[:, j] = (np.log(col) - lo) / (hi - lo)
        else:
            lo, hi = v.min(), v.max()
            Z[:, j] = (col - lo) / max(hi - lo, 1e-9)
    return Z


def to_arch(row: np.ndarray) -> tuple[dict, float]:
    n_c, f_c, xi, L, Q, width, lr = row
    arch = {"n_c": int(n_c), "f_c": int(f_c), "pool": int(xi), "n_conv": int(L),
            "fc_dims": tuple([int(width)] * (int(Q) - 1))}
    return arch, float(lr)


def analytic_params(n_elem: int, arch: dict, out_dim: int = NTH * NPH) -> int:
    """Parameter count of BOCNNBaseline(arch) without building it."""
    n_c, f_c, L, pool = arch["n_c"], arch["f_c"], arch["n_conv"], arch["pool"]
    total = 0; in_ch = 4; length = n_elem
    for _ in range(L):
        total += in_ch * n_c * f_c + n_c
        if pool > 1 and length >= 2 * pool:
            length //= pool
        in_ch = n_c
    d = n_c * length
    for w in arch["fc_dims"]:
        total += d * w + w; d = w
    return total + d * out_dim + out_dim


def expected_improvement(mu, sigma, best, xi_ei=0.01):
    sigma = np.maximum(sigma, 1e-9)
    z = (best - xi_ei - mu) / sigma
    return (best - xi_ei - mu) * norm.cdf(z) + sigma * norm.pdf(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=28, help="array side N")
    ap.add_argument("--trials", type=int, default=30, help="total BO evaluations")
    ap.add_argument("--init", type=int, default=6, help="random initial designs")
    ap.add_argument("--epochs", type=int, default=120, help="epoch cap per trial")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--min-epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-params", type=float, default=400e6,
                    help="skip candidates above this parameter count (GPU memory limit)")
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()
    tag = args.tag or f"bo_{args.size}"
    scale = args.size; n_elem = scale * scale

    # ── data: the search only uses the training pool ───────────────────────────
    R_all, y01, hfss_pn, scaler, pool_l, te_l = prepare_tensors(scale, load_pool_data(scale))
    true_pn_te = hfss_pn[te_l]   # the trainer also reports a test MAE; it is logged, not used

    # BO inner split of the pool: 12% validation, fixed across trials
    rng = np.random.default_rng(args.seed)
    ps = pool_l.copy(); rng.shuffle(ps)
    n_val = int(round(0.12 * len(ps)))
    val_l, train_l = ps[:n_val], ps[n_val:]
    print(f"[{scale}x{scale}] BO over {len(cand_grid())} candidates; inner train {len(train_l)} / val {len(val_l)}; "
          f"{args.trials} trials ({args.init} random), {args.epochs} epochs cap", flush=True)

    X = cand_grid(); Z = encode(X)
    # parameter-count limit (analytic; instantiating every candidate would be far too slow)
    keep = np.ones(len(X), bool)
    for i, row in enumerate(X):
        arch, _ = to_arch(row)
        if analytic_params(n_elem, arch) > args.max_params:
            keep[i] = False
    X, Z = X[keep], Z[keep]
    print(f"  {len(X)} candidates within {args.max_params/1e6:.0f}M params", flush=True)

    evaluated: list[int] = []
    ys: list[float] = []
    log: list[dict] = []
    out_json = OUT_DIR / f"bocnn_bo_{tag}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def run_trial(i: int, how: str):
        arch, lr = to_arch(X[i])
        t0 = time.time()
        try:
            res = train_subset(
                R_all, y01, true_pn_te, scaler, train_idx=train_l, val_idx=val_l,
                te_local=te_l, n_elem=n_elem, arch=arch, model_name="bocnn",
                epochs=args.epochs, batch=args.batch, lr=lr, patience=args.patience,
                seed=args.seed, min_epochs=args.min_epochs, loss="mse")
            vmae = res["val_mae"]; tmae = res["test_mae"]; ep = res["epochs"]; npar = res["n_params"]
        except RuntimeError as e:      # CUDA OOM etc. -> penalise
            import torch; torch.cuda.empty_cache()
            vmae = 60.0; tmae = float("nan"); ep = 0; npar = -1
            print(f"    trial failed: {str(e)[:80]}", flush=True)
        evaluated.append(i); ys.append(vmae)
        entry = {"trial": len(ys), "how": how, "n_c": arch["n_c"], "f_c": arch["f_c"], "xi": arch["pool"],
                 "L": arch["n_conv"], "Q": len(arch["fc_dims"]) + 1,
                 "width": (arch["fc_dims"][0] if arch["fc_dims"] else 0), "lr": lr,
                 "val_mae": vmae, "test_mae_info_only": tmae, "epochs": ep, "n_params": npar,
                 "sec": time.time() - t0}
        log.append(entry)
        print(f"  [{entry['trial']:2d}/{args.trials}] {how:6s} n_c={arch['n_c']:2d} f_c={arch['f_c']} xi={arch['pool']} "
              f"L={arch['n_conv']} Q={entry['Q']} w={entry['width']:4d} lr={lr:.1e}  "
              f"val MAE={vmae:7.3f} dB  ({ep} ep, {entry['sec']:.0f}s)", flush=True)
        out_json.write_text(json.dumps({"scale": scale, "space": SPACE, "log": log}, indent=2))

    # ── initial random design ──────────────────────────────────────────────────
    init_ids = rng.choice(len(X), size=min(args.init, len(X)), replace=False)
    for i in init_ids:
        run_trial(int(i), "random")

    # ── GP + EI loop ───────────────────────────────────────────────────────────
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=np.ones(Z.shape[1]),
                                                       length_scale_bounds=(1e-2, 1e1), nu=2.5) \
        + WhiteKernel(1e-2, (1e-5, 1e0))
    while len(ys) < args.trials:
        y = np.array(ys); ymu, ysd = y.mean(), max(y.std(), 1e-6)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=3,
                                      random_state=args.seed)
        gp.fit(Z[evaluated], (y - ymu) / ysd)
        mu, sd = gp.predict(Z, return_std=True)
        ei = expected_improvement(mu, sd, best=(y.min() - ymu) / ysd)
        ei[evaluated] = -1.0
        run_trial(int(np.argmax(ei)), "EI")

    # ── report ─────────────────────────────────────────────────────────────────
    best = min(log, key=lambda e: e["val_mae"])
    print("\nBEST BO configuration (by inner-val MAE):")
    print(json.dumps(best, indent=2))
    fc = " ".join([str(best["width"])] * (best["Q"] - 1))
    print(f"\nRun the data-budget sweep with:\n"
          f"  python -m baselines.data_efficiency --size {scale} --model bocnn "
          f"--nc {best['n_c']} --fc {best['f_c']} --pool {best['xi']} --nconv {best['L']} "
          f"--fc-hidden {fc} --lr {best['lr']} --tag bocnn_{scale}")
    out_json.write_text(json.dumps({"scale": scale, "space": SPACE, "log": log, "best": best}, indent=2))
    print(f"BO log: {out_json}")


if __name__ == "__main__":
    main()

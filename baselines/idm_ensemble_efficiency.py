"""Data-budget sweep of the IDM-NN baseline (Onat et al., IEEE TAP 2024) at one
array size. Same protocol as baselines/data_efficiency.py: held-out test
steering directions, 1,575-pattern training pool, random subsets, 3 seeds, 12 %
of each subset held out for early stopping, full-pattern peak-normalised MAE in dB.

Per (N, seed):
  direct branch   DirectEEPNet, trained on [0,1]-scaled dB patterns (MSE by
                  default; the published SSIM loss is available as
                  --direct-loss ssim)
  IDM branch      IDMCoefNet, trained with MSE on the dipole coefficients built
                  by baselines/build_idm_targets.py; pattern = linear dipole
                  forward operator
  ensemble        P = beta * P_direct + (1 - beta) * P_IDM in peak-normalised
                  dB, beta chosen on the validation patterns

Usage:
    python -m baselines.build_idm_targets --size 28
    python -m baselines.idm_ensemble_efficiency --size 28
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import baselines.build_idm_targets as idm_targets
from baselines.data_efficiency import (OUT_DIR, load_pool_data, prepare_tensors, rapid_reference,
                                       subset_split)
from baselines.models.bocnn import apply_R_minmax, fit_R_minmax, peaknorm
from baselines.models.idm_ensemble import DirectEEPNet, IDMCoefNet, IDMForward, ssim_loss
from rapid.config import DEVICE, NPH, NTH, PROCESSED_DIR

OUT_DIM = NTH * NPH
BETAS = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def _paper_metric(pred_db: np.ndarray, true_db: np.ndarray, theta_max: int = 45) -> float:
    """Paper Eq. 4 on the equivalent window (their 45<=theta<=135 around broadside
    theta=90 -> ours theta<=45 around broadside theta=0), linear-magnitude MSE in dB."""
    p = 10 ** (pred_db[:, : theta_max + 1] / 20.0)
    t = 10 ** (true_db[:, : theta_max + 1] / 20.0)
    return float(10 * np.log10(np.mean((p - t) ** 2)))


def _mae_db(pred_db, true_db):
    return float(np.mean(np.abs(pred_db - true_db)))


def train_direct(Rn, y01, train_idx, val_idx, val_true_pn, scaler, n_elem, *, epochs, min_epochs,
                 batch, lr, patience, seed, loss_name="ssim"):
    torch.manual_seed(seed)
    model = DirectEEPNet(n_elem).to(DEVICE)
    model.set_output_bias(y01[train_idx].mean(0))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(8, patience // 3))
    loader = DataLoader(TensorDataset(torch.from_numpy(Rn[train_idx]), torch.from_numpy(y01[train_idx])),
                        batch_size=batch, shuffle=True)
    Rva = torch.from_numpy(Rn[val_idx]).to(DEVICE)
    mse = nn.MSELoss()
    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for Rb, yb in loader:
            Rb, yb = Rb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(Rb)
            loss = ssim_loss(out, yb) if loss_name == "ssim" else mse(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Rva).clamp(0, 1).cpu().numpy()
        vmae = _mae_db(peaknorm(scaler.to_db(pv).reshape(-1, NTH, NPH)), val_true_pn)
        sched.step(vmae)
        if vmae < best_val - 1e-4:
            best_val, bad = vmae, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if ep + 1 >= min_epochs and bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_val, ep + 1


def predict_direct(model, Rn, idx, scaler, bs=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            R = torch.from_numpy(Rn[idx[i:i + bs]]).to(DEVICE)
            out.append(model(R).clamp(0, 1).cpu().numpy())
    return peaknorm(scaler.to_db(np.concatenate(out)).reshape(-1, NTH, NPH))


def train_idm(Rn, c_std, train_idx, val_idx, val_true_pn, fwd, floor_db, n_elem, n_dip, *, epochs, min_epochs,
              batch, lr, patience, seed):
    torch.manual_seed(seed)
    model = IDMCoefNet(n_elem, n_dip).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(8, patience // 3))
    loader = DataLoader(TensorDataset(torch.from_numpy(Rn[train_idx]), torch.from_numpy(c_std[train_idx])),
                        batch_size=batch, shuffle=True)
    mse = nn.MSELoss()
    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for Rb, cb in loader:
            Rb, cb = Rb.to(DEVICE), cb.to(DEVICE)
            opt.zero_grad()
            loss = mse(model(Rb), cb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        pv = predict_idm(model, Rn, val_idx, fwd, floor_db)
        vmae = _mae_db(pv, val_true_pn)
        sched.step(vmae)
        if vmae < best_val - 1e-4:
            best_val, bad = vmae, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if ep + 1 >= min_epochs and bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_val, ep + 1


def predict_idm(model, Rn, idx, fwd, floor_db, bs=64):
    """Network output = standardised DE-STEERED envelope g; c = g * exp(+j(x dpx + y dpy))."""
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            sel = idx[i:i + bs]
            R = torch.from_numpy(Rn[sel]).to(DEVICE)
            g = fwd.coef_from_output(model(R).float() * fwd.c_scale)
            c = g * torch.from_numpy(fwd.steer[sel]).to(DEVICE)
            out.append(fwd.pattern_db(c, floor_db=floor_db).cpu().numpy())
    return np.concatenate(out).reshape(-1, NTH, NPH)


def run_scale(scale: int, args) -> dict:
    n_elem = scale * scale
    sd = load_pool_data(scale)
    R_all, y01_all, hfss_pn_all, scaler, pool_local, te_local = prepare_tensors(scale, sd)
    all_idx = np.concatenate([sd["pool"], sd["test"]])
    true_pn_te = hfss_pn_all[te_local]
    reference = rapid_reference(scale)
    print(f"\n[{scale}x{scale}] training pool {len(pool_local)}, test {len(te_local)}"
          + (f", RaPID zero-shot reference {reference:.3f} dB" if reference else ""), flush=True)

    # dipole coefficients and basis from baselines/build_idm_targets.py
    idm = np.load(PROCESSED_DIR / f"idm_{scale}x{scale}.npz")
    assert np.allclose(idm["dpx"], sd["dpx"]) and np.allclose(idm["dpy"], sd["dpy"]), "sample order mismatch"
    coef_key = "coef" if args.idm_reg == "noise" else "coef_tikh"
    coef_all = idm[coef_key].astype(np.complex64)
    n_dip = coef_all.shape[1]
    layout = str(idm["layout"])
    idm_targets.N = scale                          # dipole_layout default extent follows the size
    A_th, A_ph, _, _, _ = idm_targets.build_basis(layout)
    fwd = IDMForward(A_th, A_ph)
    del A_th, A_ph; torch.cuda.empty_cache()
    print(f"  dipole layout {layout} ({n_dip} dipoles), regularisation {args.idm_reg}", flush=True)

    floor_db = scaler.vmin
    coef_local = coef_all[all_idx]
    # De-steered targets: a dipole 14 half-lambda off-centre rotates its coefficient
    # phase by ~50 deg per 3.7 deg of steering, so the raw coefficients are a highly
    # oscillatory function of the input (median neighbour difference 0.38 of the norm;
    # the network fits them only partially and the residual fills the deep pattern
    # regions). Removing the known steering phase exp(j(x dpx + y dpy)) leaves a smooth
    # envelope (neighbour difference 0.08) that the network regresses; the phase is
    # reapplied in predict_idm. Column order follows build_basis: [front: per orient | back: per orient].
    n_or = n_dip // (2 * len(idm["x"]))
    xs = np.tile(idm["x"].astype(np.float64), 2 * n_or); ys = np.tile(idm["y"].astype(np.float64), 2 * n_or)
    assert len(xs) == n_dip
    ph = xs[None, :] * np.deg2rad(sd["dpx"][all_idx])[:, None] + ys[None, :] * np.deg2rad(sd["dpy"][all_idx])[:, None]
    fwd.steer = np.exp(1j * ph).astype(np.complex64)                  # (n_local, Nd)
    g_local = coef_local * np.conj(fwd.steer)
    c_ri = np.concatenate([g_local.real, g_local.imag], 1).astype(np.float32)   # (n, 2Nd) de-steered
    oracle_te = fwd.pattern_db(torch.from_numpy(coef_local[te_local]).to(DEVICE), floor_db=floor_db).cpu().numpy().reshape(-1, NTH, NPH)
    oracle_te_mae = _mae_db(oracle_te, true_pn_te)
    print(f"  IDM with perfect coefficients: test MAE = {oracle_te_mae:.4f} dB", flush=True)

    rows = []
    for N in args.sizes:
        if N > len(pool_local):
            continue
        seed_res = []
        for seed in args.seeds:
            train_l, val_l = subset_split(pool_local, N, seed)
            val_true_pn = hfss_pn_all[val_l]

            lo, span = fit_R_minmax(R_all[train_l]); Rn = apply_R_minmax(R_all, lo, span)
            # coefficient targets: one global scale (train subset) so raw MSE is well-conditioned
            c_scale = float(np.sqrt(np.mean(c_ri[train_l] ** 2)) + 1e-12)
            c_std = (c_ri / c_scale).astype(np.float32)
            fwd.c_scale = c_scale

            t1 = time.time()
            mA, vA, epA = train_direct(Rn, y01_all, train_l, val_l, val_true_pn, scaler, n_elem,
                                       epochs=args.epochs, min_epochs=args.min_epochs, batch=args.batch,
                                       lr=args.lr_direct, patience=args.patience, seed=seed, loss_name=args.direct_loss)
            tA = time.time() - t1; t1 = time.time()
            mB, vB, epB = train_idm(Rn, c_std, train_l, val_l, val_true_pn, fwd, floor_db, n_elem, n_dip,
                                    epochs=args.epochs, min_epochs=args.min_epochs, batch=args.batch,
                                    lr=args.lr, patience=args.patience, seed=seed)
            tB = time.time() - t1

            pA_va = predict_direct(mA, Rn, val_l, scaler); pB_va = predict_idm(mB, Rn, val_l, fwd, floor_db)
            pA_te = predict_direct(mA, Rn, te_local, scaler); pB_te = predict_idm(mB, Rn, te_local, fwd, floor_db)
            val_curve = {float(b): _mae_db(b * pA_va + (1 - b) * pB_va, val_true_pn) for b in BETAS}
            beta = min(val_curve, key=val_curve.get)
            pE_te = beta * pA_te + (1 - beta) * pB_te
            res = {"seed": int(seed), "beta": beta,
                   "nn_test_mae": _mae_db(pA_te, true_pn_te), "idm_test_mae": _mae_db(pB_te, true_pn_te),
                   "ens_test_mae": _mae_db(pE_te, true_pn_te),
                   "ens_paper_metric_db": _paper_metric(pE_te, true_pn_te),
                   "nn_val_mae": vA, "idm_val_mae": vB, "ens_val_mae": val_curve[beta],
                   "epochs_nn": int(epA), "epochs_idm": int(epB)}
            seed_res.append(res)
            print(f"  N={N:5d} seed={seed} (train {len(train_l):4d}/val {len(val_l):3d})  test MAE: "
                  f"NN={res['nn_test_mae']:.3f}  IDM={res['idm_test_mae']:.3f}  ENS={res['ens_test_mae']:.3f} dB "
                  f"(beta={beta:.2f}; val NN {vA:.2f}/IDM {vB:.2f}; ep {epA}/{epB}; {tA:.0f}s+{tB:.0f}s)", flush=True)
            del mA, mB; torch.cuda.empty_cache()
        row = {"N": int(N), "n_train": int(len(train_l)), "n_val": int(len(val_l)), "seed_diag": seed_res}
        for k in ("nn", "idm", "ens"):
            arr = np.array([r[f"{k}_test_mae"] for r in seed_res])
            row[f"{k}_test_mae"] = float(arr.mean()); row[f"{k}_test_mae_std"] = float(arr.std())
            row[f"{k}_test_mae_best"] = float(arr.min())
        row["betas"] = [r["beta"] for r in seed_res]
        row["ens_paper_metric_db"] = float(np.mean([r["ens_paper_metric_db"] for r in seed_res]))
        rows.append(row)
        print(f"  -> N={N:5d}  ENS mean={row['ens_test_mae']:.4f} ± {row['ens_test_mae_std']:.4f} (best {row['ens_test_mae_best']:.4f})  "
              f"NN {row['nn_test_mae']:.3f}  IDM {row['idm_test_mae']:.3f}  beta={row['betas']}", flush=True)

    demoA, demoB = DirectEEPNet(n_elem), IDMCoefNet(n_elem, n_dip)
    return {"size": scale, "model": "idm_nn", "rapid_zero_shot_mae": reference, "rows": rows,
            "idm_layout": layout, "n_dipoles": int(n_dip), "idm_reg": args.idm_reg,
            "idm_perfect_coefficients_test_mae": oracle_te_mae, "direct_loss": args.direct_loss,
            "params_direct": demoA.count_params(), "params_idm": demoB.count_params()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--size", type=int, nargs="+", default=[28], help="array side(s) N")
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 400, 800, 1200, 1575],
                    help="training budgets (number of target-size patterns)")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--min-epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="IDM branch learning rate")
    ap.add_argument("--lr-direct", type=float, default=3e-4, help="direct branch learning rate")
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--direct-loss", choices=["ssim", "mse"], default="mse",
                    help="published: SSIM. On whole-array patterns with a ~100 dB range SSIM is "
                         "poorly aligned with the dB-MAE metric, so MSE is the default, as for the "
                         "other baselines")
    ap.add_argument("--idm-reg", choices=["noise", "tikhonov"], default="noise")
    ap.add_argument("--tag", type=str, default="idm_nn")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {str(s): run_scale(s, args) for s in args.size}
    out_json = OUT_DIR / f"results_{args.tag}_{'_'.join(map(str, args.size))}.json"
    out_json.write_text(json.dumps(results, indent=2))

    for s, r in results.items():
        print(f"\n[{s}x{s}] IDM-NN: direct branch {r['params_direct'] / 1e6:.1f} M + IDM branch "
              f"{r['params_idm'] / 1e6:.1f} M parameters, {r['n_dipoles']} dipoles")
        print(f"  {'N':>6s} {'direct':>8s} {'IDM':>8s} {'ensemble':>9s} {'std':>6s}  beta")
        for row in r["rows"]:
            print(f"  {row['N']:6d} {row['nn_test_mae']:8.3f} {row['idm_test_mae']:8.3f} "
                  f"{row['ens_test_mae']:9.3f} {row['ens_test_mae_std']:6.3f}  {row['betas']}")
        if r["rapid_zero_shot_mae"] is not None:
            print(f"  RaPID, zero-shot (0 patterns of this size): {r['rapid_zero_shot_mae']:.4f} dB")
    print(f"\nresults: {out_json}")


if __name__ == "__main__":
    main()

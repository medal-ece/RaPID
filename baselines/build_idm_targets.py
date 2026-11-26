"""Infinitesimal-dipole-model (IDM) basis and coefficient targets for the IDM-NN
baseline (Onat et al., IEEE TAP 2024) at one array size.

The published method inverts a Vandermonde-type IDM matrix against the complex
full-wave pattern. The NxN ground truth used here is magnitude-only, so the
inversion target is the hybrid field  E_hyb = E_compose * |E_HFSS| / |E_compose|
(full-wave magnitude, phase and polarisation ratio of the physics-based
composition).

Basis: magnetic infinitesimal dipoles (x and y oriented) on a uniform grid that
extends beyond the aperture, with two independent equivalent-source sheets, one
radiating into the forward hemisphere and one into the back hemisphere. A single
planar sheet cannot represent these patterns: its columns depend on theta only
through sin(theta) and are therefore symmetric under theta -> 180 - theta, while
the back hemisphere of the true pattern is uncorrelated with the front. (The
published element patterns cover one hemisphere, so the issue does not arise
there.)

Inputs:  data/6x6_subblock_fields/, data/NxN_array/, processed/compose_NxN.npz
Output:  processed/idm_NxN.npz  (dipole coefficients per steering direction,
         dipole layout, reconstruction error with perfect coefficients)

Usage:
    python -m baselines.build_idm_targets --size 28
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rapid.composition import (KD, PH, TH, compose, parse_magnitude_csv,  # noqa: E402
                               parse_subblock_field_csv)
from rapid.config import (NPH, NTH, PROCESSED_DIR, SUBBLOCK_FIELD_DIR, array_dir,  # noqa: E402
                          compose_npz)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 28            # array side; set by --size (module-level so dipole_layout's default follows)


def default_layout(n: int) -> str:
    """Dipole grid with 1.0 half-wavelength spacing that extends 4 half-wavelengths
    beyond the aperture on every side, x and y orientations, two sheets."""
    return f"g{n + 8}_xy_e{n // 2 + 4}_h2"


def dipole_layout(n_side: int, half_extent: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """n_side x n_side uniform dipole grid covering the aperture [-half_extent,
    +half_extent] (half-wavelength units). n_side=20 -> one dipole per element at
    the lattice sites; 30 -> lambda/3
    spacing; 40 -> lambda/4 spacing (the paper's dense equivalent-source IDM)."""
    if half_extent is None:
        half_extent = N / 2.0
    s = 2.0 * half_extent / n_side
    g = -half_extent + s / 2.0 + s * np.arange(n_side)
    ix, iy = np.meshgrid(g, g, indexing="ij")
    return ix.ravel(), iy.ravel()


def build_basis(spec: str):
    """spec 'g{n_side}_{orient}[_e{half_extent}]' with orient in x|y|xy; half_extent in
    half-wavelengths (default: the element aperture; a larger value also covers
    the ground plane). Returns (A_th, A_ph, x, y, orient)."""
    parts_ = spec[1:].split("_")
    n_side, orient = parts_[0], parts_[1]
    half_extent = N / 2.0
    two_sheets = False
    for tok in parts_[2:]:
        if tok.startswith("e"):
            half_extent = float(tok[1:])
        elif tok == "h2":
            # Two equivalent-source sheets with independent coefficients: one for
            # the forward hemisphere (image theory on the ground plane) and one for
            # the back hemisphere (edge-diffraction / back radiation). A single
            # planar sheet is theta->180-theta symmetric (columns depend on sin
            # theta only) and cannot represent the uncorrelated back lobe.
            two_sheets = True
    x, y = dipole_layout(int(n_side), half_extent)
    parts = [dipole_basis(o, False, x, y) for o in orient]
    A_th = torch.cat([p[0] for p in parts], 1)
    A_ph = torch.cat([p[1] for p in parts], 1)
    if two_sheets:
        th = torch.tensor(TH, dtype=torch.float32, device=DEV)
        front = (th <= np.pi / 2 + 1e-9).float().expand(NTH, NPH).reshape(-1, 1)
        A_th = torch.cat([A_th * front, A_th * (1 - front)], 1)
        A_ph = torch.cat([A_ph * front, A_ph * (1 - front)], 1)
    return A_th, A_ph, x, y, orient


def dipole_basis(orient: str, ground: bool, x: np.ndarray, y: np.ndarray):
    """Far-field basis of magnetic infinitesimal dipoles at (x, y) on the z=0
    plane, array broadside +z.  E ~ r_hat x m_hat  =>
      m || y:  E_th = -cos(phi),        E_ph =  cos(th) sin(phi)
      m || x:  E_th =  sin(phi),        E_ph =  cos(th) cos(phi)
    ground=True keeps the upper hemisphere only (image theory on a PEC plane).
    Returns (A_th, A_ph): (NTH*NPH, Nd) complex64 torch tensors on DEV."""
    th = torch.tensor(TH, dtype=torch.float32, device=DEV)   # (NTH,1)
    ph = torch.tensor(PH, dtype=torch.float32, device=DEV)   # (1,NPH)
    if orient == "y":
        f_th = (-torch.cos(ph)).expand(NTH, NPH)
        f_ph = torch.cos(th) * torch.sin(ph)
    elif orient == "x":
        f_th = torch.sin(ph).expand(NTH, NPH)
        f_ph = torch.cos(th) * torch.cos(ph)
    else:
        raise ValueError(orient)
    if ground:
        mask = (th <= np.pi / 2 + 1e-9).float().expand(NTH, NPH)
        f_th = f_th * mask
        f_ph = f_ph * mask
    sin_th = torch.sin(th)
    cos_ph = torch.cos(ph)
    sin_ph = torch.sin(ph)
    xs = torch.tensor(x, dtype=torch.float32, device=DEV)
    ys = torch.tensor(y, dtype=torch.float32, device=DEV)
    # psi_i(th,ph) = KD * sin(th) * (x_i cos ph + y_i sin ph)   (Nd, NTH, NPH)
    psi = KD * sin_th[None] * (xs[:, None, None] * cos_ph[None] + ys[:, None, None] * sin_ph[None])
    ph_fac = torch.polar(torch.ones_like(psi), psi)          # complex64
    A_th = (f_th[None].to(torch.complex64) * ph_fac).reshape(len(x), -1).T.contiguous()
    A_ph = (f_ph[None].to(torch.complex64) * ph_fac).reshape(len(x), -1).T.contiguous()
    return A_th, A_ph


class IDMFitter:
    """Least-squares c = argmin ||A c - b||  with A = [A_th; A_ph] stacked.
    reg='noise'   : paper's Gaussian-noise relaxation of the matrix
    reg='tikhonov': (A^H A + lam I) c = A^H b"""

    def __init__(self, A_th, A_ph, reg: str, sigma: float = 1e-3, lam: float = 1e-3, seed: int = 0):
        self.A_th, self.A_ph = A_th, A_ph
        A = torch.cat([A_th, A_ph], 0)                            # (2M, Nd)
        if reg == "noise":
            g = torch.Generator(device="cpu").manual_seed(seed)
            rms = float(A.abs().pow(2).mean().sqrt())
            noise = (torch.randn(A.shape, generator=g) + 1j * torch.randn(A.shape, generator=g)).to(A.dtype) * (sigma * rms / np.sqrt(2))
            A_fit = A + noise.to(A.device)
        else:
            A_fit = A
        self.A_fit = A_fit
        self.chunk = 8192
        t0 = time.time()
        G = torch.zeros(A.shape[1], A.shape[1], dtype=torch.complex128, device=A.device)
        for i in range(0, A_fit.shape[0], self.chunk):
            blk = A_fit[i:i + self.chunk].to(torch.complex128)
            G += blk.conj().T @ blk
        if reg == "tikhonov":
            G = G + lam * G.diagonal().real.mean() * torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        self.G = G
        torch.cuda.synchronize(); t1 = time.time()
        # condition number via eigvalsh on CPU (GPU complex128 syevd is slow on consumer cards)
        ev = torch.linalg.eigvalsh(G.cpu()).real
        self.cond = float(ev.max() / ev.min().clamp_min(1e-30))
        t2 = time.time()
        self.L = torch.linalg.cholesky(G)
        torch.cuda.synchronize()
        print(f"      [IDMFitter {reg}] Nd={A.shape[1]}  G {t1-t0:.1f}s  eig {t2-t1:.1f}s  chol {time.time()-t2:.1f}s",
              flush=True)

    def fit(self, E_th, E_ph):
        """E_th, E_ph: (S, NTH, NPH) complex numpy -> c: (S, Nd) complex64 numpy."""
        S = E_th.shape[0]
        b = torch.cat([torch.as_tensor(E_th.reshape(S, -1)), torch.as_tensor(E_ph.reshape(S, -1))], 1).to(DEV).T  # (2M, S)
        rhs = torch.zeros(self.A_fit.shape[1], S, dtype=torch.complex128, device=DEV)
        for i in range(0, self.A_fit.shape[0], self.chunk):
            rhs += self.A_fit[i:i + self.chunk].to(torch.complex128).conj().T @ b[i:i + self.chunk].to(torch.complex128)
        c = torch.cholesky_solve(rhs, self.L)
        return c.T.to(torch.complex64).cpu().numpy()

    def forward_db(self, c):
        """c: (S, Nd) complex numpy -> peak-normalised dB pattern (S, NTH, NPH)."""
        ct = torch.as_tensor(c).to(DEV)
        e_th = ct @ self.A_th.T
        e_ph = ct @ self.A_ph.T
        mag = torch.sqrt(e_th.abs() ** 2 + e_ph.abs() ** 2).clamp_min(1e-12)
        db = 20.0 * torch.log10(mag)
        db = db - db.max(dim=1, keepdim=True).values
        return db.reshape(-1, NTH, NPH).cpu().numpy().astype(np.float32)


def hybrid_field(E_th_c, E_ph_c, truth_db):
    mag_c = np.sqrt(np.abs(E_th_c) ** 2 + np.abs(E_ph_c) ** 2)
    mag_h = 10.0 ** (truth_db.astype(np.float64) / 20.0)
    scale = (mag_h / np.maximum(mag_c, 1e-20)).astype(np.float32)
    return (E_th_c * scale).astype(np.complex64), (E_ph_c * scale).astype(np.complex64)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--size", type=int, default=28, help="array side N")
    ap.add_argument("--truth-dir", default=None, help="NxN ground-truth folder (default data/NxN_array)")
    ap.add_argument("--sigma", type=float, default=1e-3, help="noise relaxation level (rel. RMS of A)")
    ap.add_argument("--lam", type=float, default=1e-3, help="Tikhonov level (rel. mean diag of A^H A)")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--layouts", nargs="+", default=None,
                    help="dipole layouts to compare on the first file: "
                         "g{n_side}_{x|y|xy}[_e{half_extent}][_h2]; the best one is used")
    ap.add_argument("--layout", default=None, help="use this layout without comparing")
    ap.add_argument("--out", default=None, help="output npz (default processed/idm_NxN.npz)")
    ap.add_argument("--no-noise", action="store_true",
                    help="skip the noise-relaxed fit (halves GPU memory); 'coef' then equals 'coef_tikh'")
    args = ap.parse_args()
    global N
    N = args.size
    wh_dir = Path(args.truth_dir) if args.truth_dir else array_dir(N)
    out_npz = Path(args.out) if args.out else PROCESSED_DIR / f"idm_{N}x{N}.npz"
    out_json = out_npz.with_suffix(".json")
    layouts = args.layouts or [default_layout(N)]

    ref_ids = np.load(compose_npz(N))["ids"]
    sub_files = sorted(Path(SUBBLOCK_FIELD_DIR).glob("*.csv"))
    if not sub_files:
        raise SystemExit(f"no CSV files in {SUBBLOCK_FIELD_DIR} (see data/README.md)")

    coef_noise, coef_tikh, ids_all, dpx_all, dpy_all = [], [], [], [], []
    mae_noise, mae_tikh, relres_noise = [], [], []
    fitters = None
    chosen = None
    probe = {}
    t0 = time.time()
    n_done = 0
    for fi, sf in enumerate(sub_files, 1):
        try:
            E_theta, E_phi, dpx, dpy, ids_s = parse_subblock_field_csv(sf)
        except ValueError:
            print(f"[{fi:2d}/{len(sub_files)}] {sf.name}  skipped (truncated)", flush=True)
            continue
        truth, dpx_t, dpy_t, ids_t = parse_magnitude_csv(wh_dir / sf.name)
        assert np.array_equal(ids_s, ids_t)
        E_th_c, E_ph_c = compose(E_theta, E_phi, dpx, dpy, N, return_complex=True)
        Eh_th, Eh_ph = hybrid_field(E_th_c, E_ph_c, truth)
        del E_theta, E_phi

        if chosen is None:
            if args.layout is None and len(layouts) > 1:
                # Following the published procedure: compare dipole densities /
                # orientations on the first file and keep the best fit.
                print("Comparing dipole layouts on the first file ...", flush=True)
                for spec in layouts:
                    A_th, A_ph, xx, yy, _ = build_basis(spec)
                    ft = IDMFitter(A_th, A_ph, "tikhonov", lam=args.lam)
                    m = float(np.abs(ft.forward_db(ft.fit(Eh_th, Eh_ph)) - truth).mean())
                    probe[spec] = {"n_dipoles": int(A_th.shape[1]), "reconstruction_mae": m, "cond": ft.cond}
                    print(f"   {spec}: {A_th.shape[1]} dipoles, reconstruction MAE {m:.3f} dB, "
                          f"cond {ft.cond:.2e}", flush=True)
                    del A_th, A_ph, ft
                    torch.cuda.empty_cache()
                chosen = min(probe, key=lambda k: probe[k]["reconstruction_mae"])
            else:
                chosen = args.layout or layouts[0]
            print(f"-> dipole layout: {chosen}", flush=True)
            A_th, A_ph, x, y, orient = build_basis(chosen)
            fitters = {"tikhonov": IDMFitter(A_th, A_ph, "tikhonov", lam=args.lam)}
            fitters["noise"] = (fitters["tikhonov"] if args.no_noise
                                else IDMFitter(A_th, A_ph, "noise", sigma=args.sigma))

        cn = fitters["noise"].fit(Eh_th, Eh_ph)
        ck = fitters["tikhonov"].fit(Eh_th, Eh_ph)
        mn = np.abs(fitters["noise"].forward_db(cn) - truth).mean(axis=(1, 2))
        mk = np.abs(fitters["tikhonov"].forward_db(ck) - truth).mean(axis=(1, 2))
        # relative linear-field residual (noise fit)
        b = np.concatenate([Eh_th.reshape(len(cn), -1), Eh_ph.reshape(len(cn), -1)], 1)
        with torch.no_grad():
            ct = torch.as_tensor(cn).to(DEV)
            pred = torch.cat([ct @ fitters["noise"].A_th.T, ct @ fitters["noise"].A_ph.T], 1).cpu().numpy()
        rr = np.linalg.norm(pred - b, axis=1) / np.linalg.norm(b, axis=1)
        del E_th_c, E_ph_c
        coef_noise.append(cn); coef_tikh.append(ck); ids_all.append(ids_s); dpx_all.append(dpx); dpy_all.append(dpy)
        mae_noise.append(mn); mae_tikh.append(mk); relres_noise.append(rr)
        n_done += 1
        print(f"[{fi:2d}/{len(sub_files)}] {sf.name}  reconstruction MAE noise={mn.mean():.3f} "
              f"tikhonov={mk.mean():.3f} dB  ({time.time()-t0:.0f}s)", flush=True)
        if args.max_files and n_done >= args.max_files:
            break

    ids = np.concatenate(ids_all)
    if args.max_files is None:
        assert np.array_equal(ids, ref_ids), f"sample order differs from {compose_npz(N).name}"
    coef_noise = np.concatenate(coef_noise); coef_tikh = np.concatenate(coef_tikh)
    mae_noise = np.concatenate(mae_noise); mae_tikh = np.concatenate(mae_tikh); relres = np.concatenate(relres_noise)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz, coef=coef_noise, coef_tikh=coef_tikh, ids=ids,
        dpx=np.concatenate(dpx_all), dpy=np.concatenate(dpy_all),
        x=x.astype(np.float32), y=y.astype(np.float32), orient=orient, layout=chosen,
        oracle_mae=mae_noise, oracle_mae_tikh=mae_tikh, relres=relres,
        sigma=args.sigma, lam=args.lam,
        cond_noise=fitters["noise"].cond, cond_tikh=fitters["tikhonov"].cond,
    )
    summary = {
        "n_samples": int(len(ids)),
        "basis": {"layout": chosen, "orient": orient, "n_dipoles": int(coef_noise.shape[1])},
        "layout_comparison": probe, "sigma": args.sigma, "lam": args.lam,
        "reconstruction_mae_noise_db": float(mae_noise.mean()),
        "reconstruction_mae_tikhonov_db": float(mae_tikh.mean()),
        "relative_residual_noise": float(relres.mean()),
    }
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved {out_npz}")


if __name__ == "__main__":
    main()

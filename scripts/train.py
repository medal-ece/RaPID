"""Train the RaPID CNN on the <= 6x6 training set.

The network learns the correction (ground truth minus base pattern, in dB) on
the two training tasks built by scripts/build_training_set.py. Recipe: Adam
(lr 2e-4), batch 16, at most 200 epochs, learning rate halved when the
validation score plateaus, early stopping after 40 epochs without improvement,
EMA of the weights (the EMA weights are saved).

Usage:
    python -m scripts.train --run rapid_retrain
    python -m scripts.train --run ablation_no_skips --no-skips
    python -m scripts.train --run ablation_no_heads --no-heads
    python -m scripts.train --run ablation_no_composition --no-composition

The three flags reproduce the arms of the ablation study (Table IV).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rapid.config import (BATCH_SIZE, CHECKPOINTS_DIR, DEVICE, NULL_THRESHOLD_DB,
                          PROCESSED_DIR, RANDOM_SEED)
from rapid.inference import build_inputs, input_stats_from_patterns, pixel_stats
from rapid.losses import EMA, RaPIDLoss, topk_null_depths
from rapid.model import CHANNELS, RaPIDNet, count_parameters

LR = 2e-4
MAX_EPOCHS = 200
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 4
PLATEAU_MIN_LR = 1e-6
VALIDATE_EVERY = 2
EARLY_STOP_PATIENCE = 40
AUX_RAMP_EPOCHS = 20
NOISE_STD = 0.05
NOISE_CHANNELS = {"composition", "af"}
# validation score = weighted RMS error of the correction per region
VAL_WEIGHT_MAIN, VAL_WEIGHT_MAIN_SL, VAL_WEIGHT_NULL = 0.30, 0.30, 0.40


class PatternDataset(Dataset):
    def __init__(self, rows, data, stats, channels, base_name, augment):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.d, self.s = data, stats
        self.channels, self.base_name = channels, base_name
        self.noise_pos = [i for i, c in enumerate(channels) if c in NOISE_CHANNELS] if augment else []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = int(self.rows[i])
        comp = self.d["composition"][r].astype(np.float32)
        af = self.d["af"][r].astype(np.float32)
        truth = self.d["truth"][r].astype(np.float32)
        base = comp if self.base_name == "composition" else af

        x = build_inputs(comp[None], af[None], self.s, self.channels)[0]
        for p in self.noise_pos:
            x[p] += np.random.normal(0, NOISE_STD, x[p].shape).astype(np.float32)

        res_n = (truth - base - self.s["res_mean"]) / self.s["res_std"]
        null_mask = (truth < truth.max() + NULL_THRESHOLD_DB).astype(np.float32)
        return (torch.from_numpy(x), torch.from_numpy(res_n[None].astype(np.float32)),
                torch.from_numpy(null_mask[None]), torch.from_numpy(topk_null_depths(truth)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="output folder name under checkpoints/")
    ap.add_argument("--data", default=str(PROCESSED_DIR / "train_set.npz"))
    ap.add_argument("--split", default=str(PROCESSED_DIR / "train_split.npz"))
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--no-skips", action="store_true", help="ablation: no U-Net skip connections")
    ap.add_argument("--no-heads", action="store_true", help="ablation: single output head")
    ap.add_argument("--no-composition", action="store_true",
                    help="ablation: the ideal array factor replaces the physics-based composition")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    channels = ["af"] if args.no_composition else list(CHANNELS)
    base_name = channels[0]
    config = {"channels": channels, "base_idx": 0,
              "use_skips": not args.no_skips, "use_heads": not args.no_heads}

    raw = np.load(args.data)
    data = {k: raw[k] for k in ("composition", "af", "truth")}
    sp = np.load(args.split)
    tr, va = sp["train"], sp["val"]

    stats = input_stats_from_patterns(data["composition"][tr], data["af"][tr])
    base = data[base_name]
    stats["res_mean"], stats["res_std"] = pixel_stats(
        data["truth"][tr].astype(np.float32) - base[tr].astype(np.float32))
    base_mean, base_std = stats[f"{base_name}_mean"], stats[f"{base_name}_std"]

    ckpt_dir = CHECKPOINTS_DIR / args.run
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = RaPIDNet(in_ch=len(channels), base_idx=0, base_mean=base_mean, base_std=base_std,
                     res_mean=stats["res_mean"], res_std=stats["res_std"],
                     use_skips=config["use_skips"], use_heads=config["use_heads"]).to(DEVICE)
    print(f"[{args.run}] channels={channels} skips={config['use_skips']} heads={config['use_heads']} "
          f"train={len(tr)} val={len(va)} params={count_parameters(model) / 1e6:.2f}M", flush=True)

    pin = DEVICE.type == "cuda"
    tr_loader = DataLoader(PatternDataset(tr, data, stats, channels, base_name, augment=True),
                           batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin, drop_last=True)
    va_loader = DataLoader(PatternDataset(va, data, stats, channels, base_name, augment=False),
                           batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)

    criterion = RaPIDLoss(stats["res_mean"], stats["res_std"], base_mean, base_std,
                          anchor_main_beam=not args.no_composition).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE, min_lr=PLATEAU_MIN_LR)
    ema = EMA(model)

    @torch.no_grad()
    def validate() -> float:
        live = ema.swap_into(model)
        try:
            model.eval()
            main_e, main_sl_e, null_e = [], [], []
            rms = lambda a: float(np.sqrt((a ** 2).mean()))
            for x, y, null_mask, _ in va_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                err = ((model(x) - y) * model.res_std).float()
                base_db = x[:, :1] * model.base_std + model.base_mean
                truth_db = base_db + y * model.res_std + model.res_mean
                peak = base_db.amax(dim=(-2, -1), keepdim=True)
                main, main_sl = truth_db >= peak - 3.0, truth_db >= peak - 20.0
                for i in range(err.shape[0]):
                    e = err[i, 0].cpu().numpy()
                    for mask, acc in ((main[i, 0], main_e), (main_sl[i, 0], main_sl_e),
                                      (null_mask[i, 0] > 0.5, null_e)):
                        m = mask.cpu().numpy()
                        if m.any():
                            acc.append(rms(e[m]))
            return (VAL_WEIGHT_MAIN * np.mean(main_e) + VAL_WEIGHT_MAIN_SL * np.mean(main_sl_e)
                    + VAL_WEIGHT_NULL * np.mean(null_e))
        finally:
            ema.restore(model, live)

    best, best_epoch, stale = float("inf"), -1, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        aux_scale = min(1.0, epoch / AUX_RAMP_EPOCHS)
        losses = []
        for x, y, null_mask, null_depths in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            null_mask, null_depths = null_mask.to(DEVICE), null_depths.to(DEVICE)
            loss = criterion(model(x, return_aux=True), y, null_mask, null_depths,
                             x[:, :1], aux_scale=aux_scale)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema.update(model)
            losses.append(float(loss.detach()))

        if epoch % VALIDATE_EVERY == 0 or epoch == 1 or epoch == args.epochs:
            score = validate()
            improved = score < best - 1e-5
            print(f"[{args.run}] epoch {epoch:03d} loss={np.mean(losses):.4f} val={score:.4f} "
                  f"lr={opt.param_groups[0]['lr']:.1e}{' *' if improved else ''}", flush=True)
            if improved:
                best, best_epoch, stale = score, epoch, 0
                torch.save({"state_dict": ema.state_dict(), "config": config}, ckpt_dir / "rapid.pt")
            else:
                stale += VALIDATE_EVERY
            sched.step(score)
            if stale >= EARLY_STOP_PATIENCE:
                print(f"[{args.run}] early stop at epoch {epoch}", flush=True)
                break

    print(f"[{args.run}] done in {(time.time() - t0) / 60:.1f} min; best epoch {best_epoch} "
          f"(val {best:.5f}) -> {ckpt_dir / 'rapid.pt'}", flush=True)


if __name__ == "__main__":
    main()

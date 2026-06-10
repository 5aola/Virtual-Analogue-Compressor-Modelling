"""Training entry point.

Examples
--------
Train on a manifest of WAV pairs::

    python -m mamba_va.train --manifest data/cl1b/manifest.csv \
        --d_model 24 --n_layers 3 --d_state 16 --n_params 4 \
        --seq_len 2048 --batch_size 8 --epochs 200 --out runs/cl1b

Smoke-train on synthetic data (no files needed)::

    python -m mamba_va.train --synthetic --epochs 3 --out runs/synth
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import torch

from .model import CompSSM
from .losses import CombinedLoss, esr
from .data import WavPairsDataset
from .utils import detach_state, count_params


def build_synthetic_dataset(n_settings=6, seconds=6.0, sr=48000):
    from .synth import make_dataset
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_settings):
        params = dict(
            threshold_db=float(rng.uniform(-40, -6)),
            ratio=float(rng.uniform(2, 10)),
            attack_ms=float(10 ** rng.uniform(-0.3, 2.3)),
            release_ms=float(10 ** rng.uniform(1.3, 3.3)),
        )
        dry, wet, pvec = make_dataset(seconds=seconds, sr=sr, params=params, seed=i)
        rows.append((dry, wet, pvec))
    ds = WavPairsDataset.__new__(WavPairsDataset)
    ds.dry = [r[0] for r in rows]
    ds.wet = [r[1] for r in rows]
    ds.params = [np.asarray(r[2], np.float32) for r in rows]
    ds.n_params = 4
    ds.sr = sr
    return ds


def evaluate(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        state = None
        for b in loader:
            if b["reset"]:
                state = None
            y_hat, state = model(b["x"], b["params"], state, parallel=True)
            state = detach_state(state)
            tot += esr(y_hat, b["y"]).item()
            n += 1
    return tot / max(1, n)


def train(args):
    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.synthetic:
        full = build_synthetic_dataset(seconds=args.synth_seconds, sr=args.synth_sr)
        n_params = full.n_params
    else:
        full = WavPairsDataset(args.manifest, n_params=args.n_params, sr=args.sr)
        n_params = full.n_params
    train_ds, val_ds = full.split(frac=args.train_frac)

    model = CompSSM(
        n_params=n_params, d_model=args.d_model, d_state=args.d_state,
        n_layers=args.n_layers, expand=args.expand, conv_kernel=args.conv_kernel,
        n_bands=args.n_bands, max_db=args.max_db,
        sr=float(getattr(full, "sr", None) or args.sr or 44100),
    ).to(device)
    print(f"model parameters: {count_params(model)}")

    crit = CombinedLoss(
        w_esr=args.w_esr, w_preemph=args.w_preemph,
        w_stft=args.w_stft, w_dc=args.w_dc,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_gamma)

    os.makedirs(args.out, exist_ok=True)
    best = float("inf")
    bad = 0
    for epoch in range(args.epochs):
        model.train()
        loader = train_ds.tbptt_loader(args.batch_size, args.seq_len,
                                       shuffle=True, device=device)
        state = None
        running, steps = 0.0, 0
        for b in loader:
            if b["reset"]:
                state = None
            y_hat, state = model(b["x"], b["params"], state, parallel=True)
            loss = crit(y_hat, b["y"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            state = detach_state(state)
            running += loss.item()
            steps += 1
        sched.step()

        val = evaluate(model, val_ds.tbptt_loader(args.batch_size, args.seq_len,
                                                  shuffle=False, device=device), device)
        print(f"epoch {epoch:3d}  train_loss {running/max(1,steps):.5f}  val_ESR {val:.5f}")

        if val < best:
            best = val
            bad = 0
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "n_params": n_params, "val_esr": val,
                        "sr": float(model.sr)},
                       os.path.join(args.out, "best.pt"))
        else:
            bad += 1
            if args.patience and bad >= args.patience:
                print(f"early stopping at epoch {epoch} (best val_ESR {best:.5f})")
                break
    print(f"done. best val_ESR {best:.5f}. checkpoint: {os.path.join(args.out,'best.pt')}")
    return best


def build_argparser():
    p = argparse.ArgumentParser(description="Train CompSSM on audio-effect WAV pairs")
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--synthetic", action="store_true", help="use built-in synthetic data")
    p.add_argument("--synth_seconds", type=float, default=6.0)
    p.add_argument("--synth_sr", type=int, default=48000)
    p.add_argument("--out", type=str, default="runs/exp")
    p.add_argument("--sr", type=int, default=None)
    p.add_argument("--n_params", type=int, default=None)
    # model
    p.add_argument("--d_model", type=int, default=24)
    p.add_argument("--d_state", type=int, default=16)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--conv_kernel", type=int, default=4)
    p.add_argument("--n_bands", type=int, default=4)
    p.add_argument("--max_db", type=float, default=48.0)
    # optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_gamma", type=float, default=0.99)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=8192,
                   help="TBPTT gradient window; must cover the device's slowest "
                        "time constant (e.g. >=0.4 s of samples for a 0.4 s release)")
    p.add_argument("--train_frac", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # loss weights
    p.add_argument("--w_esr", type=float, default=1.0)
    p.add_argument("--w_preemph", type=float, default=0.5)
    p.add_argument("--w_stft", type=float, default=0.5)
    p.add_argument("--w_dc", type=float, default=0.1)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)

"""Frozen GR-predictor frontend for the two-stage (predicted-GR) cascade.

Stage 1 of the 09_end2end pipeline: load a trained ``BlackboxGRLSTM`` run from
``gr_pred_runs/`` (05_conditioning), freeze it, and stream every (song, setting)
pair's FULL dry song through it — state carried across chunks, cold start only
at the true song start, i.e. the deployment condition — to export predicted
``gr_db`` curves. The curves are written in the exact ``gr_curves/<setting>/
<song>.pt`` / ``{"gr_db": [1, T]}`` sample-rate format the 06_output
``GRCropDataModule`` reads, into a *view* dataset root whose dry/wet folders
are symlinks to the real cache. Stage 2 (the gain-prior WS model) then trains
against that root with zero dataloader changes — the only difference from the
oracle run is which curve sits under ``gr_curves/``.

Import hygiene: 05_conditioning cannot go on ``sys.path`` (its ``splits.py`` /
``dataset.py`` / ``model.py`` collide with 06_output's). ``gr_target`` and
``model_blackbox_gr`` are therefore loaded by explicit file path and registered
in ``sys.modules`` so the inner ``from gr_target import ...`` resolves.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F


# ── 05_conditioning module loading (by path — no sys.path pollution) ──────

def _load_module_from_path(path: str, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register BEFORE exec so inner imports resolve
    spec.loader.exec_module(mod)
    return mod


def load_blackbox_class(cond05_dir: str):
    """Return the ``BlackboxGRLSTM`` class from 05_conditioning without putting
    that directory (and its colliding module names) on sys.path."""
    _load_module_from_path(os.path.join(cond05_dir, "gr_target.py"), "gr_target")
    mod = _load_module_from_path(
        os.path.join(cond05_dir, "model_blackbox_gr.py"), "model_blackbox_gr"
    )
    return mod.BlackboxGRLSTM


# ── checkpoint selection & loading ─────────────────────────────────────────

def select_best_ckpt(run_dir: str) -> str:
    """Pick the saved ``best-*.ckpt`` whose epoch has the lowest ``loss/val``
    in the run's CSV log (ModelCheckpoint filenames carry no metric). Falls
    back to the newest best-*, then last.ckpt."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    bests = sorted(glob.glob(os.path.join(ckpt_dir, "best-*.ckpt")))
    if not bests:
        return os.path.join(ckpt_dir, "last.ckpt")

    epoch_of = {}
    for c in bests:
        m = re.match(r"best-(\d+)-\d+\.ckpt", os.path.basename(c))
        if m:
            epoch_of[int(m.group(1))] = c

    metrics_csv = os.path.join(run_dir, "csv", "metrics.csv")
    if epoch_of and os.path.isfile(metrics_csv):
        df = pd.read_csv(metrics_csv)
        if "loss/val" in df.columns:
            val = df.dropna(subset=["loss/val"]).groupby("epoch")["loss/val"].min()
            cands = val[val.index.isin(epoch_of)]
            if not cands.empty:
                return epoch_of[int(cands.idxmin())]
    return bests[-1]


def load_frozen_gr_predictor(
    run_dir: str,
    cond05_dir: str,
    ckpt_path: str | None = None,
    device: str = "cuda",
):
    """Build ``BlackboxGRLSTM`` from the run's hparams.json, load the checkpoint
    (Lightning ``model.`` prefix stripped), freeze, eval. Returns
    ``(model, hparams, ckpt_path)``."""
    with open(os.path.join(run_dir, "hparams.json")) as f:
        hp = json.load(f)
    assert hp["model_type"].startswith("blackbox_gr_lstm"), hp["model_type"]

    BlackboxGRLSTM = load_blackbox_class(cond05_dir)
    m = hp["model"]
    model = BlackboxGRLSTM(
        hop_size=int(hp["hop_size"]),
        sample_rate=int(hp["sample_rate"]),
        enc_channels=int(m["enc_channels"]),
        frontend_channels=int(m["frontend_channels"]),
        temporal_kernel=int(m["temporal_kernel"]),
        temporal_dilations=tuple(int(d) for d in m["temporal_dilations"]),
        hidden_size=int(m["hidden_size"]),
        film_order=int(m["film_order"]),
    )

    ckpt_path = ckpt_path or select_best_ckpt(run_dir)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
    model.load_state_dict(
        {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    )
    model.to(device).eval().requires_grad_(False)
    n = sum(p.numel() for p in model.parameters())
    print(f"Frozen GR predictor: {n:,} params  ({hp['model_type']})")
    print(f"  ckpt: {os.path.basename(ckpt_path)}")
    return model, hp, ckpt_path


# ── predicted-GR view export ───────────────────────────────────────────────

def build_view_root(view_root: str, data_root: str) -> None:
    """View dataset root: predicted gr_curves/ + symlinked dry/wet folders."""
    os.makedirs(os.path.join(view_root, "gr_curves"), exist_ok=True)
    for d in ("processed_normalized", "processed_ground_truth", "test_ground_truth"):
        src, dst = os.path.join(data_root, d), os.path.join(view_root, d)
        if os.path.isdir(src) and not (os.path.islink(dst) or os.path.exists(dst)):
            os.symlink(src, dst)


@torch.no_grad()
def predict_gr_view(
    model,
    pairs: list[dict],
    view_root: str,
    data_root: str,
    device: str = "cuda",
    chunk_sec: float = 120.0,
    test_keys: set[str] | None = None,
) -> pd.DataFrame:
    """Export predicted GR curves for every pair; return per-pair quality rows.

    Per song, all settings are batched into one streaming pass (same dry,
    different knobs). Chunks are hop-aligned and state is carried across them
    (the frontend's streaming contract makes this exact vs one full forward);
    the ONLY cold start is the real song start. The frame-rate prediction is
    concatenated across chunks first and interpolated to sample rate once, so
    there are no chunk-boundary interpolation seams.

    Quality columns compare against the ORACLE curves (``pairs[i]["gr"]``):
    ``mae_db`` full-length, ``mae_db_post1s`` skipping the cold-start second.
    """
    sr = model.sample_rate
    hop = model.hop_size
    chunk = max(hop, (int(round(chunk_sec * sr)) // hop) * hop)

    by_song: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_song[p["song"]].append(p)

    rows = []
    for song in sorted(by_song):
        plist = by_song[song]
        audio, file_sr = sf.read(plist[0]["dry"], dtype="float32", always_2d=True)
        assert file_sr == sr, f"{song}: {file_sr} Hz dry vs model {sr} Hz"
        dry = torch.from_numpy(audio.T).mean(dim=0, keepdim=True)      # [1, L]
        L = dry.shape[-1] - (dry.shape[-1] % hop)
        dry = dry[:, :L].unsqueeze(0)                                   # [1, 1, L]

        params = torch.tensor(
            [p["params"] for p in plist], dtype=torch.float32, device=device
        )                                                               # [B, 4]
        B = params.shape[0]

        state, frames = None, []
        for o in range(0, L, chunk):
            seg = dry[..., o : o + chunk].expand(B, 1, -1).to(device)
            gr, state = model(seg, params, state, return_state=True)
            frames.append(model.to_db(gr).cpu())                        # [B, 1, Tf] dB
        gr_db = torch.cat(frames, dim=-1)                               # frame rate

        for i, p in enumerate(plist):
            curve = F.interpolate(
                gr_db[i : i + 1], size=L, mode="linear", align_corners=False
            )[0].contiguous()                                           # [1, L]
            out = os.path.join(view_root, "gr_curves", p["setting"], f"{song}.pt")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            torch.save({"gr_db": curve}, out)

            true = torch.load(p["gr"], weights_only=False)["gr_db"].float()
            if true.dim() == 1:
                true = true.unsqueeze(0)
            n = min(L, true.shape[-1])
            err = (curve[:, :n] - true[:, :n]).abs()
            key = f"{song}::{p['setting']}"
            rows.append({
                "song": song,
                "setting": p["setting"],
                "split": "test" if test_keys and key in test_keys else "train/val",
                "mae_db": float(err.mean()),
                "mae_db_post1s": float(err[:, sr:].mean()) if n > sr else float("nan"),
                "minutes": n / sr / 60.0,
            })
        print(f"  {song}: {B} settings x {L/sr/60.0:.1f} min")

    df = pd.DataFrame(rows).sort_values(["split", "song", "setting"]).reset_index(drop=True)
    df.to_csv(os.path.join(view_root, "gr_pred_quality.csv"), index=False)
    return df

from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import torch


def _checkpoint_sort_key(path: str | Path) -> tuple[int, int, float]:
    """Sort Lightning checkpoints by encoded epoch/step, then filesystem mtime."""
    path = Path(path)
    match = re.match(r"(?:best|epoch)-(\d+)(?:-(\d+))?", path.stem)
    if match:
        return int(match.group(1)), int(match.group(2) or -1), path.stat().st_mtime
    return -1, -1, path.stat().st_mtime


def latest_best_checkpoint(run: str | Path) -> str:
    """Return the latest best checkpoint, falling back to last or any checkpoint."""
    run = Path(run)
    ckpt_dir = run / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(ckpt_dir)

    best_files = sorted(ckpt_dir.glob("best-*.ckpt"), key=_checkpoint_sort_key)
    if best_files:
        return str(best_files[-1])

    last_path = ckpt_dir / "last.ckpt"
    if last_path.is_file():
        return str(last_path)

    all_ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=_checkpoint_sort_key)
    if not all_ckpts:
        raise FileNotFoundError(f"No .ckpt files in {ckpt_dir}")
    return str(all_ckpts[-1])


def list_runs(runs_dir: str | Path) -> pd.DataFrame:
    """Return run folders with hparams, metrics, and the default eval checkpoint."""
    import pandas as pd

    runs_dir = Path(runs_dir)
    rows = []
    if not runs_dir.is_dir():
        return pd.DataFrame()

    for run in sorted(runs_dir.iterdir()):
        if not run.is_dir():
            continue

        ckpts = sorted(run.glob("checkpoints/*.ckpt"), key=_checkpoint_sort_key)
        csv_path = run / "csv" / "metrics.csv"
        best_val, latest_epoch = None, None
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if "loss/val" in df.columns and df["loss/val"].notna().any():
                    best_val = float(df["loss/val"].dropna().min())
                if "epoch" in df.columns and df["epoch"].notna().any():
                    latest_epoch = int(df["epoch"].dropna().max())
            except Exception:
                pass

        hp_path = run / "hparams.json"
        hp = {}
        if hp_path.exists():
            try:
                hp = json.loads(hp_path.read_text())
            except Exception:
                hp = {}

        eval_ckpt_name = None
        if ckpts:
            try:
                eval_ckpt_name = Path(latest_best_checkpoint(run)).name
            except Exception:
                eval_ckpt_name = ckpts[-1].name

        rows.append(
            {
                "run": run.name,
                "type": hp.get("model_type", "?"),
                "setting": hp.get("setting", "?"),
                "dataset": hp.get("dataset", "?"),
                "epoch": latest_epoch,
                "best_val": best_val,
                "n_ckpts": len(ckpts),
                "eval_ckpt": eval_ckpt_name,
                "modified": pd.to_datetime(run.stat().st_mtime, unit="s"),
                "mtime": run.stat().st_mtime,
                "path": str(run),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("mtime").reset_index(drop=True)


def plot_loss_curves(run_dir: str | Path, ymin: float | None = None) -> pd.DataFrame | None:
    import matplotlib.pyplot as plt
    import pandas as pd

    csv_path = Path(run_dir) / "csv" / "metrics.csv"
    if not csv_path.exists():
        print(f"No metrics.csv yet at {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    has_mae = any(col in df.columns for col in ("mae_db/train", "mae_db/val"))
    ncols = 2 if has_mae else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4), squeeze=False)
    axes = axes[0]

    ax = axes[0]
    train = df.dropna(subset=["loss/train"])[["epoch", "loss/train"]] if "loss/train" in df.columns else pd.DataFrame()
    val = df.dropna(subset=["loss/val"])[["epoch", "loss/val"]] if "loss/val" in df.columns else pd.DataFrame()
    if len(train):
        ax.plot(train["epoch"], train["loss/train"], label="train", lw=1.5, color="#1f77b4")
    if len(val):
        ax.plot(val["epoch"], val["loss/val"], label="val", lw=1.5, color="#d62728")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    last_epoch = int(
        max(
            train["epoch"].max() if len(train) else -1,
            val["epoch"].max() if len(val) else -1,
        )
    )
    best_val = float(val["loss/val"].min()) if len(val) else float("nan")
    ax.set_title(f"{Path(run_dir).name}  |  epoch={last_epoch}  |  best val={best_val:.4f}")
    if ymin is not None:
        ax.set_ylim(bottom=ymin)

    if has_mae:
        ax2 = axes[1]
        for split, color in (("train", "#1f77b4"), ("val", "#d62728")):
            col = f"mae_db/{split}"
            if col in df.columns:
                rows = df.dropna(subset=[col])[["epoch", col]]
                if len(rows):
                    ax2.plot(rows["epoch"], rows[col], label=split, lw=1.5, color=color)
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("GR MAE (dB)")
        ax2.grid(alpha=0.3)
        ax2.legend()
        ax2.set_title("GR MAE")

    plt.tight_layout()
    plt.show()
    return df


def _find_hparams_json(ckpt_path: str | Path) -> str | None:
    """Look for hparams.json next to the checkpoint or one/two levels up."""
    ckpt_dir = Path(ckpt_path).expanduser().resolve().parent
    candidates = [
        ckpt_dir / "hparams.json",
        ckpt_dir.parent / "hparams.json",
        ckpt_dir.parent.parent / "hparams.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def wet_dir_for_setting(data_root: str | Path, setting: str) -> str:
    data_root = Path(data_root)
    candidates = [
        data_root / setting,
        data_root / "processed_ground_truth" / setting,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    raise FileNotFoundError(f"No wet directory found for setting={setting}. Tried: {candidates}")


def _song_from_wet_path(wet_path: str | Path) -> str:
    name = Path(wet_path).name
    if name.endswith("-exported.wav"):
        return name.replace("-exported.wav", "")

    stem = Path(name).stem
    match = re.match(r"^Bounce\s+\d+-(.+?)_UnmasteredWAV(?:\s+\[.*\])?(?:-\d+)?$", stem)
    if match:
        return match.group(1)

    if "_UnmasteredWAV" in stem:
        return stem.split("_UnmasteredWAV", 1)[0]
    return stem


def same_setting_pairs(data_root: str | Path, setting: str) -> list[tuple[str, str]]:
    dry_dir = Path(data_root) / "processed_normalized"
    wet_dir = Path(wet_dir_for_setting(data_root, setting))
    assert dry_dir.is_dir(), f"Missing: {dry_dir}"
    assert wet_dir.is_dir(), f"Missing: {wet_dir}"

    dry_lookup = {
        Path(path).name.replace("_UnmasteredWAV.wav", ""): path
        for path in sorted(glob.glob(str(dry_dir / "*_UnmasteredWAV.wav")))
    }

    pairs = []
    unmatched = []
    for wet_path in sorted(glob.glob(str(wet_dir / "*.wav"))):
        song = _song_from_wet_path(wet_path)
        if song in dry_lookup:
            pairs.append((dry_lookup[song], wet_path))
        else:
            unmatched.append((song, Path(wet_path).name))

    if not pairs:
        preview = unmatched[:10]
        raise AssertionError(
            f"No dry/wet pairs for setting={setting} using wet_dir={wet_dir}. "
            f"Dry examples={list(dry_lookup)[:10]}; parsed wet examples={preview}"
        )
    if unmatched:
        print(f"Matched {len(pairs)} wet files; skipped {len(unmatched)} without dry match.")
    return pairs


def _read_dry_wet_segment(
    dry_path: str | Path,
    wet_path: str | Path,
    start: int,
    stop: int,
    sr: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch
    import torchaudio
    import soundfile as sf

    dry_np, sr_d = sf.read(dry_path, start=start, stop=stop, dtype="float32", always_2d=True)
    wet_np, sr_w = sf.read(wet_path, start=start, stop=stop, dtype="float32", always_2d=True)
    dry = torch.from_numpy(dry_np.T)
    wet = torch.from_numpy(wet_np.T)
    if sr_d != sr:
        dry = torchaudio.functional.resample(dry, sr_d, sr)
    if sr_w != sr:
        wet = torchaudio.functional.resample(wet, sr_w, sr)
    if dry.shape[0] > 1:
        dry = dry.mean(dim=0, keepdim=True)
    if wet.shape[0] > 1:
        wet = wet.mean(dim=0, keepdim=True)
    min_len = min(dry.shape[-1], wet.shape[-1])
    return dry[..., :min_len], wet[..., :min_len]


def _db_to_amplitude(db: torch.Tensor) -> torch.Tensor:
    import torch

    return torch.pow(10.0, db / 20.0)


def _pair_num_frames(dry_path: str | Path, wet_path: str | Path, sr: int) -> int:
    import soundfile as sf

    dry_info = sf.info(dry_path)
    wet_info = sf.info(wet_path)
    if dry_info.samplerate != sr or wet_info.samplerate != sr:
        print(
            f"[warning] {Path(dry_path).stem}: expected {sr} Hz, "
            f"got dry={dry_info.samplerate}, wet={wet_info.samplerate}; "
            "segment offsets assume matching sample rates."
        )
    return min(dry_info.frames, wet_info.frames)


def _song_from_dry_path(dry_path: str | Path) -> str:
    return Path(dry_path).name.replace("_UnmasteredWAV.wav", "")


def _validation_pairs_from_training_split(
    all_pairs: list[tuple[str, str]],
    hp: dict,
) -> list[tuple[str, str]]:
    import torch

    songs = sorted({_song_from_dry_path(dry_path) for dry_path, _ in all_pairs})
    split_seed = int(hp.get("split_seed", 42))
    train_split = float(hp.get("train_split", 0.8))

    generator = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(len(songs), generator=generator).tolist()
    n_train = int(len(songs) * train_split)
    n_train = min(max(1, n_train), len(songs) - 1)

    train_songs = {songs[i] for i in perm[:n_train]}
    val_songs = set(songs) - train_songs

    print(
        f"Validation split: {len(val_songs)} validation songs, "
        f"{len(train_songs)} train songs (seed={split_seed}, train_split={train_split})"
    )
    print(f"Validation songs: {sorted(val_songs)}")

    return [
        (dry_path, wet_path)
        for dry_path, wet_path in all_pairs
        if _song_from_dry_path(dry_path) in val_songs
    ]


def _weighted_average_metric_rows(chunk_rows: list[dict], metric_cols: list[str]) -> dict:
    total_frames = sum(row["Frames"] for row in chunk_rows)
    return {
        col: sum(row[col] * row["Frames"] for row in chunk_rows) / total_frames
        for col in metric_cols
    }

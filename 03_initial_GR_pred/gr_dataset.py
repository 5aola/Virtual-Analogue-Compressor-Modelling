"""
Gain Reduction (GR) dataset utilities and pre-computation export.

Computes GR envelopes (dB) from dry/wet audio pairs using 1024-sample
windowed RMS.  Supports three datasets:

  - **Diff-SSL-G-Comp** — SSL G-Bus hardware (44.1 kHz, separate dry/wet WAVs)
  - **CL1B** — TubeTech CL 1B optical compressor (48 kHz, stereo L=dry R=wet)
  - **LA2A** — Teletronix LA-2A hardware (44.1 kHz, separate input/target WAVs)

Export format: one ``.pt`` file per pair containing::

    {"gr_db": Tensor[1, T], "sample_rate": int, "rms_window": int,
     "dry_path": str, "wet_path": str, ...metadata}

GR is stored in **dB** (negative = compression, 0 = unity gain).
"""

import os
import re
import glob
import numpy as np
import torch
import torchaudio
import soundfile as sf
import lightning as pl
from torch.utils.data import Dataset, DataLoader
from typing import Optional

from src.dsp_torch import (
    GR_DB_MIN,
    GR_DB_MAX,
    RMS_WINDOW,
    windowed_rms,
    gain_reduction_db,
    normalize_gr,
    denormalize_gr,
)

DEFAULT_DATA_ROOT = "/Volumes/Saola's Drive/AllCode/thesis/data/Diff-SSL-G-Comp"
DEFAULT_SETTING = "threshold_-12_attack_10_release_0.4_ratio_10"

ALL_DATA_ROOT = "/Volumes/Saola's Drive/AllCode/thesis/data"

SAMPLE_RATE = 44100
SAMPLE_LENGTH = 132300  # 3 s at 44100 Hz


# ---------------------------------------------------------------------------
# Dataset (legacy — computes GR on-the-fly)
# ---------------------------------------------------------------------------


class GainReductionDataset(Dataset):
    """PyTorch dataset that yields ``(dry_audio, gr_target)`` pairs.

    ``gr_target`` is the normalised gain-reduction envelope computed on the
    fly from the dry/wet audio pair using a 1024-sample RMS window.

    Only files for **one** compressor-settings folder are loaded.
    """

    def __init__(
        self,
        data_root: str = DEFAULT_DATA_ROOT,
        settings_folder: str = DEFAULT_SETTING,
        sample_length: int = SAMPLE_LENGTH,
        sample_rate: int = SAMPLE_RATE,
        rms_window: int = RMS_WINDOW,
    ):
        self.sample_length = sample_length
        self.sample_rate = sample_rate
        self.rms_window = rms_window

        dry_dir = os.path.join(data_root, "processed_normalized")
        wet_dir = os.path.join(data_root, "processed_ground_truth", settings_folder)

        if not os.path.isdir(dry_dir):
            raise FileNotFoundError(f"Dry directory not found: {dry_dir}")
        if not os.path.isdir(wet_dir):
            raise FileNotFoundError(f"Wet directory not found: {wet_dir}")

        dry_lookup: dict[str, str] = {}
        for p in sorted(glob.glob(os.path.join(dry_dir, "*_UnmasteredWAV.wav"))):
            song = os.path.basename(p).replace("_UnmasteredWAV.wav", "")
            dry_lookup[song] = p

        pairs: list[tuple[str, str]] = []
        for wet_path in sorted(glob.glob(os.path.join(wet_dir, "*-exported.wav"))):
            song = os.path.basename(wet_path).replace("-exported.wav", "")
            if song in dry_lookup:
                pairs.append((dry_lookup[song], wet_path))

        if not pairs:
            raise ValueError(
                f"No matching dry/wet pairs for setting '{settings_folder}'"
            )

        self.samples: list[dict] = []
        for dry_path, wet_path in pairs:
            md = sf.info(dry_path)
            n_frames = md.frames
            if sample_length == -1:
                self.samples.append(
                    {"dry": dry_path, "wet": wet_path, "offset": 0, "frames": n_frames}
                )
            else:
                for n in range(n_frames // sample_length):
                    self.samples.append(
                        {
                            "dry": dry_path,
                            "wet": wet_path,
                            "offset": n * sample_length,
                            "frames": sample_length,
                        }
                    )

        print(
            f"GainReductionDataset: {len(pairs)} songs, "
            f"{len(self.samples)} chunks  "
            f"[setting={settings_folder}]"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        nf = s["frames"] if self.sample_length != -1 else -1

        dry, sr = sf.read(
            s["dry"],
            start=s["offset"],
            stop=None if nf == -1 else s["offset"] + nf,
            dtype="float32",
            always_2d=True,
        )
        dry = torch.from_numpy(dry.T)

        wet, sr_w = sf.read(
            s["wet"],
            start=s["offset"],
            stop=None if nf == -1 else s["offset"] + nf,
            dtype="float32",
            always_2d=True,
        )
        wet = torch.from_numpy(wet.T)

        if sr != self.sample_rate:
            dry = torchaudio.functional.resample(dry, sr, self.sample_rate)
        if sr_w != self.sample_rate:
            wet = torchaudio.functional.resample(wet, sr_w, self.sample_rate)

        if dry.shape[0] > 1:
            dry = dry.mean(dim=0, keepdim=True)
        if wet.shape[0] > 1:
            wet = wet.mean(dim=0, keepdim=True)

        min_len = min(dry.shape[-1], wet.shape[-1])
        dry = dry[..., :min_len]
        wet = wet[..., :min_len]

        gr = gain_reduction_db(dry, wet, self.rms_window)
        gr = normalize_gr(gr).clamp(-1.0, 1.0)

        return dry, gr


# ---------------------------------------------------------------------------
# DataModule (legacy)
# ---------------------------------------------------------------------------


class GainReductionDataModule(pl.LightningDataModule):
    """Lightning DataModule wrapping :class:`GainReductionDataset`."""

    def __init__(
        self,
        data_root: str = DEFAULT_DATA_ROOT,
        settings_folder: str = DEFAULT_SETTING,
        sample_length: int = SAMPLE_LENGTH,
        sample_rate: int = SAMPLE_RATE,
        rms_window: int = RMS_WINDOW,
        train_split: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_root = data_root
        self.settings_folder = settings_folder
        self.sample_length = sample_length
        self.sample_rate = sample_rate
        self.rms_window = rms_window
        self.train_split = train_split
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        full = GainReductionDataset(
            data_root=self.data_root,
            settings_folder=self.settings_folder,
            sample_length=self.sample_length,
            sample_rate=self.sample_rate,
            rms_window=self.rms_window,
        )
        n_train = int(len(full) * self.train_split)
        n_val = len(full) - n_train
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full, [n_train, n_val]
        )
        print(f"Train: {n_train}  Val: {n_val}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# ===========================================================================
# Pre-computed GR dataset (loads from exported .pt files)
# ===========================================================================


class PrecomputedGRDataset(Dataset):
    """Dataset that loads dry audio + pre-computed GR curves from ``.pt`` files.

    GR curves are cached in memory at construction time (~33 MB per song
    at 44.1 kHz).  Dry audio is read from the original WAV on each access.

    Returns ``(dry [1, T], gr_db [1, T])`` — GR in dB, not normalised.
    """

    def __init__(
        self,
        data_root: str,
        settings_folder: str,
        sample_length: int = SAMPLE_LENGTH,
        sample_stride: int | None = None,
        sample_rate: int = SAMPLE_RATE,
        random_crop: bool = False,
        samples: list[dict] | None = None,
        _gr_cache: dict | None = None,
        cache_dry: bool = False,
        _dry_cache: dict | None = None,
    ):
        self.data_root = data_root
        self.settings_folder = settings_folder
        self.sample_length = sample_length
        self.sample_stride = sample_stride or sample_length
        self.sample_rate = sample_rate
        self.random_crop = random_crop
        self.cache_dry = cache_dry

        if samples is not None:
            self.samples = samples
            self._gr_cache = _gr_cache or {}
            self._dry_cache = _dry_cache or {}
            return

        dry_dir = os.path.join(data_root, "processed_normalized")
        gr_dir = os.path.join(data_root, "gr_curves", settings_folder)

        if not os.path.isdir(gr_dir):
            raise FileNotFoundError(
                f"No pre-computed GR curves at {gr_dir}. "
                f"Run export first: export_dataset(discover_diffssl_pairs(...))"
            )

        dry_lookup: dict[str, str] = {}
        for p in sorted(glob.glob(os.path.join(dry_dir, "*_UnmasteredWAV.wav"))):
            song = os.path.basename(p).replace("_UnmasteredWAV.wav", "")
            dry_lookup[song] = p

        self._gr_cache: dict[str, torch.Tensor] = {}
        for pt_file in sorted(glob.glob(os.path.join(gr_dir, "*.pt"))):
            song = os.path.splitext(os.path.basename(pt_file))[0]
            if song not in dry_lookup:
                continue
            rec = torch.load(pt_file, weights_only=False)
            self._gr_cache[song] = rec["gr_db"]

        self._dry_cache: dict[str, torch.Tensor] = {}
        if self.cache_dry:
            print("Caching dry audio files in RAM...")
            for song in sorted(self._gr_cache.keys()):
                dry_path = dry_lookup[song]
                dry_data, sr = sf.read(dry_path, dtype="float32", always_2d=True)
                dry_tensor = torch.from_numpy(dry_data.T)
                if sr != self.sample_rate:
                    dry_tensor = torchaudio.functional.resample(dry_tensor, sr, self.sample_rate)
                if dry_tensor.shape[0] > 1:
                    dry_tensor = dry_tensor.mean(dim=0, keepdim=True)
                self._dry_cache[song] = dry_tensor

        self.samples: list[dict] = []
        for song in sorted(self._gr_cache.keys()):
            n_frames = int(self._gr_cache[song].shape[-1])
            dry_path = dry_lookup[song]
            if sample_length == -1:
                self.samples.append(
                    {
                        "song": song,
                        "dry": dry_path,
                        "offset": 0,
                        "frames": n_frames,
                        "n_frames": n_frames,
                    }
                )
            else:
                if n_frames < sample_length:
                    continue
                max_start = n_frames - sample_length
                for offset in range(0, max_start + 1, self.sample_stride):
                    self.samples.append(
                        {
                            "song": song,
                            "dry": dry_path,
                            "offset": offset,
                            "frames": sample_length,
                            "n_frames": n_frames,
                        }
                    )
                if self.samples[-1]["offset"] != max_start:
                    self.samples.append(
                        {
                            "song": song,
                            "dry": dry_path,
                            "offset": max_start,
                            "frames": sample_length,
                            "n_frames": n_frames,
                        }
                    )

        cache_mb = sum(t.numel() * 4 for t in self._gr_cache.values()) / 1024 / 1024
        dry_cache_mb = sum(t.numel() * 4 for t in self._dry_cache.values()) / 1024 / 1024 if self.cache_dry else 0.0
        print(
            f"PrecomputedGRDataset: {len(self._gr_cache)} songs, "
            f"{len(self.samples)} crops  "
            f"[setting={settings_folder}, length={sample_length}, "
            f"stride={self.sample_stride}, GR cache={cache_mb:.0f} MB, dry cache={dry_cache_mb:.0f} MB]"
        )

    def with_samples(
        self, samples: list[dict], random_crop: bool
    ) -> "PrecomputedGRDataset":
        return PrecomputedGRDataset(
            data_root=self.data_root,
            settings_folder=self.settings_folder,
            sample_length=self.sample_length,
            sample_stride=self.sample_stride,
            sample_rate=self.sample_rate,
            random_crop=random_crop,
            samples=samples,
            _gr_cache=self._gr_cache,
            cache_dry=self.cache_dry,
            _dry_cache=self._dry_cache,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _crop_offset(self, sample: dict) -> int:
        if self.sample_length == -1 or not self.random_crop:
            return sample["offset"]
        max_start = max(0, sample["n_frames"] - sample["frames"])
        if max_start == 0:
            return 0
        return int(torch.randint(0, max_start + 1, ()).item())

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        offset = self._crop_offset(s)
        nf = s["frames"] if self.sample_length != -1 else s["n_frames"]

        if self.cache_dry and s["song"] in self._dry_cache:
            dry = self._dry_cache[s["song"]][..., offset : offset + nf]
        else:
            dry, sr = sf.read(
                s["dry"],
                start=offset,
                stop=offset + nf,
                dtype="float32",
                always_2d=True,
            )
            dry = torch.from_numpy(dry.T)
            if sr != self.sample_rate:
                dry = torchaudio.functional.resample(dry, sr, self.sample_rate)
            if dry.shape[0] > 1:
                dry = dry.mean(dim=0, keepdim=True)

        gr_db = self._gr_cache[s["song"]][..., offset : offset + nf]

        min_len = min(dry.shape[-1], gr_db.shape[-1])
        if self.cache_dry:
            return dry[..., :min_len].clone(), gr_db[..., :min_len].clone()
        return dry[..., :min_len], gr_db[..., :min_len]


class PrecomputedGRDataModule(pl.LightningDataModule):
    """DataModule for pre-computed GR curves with song-level splitting."""

    def __init__(
        self,
        data_root: str,
        settings_folder: str,
        sample_length: int = SAMPLE_LENGTH,
        sample_stride: int | None = None,
        sample_rate: int = SAMPLE_RATE,
        train_split: float = 0.8,
        batch_size: int = 8,
        num_workers: int = 2,
        cache_dry: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_root = data_root
        self.settings_folder = settings_folder
        self.sample_length = sample_length
        self.sample_stride = sample_stride or sample_length
        self.sample_rate = sample_rate
        self.train_split = train_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dry = cache_dry

    def setup(self, stage: Optional[str] = None) -> None:
        full = PrecomputedGRDataset(
            data_root=self.data_root,
            settings_folder=self.settings_folder,
            sample_length=self.sample_length,
            sample_stride=self.sample_stride,
            sample_rate=self.sample_rate,
            random_crop=False,
            cache_dry=self.cache_dry,
        )

        songs = sorted({s["song"] for s in full.samples})
        if len(songs) < 2:
            raise ValueError("Need >= 2 songs for train/val split.")
        generator = torch.Generator().manual_seed(42)
        perm = torch.randperm(len(songs), generator=generator).tolist()
        n_train = int(len(songs) * self.train_split)
        n_train = min(max(1, n_train), len(songs) - 1)
        train_songs = {songs[i] for i in perm[:n_train]}
        val_songs = set(songs) - train_songs

        self.train_dataset = full.with_samples(
            [s for s in full.samples if s["song"] in train_songs],
            random_crop=True,
        )
        self.val_dataset = full.with_samples(
            [s for s in full.samples if s["song"] in val_songs],
            random_crop=False,
        )
        print(
            f"Train: {len(self.train_dataset)} crops from "
            f"{len(train_songs)} songs {sorted(train_songs)}"
        )
        print(
            f"Val:   {len(self.val_dataset)} crops from "
            f"{len(val_songs)} songs {sorted(val_songs)}"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )


# ===========================================================================
# GR curve export — pre-compute and save as .pt files
# ===========================================================================


def _windowed_rms_numpy(x: np.ndarray, window_size: int) -> np.ndarray:
    """Causal sliding-window RMS using cumsum — O(n) time and memory.

    Matches the zero-left-padding convention of ``src.dsp_torch.windowed_rms``
    so outputs are identical sample-for-sample.
    """
    x2 = x.astype(np.float64) ** 2
    padded = np.concatenate([np.zeros(window_size - 1, dtype=np.float64), x2])
    cs = np.empty(len(padded) + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(padded, out=cs[1:])
    mean_sq = (cs[window_size:] - cs[:-window_size]) / window_size
    return np.sqrt(np.maximum(mean_sq, 1e-10)).astype(np.float32)


EXPORT_CHUNK_SAMPLES = 4_000_000


def compute_gr_for_pair(
    dry_path: str,
    wet_path: str | None,
    rms_window: int = RMS_WINDOW,
    stereo_lr: bool = False,
) -> tuple[torch.Tensor, int]:
    """Compute gain-reduction in dB for one dry/wet pair.

    Uses numpy cumsum-based RMS (not PyTorch conv1d) and reads from
    disk in chunks, so peak memory stays well under 1 GB even for
    20-minute files.

    Args:
        dry_path: Path to the dry audio WAV file.
        wet_path: Path to the wet audio WAV file.  Ignored when
            ``stereo_lr=True`` (CL1B format where one file has L=dry, R=wet).
        rms_window: RMS analysis window in samples.
        stereo_lr: If ``True``, treat ``dry_path`` as a stereo file with
            left channel = dry, right channel = wet.

    Returns:
        ``(gr_db [1, T], sample_rate)`` — gain reduction in dB.
    """
    info_d = sf.info(dry_path)
    sr = info_d.samplerate

    if not stereo_lr:
        info_w = sf.info(wet_path)
        sr_w = info_w.samplerate
        if sr != sr_w:
            target_sr = min(sr, sr_w)
            dry_np, _ = sf.read(dry_path, dtype="float32", always_2d=True)
            wet_np, _ = sf.read(wet_path, dtype="float32", always_2d=True)
            dry_mono = dry_np.mean(axis=1) if dry_np.shape[1] > 1 else dry_np[:, 0]
            wet_mono = wet_np.mean(axis=1) if wet_np.shape[1] > 1 else wet_np[:, 0]
            del dry_np, wet_np
            if sr != target_sr:
                dry_mono = torchaudio.functional.resample(
                    torch.from_numpy(dry_mono), sr, target_sr
                ).numpy()
            if sr_w != target_sr:
                wet_mono = torchaudio.functional.resample(
                    torch.from_numpy(wet_mono), sr_w, target_sr
                ).numpy()
            min_len = min(len(dry_mono), len(wet_mono))
            dry_rms = _windowed_rms_numpy(dry_mono[:min_len], rms_window)
            wet_rms = _windowed_rms_numpy(wet_mono[:min_len], rms_window)
            gr_full = (20.0 * np.log10(wet_rms) - 20.0 * np.log10(dry_rms)).astype(np.float32)
            return torch.from_numpy(gr_full).unsqueeze(0), target_sr
        total_frames = min(info_d.frames, info_w.frames)
    else:
        total_frames = info_d.frames

    overlap = rms_window - 1
    gr_parts: list[np.ndarray] = []
    pos = 0

    while pos < total_frames:
        read_start = max(0, pos - overlap)
        read_end = min(total_frames, pos + EXPORT_CHUNK_SAMPLES)

        if stereo_lr:
            chunk, _ = sf.read(
                dry_path,
                start=read_start,
                stop=read_end,
                dtype="float32",
                always_2d=True,
            )
            dry_mono = chunk[:, 0] if chunk.shape[1] == 1 else chunk[:, 0]
            wet_mono = chunk[:, 1] if chunk.shape[1] >= 2 else chunk[:, 0]
            del chunk
        else:
            dry_np, _ = sf.read(
                dry_path,
                start=read_start,
                stop=read_end,
                dtype="float32",
                always_2d=True,
            )
            wet_np, _ = sf.read(
                wet_path,
                start=read_start,
                stop=read_end,
                dtype="float32",
                always_2d=True,
            )
            dry_mono = dry_np.mean(axis=1) if dry_np.shape[1] > 1 else dry_np[:, 0]
            wet_mono = wet_np.mean(axis=1) if wet_np.shape[1] > 1 else wet_np[:, 0]
            del dry_np, wet_np

        dry_rms = _windowed_rms_numpy(dry_mono, rms_window)
        wet_rms = _windowed_rms_numpy(wet_mono, rms_window)
        del dry_mono, wet_mono

        gr_db = 20.0 * np.log10(wet_rms) - 20.0 * np.log10(dry_rms)
        del dry_rms, wet_rms

        keep_from = pos - read_start
        gr_parts.append(gr_db[keep_from:])
        del gr_db

        pos = read_end

    gr_full = np.concatenate(gr_parts).astype(np.float32)
    return torch.from_numpy(gr_full).unsqueeze(0), sr  # [1, T]


def export_pair(
    dry_path: str,
    wet_path: str | None,
    output_path: str,
    rms_window: int = RMS_WINDOW,
    stereo_lr: bool = False,
    metadata: dict | None = None,
) -> str:
    """Compute GR for one pair and save as ``.pt`` file.

    The saved dict always contains ``gr_db``, ``sample_rate``,
    ``rms_window``, ``dry_path``, ``wet_path``, plus any extra
    fields from *metadata*.
    """
    gr_db, sr = compute_gr_for_pair(
        dry_path,
        wet_path,
        rms_window=rms_window,
        stereo_lr=stereo_lr,
    )

    record = {
        "gr_db": gr_db,
        "sample_rate": sr,
        "rms_window": rms_window,
        "dry_path": dry_path,
        "wet_path": wet_path if wet_path else dry_path,
    }
    if metadata:
        record.update(metadata)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(record, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Dataset-specific pair discovery
# ---------------------------------------------------------------------------


def discover_diffssl_pairs(
    data_root: str | None = None,
    setting: str | None = None,
) -> list[dict]:
    """Find all dry/wet pairs for Diff-SSL-G-Comp.

    If *setting* is ``None``, discovers pairs for **all** settings.
    """
    root = os.path.join(data_root or ALL_DATA_ROOT, "Diff-SSL-G-Comp")
    dry_dir = os.path.join(root, "processed_normalized")
    gt_dir = os.path.join(root, "processed_ground_truth")

    dry_lookup: dict[str, str] = {}
    for p in sorted(glob.glob(os.path.join(dry_dir, "*_UnmasteredWAV.wav"))):
        song = os.path.basename(p).replace("_UnmasteredWAV.wav", "")
        dry_lookup[song] = p

    if setting:
        settings = [setting]
    else:
        settings = sorted(
            s
            for s in os.listdir(gt_dir)
            if os.path.isdir(os.path.join(gt_dir, s)) and not s.startswith(".")
        )

    pairs: list[dict] = []
    for s in settings:
        wet_dir = os.path.join(gt_dir, s)
        for wet_path in sorted(glob.glob(os.path.join(wet_dir, "*-exported.wav"))):
            song = os.path.basename(wet_path).replace("-exported.wav", "")
            if song in dry_lookup:
                pairs.append(
                    {
                        "dry_path": dry_lookup[song],
                        "wet_path": wet_path,
                        "song": song,
                        "setting": s,
                        "dataset": "diffssl",
                        "sample_rate": 44100,
                        "stereo_lr": False,
                    }
                )
    return pairs


def discover_ableton_pairs(
    data_root: str | None = None,
    setting: str = "ableton_compressed",
) -> list[dict]:
    """Find all dry/wet pairs for Ableton-processed (ideal compressor) dataset.

    Wet files follow ``Bounce N-SongName_UnmasteredWAV [date]-1.wav``.
    Dry files are in ``processed_normalized/SongName_UnmasteredWAV.wav``.
    """
    root = os.path.join(data_root or ALL_DATA_ROOT, "Diff-SSL-G-Comp")
    dry_dir = os.path.join(root, "processed_normalized")
    wet_dir = os.path.join(root, setting)

    if not os.path.isdir(wet_dir):
        raise FileNotFoundError(f"Ableton wet directory not found: {wet_dir}")

    dry_lookup: dict[str, str] = {}
    for p in sorted(glob.glob(os.path.join(dry_dir, "*_UnmasteredWAV.wav"))):
        song = os.path.basename(p).replace("_UnmasteredWAV.wav", "")
        dry_lookup[song] = p

    pairs: list[dict] = []
    for wet_path in sorted(glob.glob(os.path.join(wet_dir, "*.wav"))):
        bn = os.path.basename(wet_path)
        m = re.match(r"Bounce \d+-(.+?)_UnmasteredWAV \[.*\]-\d+\.wav", bn)
        if not m:
            continue
        song = m.group(1)
        if song in dry_lookup:
            pairs.append(
                {
                    "dry_path": dry_lookup[song],
                    "wet_path": wet_path,
                    "song": song,
                    "setting": setting,
                    "dataset": "ableton",
                    "sample_rate": 44100,
                    "stereo_lr": False,
                }
            )
    return pairs


def discover_cl1b_pairs(
    data_root: str | None = None,
    folder: int | None = None,
) -> list[dict]:
    """Find all stereo (L=dry, R=wet) pairs for CL1B.

    *folder* selects a single release-value subfolder (0–4).  ``None``
    discovers all five folders.
    """
    root = os.path.join(data_root or ALL_DATA_ROOT, "CL1B")
    folders = [str(folder)] if folder is not None else [str(i) for i in range(5)]

    pairs: list[dict] = []
    for fld in folders:
        fld_path = os.path.join(root, fld)
        if not os.path.isdir(fld_path):
            continue
        for wav in sorted(glob.glob(os.path.join(fld_path, "*.wav"))):
            bn = os.path.basename(wav)
            m = re.match(r"TubeTech_a_(\d+)_r_(\d+)_r_(\d+)_t_(\d+)_g_(\d+)\.wav", bn)
            if not m:
                continue
            pairs.append(
                {
                    "dry_path": wav,
                    "wet_path": None,
                    "stereo_lr": True,
                    "folder": int(fld),
                    "attack": int(m.group(1)),
                    "release": int(m.group(2)),
                    "ratio": int(m.group(3)),
                    "threshold": int(m.group(4)),
                    "gain": int(m.group(5)),
                    "dataset": "cl1b",
                    "sample_rate": 48000,
                }
            )
    return pairs


def discover_la2a_pairs(
    data_root: str | None = None,
) -> list[dict]:
    """Find all input/target pairs for LA2A.

    Handles both ``LA2A_3c`` and ``LA2A_2c`` target naming variants.
    """
    root = os.path.join(data_root or ALL_DATA_ROOT, "LA2A", "all")

    inputs: dict[str, str] = {}
    for f in sorted(os.listdir(root)):
        m = re.match(r"input_(\d+)_\.wav", f)
        if m:
            inputs[m.group(1)] = os.path.join(root, f)

    pairs: list[dict] = []
    for f in sorted(os.listdir(root)):
        m = re.match(r"target_(\d+)_LA2A_(\w+)__(\d+)__(\d+)\.wav", f)
        if m and m.group(1) in inputs:
            pairs.append(
                {
                    "dry_path": inputs[m.group(1)],
                    "wet_path": os.path.join(root, f),
                    "pair_id": int(m.group(1)),
                    "channel_config": m.group(2),
                    "comp_limit": int(m.group(3)),
                    "peak_reduction": int(m.group(4)),
                    "dataset": "la2a",
                    "sample_rate": 44100,
                    "stereo_lr": False,
                }
            )
    return pairs


# ---------------------------------------------------------------------------
# Batch export
# ---------------------------------------------------------------------------


def _output_rel_path(pair: dict) -> str:
    """Determine the relative .pt path for a discovered pair."""
    dataset = pair["dataset"]
    if dataset == "diffssl":
        return os.path.join(
            "Diff-SSL-G-Comp", "gr_curves", pair["setting"], f"{pair['song']}.pt"
        )
    elif dataset == "cl1b":
        stem = os.path.splitext(os.path.basename(pair["dry_path"]))[0]
        return os.path.join("CL1B", "gr_curves", str(pair["folder"]), f"{stem}.pt")
    elif dataset == "la2a":
        stem = (
            f"pair_{pair['pair_id']}_cl{pair['comp_limit']}_pr{pair['peak_reduction']}"
        )
        return os.path.join("LA2A", "gr_curves", f"{stem}.pt")
    elif dataset == "ableton":
        return os.path.join(
            "Diff-SSL-G-Comp", "gr_curves", pair["setting"], f"{pair['song']}.pt"
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def export_dataset(
    pairs: list[dict],
    output_base: str,
    rms_window: int = RMS_WINDOW,
    skip_existing: bool = True,
) -> list[str]:
    """Export GR curves for a list of discovered pairs.

    Args:
        pairs: List of pair dicts from ``discover_*_pairs()``.
        output_base: Root directory under which ``.pt`` files are saved
            (typically the same as the data root).
        rms_window: RMS window size in samples.
        skip_existing: If ``True``, skip pairs whose output file already exists.

    Returns:
        List of output file paths.
    """
    saved: list[str] = []
    for i, pair in enumerate(pairs):
        rel = _output_rel_path(pair)
        out_path = os.path.join(output_base, rel)

        if skip_existing and os.path.isfile(out_path):
            saved.append(out_path)
            continue

        metadata = {
            k: v
            for k, v in pair.items()
            if k not in ("dry_path", "wet_path", "stereo_lr")
        }

        print(f"  [{i + 1}/{len(pairs)}] {rel}")
        export_pair(
            dry_path=pair["dry_path"],
            wet_path=pair.get("wet_path"),
            output_path=out_path,
            rms_window=rms_window,
            stereo_lr=pair.get("stereo_lr", False),
            metadata=metadata,
        )
        saved.append(out_path)

    return saved


def export_all(
    data_root: str | None = None,
    output_base: str | None = None,
    rms_window: int = RMS_WINDOW,
    skip_existing: bool = True,
) -> dict[str, list[str]]:
    """Export GR curves for Diff-SSL, CL1B, and LA2A datasets.

    Returns a dict mapping dataset name to the list of saved paths.
    """
    data_root = data_root or ALL_DATA_ROOT
    output_base = output_base or data_root

    results: dict[str, list[str]] = {}

    print("=== Diff-SSL-G-Comp ===")
    diffssl = discover_diffssl_pairs(data_root)
    print(f"Found {len(diffssl)} pairs")
    results["diffssl"] = export_dataset(diffssl, output_base, rms_window, skip_existing)

    print("\n=== CL1B ===")
    cl1b = discover_cl1b_pairs(data_root)
    print(f"Found {len(cl1b)} pairs")
    results["cl1b"] = export_dataset(cl1b, output_base, rms_window, skip_existing)

    print("\n=== LA2A ===")
    la2a = discover_la2a_pairs(data_root)
    print(f"Found {len(la2a)} pairs")
    results["la2a"] = export_dataset(la2a, output_base, rms_window, skip_existing)

    print("\n=== Ableton (ideal compressor) ===")
    ableton = discover_ableton_pairs(data_root)
    print(f"Found {len(ableton)} pairs")
    results["ableton"] = export_dataset(ableton, output_base, rms_window, skip_existing)

    return results


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_gr_export(pt_path: str) -> dict:
    """Load an exported ``.pt`` GR record."""
    return torch.load(pt_path, weights_only=False)


# ---------------------------------------------------------------------------
# Visualization / test
# ---------------------------------------------------------------------------


def plot_gr_curve(pt_path: str, ax=None, show: bool = True):
    """Load and plot a single exported GR curve.

    Args:
        pt_path: Path to a ``.pt`` file saved by :func:`export_pair`.
        ax: Optional matplotlib ``Axes``.  Created if ``None``.
        show: Call ``plt.show()`` when done.

    Returns:
        The ``Axes`` object.
    """
    import matplotlib.pyplot as plt

    record = load_gr_export(pt_path)
    gr_db = record["gr_db"].squeeze().numpy()
    sr = record["sample_rate"]
    t = np.arange(len(gr_db)) / sr

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 3))

    ax.plot(t, gr_db, linewidth=0.5, color="tab:blue")
    ax.set_ylabel("GR (dB)")
    ax.set_xlabel("Time (s)")

    title_parts = []
    if "dataset" in record:
        title_parts.append(record["dataset"].upper())
    if "song" in record:
        title_parts.append(record["song"])
    if "setting" in record:
        title_parts.append(record["setting"])
    if "pair_id" in record:
        title_parts.append(f"pair {record['pair_id']}")
    ax.set_title(" — ".join(title_parts) if title_parts else os.path.basename(pt_path))

    ax.set_ylim(min(float(gr_db.min()) - 2, -32), 2)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_gr_grid(
    pt_paths: list[str],
    cols: int = 1,
    figsize: tuple[int, int] | None = None,
    show: bool = True,
):
    """Plot a grid of exported GR curves for visual comparison.

    Args:
        pt_paths: List of ``.pt`` file paths.
        cols: Number of columns in the grid.
        figsize: Figure size; auto-computed if ``None``.
        show: Call ``plt.show()`` when done.

    Returns:
        ``(fig, axes)`` tuple.
    """
    import math
    import matplotlib.pyplot as plt

    n = len(pt_paths)
    rows = math.ceil(n / cols)
    if figsize is None:
        figsize = (14 * cols, 3 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    for i, pt_path in enumerate(pt_paths):
        ax = axes[i // cols][i % cols]
        plot_gr_curve(pt_path, ax=ax, show=False)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def verify_export(
    pt_path: str,
    rms_window: int = RMS_WINDOW,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    verbose: bool = True,
    plot: bool = False,
    plot_seconds: float | None = 10.0,
) -> dict:
    """Compare an exported .pt GR curve against a freshly computed one.

    Loads the exported record, re-computes GR from the original dry/wet
    paths stored in it, and reports the numerical difference.

    Args:
        pt_path: Path to an exported ``.pt`` file.
        rms_window: RMS window (should match what was used for export).
        atol: Absolute tolerance for the allclose check.
        rtol: Relative tolerance for the allclose check.
        verbose: Print summary to stdout.
        plot: If ``True``, show a 3-panel plot (exported, recomputed, error).
        plot_seconds: Limit the plot to the first N seconds for readability.
            ``None`` plots the full signal.

    Returns:
        Dict with keys: ``match`` (bool), ``max_abs_err``, ``mean_abs_err``,
        ``exported_shape``, ``recomputed_shape``, ``pt_path``.
    """
    rec = load_gr_export(pt_path)
    gr_exported = rec["gr_db"]
    sr = rec["sample_rate"]

    gr_recomputed, _ = compute_gr_for_pair(
        dry_path=rec["dry_path"],
        wet_path=rec["wet_path"],
        rms_window=rec.get("rms_window", rms_window),
        stereo_lr=rec.get("stereo_lr", False),
    )

    min_len = min(gr_exported.shape[-1], gr_recomputed.shape[-1])
    gr_exported = gr_exported[..., :min_len]
    gr_recomputed = gr_recomputed[..., :min_len]

    diff = (gr_exported - gr_recomputed).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    match = torch.allclose(gr_exported, gr_recomputed, atol=atol, rtol=rtol)

    result = {
        "match": match,
        "max_abs_err": max_err,
        "mean_abs_err": mean_err,
        "exported_shape": list(gr_exported.shape),
        "recomputed_shape": list(gr_recomputed.shape),
        "pt_path": pt_path,
    }

    if verbose:
        status = "PASS" if match else "FAIL"
        print(
            f"[{status}] {os.path.basename(pt_path)}  "
            f"max_err={max_err:.2e}  mean_err={mean_err:.2e}  "
            f"shape={list(gr_exported.shape)}"
        )

    if plot:
        import matplotlib.pyplot as plt

        exp_np = gr_exported.squeeze().numpy()
        rec_np = gr_recomputed.squeeze().numpy()
        err_np = diff.squeeze().numpy()
        t = np.arange(len(exp_np)) / sr

        if plot_seconds is not None:
            n_samples = min(len(exp_np), int(plot_seconds * sr))
            exp_np = exp_np[:n_samples]
            rec_np = rec_np[:n_samples]
            err_np = err_np[:n_samples]
            t = t[:n_samples]

        fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)

        axes[0].plot(t, exp_np, linewidth=0.5, color="tab:blue", label="Exported")
        axes[0].plot(t, rec_np, linewidth=0.5, color="tab:orange", alpha=0.7, label="Recomputed")
        axes[0].set_ylabel("GR (dB)")
        axes[0].legend(loc="lower left")
        axes[0].set_title(
            f"{os.path.basename(pt_path)}  —  "
            f"{'PASS' if match else 'FAIL'}  "
            f"(max_err={max_err:.2e})"
        )

        axes[1].plot(t, exp_np - rec_np, linewidth=0.5, color="tab:red")
        axes[1].set_ylabel("Error (dB)")
        axes[1].axhline(0, color="gray", linewidth=0.5, linestyle="--")

        axes[2].semilogy(t, err_np + 1e-12, linewidth=0.5, color="tab:purple")
        axes[2].set_ylabel("|Error| (dB, log)")
        axes[2].set_xlabel("Time (s)")

        fig.tight_layout()
        plt.show()

    return result


def verify_exports(
    pt_paths: list[str],
    rms_window: int = RMS_WINDOW,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> list[dict]:
    """Batch-verify multiple exported .pt files.

    Returns list of result dicts (same as :func:`verify_export`).
    Prints a summary at the end.
    """
    results = []
    for p in pt_paths:
        results.append(verify_export(p, rms_window=rms_window, atol=atol, rtol=rtol))

    n_pass = sum(r["match"] for r in results)
    n_fail = len(results) - n_pass
    print(f"\n--- Summary: {n_pass}/{len(results)} passed, {n_fail} failed ---")
    if n_fail > 0:
        for r in results:
            if not r["match"]:
                print(f"  FAIL: {r['pt_path']}  max_err={r['max_abs_err']:.2e}")
    return results


def plot_alignment(
    pt_path: str,
    plot_seconds: float | None = 5.0,
    offset_seconds: float = 0.0,
    show: bool = True,
):
    """Plot dry waveform, wet waveform, and exported GR curve aligned in time.

    Use this to visually check for time shifts or misalignment between
    the GR envelope and the source audio signals.

    Args:
        pt_path: Path to an exported ``.pt`` file.
        plot_seconds: Duration to display. ``None`` for full signal.
        offset_seconds: Start time offset into the file (useful for
            inspecting transients mid-song).
        show: Call ``plt.show()`` when done.

    Returns:
        ``(fig, axes)`` tuple.
    """
    import matplotlib.pyplot as plt

    rec = load_gr_export(pt_path)
    gr_db = rec["gr_db"].squeeze().numpy()
    sr = rec["sample_rate"]
    stereo_lr = rec.get("stereo_lr", False)

    start_sample = int(offset_seconds * sr)
    if plot_seconds is not None:
        end_sample = start_sample + int(plot_seconds * sr)
    else:
        end_sample = len(gr_db)
    end_sample = min(end_sample, len(gr_db))

    if stereo_lr:
        chunk, _ = sf.read(
            rec["dry_path"],
            start=start_sample,
            stop=end_sample,
            dtype="float32",
            always_2d=True,
        )
        dry_np = chunk[:, 0]
        wet_np = chunk[:, 1] if chunk.shape[1] >= 2 else chunk[:, 0]
    else:
        dry_np, _ = sf.read(
            rec["dry_path"],
            start=start_sample,
            stop=end_sample,
            dtype="float32",
            always_2d=True,
        )
        dry_np = dry_np.mean(axis=1) if dry_np.shape[1] > 1 else dry_np[:, 0]

        wet_np, _ = sf.read(
            rec["wet_path"],
            start=start_sample,
            stop=end_sample,
            dtype="float32",
            always_2d=True,
        )
        wet_np = wet_np.mean(axis=1) if wet_np.shape[1] > 1 else wet_np[:, 0]

    gr_slice = gr_db[start_sample:end_sample]
    n = min(len(dry_np), len(wet_np), len(gr_slice))
    dry_np = dry_np[:n]
    wet_np = wet_np[:n]
    gr_slice = gr_slice[:n]

    t = (np.arange(n) + start_sample) / sr

    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(t, dry_np, linewidth=0.3, color="tab:blue")
    axes[0].set_ylabel("Dry")
    axes[0].set_ylim(-1, 1)

    axes[1].plot(t, wet_np, linewidth=0.3, color="tab:green")
    axes[1].set_ylabel("Wet")
    axes[1].set_ylim(-1, 1)

    axes[2].plot(t, dry_np, linewidth=0.3, color="tab:blue", alpha=0.5, label="Dry")
    axes[2].plot(t, wet_np, linewidth=0.3, color="tab:green", alpha=0.5, label="Wet")
    axes[2].set_ylabel("Overlay")
    axes[2].set_ylim(-1, 1)
    axes[2].legend(loc="upper right", fontsize=8)

    axes[3].plot(t, gr_slice, linewidth=0.5, color="tab:red")
    axes[3].set_ylabel("GR (dB)")
    axes[3].set_xlabel("Time (s)")
    axes[3].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[3].set_ylim(min(float(gr_slice.min()) - 2, -32), 2)

    title_parts = [os.path.basename(pt_path)]
    if "dataset" in rec:
        title_parts.insert(0, rec["dataset"].upper())
    if "setting" in rec:
        title_parts.append(rec["setting"][:40])
    fig.suptitle(" — ".join(title_parts), fontsize=10)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def test_export_one_each(
    data_root: str | None = None,
    output_base: str | None = None,
    rms_window: int = RMS_WINDOW,
    show: bool = True,
) -> list[str]:
    """Export and plot one pair from each dataset as a smoke test.

    Returns the list of saved ``.pt`` paths.
    """
    data_root = data_root or ALL_DATA_ROOT
    output_base = output_base or data_root

    paths: list[str] = []

    diffssl = discover_diffssl_pairs(data_root)
    if diffssl:
        p = export_dataset(diffssl[:1], output_base, rms_window, skip_existing=False)
        paths.extend(p)
        print(f"Diff-SSL: exported {p[0]}")

    cl1b = discover_cl1b_pairs(data_root)
    if cl1b:
        p = export_dataset(cl1b[:1], output_base, rms_window, skip_existing=False)
        paths.extend(p)
        print(f"CL1B:     exported {p[0]}")

    la2a = discover_la2a_pairs(data_root)
    if la2a:
        p = export_dataset(la2a[:1], output_base, rms_window, skip_existing=False)
        paths.extend(p)
        print(f"LA2A:     exported {p[0]}")

    if paths and show:
        plot_gr_grid(paths)

    return paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export GR curves for all datasets")
    parser.add_argument(
        "--data-root",
        default=ALL_DATA_ROOT,
        help="Root directory containing CL1B/, LA2A/, Diff-SSL-G-Comp/ folders",
    )
    parser.add_argument(
        "--output-base",
        default=None,
        help="Output root (default: same as --data-root)",
    )
    parser.add_argument(
        "--rms-window",
        type=int,
        default=RMS_WINDOW,
        help=f"RMS window size in samples (default: {RMS_WINDOW})",
    )
    parser.add_argument(
        "--dataset",
        choices=["diffssl", "cl1b", "la2a", "ableton", "all"],
        default="all",
        help="Which dataset to export (default: all)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Export and plot one pair from each dataset (smoke test)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-export even if the .pt file already exists",
    )
    args = parser.parse_args()

    output = args.output_base or args.data_root
    skip = not args.no_skip

    if args.test:
        test_export_one_each(args.data_root, output, args.rms_window)
    elif args.dataset == "all":
        export_all(args.data_root, output, args.rms_window, skip)
    else:
        discover_fn = {
            "diffssl": discover_diffssl_pairs,
            "cl1b": discover_cl1b_pairs,
            "la2a": discover_la2a_pairs,
            "ableton": discover_ableton_pairs,
        }[args.dataset]
        pairs = discover_fn(args.data_root)
        print(f"Found {len(pairs)} pairs")
        export_dataset(pairs, output, args.rms_window, skip)

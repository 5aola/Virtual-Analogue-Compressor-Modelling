"""
Gain Reduction (GR) Dataset for the Diff-SSL-G-Comp compressor dataset.

Loads dry/wet audio pairs for a single compressor setting, computes the
gain reduction envelope (dB) from 1024-sample windowed RMS, and normalises
it to [-1, 1] so it can serve as the target for a TCN with tanh output.

Normalisation:  gr_norm = (gr_db - GR_DB_MIN) / (GR_DB_MAX - GR_DB_MIN) * 2 - 1
                maps [GR_DB_MIN, GR_DB_MAX] → [-1, 1]
"""

import os
import glob

import torch
import torch.nn.functional as F
import torchaudio
import lightning as pl
from torch.utils.data import Dataset, DataLoader
from typing import Optional

DEFAULT_DATA_ROOT = (
    "/Volumes/Saola's Drive/AllCode/thesis/data/Diff-SSL-G-Comp"
)
DEFAULT_SETTING = "threshold_-12_attack_10_release_0.4_ratio_10"

GR_DB_MIN = -30.0
GR_DB_MAX = 0.0
RMS_WINDOW = 1024
SAMPLE_RATE = 44100
SAMPLE_LENGTH = 132300  # 3 s at 44100 Hz


# ---------------------------------------------------------------------------
# GR computation helpers (PyTorch, no numpy dependency)
# ---------------------------------------------------------------------------


def windowed_rms(signal: torch.Tensor, window_size: int) -> torch.Tensor:
    """Sample-rate windowed RMS via 1-D convolution.

    Args:
        signal: ``[1, T]`` mono audio.
        window_size: RMS analysis window in samples.

    Returns:
        ``[1, T]`` RMS envelope (linear amplitude).
    """
    # signal: [1, T] -> [1, 1, T] for conv1d
    sq = (signal.unsqueeze(0)) ** 2
    kernel = torch.ones(1, 1, window_size, device=signal.device) / window_size
    pad = window_size - 1
    rms_sq = F.conv1d(sq, kernel, padding=pad)
    rms_sq = rms_sq[..., :signal.shape[-1]]
    return torch.sqrt(rms_sq.squeeze(0).clamp(min=1e-10))


def gain_reduction_db(
    dry: torch.Tensor, wet: torch.Tensor, window_size: int = RMS_WINDOW
) -> torch.Tensor:
    """GR in dB = RMS_dB(wet) − RMS_dB(dry).  Negative means compression.

    Args:
        dry: ``[1, T]`` dry (input) audio.
        wet: ``[1, T]`` wet (compressed output) audio.
        window_size: RMS window size.

    Returns:
        ``[1, T]`` gain-reduction signal in dB.
    """
    dry_rms = windowed_rms(dry, window_size)
    wet_rms = windowed_rms(wet, window_size)
    dry_db = 20 * torch.log10(dry_rms)
    wet_db = 20 * torch.log10(wet_rms)
    return wet_db - dry_db


def normalize_gr(gr_db: torch.Tensor) -> torch.Tensor:
    """Map ``[GR_DB_MIN, GR_DB_MAX]`` → ``[-1, 1]``."""
    return (gr_db - GR_DB_MIN) / (GR_DB_MAX - GR_DB_MIN) * 2 - 1


def denormalize_gr(gr_norm: torch.Tensor) -> torch.Tensor:
    """Map ``[-1, 1]`` → ``[GR_DB_MIN, GR_DB_MAX]``."""
    return (gr_norm + 1) / 2 * (GR_DB_MAX - GR_DB_MIN) + GR_DB_MIN


# ---------------------------------------------------------------------------
# Dataset
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
        wet_dir = os.path.join(
            data_root, "processed_ground_truth", settings_folder
        )

        if not os.path.isdir(dry_dir):
            raise FileNotFoundError(f"Dry directory not found: {dry_dir}")
        if not os.path.isdir(wet_dir):
            raise FileNotFoundError(f"Wet directory not found: {wet_dir}")

        # song_name → dry path
        dry_lookup: dict[str, str] = {}
        for p in sorted(glob.glob(os.path.join(dry_dir, "*_UnmasteredWAV.wav"))):
            song = os.path.basename(p).replace("_UnmasteredWAV.wav", "")
            dry_lookup[song] = p

        # match wet files to dry files
        pairs: list[tuple[str, str]] = []
        for wet_path in sorted(glob.glob(os.path.join(wet_dir, "*-exported.wav"))):
            song = os.path.basename(wet_path).replace("-exported.wav", "")
            if song in dry_lookup:
                pairs.append((dry_lookup[song], wet_path))

        if not pairs:
            raise ValueError(
                f"No matching dry/wet pairs for setting '{settings_folder}'"
            )

        # chunk each file into fixed-length segments
        self.samples: list[dict] = []
        for dry_path, wet_path in pairs:
            md = torchaudio.info(dry_path)
            n_frames = md.num_frames
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

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        nf = s["frames"] if self.sample_length != -1 else -1

        dry, sr = torchaudio.load(
            s["dry"], frame_offset=s["offset"], num_frames=nf, normalize=True
        )
        wet, sr_w = torchaudio.load(
            s["wet"], frame_offset=s["offset"], num_frames=nf, normalize=True
        )

        # resample if needed
        if sr != self.sample_rate:
            dry = torchaudio.functional.resample(dry, sr, self.sample_rate)
        if sr_w != self.sample_rate:
            wet = torchaudio.functional.resample(wet, sr_w, self.sample_rate)

        # to mono
        if dry.shape[0] > 1:
            dry = dry.mean(dim=0, keepdim=True)
        if wet.shape[0] > 1:
            wet = wet.mean(dim=0, keepdim=True)

        # ensure same length
        min_len = min(dry.shape[-1], wet.shape[-1])
        dry = dry[..., :min_len]
        wet = wet[..., :min_len]

        gr = gain_reduction_db(dry, wet, self.rms_window)
        gr = normalize_gr(gr).clamp(-1.0, 1.0)

        return dry, gr


# ---------------------------------------------------------------------------
# DataModule
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

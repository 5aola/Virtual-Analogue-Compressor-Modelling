"""diffssl-style crop dataloader (mirrors ``02b_sota_training/dataset.py``) that
**also** yields the sample-aligned GR curve, for the GR-TFiLM model.

Kept identical to 02b (ablation contract) except ``batch_size`` (raised for speed):
  - pair inventory: 10 settings × 10 songs (``discover_gr_pairs``)
  - split: ``splits.build_split_manifest`` (seed 42)
  - non-overlapping ``sample_length`` crops, ``batch_size=64`` (diffssl used 16;
    raised for GPU throughput), train shuffle + ``drop_last``; fresh crop per
    ``__getitem__`` (state reset every batch).

One addition vs 02b: each item carries the GR crop so the model can use it as a
time-varying conditioning signal.

    dry [B, 1, S]   raw dry crop
    gr  [B, 1, S]   GR curve in dB, sample-aligned to dry/wet   (conditioning)
    wet [B, 1, S]   compressor output                           (target)
    p   [B, C]      normalised static knobs                     (SOTA conditioning)
"""

from __future__ import annotations

import glob
import os
from typing import Optional

import lightning as pl
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from splits import (
    SplitManifest,
    build_split_manifest,
    filter_pairs_by_keys,
    normalize_setting_params,
)

SAMPLE_RATE = 44100
SAMPLE_LENGTH = 132300  # diffssl LSTM32TVC: 3 s @ 44.1 kHz
BATCH_SIZE = 64  # raised from diffssl's 16: the hidden=32 sample-rate LSTM is
                 # launch/latency-bound and parallelises over the batch dim, so a
                 # larger batch cuts batches/epoch ~4x at near-equal step time.


def discover_gr_pairs(data_root: str) -> list[dict]:
    """All (song, setting) pairs with dry, wet, GR curve, and static params on disk.

    Same inventory as ``02b_sota_training.discover_diffssl_wet_pairs``.
    """
    dry_lookup = {
        os.path.basename(p).replace("_UnmasteredWAV.wav", ""): p
        for p in glob.glob(
            os.path.join(data_root, "processed_normalized", "*_UnmasteredWAV.wav")
        )
    }

    pairs: list[dict] = []
    gr_root = os.path.join(data_root, "gr_curves")
    for setting in sorted(os.listdir(gr_root)):
        if not setting.startswith("threshold_"):
            continue
        wet_dir = os.path.join(data_root, "processed_ground_truth", setting)
        for pt in sorted(glob.glob(os.path.join(gr_root, setting, "*.pt"))):
            song = os.path.splitext(os.path.basename(pt))[0]
            wet = os.path.join(wet_dir, f"{song}-exported.wav")
            if song in dry_lookup and os.path.isfile(wet):
                pairs.append(
                    {
                        "song": song,
                        "setting": setting,
                        "dry": dry_lookup[song],
                        "wet": wet,
                        "gr": pt,
                        "params": normalize_setting_params(setting),
                    }
                )
    return sorted(pairs, key=lambda p: (p["song"], p["setting"]))


class GRCropDataset(Dataset):
    """One item == aligned dry/gr/wet crop + static params (diffssl chunk index).

    GR curves (``.pt``, key ``gr_db``) are cached in RAM once per unique pair and
    sliced per crop — sample-aligned to dry/wet (see ``amplitude_match.py``).
    """

    def __init__(
        self,
        pair_meta: list[dict],
        sample_length: int = SAMPLE_LENGTH,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.sample_length = sample_length
        self.sample_rate = sample_rate
        self.samples: list[dict] = []
        self.gr_cache: dict[str, torch.Tensor] = {}

        for m in pair_meta:
            try:
                dry_frames = sf.info(m["dry"]).frames
                wet_frames = sf.info(m["wet"]).frames
            except (RuntimeError, OSError) as e:
                print(f"Skipping pair {m['song']}::{m['setting']}: {e}")
                continue

            if m["gr"] not in self.gr_cache:
                gr = torch.load(m["gr"], weights_only=False)["gr_db"].float()
                if gr.dim() == 1:
                    gr = gr.unsqueeze(0)  # [1, T]
                self.gr_cache[m["gr"]] = gr
            gr_frames = self.gr_cache[m["gr"]].shape[-1]

            num_frames = min(dry_frames, wet_frames, gr_frames)
            n_chunks = num_frames // sample_length
            if n_chunks == 0:
                continue

            params = torch.tensor(m["params"], dtype=torch.float32)
            for n in range(n_chunks):
                self.samples.append(
                    {
                        "dry": m["dry"],
                        "wet": m["wet"],
                        "gr": m["gr"],
                        "offset": n * sample_length,
                        "params": params,
                    }
                )

        minutes = len(self.samples) * sample_length / sample_rate / 60.0
        print(
            f"GRCropDataset: {len(self.samples)} crops from {len(pair_meta)} pairs  "
            f"[sample_length={sample_length}, {minutes:.1f} min audio]"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_slice(self, path: str, offset: int) -> torch.Tensor:
        audio, sr = sf.read(
            path,
            start=offset,
            stop=offset + self.sample_length,
            dtype="float32",
            always_2d=True,
        )
        x = torch.from_numpy(audio.T)
        if sr != self.sample_rate:
            x = torchaudio.functional.resample(x, sr, self.sample_rate)
            if x.shape[-1] > self.sample_length:
                x = x[..., : self.sample_length]
            elif x.shape[-1] < self.sample_length:
                x = torch.nn.functional.pad(x, (0, self.sample_length - x.shape[-1]))
        if x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)
        return x

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        dry = self._load_slice(s["dry"], s["offset"])
        wet = self._load_slice(s["wet"], s["offset"])
        gr = self.gr_cache[s["gr"]][:, s["offset"] : s["offset"] + self.sample_length]
        return dry, gr.contiguous(), wet, s["params"]


class GRCropDataModule(pl.LightningDataModule):
    """Custom split manifest + diffssl crop/batch recipe, emitting the GR curve."""

    def __init__(
        self,
        data_root: str,
        sample_length: int = SAMPLE_LENGTH,
        sample_rate: int = SAMPLE_RATE,
        batch_size: int = BATCH_SIZE,
        split_seed: int = 42,
        n_train_songs: int | None = None,
        n_val_songs: int = 1,
        n_test_songs: int = 2,
        split_manifest_path: str | None = None,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_root = data_root
        self.sample_length = sample_length
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.split_seed = split_seed
        self.n_train_songs = n_train_songs
        self.n_val_songs = n_val_songs
        self.n_test_songs = n_test_songs
        self.split_manifest_path = split_manifest_path
        self.num_workers = num_workers
        self.split: SplitManifest | None = None
        self._meta: dict[str, list[dict]] = {}
        self.train_dataset: GRCropDataset | None = None
        self.val_dataset: GRCropDataset | None = None
        self.test_dataset: GRCropDataset | None = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.split is None:
            all_pairs = discover_gr_pairs(self.data_root)

            if self.split_manifest_path and os.path.isfile(self.split_manifest_path):
                self.split = SplitManifest.load(self.split_manifest_path)
                print(f"Loaded split manifest: {self.split_manifest_path}")
            else:
                self.split = build_split_manifest(
                    all_pairs,
                    seed=self.split_seed,
                    n_train_songs=self.n_train_songs,
                    n_val_songs=self.n_val_songs,
                    n_test_songs=self.n_test_songs,
                )
                if self.split_manifest_path:
                    self.split.save(self.split_manifest_path)

            self._meta = {
                "train": filter_pairs_by_keys(all_pairs, set(self.split.train_pair_keys)),
                "val": filter_pairs_by_keys(all_pairs, set(self.split.val_pair_keys)),
                "test": filter_pairs_by_keys(all_pairs, set(self.split.test_pair_keys)),
            }
            print(f"Split seed     : {self.split.seed}")
            print(f"Train songs    : {self.split.train_songs}")
            print(f"Val songs      : {self.split.val_songs}")
            print(f"Test songs     : {self.split.test_songs}")
            print(f"Test settings  : {self.split.test_settings} (lowest threshold)")
            print(
                f"Pair counts    : train={len(self._meta['train'])} "
                f"val={len(self._meta['val'])} test={len(self._meta['test'])} "
                f"/ {len(all_pairs)} total"
            )

        if stage in (None, "fit") and self.train_dataset is None:
            self.train_dataset = GRCropDataset(
                self._meta["train"], self.sample_length, self.sample_rate
            )
        if stage in (None, "fit", "validate") and self.val_dataset is None:
            self.val_dataset = GRCropDataset(
                self._meta["val"], self.sample_length, self.sample_rate
            )
        if stage in (None, "test") and self.test_dataset is None:
            self.test_dataset = GRCropDataset(
                self._meta["test"], self.sample_length, self.sample_rate
            )

    def _loader_kwargs(self) -> dict:
        kw = dict(num_workers=self.num_workers, pin_memory=True)
        if self.num_workers > 0:
            kw["persistent_workers"] = True
            kw["prefetch_factor"] = 4
        return kw

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **self._loader_kwargs(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **self._loader_kwargs(),
        )

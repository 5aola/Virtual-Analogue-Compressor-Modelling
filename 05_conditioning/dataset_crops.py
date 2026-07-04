"""3 s crop dataloader for frame-rate GR prediction (dry + GR only, no wet).

Mirrors the `06_output/dataset_tfilm.py` diffssl crop recipe — non-overlapping
`sample_length` crops, batch 16, train shuffle + drop_last, fresh model state
every batch — so the GR predictor trains under the same regime as the
downstream gain-prior model it feeds. Replaces the stateful-TBPTT +
cold-start machinery in `dataset.py`: with a windowed detector frontend and a
DSP ballistics prior the model's memory horizon is ~one release time, so
random crops with a loss warmup mask cover state handling entirely.

Batch: dry [B, 1, S], gr [B, 1, S] (dB, sample-aligned), params [B, 4].
"""

from __future__ import annotations

import os
from typing import Optional

import lightning as pl
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from dataset import discover_diffssl_gr_pairs
from splits import (
    SplitManifest,
    build_split_manifest,
    discover_test_ground_truth_keys,
    filter_pairs_by_keys,
)

SAMPLE_RATE = 44100
SAMPLE_LENGTH = 132300  # 3 s @ 44.1 kHz, same as 02b / 06_output
BATCH_SIZE = 16


class GRPredCropDataset(Dataset):
    """One item == aligned (dry, gr, params) crop. GR curves cached in RAM."""

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
            except (RuntimeError, OSError) as e:
                print(f"Skipping pair {m['song']}::{m['setting']}: {e}")
                continue

            if m["gr"] not in self.gr_cache:
                gr = torch.load(m["gr"], weights_only=False)["gr_db"].float()
                if gr.dim() == 1:
                    gr = gr.unsqueeze(0)  # [1, T]
                self.gr_cache[m["gr"]] = gr

            num_frames = min(dry_frames, self.gr_cache[m["gr"]].shape[-1])
            params = torch.tensor(m["params"], dtype=torch.float32)
            for n in range(num_frames // sample_length):
                self.samples.append(
                    {
                        "dry": m["dry"],
                        "gr": m["gr"],
                        "offset": n * sample_length,
                        "params": params,
                    }
                )

        minutes = len(self.samples) * sample_length / sample_rate / 60.0
        print(
            f"GRPredCropDataset: {len(self.samples)} crops from {len(pair_meta)} pairs  "
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
        gr = self.gr_cache[s["gr"]][:, s["offset"] : s["offset"] + self.sample_length]
        return dry, gr.contiguous(), s["params"]


class GRPredCropDataModule(pl.LightningDataModule):
    """Custom split manifest + diffssl crop/batch recipe for GR prediction.

    Test policy: the external ``test_ground_truth/`` pairs (scanned under
    ``test_gt_root``, default ``data_root``) are the ONLY held-out test set
    and are excluded from train/val by pair key — see
    ``splits.discover_test_ground_truth_keys``. On Colab pass the Drive
    dataset root as ``test_gt_root`` (the local cache holds dry + gr_curves
    only). Loaded manifests are re-validated against the test keys, so a
    stale contaminated manifest cannot be resumed silently.
    """

    def __init__(
        self,
        data_root: str,
        sample_length: int = SAMPLE_LENGTH,
        sample_rate: int = SAMPLE_RATE,
        batch_size: int = BATCH_SIZE,
        split_seed: int = 42,
        n_train_songs: int | None = None,
        n_val_songs: int = 1,
        test_gt_root: str | None = None,
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
        self.test_gt_root = test_gt_root
        self.split_manifest_path = split_manifest_path
        self.num_workers = num_workers
        self.split: SplitManifest | None = None
        self.meta: dict[str, list[dict]] = {}
        self.train_dataset: GRPredCropDataset | None = None
        self.val_dataset: GRPredCropDataset | None = None
        self.test_dataset: GRPredCropDataset | None = None

    def setup(self, stage: Optional[str] = None) -> None:
        # idempotent: trainer.fit / trainer.test re-invoke this — build once
        if self.split is None:
            all_pairs = discover_diffssl_gr_pairs(self.data_root)
            test_keys = discover_test_ground_truth_keys(self.test_gt_root or self.data_root)

            if self.split_manifest_path and os.path.isfile(self.split_manifest_path):
                self.split = SplitManifest.load(self.split_manifest_path)
                print(f"Loaded split manifest: {self.split_manifest_path}")
            else:
                self.split = build_split_manifest(
                    all_pairs,
                    seed=self.split_seed,
                    n_train_songs=self.n_train_songs,
                    n_val_songs=self.n_val_songs,
                    test_pair_keys=test_keys,
                )
                if self.split_manifest_path:
                    self.split.save(self.split_manifest_path)

            leaked = (
                set(self.split.train_pair_keys) | set(self.split.val_pair_keys)
            ) & test_keys
            assert not leaked, (
                f"test_ground_truth pairs leaked into train/val: {sorted(leaked)} — "
                "stale/contaminated split manifest? Delete it and rebuild."
            )

            self.meta = {
                "train": filter_pairs_by_keys(all_pairs, set(self.split.train_pair_keys)),
                "val": filter_pairs_by_keys(all_pairs, set(self.split.val_pair_keys)),
                "test": filter_pairs_by_keys(all_pairs, set(self.split.test_pair_keys)),
            }
            print(f"Split seed     : {self.split.seed}")
            print(f"Train songs    : {self.split.train_songs}")
            print(f"Val songs      : {self.split.val_songs}")
            print(f"Test pairs     : {self.split.test_pair_keys} (external test_ground_truth)")
            print(
                f"Pair counts    : train={len(self.meta['train'])} "
                f"val={len(self.meta['val'])} test={len(self.meta['test'])} "
                f"/ {len(all_pairs)} total"
            )

        if stage in (None, "fit") and self.train_dataset is None:
            self.train_dataset = GRPredCropDataset(
                self.meta["train"], self.sample_length, self.sample_rate
            )
        if stage in (None, "fit", "validate") and self.val_dataset is None:
            self.val_dataset = GRPredCropDataset(
                self.meta["val"], self.sample_length, self.sample_rate
            )
        if stage in (None, "test") and self.test_dataset is None:
            self.test_dataset = GRPredCropDataset(
                self.meta["test"], self.sample_length, self.sample_rate
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

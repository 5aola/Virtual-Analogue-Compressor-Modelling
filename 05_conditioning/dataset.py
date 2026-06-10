"""Multi-setting stateful GR dataset for TFiLM-conditioned LSTM training."""

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
HOP_SIZE = 256
SEGMENT_LEN = 32768


def discover_diffssl_gr_pairs(data_root: str) -> list[dict]:
    """All (song, setting) pairs with pre-computed GR curves on disk."""
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
        for pt in sorted(glob.glob(os.path.join(gr_root, setting, "*.pt"))):
            song = os.path.splitext(os.path.basename(pt))[0]
            if song in dry_lookup:
                pairs.append(
                    {
                        "song": song,
                        "setting": setting,
                        "dry": dry_lookup[song],
                        "gr": pt,
                        "params": normalize_setting_params(setting),
                    }
                )
    return sorted(pairs, key=lambda p: (p["song"], p["setting"]))


class StatefulMultiSettingGRDataset(Dataset):
    """One item == one TBPTT step across B parallel (song, setting) streams.

    Dry audio is loaded once per song and shared (views) across settings.
    """

    def __init__(
        self,
        pair_meta: list[dict],
        segment_len: int = SEGMENT_LEN,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.S = segment_len
        self.sr = sample_rate
        dry_by_song: dict[str, torch.Tensor] = {}
        self.cache: list[dict] = []

        for m in pair_meta:
            if m["song"] not in dry_by_song:
                dry_by_song[m["song"]] = self._load_dry(m["dry"])
            dry = dry_by_song[m["song"]]
            gr_db = torch.load(m["gr"], weights_only=False)["gr_db"].float()
            n = min(dry.shape[-1], gr_db.shape[-1])
            n -= n % self.S
            self.cache.append(
                {
                    "song": m["song"],
                    "setting": m["setting"],
                    "params": torch.tensor(m["params"], dtype=torch.float32),
                    "dry": dry[..., :n],
                    "gr_db": gr_db[..., :n],
                    "K": n // self.S,
                }
            )

        self.B = len(self.cache)
        self.K = max(c["K"] for c in self.cache)
        ram = (
            sum(d.numel() for d in dry_by_song.values())
            + sum(c["gr_db"].numel() for c in self.cache)
        ) * 4 / 1024**2
        print(
            f"StatefulMultiSettingGRDataset: B={self.B} streams "
            f"({len(dry_by_song)} songs), {self.K} steps/epoch, {ram:.0f} MB  "
            f"[segment_len={self.S}]"
        )

    def _load_dry(self, path: str) -> torch.Tensor:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        x = torch.from_numpy(audio.T)
        if sr != self.sr:
            x = torchaudio.functional.resample(x, sr, self.sr)
        if x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)
        return x

    def __len__(self) -> int:
        return self.K

    def __getitem__(self, s: int):
        S, B = self.S, self.B
        dry_b = torch.zeros(B, 1, S)
        gr_b = torch.zeros(B, 1, S)
        params_b = torch.zeros(B, 4)
        mask = torch.zeros(B)
        for r, c in enumerate(self.cache):
            if s < c["K"]:
                o = s * S
                dry_b[r] = c["dry"][:, o : o + S]
                gr_b[r] = c["gr_db"][:, o : o + S]
                params_b[r] = c["params"]
                mask[r] = 1.0
        return dry_b, gr_b, params_b, mask, (s == 0)


class MultiSettingGRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        segment_len: int = SEGMENT_LEN,
        sample_rate: int = SAMPLE_RATE,
        split_seed: int = 42,
        n_train_songs: int | None = None,
        n_val_songs: int = 1,
        n_test_songs: int = 2,
        split_manifest_path: str | None = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.segment_len = segment_len
        self.sample_rate = sample_rate
        self.split_seed = split_seed
        self.n_train_songs = n_train_songs
        self.n_val_songs = n_val_songs
        self.n_test_songs = n_test_songs
        self.split_manifest_path = split_manifest_path
        self.split: SplitManifest | None = None
        self._meta: dict[str, list[dict]] = {}
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        # idempotent: trainer.fit / trainer.test re-invoke this — build once
        if self.split is None:
            all_pairs = discover_diffssl_gr_pairs(self.data_root)

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
            self.train_dataset = StatefulMultiSettingGRDataset(
                self._meta["train"], self.segment_len, self.sample_rate
            )
        if stage in (None, "fit", "validate") and self.val_dataset is None:
            self.val_dataset = StatefulMultiSettingGRDataset(
                self._meta["val"], self.segment_len, self.sample_rate
            )
        if stage in (None, "test") and self.test_dataset is None:
            self.test_dataset = StatefulMultiSettingGRDataset(
                self._meta["test"], self.segment_len, self.sample_rate
            )

    # batch_size=None: the dataset already returns whole step-batches.
    # shuffle MUST be False to preserve time-order for state continuity.
    def _loader(self, dataset):
        return DataLoader(
            dataset, batch_size=None, shuffle=False, num_workers=0, pin_memory=True
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset)

    def val_dataloader(self):
        return self._loader(self.val_dataset)

    def test_dataloader(self):
        return self._loader(self.test_dataset)

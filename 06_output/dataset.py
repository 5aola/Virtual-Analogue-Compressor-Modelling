"""Multi-setting stateful dataset for the GR **output-transformer** LSTM.

Audio→audio in the time domain, **no conditioning**:

    input  = amplitude_match(dry, gr_db)   (dry × exported GR gain envelope)
    target = wet                           (true compressor output)

It fuses two existing pipelines:

  * ``05_conditioning/dataset.py`` — the multi-setting Diff-SSL layout
    (10 settings × 10 songs), shared dry audio per song, and the song-level
    split via ``splits.build_split_manifest`` (seed 42). The split is therefore
    identical to ``train_lstm_tfilm_gr.ipynb``.
  * ``02b_sota_training`` — the SOTA stateful, windowed audio→audio streams:
    each batch row is one (song, setting) track fed sequentially in
    ``segment_len`` chunks with the LSTM state carried chunk→chunk (TBPTT); the
    input chunk carries ``window-1`` samples of left context for the strided=1
    conv front-end.

The amplitude-matched input is computed **on the fly** per chunk from the
cached shared dry + per-pair GR curve (a cheap multiply), so only dry (shared),
``gr_db`` (per pair) and ``wet`` (per pair) are held in RAM.
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

from amplitude_match import amplitude_match
from splits import (
    SplitManifest,
    build_split_manifest,
    filter_pairs_by_keys,
)

SAMPLE_RATE = 44100
WINDOW = 64          # conv front-end receptive field (samples) — SOTA Optical-DRC
SEGMENT_LEN = 32768  # TBPTT chunk length (samples)


def discover_output_transformer_pairs(data_root: str) -> list[dict]:
    """All (song, setting) pairs that have BOTH a GR curve and a wet WAV.

    Yields the same (song, setting) set as ``05_conditioning`` (every setting
    has full dry/GR/wet parity), so the resulting split is identical.
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
                        "gr": pt,
                        "wet": wet,
                    }
                )
    return sorted(pairs, key=lambda p: (p["song"], p["setting"]))


class StatefulOutputTransformerDataset(Dataset):
    """One item == one TBPTT step across B parallel (song, setting) streams.

    ``__getitem__(s)`` returns the whole step-batch:

        inp   [B, 1, segment_len + window - 1]  amplitude-matched, left-padded
        wet   [B, 1, segment_len]               aligned target (window's last sample)
        mask  [B]                               1 = real, 0 = past this track's end
        reset (bool)                            True at s == 0  → zero LSTM state

    Dry audio is loaded once per song and shared (views) across its settings;
    the matched input is built per chunk as ``dry_slice × 10**(gr_slice/20)``.
    """

    def __init__(
        self,
        pair_meta: list[dict],
        segment_len: int = SEGMENT_LEN,
        window: int = WINDOW,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.S = segment_len
        self.w = window
        self.sr = sample_rate
        self.dry_by_song: dict[str, torch.Tensor] = {}
        self.cache: list[dict] = []

        for m in pair_meta:
            if m["song"] not in self.dry_by_song:
                self.dry_by_song[m["song"]] = self._load_audio(m["dry"])
            dry = self.dry_by_song[m["song"]]
            gr_db = torch.load(m["gr"], weights_only=False)["gr_db"].float()
            wet = self._load_audio(m["wet"])
            n = min(dry.shape[-1], gr_db.shape[-1], wet.shape[-1])
            if n < self.S:
                continue
            self.cache.append(
                {
                    "song": m["song"],
                    "setting": m["setting"],
                    "gr_db": gr_db[..., :n].contiguous(),
                    "wet": wet[..., :n].contiguous(),
                    "K": n // self.S,
                }
            )

        self.B = len(self.cache)
        self.K = max(c["K"] for c in self.cache) if self.cache else 0
        ram = (
            sum(d.numel() for d in self.dry_by_song.values())
            + sum(c["gr_db"].numel() + c["wet"].numel() for c in self.cache)
        ) * 4 / 1024**2
        print(
            f"StatefulOutputTransformerDataset: B={self.B} streams "
            f"({len(self.dry_by_song)} songs), {self.K} steps/epoch, {ram:.0f} MB  "
            f"[segment_len={self.S}, window={self.w}]"
        )

    def _load_audio(self, path: str) -> torch.Tensor:
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
        S, w, B = self.S, self.w, self.B
        inp_b = torch.zeros(B, 1, S + w - 1)
        wet_b = torch.zeros(B, 1, S)
        mask = torch.zeros(B)
        for r, c in enumerate(self.cache):
            if s < c["K"]:
                o = s * S
                dry = self.dry_by_song[c["song"]]
                wet_b[r] = c["wet"][:, o : o + S]
                lo = o - (w - 1)
                if lo < 0:  # first chunk: zero-pad the left context
                    amp = amplitude_match(dry[:, : o + S], c["gr_db"][:, : o + S])
                    inp_b[r, :, (w - 1) - o :] = amp
                else:
                    inp_b[r] = amplitude_match(
                        dry[:, lo : o + S], c["gr_db"][:, lo : o + S]
                    )
                mask[r] = 1.0
        return inp_b, wet_b, mask, (s == 0)


class OutputTransformerDataModule(pl.LightningDataModule):
    """Same split as ``05_conditioning`` (seed 42, n_val=1, n_test=2; test =
    held-out songs × lowest-threshold settings)."""

    # subclasses (e.g. the GR-conditioned variant) override this to swap in a
    # dataset that returns extra tensors; the split logic stays shared.
    dataset_cls = StatefulOutputTransformerDataset

    def __init__(
        self,
        data_root: str,
        segment_len: int = SEGMENT_LEN,
        window: int = WINDOW,
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
        self.window = window
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
        if self.split is None:
            all_pairs = discover_output_transformer_pairs(self.data_root)

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
            self.train_dataset = self.dataset_cls(
                self._meta["train"], self.segment_len, self.window, self.sample_rate
            )
        if stage in (None, "fit", "validate") and self.val_dataset is None:
            self.val_dataset = self.dataset_cls(
                self._meta["val"], self.segment_len, self.window, self.sample_rate
            )
        if stage in (None, "test") and self.test_dataset is None:
            self.test_dataset = self.dataset_cls(
                self._meta["test"], self.segment_len, self.window, self.sample_rate
            )

    # batch_size=None: dataset already returns whole step-batches.
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

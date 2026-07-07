"""diffssl-style crop dataloader for the SignalTrain **LA2A** dataset, emitting
the sample-aligned gain-reduction curve for the gain-prior model.

Mirrors ``06_output/dataset_tfilm.py`` (same crop/batch recipe, same
``(dry, gr, wet, params)`` item contract, drop-in for
``system_gainprior.GainPriorSystem``) with two LA2A-specific changes:

1. **Flat pair inventory** — the SignalTrain dataset is 84 numbered recordings
   ``all/input_<id>_.wav`` + ``all/target_<id>_LA2A_<Nc>__<cl>__<pr>.wav``, one
   per ``(comp_limit, peak_reduction)`` setting, not the Diff-SSL
   song × setting-folder grid. See ``discover_la2a_pairs``.

2. **GR computed on-the-fly** — the exported ``gr_curves/pair_*.pt`` are full
   length (52.9 M samples ≈ 211 MB each, ~18 GB total): impractical to ship /
   cache. Because GR is a deterministic function of ``(dry, wet)``, each crop's
   GR is recomputed with **the exact export function** —
   ``src.dsp_torch.gain_reduction_db(dry, wet, 1024)`` (causal, zero-left-padded
   1024-sample trailing RMS; the very function ``03_initial_GR_pred`` used to
   write the ``.pt`` files). A ``rms_window-1`` sample lookback is prepended to
   each crop before the RMS so the causal window is filled from real preceding
   samples — making the crop's GR **bit-identical** to slicing the full-file
   curve (both zero-pad only at the absolute file start). No ``.pt`` needed.

    dry [B, 1, S]   raw dry crop
    gr  [B, 1, S]   GR curve in dB, sample-aligned (== the exported .pt, sliced)
    wet [B, 1, S]   compressor output (target)
    p   [B, 2]      normalised [comp_limit, peak_reduction] knobs

Split: temporal-within-recording, see ``splits_la2a`` (every setting in every
split; test = unseen audio regions at known settings — the LA2A convention).
"""

from __future__ import annotations

import glob
import os
import re
from typing import Optional

import lightning as pl
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from src.dsp_torch import gain_reduction_db
from splits_la2a import (
    La2aSplitManifest,
    build_la2a_split_manifest,
    evenly_spaced_offsets,
    normalize_la2a_params,
    region_bounds,
)

SAMPLE_RATE = 44100
SAMPLE_LENGTH = 132300  # 3 s @ 44.1 kHz — matches the diffssl LSTM32TVC crop
BATCH_SIZE = 16
RMS_WINDOW = 1024  # GR export window (03_initial_GR_pred) — do not change

_TARGET_RE = re.compile(r"target_(\d+)_LA2A_\w+__(\d+)__(\d+)\.wav$")
_INPUT_RE = re.compile(r"input_(\d+)_.*\.wav$")


def discover_la2a_pairs(data_root: str, audio_subdir: str = "all") -> list[dict]:
    """All complete ``(input, target)`` recordings with their knob setting.

    Requires only the two WAVs per id — GR is recomputed on-the-fly, so the
    18 GB ``gr_curves/`` tree need not be present on the training machine.
    Returns dicts with the min-length frame count (dry/wet trimmed to ``min``,
    as diffssl does) used by the temporal split.
    """
    adir = os.path.join(data_root, audio_subdir)
    inputs: dict[int, str] = {}
    for p in glob.glob(os.path.join(adir, "input_*.wav")):
        m = _INPUT_RE.search(os.path.basename(p))
        if m:
            inputs[int(m.group(1))] = p
    targets: dict[int, tuple[str, int, int]] = {}
    for p in glob.glob(os.path.join(adir, "target_*.wav")):
        m = _TARGET_RE.search(os.path.basename(p))
        if m:
            targets[int(m.group(1))] = (p, int(m.group(2)), int(m.group(3)))

    pairs: list[dict] = []
    for pid in sorted(set(inputs) & set(targets)):
        dry = inputs[pid]
        wet, cl, pr = targets[pid]
        try:
            frames = min(sf.info(dry).frames, sf.info(wet).frames)
        except (RuntimeError, OSError) as e:
            print(f"Skipping pair {pid}: {e}")
            continue
        pairs.append(
            {
                "id": pid,
                "comp_limit": cl,
                "peak_reduction": pr,
                "dry": dry,
                "wet": wet,
                "frames": frames,
                "params": normalize_la2a_params(cl, pr),
            }
        )
    return sorted(pairs, key=lambda p: p["id"])


class La2aCropDataset(Dataset):
    """One item == aligned dry/gr/wet crop + params, from one temporal region.

    Crops are the evenly-spaced offsets of ``splits_la2a`` inside this split's
    region of each recording. GR is recomputed per crop from the dry/wet slices
    (with an ``rms_window-1`` lookback) — no GR cache, tiny memory footprint.
    """

    def __init__(
        self,
        pair_meta: list[dict],
        split: str,
        sample_length: int = SAMPLE_LENGTH,
        sample_rate: int = SAMPLE_RATE,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        crops_per_pair: int = 24,
        rms_window: int = RMS_WINDOW,
    ):
        self.split = split
        self.sample_length = sample_length
        self.sample_rate = sample_rate
        self.rms_window = rms_window
        self.lookback = rms_window - 1
        self.samples: list[dict] = []

        for m in pair_meta:
            bounds = region_bounds(m["frames"], train_frac, val_frac, test_frac)
            start, end = bounds[split]
            offsets = evenly_spaced_offsets(start, end, sample_length, crops_per_pair)
            params = torch.tensor(m["params"], dtype=torch.float32)
            for off in offsets:
                self.samples.append(
                    {"dry": m["dry"], "wet": m["wet"], "offset": off, "params": params}
                )

        minutes = len(self.samples) * sample_length / sample_rate / 60.0
        print(
            f"La2aCropDataset[{split}]: {len(self.samples)} crops from "
            f"{len(pair_meta)} recordings  [<= {crops_per_pair}/rec, "
            f"sample_length={sample_length}, {minutes:.1f} min audio]"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_slice(self, path: str, start: int, n: int) -> torch.Tensor:
        audio, sr = sf.read(
            path, start=start, stop=start + n, dtype="float32", always_2d=True
        )
        x = torch.from_numpy(audio.T)  # [C, n]
        if x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:  # LA2A is native 44.1 kHz — guard only
            x = torchaudio.functional.resample(x, sr, self.sample_rate)
        return x

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        start, L, lb = s["offset"], self.sample_length, self.lookback
        read_start = max(0, start - lb)
        n = start + L - read_start  # L (+ up to lb lookback) samples

        dry = self._load_slice(s["dry"], read_start, n)  # [1, n]
        wet = self._load_slice(s["wet"], read_start, n)  # [1, n]
        # exact export convention: causal zero-left-padded 1024-RMS GR in dB.
        # The lookback fills the causal window from real samples, so the last L
        # samples equal the full-file curve sliced to [start, start+L).
        gr = gain_reduction_db(dry, wet, self.rms_window)  # [1, n]

        return (
            dry[..., -L:].contiguous(),
            gr[..., -L:].contiguous(),
            wet[..., -L:].contiguous(),
            s["params"],
        )


class La2aCropDataModule(pl.LightningDataModule):
    """Temporal split (``splits_la2a``) + diffssl crop/batch recipe for LA2A.

    Same public API as ``06_output.dataset_tfilm.GRCropDataModule`` so the
    training notebook is a near drop-in. A saved ``split_manifest_path`` pins
    the split parameters (fractions, ``crops_per_pair``, ``sample_length``) for
    exact reproduction in eval; when present it is loaded and its parameters
    take precedence over the constructor arguments.

    ``DATASET_CLS`` is the per-crop dataset built for each split — a subclass
    can override it to change the emitted item (e.g. ``dataset_la2a_gr`` drops
    ``wet`` for the GR-predictor contract) without reimplementing ``setup``.
    """

    DATASET_CLS = La2aCropDataset

    def __init__(
        self,
        data_root: str,
        sample_length: int = SAMPLE_LENGTH,
        sample_rate: int = SAMPLE_RATE,
        batch_size: int = BATCH_SIZE,
        split_seed: int = 42,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        crops_per_pair: dict[str, int] | None = None,
        rms_window: int = RMS_WINDOW,
        split_manifest_path: str | None = None,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_root = data_root
        self.sample_length = sample_length
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.split_seed = split_seed
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.crops_per_pair = crops_per_pair or {"train": 24, "val": 4, "test": 8}
        self.rms_window = rms_window
        self.split_manifest_path = split_manifest_path
        self.num_workers = num_workers
        self.split: La2aSplitManifest | None = None
        self._pairs: list[dict] = []
        self.train_dataset: La2aCropDataset | None = None
        self.val_dataset: La2aCropDataset | None = None
        self.test_dataset: La2aCropDataset | None = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.split is None:
            self._pairs = discover_la2a_pairs(self.data_root)
            if not self._pairs:
                raise FileNotFoundError(
                    f"No LA2A input/target pairs under {self.data_root}/all"
                )

            if self.split_manifest_path and os.path.isfile(self.split_manifest_path):
                self.split = La2aSplitManifest.load(self.split_manifest_path)
                print(f"Loaded split manifest: {self.split_manifest_path}")
                # manifest parameters win, for exact reproduction
                self.sample_length = self.split.sample_length
                self.train_frac = self.split.train_frac
                self.val_frac = self.split.val_frac
                self.test_frac = self.split.test_frac
                self.crops_per_pair = self.split.crops_per_pair
            else:
                self.split = build_la2a_split_manifest(
                    self._pairs,
                    seed=self.split_seed,
                    sample_length=self.sample_length,
                    train_frac=self.train_frac,
                    val_frac=self.val_frac,
                    test_frac=self.test_frac,
                    crops_per_pair=self.crops_per_pair,
                )
                if self.split_manifest_path:
                    self.split.save(self.split_manifest_path)

            print(f"Recordings     : {len(self._pairs)}")
            print(f"Settings       : {len(self.split.settings)} unique (comp_limit, peak_reduction)")
            print(f"Split fractions: train={self.train_frac} val={self.val_frac} test={self.test_frac}")
            print(f"Crops per rec  : {self.crops_per_pair}")
            print(f"Crop totals    : {self.split.crop_counts}")

        common = dict(
            sample_length=self.sample_length,
            sample_rate=self.sample_rate,
            train_frac=self.train_frac,
            val_frac=self.val_frac,
            test_frac=self.test_frac,
            rms_window=self.rms_window,
        )
        if stage in (None, "fit") and self.train_dataset is None:
            self.train_dataset = self.DATASET_CLS(
                self._pairs, "train", crops_per_pair=self.crops_per_pair["train"], **common
            )
        if stage in (None, "fit", "validate") and self.val_dataset is None:
            self.val_dataset = self.DATASET_CLS(
                self._pairs, "val", crops_per_pair=self.crops_per_pair["val"], **common
            )
        if stage in (None, "test") and self.test_dataset is None:
            self.test_dataset = self.DATASET_CLS(
                self._pairs, "test", crops_per_pair=self.crops_per_pair["test"], **common
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
            self.val_dataset, batch_size=self.batch_size, shuffle=False, **self._loader_kwargs()
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, **self._loader_kwargs()
        )

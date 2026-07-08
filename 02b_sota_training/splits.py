"""Song-level train/val/test splits for multi-setting Diff-SSL training.

Copied verbatim from ``06_output/splits.py`` (the frozen split logic) so this
folder is self-contained on ``sys.path`` (no import collision with
``05_conditioning`` / ``06_output``, which also define ``dataset``/``model``/
``system``). Keeping the code byte-identical guarantees the split matches the
GR notebooks — including the v2 external-test policy
(``discover_test_ground_truth_keys`` + ``build_split_manifest(test_pair_keys=...)``,
which excludes the ``test_ground_truth/`` pairs from train/val by key) and the
legacy lowest-threshold policy (seed 42, n_val=1, n_test=2).
"""

from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Any

from src.dsp import PARAM_ORDER, parse_settings_from_folder_name

# nablafx / Diff-SSL-G-Comp param ranges (folder-name physical values)
DIFFSSL_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "threshold": (-20.0, 20.0),
    "attack": (0.0, 30.0),
    "release": (0.0, 1.6),
    "ratio": (0.0, 10.0),
}


def setting_threshold(setting: str) -> float:
    return float(parse_settings_from_folder_name(setting)["threshold"])


def normalize_setting_params(setting: str) -> list[float]:
    """Map folder-name compressor settings to [0, 1] — see NORMALISATION.md."""
    parsed = parse_settings_from_folder_name(setting)
    out: list[float] = []
    for key in PARAM_ORDER:
        lo, hi = DIFFSSL_PARAM_RANGES[key]
        out.append((float(parsed[key]) - lo) / (hi - lo))
    return out


def lowest_threshold_settings(settings: list[str]) -> list[str]:
    """Return all setting folders tied for the minimum threshold value."""
    min_t = min(setting_threshold(s) for s in settings)
    return sorted(s for s in settings if setting_threshold(s) == min_t)


def discover_test_ground_truth_keys(data_root: str) -> set[str]:
    """Pair keys (``song::setting``) of the external held-out test set.

    The ``test_ground_truth/`` folder is the single source of truth for what
    is held out: any wet export placed there marks its (song, setting) pair
    as test-only, and ``build_split_manifest(test_pair_keys=...)`` excludes
    it from train/val — even when a GR curve for it exists under
    ``gr_curves/`` (those curves then serve as test labels only). This is the
    guard against the contamination found in run
    ``lstm_gr_20260702_174039``: exported test GR curves were silently swept
    into train (4 pairs) and val (1 pair) by the song-level split.
    """
    tgt_dir = os.path.join(data_root, "test_ground_truth")
    if not os.path.isdir(tgt_dir):
        raise FileNotFoundError(
            f"No test_ground_truth/ under {data_root} — it defines the held-out "
            "test set; refusing to build a split without it."
        )
    keys: set[str] = set()
    for setting in sorted(os.listdir(tgt_dir)):
        sdir = os.path.join(tgt_dir, setting)
        if not setting.startswith("threshold_") or not os.path.isdir(sdir):
            continue
        for wet in glob.glob(os.path.join(sdir, "*-exported.wav")):
            song = os.path.basename(wet).replace("-exported.wav", "")
            keys.add(pair_key(song, setting))
    if not keys:
        raise FileNotFoundError(f"test_ground_truth/ under {data_root} holds no wet exports")
    return keys


@dataclass
class SplitManifest:
    """Reproducible split metadata — load this in eval notebooks."""

    seed: int
    train_songs: list[str]
    val_songs: list[str]
    test_songs: list[str]
    test_settings: list[str]
    all_settings: list[str]
    train_pair_keys: list[str]
    val_pair_keys: list[str]
    test_pair_keys: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SplitManifest":
        with open(path) as f:
            return cls(**json.load(f))


def pair_key(song: str, setting: str) -> str:
    return f"{song}::{setting}"


def make_song_split(
    songs: list[str],
    *,
    n_train: int | None = None,
    n_val: int = 1,
    n_test: int = 2,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Random song-level partition (not fixed 80/10/10)."""
    songs = sorted(songs)
    n_train = n_train if n_train is not None else len(songs) - n_val - n_test
    rng = random.Random(seed)
    perm = list(range(len(songs)))
    rng.shuffle(perm)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val : n_train + n_val + n_test]
    return (
        [songs[i] for i in train_idx],
        [songs[i] for i in val_idx],
        [songs[i] for i in test_idx],
    )


def build_split_manifest(
    pairs: list[dict],
    *,
    seed: int = 42,
    n_train_songs: int | None = None,
    n_val_songs: int = 1,
    n_test_songs: int = 2,
    test_pair_keys: set[str] | None = None,
) -> SplitManifest:
    """Build train/val/test pair lists. Two test policies:

  - **``test_pair_keys`` given** (current recipe): those exact pairs — the
    external ``test_ground_truth/`` exports, see
    ``discover_test_ground_truth_keys`` — are the held-out test set and are
    excluded from train/val **by key**. Every remaining (song, setting) pair
    is fair training material (including the former lowest-threshold test
    songs); the remaining songs are split song-level into train/val.
    ``n_test_songs`` is ignored.
  - **``test_pair_keys`` None** (legacy notebooks): test = lowest-threshold
    settings × ``n_test_songs`` held-out songs; train/val = all settings ×
    remaining song subsets.
    """
    settings = sorted({p["setting"] for p in pairs})

    if test_pair_keys is not None:
        test_pairs = [p for p in pairs if pair_key(p["song"], p["setting"]) in test_pair_keys]
        rest = [p for p in pairs if pair_key(p["song"], p["setting"]) not in test_pair_keys]
        rest_songs = sorted({p["song"] for p in rest})
        train_songs, val_songs, _ = make_song_split(
            rest_songs,
            n_train=n_train_songs,
            n_val=n_val_songs,
            n_test=0,
            seed=seed,
        )
        train_set, val_set = set(train_songs), set(val_songs)
        train_pairs = [p for p in rest if p["song"] in train_set]
        val_pairs = [p for p in rest if p["song"] in val_set]
        return SplitManifest(
            seed=seed,
            train_songs=sorted(train_set),
            val_songs=sorted(val_set),
            test_songs=sorted({p["song"] for p in test_pairs}),
            test_settings=sorted({p["setting"] for p in test_pairs}),
            all_settings=settings,
            train_pair_keys=sorted(pair_key(p["song"], p["setting"]) for p in train_pairs),
            val_pair_keys=sorted(pair_key(p["song"], p["setting"]) for p in val_pairs),
            test_pair_keys=sorted(pair_key(p["song"], p["setting"]) for p in test_pairs),
        )

    songs = sorted({p["song"] for p in pairs})
    test_settings = lowest_threshold_settings(settings)

    train_songs, val_songs, test_songs = make_song_split(
        songs,
        n_train=n_train_songs,
        n_val=n_val_songs,
        n_test=n_test_songs,
        seed=seed,
    )
    train_set, val_set, test_set = set(train_songs), set(val_songs), set(test_songs)
    test_setting_set = set(test_settings)

    train_pairs, val_pairs, test_pairs = [], [], []
    for p in pairs:
        key = pair_key(p["song"], p["setting"])
        if p["song"] in test_set and p["setting"] in test_setting_set:
            test_pairs.append(p)
        elif p["song"] in val_set:
            val_pairs.append(p)
        elif p["song"] in train_set:
            train_pairs.append(p)

    return SplitManifest(
        seed=seed,
        train_songs=sorted(train_set),
        val_songs=sorted(val_set),
        test_songs=sorted(test_set),
        test_settings=test_settings,
        all_settings=settings,
        train_pair_keys=sorted(pair_key(p["song"], p["setting"]) for p in train_pairs),
        val_pair_keys=sorted(pair_key(p["song"], p["setting"]) for p in val_pairs),
        test_pair_keys=sorted(pair_key(p["song"], p["setting"]) for p in test_pairs),
    )


def filter_pairs_by_keys(pairs: list[dict], keys: set[str]) -> list[dict]:
    return [p for p in pairs if pair_key(p["song"], p["setting"]) in keys]

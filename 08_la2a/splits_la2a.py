"""Temporal train/val/test split for the SignalTrain LA2A dataset.

The LA2A (SignalTrain 1.1) dataset is structurally different from Diff-SSL-G-Comp,
so the ``06_output``/``05_conditioning`` song-level split logic does **not** apply
and is deliberately *not* reused here:

- **Diff-SSL**: many short *songs* each rendered at ~10 *setting folders*; the
  same song content recurs across settings, so a **song-level** split (hold out
  whole songs, or the external ``test_ground_truth/`` pairs by key) cleanly
  separates content.
- **LA2A**: 84 long recordings (4-20 min each), one per ``(comp_limit,
  peak_reduction)`` setting. Each recording is a *distinct* audio montage — there
  is no shared "song" axis to hold out, and every setting appears in only one (a
  few in two-three) recording(s). Holding out whole recordings would remove a
  setting from training entirely, breaking the knob conditioning.

So we split **temporally within each recording** — the standard SignalTrain /
LA2A methodology (Hawley 2019; Steinmetz TCN; Comparative-Study; Optical-DRC all
test on held-out *audio regions* at the same settings, not held-out settings):

    per recording:  [0, train_frac) -> train   [.., +val_frac) -> val   [.., 1) -> test

Properties:

- **Every setting is represented in all three splits** — the 2-knob conditioning
  is fully learnable, and the test set probes generalisation to *unseen audio*
  at *known* settings (the question the LA-2A literature asks).
- **No content leakage**: split boundaries are by time *fraction*, identical for
  every recording, so a given temporal region of a montage is always in exactly
  one split (relevant only if any montages were shared, which they are not here).
- Within each region, crops are taken at **evenly-spaced** offsets and capped at
  ``crops_per_pair`` — the 24.3 h corpus (~19 h at 80 % train) is far too large
  to crop exhaustively, so equal, deterministic, montage-spanning sampling per
  recording bounds the epoch while keeping every setting balanced and every
  region's material (music / speech / tones) represented.

The manifest pins ``sample_length``, the fractions, ``crops_per_pair`` and the
pair inventory (id + setting + frame count), so ``dataset_la2a`` reproduces the
exact same crops for training *and* for a later eval notebook.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

# LA2A (SignalTrain) knob conventions — see la2a_info.ini:
#   Comp/Limit    : 0 = compress, 1 = limit  (switch)
#   Peak Reduction: 0..100                    (front-panel dial)
LA2A_PARAM_ORDER = ["comp_limit", "peak_reduction"]
LA2A_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "comp_limit": (0.0, 1.0),
    "peak_reduction": (0.0, 100.0),
}


def normalize_la2a_params(comp_limit: int, peak_reduction: int) -> list[float]:
    """Map physical LA2A knob values to ``[0, 1]`` in ``LA2A_PARAM_ORDER``."""
    vals = {"comp_limit": float(comp_limit), "peak_reduction": float(peak_reduction)}
    out: list[float] = []
    for key in LA2A_PARAM_ORDER:
        lo, hi = LA2A_PARAM_RANGES[key]
        out.append((vals[key] - lo) / (hi - lo))
    return out


def region_bounds(
    frames: int, train_frac: float, val_frac: float, test_frac: float
) -> dict[str, tuple[int, int]]:
    """Sample-index ``(start, end)`` of each temporal region within a recording."""
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fracs must sum to 1, got {total}")
    t_end = int(round(frames * train_frac))
    v_end = int(round(frames * (train_frac + val_frac)))
    return {"train": (0, t_end), "val": (t_end, v_end), "test": (v_end, frames)}


def evenly_spaced_offsets(
    start: int, end: int, sample_length: int, max_crops: int
) -> list[int]:
    """Non-overlapping crop start offsets, evenly spanning ``[start, end)``.

    Returns up to ``max_crops`` offsets chosen from the ``n_avail`` non-overlapping
    ``sample_length`` windows that fit in the region. When ``n_avail`` exceeds the
    cap, offsets are picked at ``linspace`` positions so the selection spans the
    whole region (montage breadth) rather than clustering at the start.
    """
    n_avail = (end - start) // sample_length
    if n_avail <= 0:
        return []
    if max_crops <= 0 or n_avail <= max_crops:
        idx = range(n_avail)
    else:
        # linspace over window indices [0, n_avail-1], deduplicated & rounded.
        step = (n_avail - 1) / (max_crops - 1) if max_crops > 1 else 0.0
        idx = sorted({int(round(k * step)) for k in range(max_crops)})
    return [start + i * sample_length for i in idx]


@dataclass
class La2aSplitManifest:
    """Reproducible temporal-split metadata (load this in eval notebooks)."""

    seed: int
    sample_length: int
    train_frac: float
    val_frac: float
    test_frac: float
    crops_per_pair: dict[str, int]  # {"train": .., "val": .., "test": ..}
    settings: list[list[int]]  # sorted unique [comp_limit, peak_reduction]
    pairs: list[dict[str, Any]] = field(default_factory=list)
    # {"id", "comp_limit", "peak_reduction", "frames"} per recording
    crop_counts: dict[str, int] = field(default_factory=dict)  # realised totals

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "La2aSplitManifest":
        with open(path) as f:
            return cls(**json.load(f))


def build_la2a_split_manifest(
    pairs: list[dict],
    *,
    seed: int = 42,
    sample_length: int = 132300,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    crops_per_pair: dict[str, int] | None = None,
) -> La2aSplitManifest:
    """Build a temporal split manifest from a ``discover_la2a_pairs`` inventory.

    ``crops_per_pair`` caps the evenly-spaced crops drawn from each recording's
    region (``<= 0`` means "take all non-overlapping windows"). The realised
    per-split crop totals are recorded in ``crop_counts`` for reference.
    """
    if crops_per_pair is None:
        crops_per_pair = {"train": 24, "val": 4, "test": 8}

    settings = sorted({(p["comp_limit"], p["peak_reduction"]) for p in pairs})
    counts = {"train": 0, "val": 0, "test": 0}
    for p in pairs:
        bounds = region_bounds(p["frames"], train_frac, val_frac, test_frac)
        for split in ("train", "val", "test"):
            s, e = bounds[split]
            counts[split] += len(
                evenly_spaced_offsets(s, e, sample_length, crops_per_pair[split])
            )

    return La2aSplitManifest(
        seed=seed,
        sample_length=sample_length,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        crops_per_pair=dict(crops_per_pair),
        settings=[list(s) for s in settings],
        pairs=[
            {
                "id": p["id"],
                "comp_limit": p["comp_limit"],
                "peak_reduction": p["peak_reduction"],
                "frames": p["frames"],
            }
            for p in sorted(pairs, key=lambda q: q["id"])
        ],
        crop_counts=counts,
    )

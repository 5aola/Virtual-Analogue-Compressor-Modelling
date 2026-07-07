"""GR-predictor crop dataloader for the SignalTrain **LA2A** dataset.

The LA2A analogue of ``05_conditioning/dataset_crops.py``: it emits the
``(dry, gr, params)`` 3-tuple the frame-rate GR predictor expects
(``model_blackbox_gr.BlackboxGRLSTM`` + ``system_detector_gr.DetectorGRSystem``),
instead of the ``(dry, gr, wet, params)`` 4-tuple the gain-prior model needs.

Everything about the crops — the temporal-within-recording split
(``splits_la2a``), the evenly-spaced offset sampling, and the **on-the-fly GR**
(``src.dsp_torch.gain_reduction_db(dry, wet, 1024)`` with a ``rms_window-1``
lookback, bit-identical to the exported ``gr_curves/*.pt``) — is reused verbatim
from ``dataset_la2a``. The only differences are:

1. ``La2aGRCropDataset`` drops ``wet`` from the returned tuple (the GR predictor
   is supervised against the GR *curve*, never the wet audio directly).
2. ``La2aGRCropDataModule`` additionally exposes ``self.meta`` — per-recording
   dicts with the dry/wet paths, params, id/setting and this split's temporal
   ``region`` (sample bounds) — so a notebook can stream whole test regions for
   a full-context GR MAE (the ``05_conditioning`` streaming-eval cell analogue).

Split: temporal-within-recording, every setting in every split; test = unseen
audio regions at known settings (the LA2A convention). See ``splits_la2a``.
"""

from __future__ import annotations

from typing import Optional

from dataset_la2a import (  # re-exported so the notebook imports from one module
    BATCH_SIZE,
    RMS_WINDOW,
    SAMPLE_LENGTH,
    SAMPLE_RATE,
    La2aCropDataModule,
    La2aCropDataset,
    discover_la2a_pairs,
)
from splits_la2a import normalize_la2a_params, region_bounds

__all__ = [
    "BATCH_SIZE",
    "RMS_WINDOW",
    "SAMPLE_LENGTH",
    "SAMPLE_RATE",
    "La2aGRCropDataset",
    "La2aGRCropDataModule",
    "discover_la2a_pairs",
]


class La2aGRCropDataset(La2aCropDataset):
    """``La2aCropDataset`` that emits ``(dry, gr, params)`` for GR prediction."""

    def __getitem__(self, idx: int):
        dry, gr, _wet, params = super().__getitem__(idx)
        return dry, gr, params


class La2aGRCropDataModule(La2aCropDataModule):
    """Temporal-split LA2A crops for the GR predictor (``dry, gr, params``).

    Drop-in for ``05_conditioning/dataset_crops.GRPredCropDataModule`` up to the
    LA2A-specific split (percentage-based, every setting in every split) and the
    on-the-fly GR — same public API, plus ``self.meta`` for streaming eval.
    """

    DATASET_CLS = La2aGRCropDataset

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)
        # Per-recording streaming metadata: dry/wet + this split's region bounds.
        # Built once, after the split manifest exists; mirrors GRPredCropDataModule.meta.
        if not getattr(self, "meta", None):
            self.meta = {"train": [], "val": [], "test": []}
            for m in self._pairs:
                bounds = region_bounds(
                    m["frames"], self.train_frac, self.val_frac, self.test_frac
                )
                cl, pr = m["comp_limit"], m["peak_reduction"]
                for split in ("train", "val", "test"):
                    self.meta[split].append(
                        {
                            "id": m["id"],
                            "song": str(m["id"]),  # recording id doubles as the "song" label
                            "setting": f"cl{cl}_pr{pr}",
                            "comp_limit": cl,
                            "peak_reduction": pr,
                            "dry": m["dry"],
                            "wet": m["wet"],
                            "params": normalize_la2a_params(cl, pr),
                            "region": bounds[split],
                        }
                    )

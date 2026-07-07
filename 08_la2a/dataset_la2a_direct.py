"""Direct-output crop dataloader for the SignalTrain **LA2A** dataset.

The LA2A analogue of ``02b_sota_training/dataset.py`` (``DiffSSLCropDataset``):
it emits the ``(dry, wet, params)`` 3-tuple the diffssl TVC-LSTM baseline
expects (``model.build_diffssl_tvc_lstm`` + ``system.DiffSSLTVCLSTMSystem``),
instead of the ``(dry, gr, wet, params)`` 4-tuple the gain-prior model needs.

The crop inventory — the temporal-within-recording split (``splits_la2a``) and
the evenly-spaced offset sampling — is reused verbatim from ``dataset_la2a``.
The SOTA baseline is supervised directly against the wet audio, so ``gr`` is
never needed: ``__getitem__`` reads exactly the ``sample_length`` crop and skips
both the per-item gain-reduction cumsum and the 1023-sample RMS lookback read
that the parent (GR-emitting) dataset does — pure overhead here. The returned
dry/wet crops are byte-identical to the parent's (same offset, same length), so
this experiment still trains on the exact same audio windows as the LA2A
GR-predictor run.

Split: temporal-within-recording, every setting in every split; test = unseen
audio regions at known settings (the LA2A convention). See ``splits_la2a``.
"""

from __future__ import annotations

from dataset_la2a import (  # re-exported so the notebook imports from one module
    BATCH_SIZE,
    RMS_WINDOW,
    SAMPLE_LENGTH,
    SAMPLE_RATE,
    La2aCropDataModule,
    La2aCropDataset,
    discover_la2a_pairs,
)

__all__ = [
    "BATCH_SIZE",
    "RMS_WINDOW",
    "SAMPLE_LENGTH",
    "SAMPLE_RATE",
    "La2aDirectCropDataset",
    "La2aDirectCropDataModule",
    "discover_la2a_pairs",
]


class La2aDirectCropDataset(La2aCropDataset):
    """``La2aCropDataset`` that emits ``(dry, wet, params)`` for direct output.

    Reads exactly the ``sample_length`` crop — no GR, no RMS lookback. The parent
    computes a gain-reduction curve (and reads a 1023-sample lookback to make it
    causal-exact) on every item; both are pure overhead for a model supervised
    against wet audio, so we skip straight to the two audio slices. The dry/wet
    crops are identical to what the parent would return (same offset, same L).
    """

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        start, L = s["offset"], self.sample_length
        dry = self._load_slice(s["dry"], start, L)  # [1, L]
        wet = self._load_slice(s["wet"], start, L)  # [1, L]
        return dry.contiguous(), wet.contiguous(), s["params"]


class La2aDirectCropDataModule(La2aCropDataModule):
    """Temporal-split LA2A crops for the diffssl TVC-LSTM baseline.

    Drop-in for ``02b_sota_training/dataset.DiffSSLCropDataModule`` (same
    ``(dry, wet, params)`` item contract, same crop/batch recipe) but with the
    LA2A-specific temporal-within-recording split — every setting in every
    split, test = unseen audio regions at known settings.
    """

    DATASET_CLS = La2aDirectCropDataset

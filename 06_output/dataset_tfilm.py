"""GR-conditioned variant of the output-transformer dataset.

Identical streams/split to ``dataset.py`` (amplitude-matched input + wet target),
but each TBPTT step also yields the **sample-aligned GR curve** for the chunk so
the model can use it as a continuous, time-varying conditioning signal
(``model_tfilm.OutputTransformerTFiLMLSTM``).

    inp [B, 1, S+w-1]  amplitude-matched, left-padded   (built from dry x GR gain)
    gr  [B, 1, S]      GR curve in dB, aligned to wet    (conditioning signal)
    wet [B, 1, S]      true compressor output            (target)
    mask[B]            1 = real, 0 = past this track's end
    reset (bool)       True at s == 0 -> zero LSTM state

The GR is already cached per pair (it also builds the matched input), so this
adds no extra RAM — only one extra slice per step.
"""

from __future__ import annotations

import torch

from dataset import (  # noqa: F401  (re-exported for notebook convenience)
    SAMPLE_RATE,
    SEGMENT_LEN,
    WINDOW,
    OutputTransformerDataModule,
    StatefulOutputTransformerDataset,
    discover_output_transformer_pairs,
)


class StatefulOutputTransformerTFiLMDataset(StatefulOutputTransformerDataset):
    """``StatefulOutputTransformerDataset`` that also returns the GR curve."""

    def __getitem__(self, s: int):
        inp_b, wet_b, mask, reset = super().__getitem__(s)
        S, B = self.S, self.B
        gr_b = torch.zeros(B, 1, S)
        for r, c in enumerate(self.cache):
            if s < c["K"]:
                o = s * S
                gr_b[r] = c["gr_db"][:, o : o + S]
        return inp_b, gr_b, wet_b, mask, reset


class OutputTransformerTFiLMDataModule(OutputTransformerDataModule):
    """Same split/streams as ``OutputTransformerDataModule``; emits the GR curve."""

    dataset_cls = StatefulOutputTransformerTFiLMDataset

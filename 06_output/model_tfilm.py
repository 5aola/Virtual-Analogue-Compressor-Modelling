"""diffssl LSTM32TVC mirror + GR curve as a TFiLM conditioning signal.

This is the ``02b_sota_training`` diffssl ``LSTM32TVC`` recipe
(``cond_type="tvcond"`` — ``TVFiLMCond`` + sample-rate LSTM), kept **identical**
for the four static knobs, with **one addition**: the exported gain-reduction
curve modulates the LSTM hidden features through a true **temporal FiLM**
(``GRTFiLM``), the γ/β mechanism from the nablafx fork
(``external/nablafx-for-diffssl-compressor/nablafx/modules.py::TFiLM``) — *not*
the concat-style ``tvcond``.

```
raw dry x [B,1,S] ─┐                        4 static knobs p [B,4]
                   │  tvcond (nablafx TVFiLMCond, UNCHANGED from 02b):
                   │    pool(|x|)⊕knobs → blockLSTM → cond_seq[16]  (block rate)
                   │                        │  upsample + crop
                   └──── cat(x, cond_seq) → [B, 1+16, S]
                             │
                        main LSTM(17 → H)  → hidden [B, H, S]
                             │
                   GR-TFiLM: pool(GR) → blockLSTM → γ,β[H] per block   ← GR here
                        modulate  h·γ + β  (block-constant, broadcast)
                             │
                        Linear(H → 1) → tanh → wet [B,1,S]
```

Two conditioning mechanisms: the **static knobs** via the SOTA ``tvcond`` (kept
identical to diffssl), the **GR** via a temporal FiLM γ/β modulation.

**Capacity-matched twin** (``film_signal="amp"``): the temporal FiLM ingests the
dry **peak envelope** ``pool(|x|)`` — information the model already receives via
``tvcond`` — instead of the GR curve. Same architecture, same parameter count,
same state-dict keys; the *only* difference is whether the film signal carries
privileged GR information. Comparing the two runs isolates the value of the GR
signal itself from the extra capacity of the film path.

State handling mirrors ``nablafx/processors/lstm.py`` so this model is a drop-in
for the ``02b`` TBPTT system: ``reset_states()`` / ``detach_states()`` clear /
detach all three stateful LSTMs (main, ``TVFiLMCond``, ``GRTFiLM``); ``forward``
carries state internally rather than threading a tuple.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nablafx.processors.components import TVFiLMCond

from amplitude_match import GR_DB_MAX, GR_DB_MIN


def _reduction_positive(gr_db: torch.Tensor) -> torch.Tensor:
    """Map a GR curve in dB to ~[0, 1] with **more compression → larger value**.

    ``gr_db`` is negative during compression (``20·log10(wet/dry)``), so
    ``(GR_DB_MAX − gr_db)`` grows with reduction. ``MaxPool`` over this signal
    then captures the **peak compression** in each block — the direct analog of
    diffssl's ``MaxPool(|x|)`` (peak amplitude). Pooling the *normalised* GR
    instead (as the old ``GRTVCond`` did) inverts this: max would select the
    least-compressed sample and discard the peak reduction.
    """
    return (GR_DB_MAX - gr_db.clamp(GR_DB_MIN, GR_DB_MAX)) / (GR_DB_MAX - GR_DB_MIN)


class GRTFiLM(nn.Module):
    """GR-driven temporal FiLM (γ/β modulation over time).

    Port of the nablafx fork's ``TFiLM`` with one change: the block-rate LSTM
    ingests the **pooled GR curve** (1 channel) instead of the pooled feature
    map, so the modulation is a function of the ground-truth gain reduction.
    Emits ``2·nfeatures`` per block → split into γ, β and applied
    block-constant to the ``nfeatures``-wide hidden features.

    forward: ``feat [B, C, S]``, ``cond [B, 1, S]`` → modulated ``[B, C, S]``.
    ``signal`` picks the pooled input: ``"gr"`` — ``cond`` is the GR curve in dB,
    mapped reduction-positive; ``"amp"`` — ``cond`` is a raw waveform, pooled as
    the peak envelope ``|cond|`` (capacity-matched control, no GR information).
    LSTM state is kept internally (``reset_state`` / ``detach_state``) so it
    carries across TBPTT sub-steps alongside the other LSTMs.
    """

    def __init__(
        self,
        nfeatures: int,
        block_size: int = 128,
        num_layers: int = 1,
        signal: str = "gr",
    ):
        super().__init__()
        assert signal in ("gr", "amp"), f"unknown film signal: {signal!r}"
        self.nfeatures = nfeatures
        self.block_size = block_size
        self.num_layers = num_layers
        self.signal = signal
        self.pool = nn.MaxPool1d(kernel_size=block_size)
        # GR (1 ch) -> 2*nfeatures (gamma, beta)
        self.lstm = nn.LSTM(1, nfeatures * 2, num_layers, batch_first=True)
        self.hidden_state = None

    def reset_state(self) -> None:
        self.hidden_state = None

    def detach_state(self) -> None:
        if self.hidden_state is not None:
            self.hidden_state = tuple(h.detach() for h in self.hidden_state)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, c, s = feat.shape
        pad = (self.block_size - s % self.block_size) % self.block_size
        if pad:
            feat = F.pad(feat, (0, pad))
            cond = F.pad(cond, (0, pad))
        s_pad = feat.shape[-1]
        nsteps = s_pad // self.block_size

        if self.signal == "gr":
            pooled_in = _reduction_positive(cond)           # GR dB → [0, 1]
        else:  # "amp": peak envelope of the raw waveform (analog of pool(|x|))
            pooled_in = cond.abs().clamp(max=1.0)
        gr_down = self.pool(pooled_in)                      # [B, 1, nsteps]
        mod, self.hidden_state = self.lstm(gr_down.transpose(1, 2), self.hidden_state)
        mod = mod.transpose(1, 2).reshape(b, 2 * c, nsteps, 1)  # [B, 2C, nsteps, 1]
        gamma, beta = torch.chunk(mod, 2, dim=1)            # each [B, C, nsteps, 1]

        blocks = feat.reshape(b, c, nsteps, self.block_size)
        out = (blocks * gamma + beta).reshape(b, c, s_pad)
        return out[..., :s]


class GRTFiLMDiffSSLLSTM(nn.Module):
    """diffssl LSTM(tvcond) for the static knobs + GR-driven temporal FiLM.

    ``film_signal="amp"`` is the capacity-matched twin: the temporal FiLM reads
    the dry peak envelope instead of the GR curve (``gr`` is accepted and
    ignored, keeping the training pipeline identical).
    """

    def __init__(
        self,
        num_controls: int = 4,
        hidden_size: int = 32,
        num_layers: int = 1,
        # -- static-knob conditioning (diffssl tvcond, unchanged) --
        tvcond_dim: int = 16,
        cond_block_size: int = 128,
        cond_num_layers: int = 1,
        # -- GR conditioning (temporal FiLM) --
        gr_tfilm_block_size: int = 128,
        gr_tfilm_num_layers: int = 1,
        film_signal: str = "gr",   # "gr" | "amp" (capacity-matched twin)
    ):
        super().__init__()
        self.num_controls = num_controls
        self.hidden_size = hidden_size
        self.cond_block_size = cond_block_size
        self.film_signal = film_signal

        # diffssl tvcond: pool(|x|) + concat knobs -> block LSTM -> cond_seq[16]
        self.cond_nn = TVFiLMCond(
            input_dim=1,
            output_dim=tvcond_dim,
            cond_dim=num_controls,
            block_size=cond_block_size,
            num_layers=cond_num_layers,
        )
        # main sample-rate LSTM (raw input + upsampled cond_seq)
        self.lstm = nn.LSTM(1 + tvcond_dim, hidden_size, num_layers, batch_first=True)
        self.main_state = None
        # GR temporal FiLM on the hidden features
        self.gr_tfilm = GRTFiLM(
            nfeatures=hidden_size,
            block_size=gr_tfilm_block_size,
            num_layers=gr_tfilm_num_layers,
            signal=film_signal,
        )
        self.lin = nn.Linear(hidden_size, 1)

    # -- state API matching nablafx / the 02b TBPTT system ------------------
    def reset_states(self) -> None:
        self.main_state = None
        self.cond_nn.reset_state()
        self.gr_tfilm.reset_state()

    def detach_states(self) -> None:
        if self.main_state is not None:
            self.main_state = tuple(h.detach() for h in self.main_state)
        self.cond_nn.detach_state()
        self.gr_tfilm.detach_state()

    def forward(self, x: torch.Tensor, gr: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # x:  [B, 1, S] raw dry     gr: [B, 1, S] GR curve (dB)     p: [B, num_controls]
        s = x.shape[-1]

        # -- static-knob conditioning (diffssl tvcond) --
        cond = self.cond_nn(x, p)                                # [B, 16, nsteps]
        cond = cond.repeat_interleave(self.cond_block_size, dim=-1)[..., :s]
        h = torch.cat((x, cond), dim=1).transpose(1, 2)         # [B, S, 1+16]

        # -- main LSTM --
        h, self.main_state = self.lstm(h, self.main_state)      # [B, S, H]

        # -- temporal FiLM on the hidden features (GR, or dry envelope for the twin) --
        film_in = gr if self.film_signal == "gr" else x
        h = self.gr_tfilm(h.transpose(1, 2), film_in)           # [B, H, S]

        # -- output --
        y = self.lin(h.transpose(1, 2)).transpose(1, 2)         # [B, 1, S]
        return torch.tanh(y)

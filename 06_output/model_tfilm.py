"""GR-conditioned output-transformer LSTM (continuous TFiLM conditioning).

Extends ``OutputTransformerLSTM`` (the amplitude-matched grey box) by feeding the
**exported gain-reduction curve** back in as a *continuous, time-varying*
conditioning signal — not as the 4 static knobs (threshold/attack/release/ratio)
used in ``05_conditioning``.

Why this is the right conditioning, and where it comes from in the SOTA
--------------------------------------------------------------------------
nablafx / Optical-DRC provide two FiLM families (``nablafx.processors.components``):

* ``TFiLM`` — modulates blocks from a **static** ``cond=[B, cond_dim]`` (the
  knobs). Used by ``05_conditioning``.
* ``TVFiLMCond`` + ``TVFiLMMod`` (nablafx ``cond_type="tvcond"``,
  ``processors/lstm.py``) — the **time-varying** path. ``TVFiLMCond`` runs a
  block-rate LSTM over ``|x|`` to *synthesise* a dynamic conditioning sequence,
  which ``TVFiLMMod`` then applies block-wise.

``TVFiLMCond`` only exists because the SOTA has to *learn* a dynamic signal from
a static knob — it has no ground-truth envelope. **We already have it: the
exported GR curve is precisely that dynamic conditioning sequence.** So we drop
the learned generator and feed the real GR curve straight into ``TVFiLMMod``.

Two complementary, flag-gated injection points (both default on):

  1. ``use_tvfilm`` — pool ``gr_db`` to block rate, embed it (with its
     block-to-block delta ≈ attack/release rate), and **block-wise FiLM-modulate
     the LSTM hidden features** before the output head (the headline TFiLM).
  2. ``concat_gr`` — concat the sample-rate normalised GR to the LSTM input (the
     nablafx ``tvcond`` injection point) so the recurrence itself is GR-aware.

The residual-add baseline is preserved: ``dense_out`` is still zero-init, so the
correction is exactly 0 at step 0 regardless of the FiLM scaling — the net starts
at the validated amplitude-matched signal and only learns GR-gated coloration.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nablafx.processors.components import TVFiLMMod

from amplitude_match import GR_DB_MAX, GR_DB_MIN

_ACTIVATIONS = {"tanh": nn.Tanh, "none": nn.Identity}


def _normalize_gr(gr_db: torch.Tensor) -> torch.Tensor:
    """Map a GR curve in dB onto ~[0, 1] using the project-wide GR range."""
    return (gr_db.clamp(GR_DB_MIN, GR_DB_MAX) - GR_DB_MIN) / (GR_DB_MAX - GR_DB_MIN)


class GRFilmConditioner(nn.Module):
    """Block-wise time-varying FiLM driven by the exported GR curve.

    Pools the sample-aligned ``gr_db`` to block rate, lifts it (optionally with
    its block-to-block delta) to a small embedding, and applies nablafx
    ``TVFiLMMod`` to the feature stream. ``TVFiLMMod`` is stateless (a 1x1 conv
    adaptor + block-wise affine), so no recurrent conditioning state has to be
    carried across TBPTT chunks — only the two main LSTM states.
    """

    def __init__(
        self,
        nfeatures: int,
        block_size: int = 256,
        cond_channels: int = 8,
        use_delta: bool = True,
    ):
        super().__init__()
        self.block_size = block_size
        self.use_delta = use_delta
        in_ch = 2 if use_delta else 1
        self.embed = nn.Sequential(
            nn.Conv1d(in_ch, cond_channels, kernel_size=1),
            nn.PReLU(cond_channels),
        )
        self.tvfilm = TVFiLMMod(
            nfeatures=nfeatures, cond_dim=cond_channels, block_size=block_size
        )

    def forward(self, x: torch.Tensor, gr_db: torch.Tensor) -> torch.Tensor:
        # x: [B, nfeatures, S]   gr_db: [B, 1, S]  (sample-aligned)
        s = x.shape[-1]
        nsteps = math.ceil(s / self.block_size)  # matches TVFiLMMod's padded nsteps
        grn = _normalize_gr(gr_db)
        gr_blk = F.adaptive_avg_pool1d(grn, nsteps)  # [B, 1, nsteps] block-rate GR
        if self.use_delta:
            delta = gr_blk - F.pad(gr_blk, (1, 0))[..., :-1]  # rate of change (attack/release)
            cond = torch.cat((gr_blk, delta), dim=1)
        else:
            cond = gr_blk
        cond = self.embed(cond)  # [B, cond_channels, nsteps]
        return self.tvfilm(x, cond)  # block-wise affine modulation


class OutputTransformerTFiLMLSTM(nn.Module):
    def __init__(
        self,
        window: int = 64,
        encoder_channels: int = 8,
        hidden_size: int = 16,
        mid_channels: int = 8,
        num_lstm_layers: int = 2,
        output_mode: str = "residual_add",
        out_activation: str = "none",
        # -- GR conditioning --
        concat_gr: bool = True,
        use_tvfilm: bool = True,
        tvfilm_block_size: int = 256,
        gr_cond_channels: int = 8,
        gr_use_delta: bool = True,
    ):
        super().__init__()
        if output_mode not in ("residual_add", "residual_gain", "direct"):
            raise ValueError(f"unknown output_mode: {output_mode}")
        if num_lstm_layers not in (1, 2):
            raise ValueError("num_lstm_layers must be 1 or 2")
        self.window = window
        self.output_mode = output_mode
        self.num_lstm_layers = num_lstm_layers
        self.concat_gr = concat_gr
        self.use_tvfilm = use_tvfilm

        # stride=1 windowed conv: each step sees `window` samples (no downsampling)
        self.proj = nn.Conv1d(1, encoder_channels, kernel_size=window, stride=1)
        self.act = nn.PReLU(encoder_channels)
        lstm1_in = encoder_channels + (1 if concat_gr else 0)
        self.lstm1 = nn.LSTM(lstm1_in, hidden_size, batch_first=True)
        if num_lstm_layers == 2:
            self.dense_mid = nn.Linear(hidden_size, mid_channels)
            self.lstm2 = nn.LSTM(mid_channels, hidden_size, batch_first=True)
        if use_tvfilm:
            self.gr_film = GRFilmConditioner(
                nfeatures=hidden_size,
                block_size=tvfilm_block_size,
                cond_channels=gr_cond_channels,
                use_delta=gr_use_delta,
            )
        self.dense_out = nn.Linear(hidden_size, 1)
        self.out_activation = _ACTIVATIONS[out_activation]()
        self._init_output_baseline()

    def _init_output_baseline(self) -> None:
        """Start residual modes exactly at the amplitude-matched baseline.

        Zero-init ``dense_out`` so the correction is 0 at step 0 (``residual_add``
        -> out = current input; ``residual_gain`` -> gain = 1). This holds even
        with GR-FiLM active, since the FiLM only scales the hidden features that
        ``dense_out`` zeroes out. ``direct`` keeps the default init."""
        with torch.no_grad():
            if self.output_mode == "residual_add":
                nn.init.zeros_(self.dense_out.weight)
                nn.init.zeros_(self.dense_out.bias)
            elif self.output_mode == "residual_gain":
                nn.init.zeros_(self.dense_out.weight)
                nn.init.ones_(self.dense_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        gr: torch.Tensor,
        state=None,
        return_state: bool = False,
    ):
        # x:  [B, 1, S + window - 1]  amplitude-matched, with left context
        # gr: [B, 1, S]               sample-aligned GR curve (dB) for this chunk
        cur = x[:, :, self.window - 1 :]            # [B, 1, S] aligned current sample
        h = self.act(self.proj(x)).transpose(1, 2)  # [B, S, C]
        if self.concat_gr:
            grn = _normalize_gr(gr).transpose(1, 2)  # [B, S, 1]
            h = torch.cat((h, grn), dim=-1)          # [B, S, C+1]
        s1, s2 = (None, None) if state is None else state
        h, s1 = self.lstm1(h, s1)                    # [B, S, H]
        if self.num_lstm_layers == 2:
            h = self.dense_mid(h)                    # [B, S, mid]
            h, s2 = self.lstm2(h, s2)                # [B, S, H]
        if self.use_tvfilm:
            h = self.gr_film(h.transpose(1, 2), gr).transpose(1, 2)  # GR block-FiLM
        y = self.out_activation(self.dense_out(h)).transpose(1, 2)   # [B, 1, S]

        if self.output_mode == "residual_add":
            out = cur + y
        elif self.output_mode == "residual_gain":
            out = cur * y
        else:  # direct
            out = y
        return (out, (s1, s2)) if return_state else out

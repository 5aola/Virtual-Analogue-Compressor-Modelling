"""GR-conditioned output-transformer LSTM (diffssl ``tvcond`` conditioning).

Extends ``OutputTransformerLSTM`` (the amplitude-matched grey box) by feeding the
**exported gain-reduction curve** back in as a *continuous, time-varying*
conditioning signal — using the *same conditioning mechanism as the diffssl
``LSTM32TVC`` baseline*, not the 4 static knobs of ``05_conditioning``.

Conditioning mechanism — identical structure to diffssl ``cond_type="tvcond"``
--------------------------------------------------------------------------------
nablafx's LSTM ``tvcond`` path (``nablafx/processors/lstm.py``) does *not* FiLM
anything despite the name. It:

  1. runs ``TVFiLMCond`` — max-pool the signal to block rate, then a block-rate
     LSTM emits a learned ``cond_dim``-wide time-varying sequence;
  2. upsamples that sequence back to sample rate and **concatenates it to the
     LSTM input** (``x = cat([x, cond], dim=1)``).

We reproduce exactly that, with two deliberate choices:

  * **GR-driven generator.** ``TVFiLMCond`` only pools ``|x|`` (+static knobs)
    because the SOTA has no ground-truth envelope and must *learn* a dynamic
    signal. We already have it — the exported GR curve *is* that sequence — so the
    generator's block-rate LSTM reads the (normalised) GR instead of ``|x|``.
  * **Functional state.** nablafx's ``TVFiLMCond`` keeps its LSTM state internally
    (``reset_state``/``detach_state``); here the conditioning LSTM state is
    threaded through the model's ``state`` tuple as a third element, so it carries
    across TBPTT chunks alongside the two main LSTM states.

Amplitude matching is unchanged: the model input is ``dry * 10**(gr_db/20)``, and
the GR curve additionally drives the conditioning. The residual-add baseline is
preserved: ``dense_out`` is zero-init, so the correction is exactly 0 at step 0
regardless of the conditioning — the net starts at the validated amplitude-matched
signal and only learns GR-gated coloration.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from amplitude_match import GR_DB_MAX, GR_DB_MIN

_ACTIVATIONS = {"tanh": nn.Tanh, "none": nn.Identity}


def _normalize_gr(gr_db: torch.Tensor) -> torch.Tensor:
    """Map a GR curve in dB onto ~[0, 1] using the project-wide GR range."""
    return (gr_db.clamp(GR_DB_MIN, GR_DB_MAX) - GR_DB_MIN) / (GR_DB_MAX - GR_DB_MIN)


class GRTVCond(nn.Module):
    """diffssl ``tvcond`` conditioning generator, GR-driven.

    Structurally identical to nablafx ``TVFiLMCond`` (max-pool to block rate ->
    block-rate LSTM -> upsample to sample rate), but the generator reads the
    exported GR curve instead of ``|x|`` + static knobs, and its LSTM state is
    threaded functionally so it carries across TBPTT chunks.

    forward: ``gr_db [B, 1, S] -> cond_seq [B, cond_dim, S]`` plus the LSTM state.
    """

    def __init__(self, cond_dim: int = 16, block_size: int = 128, num_layers: int = 1):
        super().__init__()
        self.cond_dim = cond_dim
        self.block_size = block_size
        # max-pool over the (non-negative, normalised) GR == diffssl's abs+pool,
        # capturing the peak compression depth in each block.
        self.pool = nn.MaxPool1d(kernel_size=block_size)
        self.lstm = nn.LSTM(1, cond_dim, num_layers, batch_first=True)

    def forward(self, gr_db: torch.Tensor, state=None):
        s = gr_db.shape[-1]
        grn = _normalize_gr(gr_db)                      # [B, 1, S]
        if s % self.block_size:                         # pad to a whole #blocks
            grn = F.pad(grn, (0, self.block_size - s % self.block_size))
        pooled = self.pool(grn).transpose(1, 2)         # [B, nsteps, 1]
        cond, state = self.lstm(pooled, state)          # [B, nsteps, cond_dim]
        cond = cond.transpose(1, 2)                     # [B, cond_dim, nsteps]
        cond = cond.repeat_interleave(self.block_size, dim=-1)[..., :s]  # upsample+crop
        return cond, state


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
        # -- GR conditioning (diffssl tvcond, GR-driven) --
        use_gr_cond: bool = True,
        cond_dim: int = 16,
        cond_block_size: int = 128,
        cond_num_layers: int = 1,
    ):
        super().__init__()
        if output_mode not in ("residual_add", "residual_gain", "direct"):
            raise ValueError(f"unknown output_mode: {output_mode}")
        if num_lstm_layers not in (1, 2):
            raise ValueError("num_lstm_layers must be 1 or 2")
        self.window = window
        self.output_mode = output_mode
        self.num_lstm_layers = num_lstm_layers
        self.use_gr_cond = use_gr_cond

        # stride=1 windowed conv: each step sees `window` samples (no downsampling)
        self.proj = nn.Conv1d(1, encoder_channels, kernel_size=window, stride=1)
        self.act = nn.PReLU(encoder_channels)
        if use_gr_cond:
            self.gr_cond = GRTVCond(
                cond_dim=cond_dim, block_size=cond_block_size, num_layers=cond_num_layers
            )
        # diffssl tvcond injection point: the conditioning sequence is concatenated
        # to the LSTM input (encoder features), exactly like nablafx LSTM(C+cond_dim).
        lstm1_in = encoder_channels + (cond_dim if use_gr_cond else 0)
        self.lstm1 = nn.LSTM(lstm1_in, hidden_size, batch_first=True)
        if num_lstm_layers == 2:
            self.dense_mid = nn.Linear(hidden_size, mid_channels)
            self.lstm2 = nn.LSTM(mid_channels, hidden_size, batch_first=True)
        self.dense_out = nn.Linear(hidden_size, 1)
        self.out_activation = _ACTIVATIONS[out_activation]()
        self._init_output_baseline()

    def _init_output_baseline(self) -> None:
        """Start residual modes exactly at the amplitude-matched baseline.

        Zero-init ``dense_out`` so the correction is 0 at step 0 (``residual_add``
        -> out = current input; ``residual_gain`` -> gain = 1). This holds
        regardless of the conditioning, since the GR cond only feeds the LSTM input
        whose contribution ``dense_out`` zeroes out. ``direct`` keeps default init."""
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
        s1, s2, sc = (None, None, None) if state is None else state
        if self.use_gr_cond:
            cond, sc = self.gr_cond(gr, sc)              # [B, cond_dim, S]
            h = torch.cat((h, cond.transpose(1, 2)), dim=-1)  # [B, S, C+cond_dim]
        h, s1 = self.lstm1(h, s1)                    # [B, S, H]
        if self.num_lstm_layers == 2:
            h = self.dense_mid(h)                    # [B, S, mid]
            h, s2 = self.lstm2(h, s2)                # [B, S, H]
        y = self.out_activation(self.dense_out(h)).transpose(1, 2)   # [B, 1, S]

        if self.output_mode == "residual_add":
            out = cur + y
        elif self.output_mode == "residual_gain":
            out = cur * y
        else:  # direct
            out = y
        return (out, (s1, s2, sc)) if return_state else out

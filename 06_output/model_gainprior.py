"""Gain-prior diffssl LSTM — the GR curve as a *structural* multiplicative prior.

Motivation (from the 06_output evals): the trivial amplitude match
``dry × 10^(gr/20)`` already beats every trained model on the envelope metrics
(MR-STE 0.011, MR-STFT 0.069 vs the GR-TFiLM's 0.103 / 0.107), while the
trained models win by 20-60x on the waveform metrics (ESR, L1). Feature-space
conditioning (``model_tfilm.GRTFiLM``) asks the network to *re-learn* the
multiply through a block-constant γ/β affine on hidden features — and
empirically it doesn't (with the oracle GR as input its val GR MAE, 0.336 dB,
is no better than the unconditioned SOTA's 0.314 dB). Here the multiply is
built into the architecture and the network only has to *correct* it:

```
raw dry x [B,1,S] ────────────────────────────────┐
x·g  (amplitude-matched, g = 10^(gr/20)) ─────────┤
gr   (reduction-positive, ~[0,1]) ────────────────┼─ cat → main LSTM(19→H)
tvcond cond_seq[16]  (pool(|x|)⊕knobs, as 02b) ───┘         │
                                        ┌───────────────────┴──────────────┐
                              Δg = δmax·tanh(lin_g(h))            c = lin_c(h)
                                  (zero-init → 0 dB)            (zero-init → 0)
                                        │                              │
                          y = x · 10^((gr + Δg)/20)  +  c
```

Both output heads are **zero-initialised**, so at step 0 the model emits
*exactly* the amplitude-matched signal — it starts at the strong multiply
baseline and training can only improve on it. The two heads factor the
residual physically:

- ``Δg`` (bounded ±``delta_max`` dB): corrects what the 1024-sample trailing
  RMS window smeared — chiefly attack transients (the settings go down to
  1 ms attack ≈ 44 samples, far below the 23 ms RMS window).
- ``c`` (additive): the coloration/waveshaping residual (~8 % RMS) that no
  time-varying gain can produce from the dry signal.

Because the GR enters as an explicit, physically-meaningful input (not a
learned feature modulation), the model remains usable with *any* GR source at
inference: the oracle export, the ``05_conditioning`` predictor, or a GR curve
derived from a **different (sidechain) signal**.

Static knobs are conditioned exactly like 02b/diffssl (``TVFiLMCond`` →
``cond_seq[16]`` concatenated to the LSTM input). No output ``tanh`` — the
multiplicative prior already keeps the output in range, and ``tanh`` would
bias loud passages.

State handling mirrors ``model_tfilm`` / ``nablafx/processors/lstm.py``
(``reset_states`` / ``detach_states``) so this model is a drop-in for the
02b-style TBPTT systems. ``forward`` has the same ``(x, gr, p)`` signature as
``GRTFiLMDiffSSLLSTM`` so the eval tooling can be reused unchanged.

~8.3k parameters — parameter-matched to the 8k SOTA LSTM32TVC baseline
(unlike the 25.2k GR-TFiLM), so the ablation is fair.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nablafx.processors.components import TVFiLMCond

from amplitude_match import GR_DB_MAX, GR_DB_MIN, gr_db_to_gain
from model_tfilm import _reduction_positive


class GainPriorDiffSSLLSTM(nn.Module):
    """diffssl LSTM(tvcond) core with the GR curve as a multiplicative prior."""

    def __init__(
        self,
        num_controls: int = 4,
        hidden_size: int = 32,
        num_layers: int = 1,
        # -- static-knob conditioning (diffssl tvcond, unchanged from 02b) --
        tvcond_dim: int = 16,
        cond_block_size: int = 128,
        cond_num_layers: int = 1,
        # -- gain-prior heads --
        delta_max_db: float = 12.0,   # bound on the learned gain correction
        use_color: bool = True,       # additive coloration head (ablation flag)
    ):
        super().__init__()
        self.num_controls = num_controls
        self.hidden_size = hidden_size
        self.cond_block_size = cond_block_size
        self.delta_max_db = delta_max_db
        self.use_color = use_color

        # diffssl tvcond: pool(|x|) + concat knobs -> block LSTM -> cond_seq[16]
        self.cond_nn = TVFiLMCond(
            input_dim=1,
            output_dim=tvcond_dim,
            cond_dim=num_controls,
            block_size=cond_block_size,
            num_layers=cond_num_layers,
        )
        # main sample-rate LSTM: raw dry + amplitude-matched + GR + cond_seq
        self.lstm = nn.LSTM(3 + tvcond_dim, hidden_size, num_layers, batch_first=True)
        self.main_state = None

        # Δg head (dB, bounded via tanh) and additive coloration head.
        self.lin_gain = nn.Linear(hidden_size, 1)
        self.lin_color = nn.Linear(hidden_size, 1)
        self._init_zero_heads()

    def _init_zero_heads(self) -> None:
        """Zero-init both heads: at step 0, y == amplitude_match(x, gr) exactly."""
        with torch.no_grad():
            nn.init.zeros_(self.lin_gain.weight)
            nn.init.zeros_(self.lin_gain.bias)
            nn.init.zeros_(self.lin_color.weight)
            nn.init.zeros_(self.lin_color.bias)

    # -- state API matching nablafx / the 02b TBPTT systems ------------------
    def reset_states(self) -> None:
        self.main_state = None
        self.cond_nn.reset_state()

    def detach_states(self) -> None:
        if self.main_state is not None:
            self.main_state = tuple(h.detach() for h in self.main_state)
        self.cond_nn.detach_state()

    def forward(
        self,
        x: torch.Tensor,
        gr: torch.Tensor,
        p: torch.Tensor,
        return_parts: bool = False,
    ):
        # x: [B, 1, S] raw dry    gr: [B, 1, S] GR curve (dB)    p: [B, num_controls]
        s = x.shape[-1]

        # -- static-knob conditioning (diffssl tvcond, unchanged) --
        cond = self.cond_nn(x, p)                                # [B, 16, nsteps]
        cond = cond.repeat_interleave(self.cond_block_size, dim=-1)[..., :s]

        # -- gain prior --
        gr_c = gr.clamp(GR_DB_MIN, GR_DB_MAX)
        matched = x * gr_db_to_gain(gr_c, clamp=False)           # amplitude-matched
        gr_pos = _reduction_positive(gr)                         # ~[0,1], compression-positive

        # -- main LSTM over (dry, matched, GR, cond) --
        h = torch.cat((x, matched, gr_pos, cond), dim=1).transpose(1, 2)  # [B, S, 19]
        h, self.main_state = self.lstm(h, self.main_state)       # [B, S, H]

        # -- heads (both zero-init) --
        delta_db = self.delta_max_db * torch.tanh(self.lin_gain(h))       # [B, S, 1]
        delta_db = delta_db.transpose(1, 2)                               # [B, 1, S]
        color = self.lin_color(h).transpose(1, 2) if self.use_color else torch.zeros_like(x)

        y = x * gr_db_to_gain(gr_c + delta_db, clamp=False) + color
        if return_parts:
            return y, delta_db, color
        return y

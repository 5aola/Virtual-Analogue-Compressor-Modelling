"""Simplified gain-prior LSTM for the predicted-GR cascade — no waveshaper.

Motivation (measured on the trained ``e2e_predgr_20260705_185738`` run,
07_experiments/04 + the simplify-ablation eval, 2026-07-07): the waveshaper
learned the identity — max |W(s)−s| ≈ 3·10⁻⁵ over s∈[−1,1], curve harmonics
≤ 10⁻⁷ — so the composition stage contributes nothing and the trained model is
*already* effectively ``y = x·10^((ĝr+Δg)/20) + color``. This module drops the
dead waveshaper and keeps exactly the two mechanisms that were doing the work:

    s = x · 10^((ĝr + Δg)/20)      Δg = δmax·tanh(lin_gain(h))   (GR fix, dB)
    y = s · (1 + m)                m  = tanh(lin_color(h))       (mult. color)

The coloration head is now a MULTIPLY of the gain-matched signal instead of a
free additive waveform: coloration is guaranteed signal-locked (silence in →
silence out, no free-running noise floor), and (1+m) ∈ (0, 2) can reach a true
mute, which the bounded log-domain Δg cannot.

Honest caveat, so nobody over-reads the two heads: (1+m) and 10^(Δg/20) are
both time-varying gains read from the same LSTM state — one log-domain and
bounded (±δmax dB), one linear-domain. The model family is therefore
"multiplicative prior + learned time-varying gain"; keeping two heads is a
parameterisation convenience (bounded slow correction + fast linear colour),
not two mechanisms. ``use_mult_color=False`` collapses to the minimal
Δg-only model ``y = x·10^((ĝr+Δg)/20)`` (66 → 33 head params).

Kept from the WS variant (drop-in for ``GainPriorSystem`` and the eval /
07_experiments streaming helpers):
- diffssl tvcond knob conditioning, CONCATENATED into the main LSTM input
- ``use_matched_input`` flag (x·ĝ input channel)
- zero-init identity: at step 0, y == amplitude match of the given ĝr exactly
- ``forward(x, gr, p, return_parts=..., return_parts_full=...)`` contract;
  the non-gain residual is ``s·m`` (ws_res slot returns zeros)

Params: 8,322 (cond 1,472 + LSTM 6,784 + heads 66) vs 8,418 for the WS model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nablafx.processors.components import TVFiLMCond

from amplitude_match import GR_DB_MAX, GR_DB_MIN, gr_db_to_gain
from model_tfilm import _reduction_positive


class GainPriorSimpleE2ELSTM(nn.Module):
    """Multiplicative prior + Δg (log gain fix) + optional linear colour multiply."""

    def __init__(
        self,
        num_controls: int = 4,
        hidden_size: int = 32,
        num_layers: int = 1,
        tvcond_dim: int = 16,
        cond_block_size: int = 128,
        cond_num_layers: int = 1,
        delta_max_db: float = 12.0,
        use_mult_color: bool = True,
        use_matched_input: bool = True,
    ):
        super().__init__()
        self.num_controls = num_controls
        self.hidden_size = hidden_size
        self.cond_block_size = cond_block_size
        self.delta_max_db = delta_max_db
        self.use_mult_color = use_mult_color
        self.use_matched_input = use_matched_input

        self.cond_nn = TVFiLMCond(
            input_dim=1,
            output_dim=tvcond_dim,
            cond_dim=num_controls,
            block_size=cond_block_size,
            num_layers=cond_num_layers,
        )
        n_in = (3 if use_matched_input else 2) + tvcond_dim
        self.lstm = nn.LSTM(n_in, hidden_size, num_layers, batch_first=True)
        self.main_state = None

        self.lin_gain = nn.Linear(hidden_size, 1)
        self.lin_color = nn.Linear(hidden_size, 1) if use_mult_color else None
        with torch.no_grad():
            nn.init.zeros_(self.lin_gain.weight)
            nn.init.zeros_(self.lin_gain.bias)
            if self.lin_color is not None:
                nn.init.zeros_(self.lin_color.weight)
                nn.init.zeros_(self.lin_color.bias)

    # -- state API matching the 02b/06/09 TBPTT systems ----------------------
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
        return_parts_full: bool = False,
    ):
        # x: [B, 1, S] raw dry    gr: [B, 1, S] predicted GR (dB)    p: [B, C]
        s_len = x.shape[-1]

        cond = self.cond_nn(x, p)                                # [B, 16, nsteps]
        cond = cond.repeat_interleave(self.cond_block_size, dim=-1)[..., :s_len]

        gr_c = gr.clamp(GR_DB_MIN, GR_DB_MAX)
        gr_pos = _reduction_positive(gr)                         # ~[0,1]

        if self.use_matched_input:
            matched = x * gr_db_to_gain(gr_c, clamp=False)
            feats = (x, matched, gr_pos, cond)                   # [B, 19, S]
        else:
            feats = (x, gr_pos, cond)                            # [B, 18, S]
        h = torch.cat(feats, dim=1).transpose(1, 2)
        h, self.main_state = self.lstm(h, self.main_state)       # [B, S, H]

        delta_db = self.delta_max_db * torch.tanh(self.lin_gain(h))
        delta_db = delta_db.transpose(1, 2)                      # [B, 1, S]
        gained = x * gr_db_to_gain(gr_c + delta_db, clamp=False)

        if self.lin_color is not None:
            m = torch.tanh(self.lin_color(h)).transpose(1, 2)    # (-1, 1)
            color = gained * m
        else:
            color = torch.zeros_like(gained)

        y = gained + color                                       # = gained·(1+m)
        if return_parts_full:
            return y, delta_db, color, torch.zeros_like(color)
        if return_parts:
            return y, delta_db, color
        return y

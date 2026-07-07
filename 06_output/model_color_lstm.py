"""Black-box coloration LSTM on the amplitude-matched signal — no tricks.

Motivation (07_experiments/08 GR-source sweep + the 2026-07-07 simplify
ablation): the gain-prior family OBEYS the GR input best (const_0db sweep:
output pinned to the commanded gain) but never learns the coloration residual
(waveshaper → identity, metrics parked at amp-match); GR-TFiLM learns the
coloration to near-SOTA ESR but half-ignores the GR input (it re-derives
compression from the dry signal). This model combines the two properties by
construction:

    matched = x · 10^(gr/20)          gain = explicit multiply → obeys ANY
                                      GR source exactly (sidechain goal)
    y = tanh( lin( LSTM(matched ⊕ tvcond) ) )   ← the UNCHANGED from-scratch
                                      SOTA trunk (diffssl LSTM32TVC: cond_type
                                      "tvcond", residual=False, direct_path=
                                      False), fed the gained signal

The ONLY difference from the 02b SOTA black-box is the input signal: matched
instead of dry. The trunk provably learns saturation + filtering at this size
(SOTA ≈ 0.0025 A-wt ESR from raw dry); here it only has to learn the easier
matched → wet coloration (~18 % RMS residual). It cannot silently take over
the gain because the gain is already applied to its input — and it never sees
the GR value itself, so there is nothing to disobey.

No zero-init identity (nothing is anchored): the untrained output is NOT the
amplitude match — cell 6 of the training notebook prints the amp-match L1
separately as the baseline to beat.

Drop-in for ``GainPriorSystem`` / the eval + 07_experiments streaming helpers:
``forward(x, gr, p, return_parts=True)`` returns ``(y, Δg≡0, y − matched)``
so the system's ``gain/*`` logging reads: Δg share 0, color share = the full
learned coloration.

Params: 8,033 (cond 1,472 + LSTM(17→32) 6,528 + readout 33) — parameter-
matched to SOTA LSTM32TVC and the 8.3–8.4k gain-prior variants.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nablafx.processors.components import TVFiLMCond

from amplitude_match import GR_DB_MAX, GR_DB_MIN, gr_db_to_gain


class ColorBlackboxLSTM(nn.Module):
    """SOTA LSTM32TVC trunk on the amplitude-matched signal."""

    def __init__(
        self,
        num_controls: int = 4,
        hidden_size: int = 32,
        num_layers: int = 1,
        tvcond_dim: int = 16,
        cond_block_size: int = 128,
        cond_num_layers: int = 1,
    ):
        super().__init__()
        self.num_controls = num_controls
        self.hidden_size = hidden_size
        self.cond_block_size = cond_block_size

        self.cond_nn = TVFiLMCond(
            input_dim=1,
            output_dim=tvcond_dim,
            cond_dim=num_controls,
            block_size=cond_block_size,
            num_layers=cond_num_layers,
        )
        self.lstm = nn.LSTM(1 + tvcond_dim, hidden_size, num_layers, batch_first=True)
        self.main_state = None
        self.lin = nn.Linear(hidden_size, 1)

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
        # x: [B, 1, S] raw dry    gr: [B, 1, S] GR curve (dB)    p: [B, C]
        s_len = x.shape[-1]

        gr_c = gr.clamp(GR_DB_MIN, GR_DB_MAX)
        matched = x * gr_db_to_gain(gr_c, clamp=False)           # gain applied here

        # tvcond pools ITS input signal (SOTA contract) — here the matched one
        cond = self.cond_nn(matched, p)                          # [B, 16, nsteps]
        cond = cond.repeat_interleave(self.cond_block_size, dim=-1)[..., :s_len]

        h = torch.cat((matched, cond), dim=1).transpose(1, 2)    # [B, S, 17]
        h, self.main_state = self.lstm(h, self.main_state)       # [B, S, H]
        y = torch.tanh(self.lin(h)).transpose(1, 2)              # [B, 1, S]

        if return_parts_full:
            zeros = torch.zeros_like(y)
            return y, zeros, y - matched, zeros
        if return_parts:
            return y, torch.zeros_like(y), y - matched
        return y

"""Gain-prior waveshaper LSTM for the PREDICTED-GR cascade (09_end2end).

Architecturally this is ``model_gainprior_ws.GainPriorWSDiffSSLLSTM`` — same
prior, same Δg/color heads, same waveshaper, same zero-init identity — the
model itself doesn't know (or care) whether the ``gr`` it receives is the
oracle export or a frozen predictor's output. The subclass exists for one
ablation flag:

``use_matched_input`` (default **True** — the exact oracle-run architecture):
    True  : main LSTM sees (x, x·10^(gr/20), gr_pos, cond)   — 3+16 inputs
    False : main LSTM sees (x,               gr_pos, cond)   — 2+16 inputs

The False case tests the hypothesis that the amplitude-matched channel is
redundant because it is a deterministic function of the other two inputs
(x · f(gr)). Counter-hypothesis, why True stays the default: LSTM gates are
additive machines — a *product* of two input channels is exactly the kind of
feature they are bad at synthesising internally — and the matched signal is
the thing the waveshaper actually shapes, so hiding it forces the recurrent
path to re-derive it. With a predicted (imperfect) GR the channel is arguably
MORE useful: Δg's job becomes correcting predictor error, and seeing x·ĝ next
to x is the direct evidence of what that error does to the waveform. The flag
costs/saves only 4·hidden = 128 params. Run True first (clean single-variable
ablation vs the oracle-GR run), then flip.

Keep the prior path untouched in both modes: the multiplicative prior
``gained = x·10^((gr+Δg)/20)`` and the zero-init identity (output == amplitude
match of the GIVEN gr at step 0) are the whole point of the decomposition.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from amplitude_match import GR_DB_MAX, GR_DB_MIN, gr_db_to_gain
from model_gainprior_ws import GainPriorWSDiffSSLLSTM
from model_tfilm import _reduction_positive


class GainPriorWSE2ELSTM(GainPriorWSDiffSSLLSTM):
    """Drop-in for ``GainPriorSystem``; adds ``use_matched_input`` only."""

    def __init__(
        self,
        use_matched_input: bool = True,
        tvcond_dim: int = 16,
        hidden_size: int = 32,
        num_layers: int = 1,
        **kwargs,
    ):
        super().__init__(
            tvcond_dim=tvcond_dim, hidden_size=hidden_size, num_layers=num_layers,
            **kwargs,
        )
        self.use_matched_input = use_matched_input
        if not use_matched_input:
            # rebuild the main LSTM without the amplitude-matched input row
            self.lstm = nn.LSTM(2 + tvcond_dim, hidden_size, num_layers, batch_first=True)

    def forward(
        self,
        x: torch.Tensor,
        gr: torch.Tensor,
        p: torch.Tensor,
        return_parts: bool = False,
        return_parts_full: bool = False,
    ):
        # x: [B, 1, S] raw dry    gr: [B, 1, S] PREDICTED GR (dB)    p: [B, C]
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

        shaped = self.waveshaper(gained, h if self.ws_film else None)
        ws_res = shaped - gained
        color = self.lin_color(h).transpose(1, 2) if self.use_color else torch.zeros_like(x)

        y = shaped + color
        if return_parts_full:
            return y, delta_db, color, ws_res
        if return_parts:
            return y, delta_db, ws_res + color
        return y

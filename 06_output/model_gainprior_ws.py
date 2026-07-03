"""Gain-prior diffssl LSTM with a *waveshaper* coloration stage.

Same structural multiplicative prior as ``model_gainprior.GainPriorDiffSSLLSTM``
(output starts as the amplitude match, zero-init heads learn the residual),
with the coloration mechanism changed from additive synthesis to composition:

    model_gainprior:   y = x·10^((gr+Δg)/20) + c(h)           c: Linear(32→1)
    this module:       y = W( x·10^((gr+Δg)/20) ) [+ c(h)]    W: memoryless waveshaper

Why: the additive head must *synthesise* the ≈8 % RMS coloration residual as a
free waveform — to emit harmonics it has to track the signal's instantaneous
phase at sample rate through 32 hidden units and a 33-param linear readout.
Measured, it doesn't: the trained gain-prior run parks at amplitude-match ESR
(≈0.045 A-wt) while the from-scratch SOTA reaches 0.0025. A memoryless
transfer curve W(s) = s + a₂s² + a₃s³ + … generates harmonics *phase-locked
and level-dependent by construction* — distortion appears exactly in phase
with the signal, scaling with instantaneous level — which is what VCA /
output-stage saturation physically is. The dynamics stay in the prior + Δg +
LSTM; W only carries the static curve (the Wiener–Hammerstein factoring).

    s = x · 10^((gr + Δg)/20)                 Δg = δmax·tanh(lin_g(h)), zero-init
    y = s + [ r(s) − r(0) ]  (+ c, optional)  r = tiny tanh MLP, zero-init out layer

Properties kept from the additive variant:

- **Identity at init**: r's output layer (and the optional FiLM) is zero-init,
  so at step 0 y == amplitude_match(x, gr) exactly — the notebook's zero-init
  assert passes unchanged.
- **W(0) = 0 always**: r(0) is subtracted through the *same* (FiLM-)conditioned
  path, so the curve can never inject DC or signal into silence (the analogue
  counterpart: output coupling). r's output Linear has ``bias=False`` — a bias
  would cancel in the subtraction anyway.
- **Drop-in for ``system_gainprior.GainPriorSystem``**: ``forward(x, gr, p,
  return_parts=True)`` returns ``(y, Δg, residual)`` with residual =
  (W(s)−s) + c, so the system's ``gain/*_color`` log reads "total non-gain
  output share". ``return_parts_full=True`` splits waveshaper vs additive.

Optional ``ws_film=True``: the waveshaper's first hidden layer is FiLM-
modulated from the LSTM state (zero-init → starts as the static curve),
letting the curve drift with programme / compression depth (bias-point
movement under heavy GR). A memoryless W cannot express distortion memory
(hysteresis, thermal drift) — by design; that belongs to the recurrent path.

Params: 8,418 static / 8,946 with FiLM (additive variant 8,322; SOTA 8k) —
still parameter-matched. ``model_gainprior.py`` is untouched; this is a
parallel variant, not a replacement.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nablafx.processors.components import TVFiLMCond

from amplitude_match import GR_DB_MAX, GR_DB_MIN, gr_db_to_gain
from model_tfilm import _reduction_positive


class Waveshaper(nn.Module):
    """W(s) = s + r(s) − r(0): learnable memoryless transfer curve, identity at init.

    r is a tiny tanh MLP on the instantaneous sample value; the hidden-layer
    biases break odd symmetry, so even harmonics are reachable. Optional FiLM
    on the first hidden layer conditions the curve on the LSTM state.
    """

    def __init__(self, hidden: int = 8, film_dim: int = 0):
        super().__init__()
        self.inp = nn.Linear(1, hidden)
        self.mid = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1, bias=False)  # bias cancels in r(s) − r(0)
        nn.init.zeros_(self.out.weight)
        self.film = nn.Linear(film_dim, 2 * hidden) if film_dim > 0 else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def _r(self, s_bsc: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
        # s_bsc: [B, S, 1]   h: [B, S, film_dim] or None
        z = torch.tanh(self.inp(s_bsc))
        if self.film is not None and h is not None:
            gamma, beta = self.film(h).chunk(2, dim=-1)
            z = (1.0 + gamma) * z + beta
        return self.out(torch.tanh(self.mid(z)))

    def forward(self, s: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        # s: [B, 1, S] → pointwise transfer curve, W(0) = 0 guaranteed
        s_t = s.transpose(1, 2)
        r = self._r(s_t, h) - self._r(torch.zeros_like(s_t), h)
        return s + r.transpose(1, 2)


class GainPriorWSDiffSSLLSTM(nn.Module):
    """diffssl LSTM(tvcond) gain-prior core + waveshaper coloration stage."""

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
        use_color: bool = True,       # keep the additive head too (ablation flag)
        # -- waveshaper --
        ws_hidden: int = 8,           # MLP width of the transfer curve
        ws_film: bool = False,        # FiLM the curve from the LSTM state
    ):
        super().__init__()
        self.num_controls = num_controls
        self.hidden_size = hidden_size
        self.cond_block_size = cond_block_size
        self.delta_max_db = delta_max_db
        self.use_color = use_color
        self.ws_film = ws_film

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

        # Δg head (dB, bounded via tanh), additive coloration head, waveshaper.
        self.lin_gain = nn.Linear(hidden_size, 1)
        self.lin_color = nn.Linear(hidden_size, 1)
        self.waveshaper = Waveshaper(
            hidden=ws_hidden, film_dim=hidden_size if ws_film else 0
        )
        self._init_zero_heads()

    def _init_zero_heads(self) -> None:
        """Zero-init both linear heads (the waveshaper zero-inits itself):
        at step 0, y == amplitude_match(x, gr) exactly."""
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
        return_parts_full: bool = False,
    ):
        # x: [B, 1, S] raw dry    gr: [B, 1, S] GR curve (dB)    p: [B, num_controls]
        s_len = x.shape[-1]

        # -- static-knob conditioning (diffssl tvcond, unchanged) --
        cond = self.cond_nn(x, p)                                # [B, 16, nsteps]
        cond = cond.repeat_interleave(self.cond_block_size, dim=-1)[..., :s_len]

        # -- gain prior --
        gr_c = gr.clamp(GR_DB_MIN, GR_DB_MAX)
        matched = x * gr_db_to_gain(gr_c, clamp=False)           # amplitude-matched
        gr_pos = _reduction_positive(gr)                         # ~[0,1], compression-positive

        # -- main LSTM over (dry, matched, GR, cond) --
        h = torch.cat((x, matched, gr_pos, cond), dim=1).transpose(1, 2)  # [B, S, 19]
        h, self.main_state = self.lstm(h, self.main_state)       # [B, S, H]

        # -- gain correction (zero-init) + gained signal --
        delta_db = self.delta_max_db * torch.tanh(self.lin_gain(h))       # [B, S, 1]
        delta_db = delta_db.transpose(1, 2)                               # [B, 1, S]
        gained = x * gr_db_to_gain(gr_c + delta_db, clamp=False)

        # -- coloration: waveshape the gained signal (+ optional additive head) --
        shaped = self.waveshaper(gained, h if self.ws_film else None)
        ws_res = shaped - gained
        color = self.lin_color(h).transpose(1, 2) if self.use_color else torch.zeros_like(x)

        y = shaped + color
        if return_parts_full:
            return y, delta_db, color, ws_res
        if return_parts:
            return y, delta_db, ws_res + color
        return y

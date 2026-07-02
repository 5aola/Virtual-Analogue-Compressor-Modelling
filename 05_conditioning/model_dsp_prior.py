"""DSP-prior GR predictor: differentiable compressor gain computer + LSTM correction.

Replaces the bins recipe (`model.StatefulCondLSTMGRBins`) after the measured
val→test generalization gap (0.267 → 0.554 dB MAE): the raw-waveform conv
frontend can encode song-specific timbre, and the bins/LDS/CFG machinery
patched training-dynamics symptoms of that representation problem. Here the
frontend is constrained to the *detector family* (rectify → smooth → dB) —
phase- and timbre-blind by construction — and the knobs act through a physical
gain computer instead of a learned feature modulation:

    dry ─ x² ─ frame energy (hop 256) ─ one-pole detector bank (learnable τ) ─ dB
                                        │ ch 0 = prior level L
    knobs ─ analytic denorm + zero-init MLP correction → (T̂, R̂, knee, τ̂a, τ̂r)
                                        │
    g_static = soft-knee gain computer(L; T̂, R̂, knee)     [Giannoulis 2012]
    gr_dsp   = attack/release one-pole ballistics(g_static; τ̂a, τ̂r)
                                        │
    LSTM( envs ⊕ gr_dsp ⊕ knobs ) → zero-init head → Δgr = δmax·tanh(·)
    gr = gr_dsp + Δgr

Same philosophy as `06_output/model_gainprior.py` one level up: the DSP path
is initialized at the front-panel knob values (the setting folder names *are*
the physical values), all learned corrections start at zero, so step 0 already
equals a textbook feed-forward compressor's GR and training can only improve.
The MLP corrections are learnable precisely because the analogue unit is not
that textbook compressor (feedback topology, detector calibration, program
dependence) — the corrections and Δgr absorb the difference.

The detector one-poles are exact exponential smoothers implemented as causal
depthwise convs with kernels built from the learnable τ each forward (no scan;
differentiable in τ). Only the ballistics stage is a true frame-rate scan —
at hop 256 that is ~172 steps/s of audio, negligible.

Sub-frame attack values (1 ms < 5.8 ms frame period) saturate the ballistics
coefficient toward an instant attack at frame resolution — consistent with the
GR labels themselves, which come from a 1024-sample RMS window and cannot
resolve faster transitions either.

Stateless by design: `forward` takes and returns explicit state
(lstm_state, ballistics y, detector context tail), so training uses fresh
state per crop (cold-start-realistic) and eval streams whole songs chunked.
No module-held recurrent state, no reset/stash bookkeeping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gr_target import GR_DB_MAX, GR_DB_MIN, normalize_gr_01
from splits import DIFFSSL_PARAM_RANGES
from src.dsp import PARAM_ORDER

ENV_DB_FLOOR = -80.0  # envelope dB clamp / input normalisation floor


class OnePoleDetectorBank(nn.Module):
    """N exact one-pole energy smoothers with learnable time constants.

    Implemented as a causal depthwise conv over frame energy: for coefficient
    a = exp(-Δt/τ) the impulse response (1-a)·a^j is truncated at
    `kernel_frames` and renormalized to unit DC gain, so the conv equals the
    IIR one-pole up to the truncation tail. Differentiable in τ, no scan.
    """

    def __init__(
        self,
        taus_ms: tuple[float, ...] = (12.0, 2.0, 40.0, 150.0),
        hop_size: int = 256,
        sample_rate: int = 44100,
        kernel_frames: int = 256,
    ):
        super().__init__()
        self.num_detectors = len(taus_ms)
        self.kernel_frames = kernel_frames
        self.dt = hop_size / sample_rate
        self.log_tau = nn.Parameter(torch.log(torch.tensor(taus_ms) / 1000.0))

    @property
    def taus_ms(self) -> torch.Tensor:
        return self.log_tau.detach().exp() * 1000.0

    def kernels(self) -> torch.Tensor:
        tau = self.log_tau.exp().clamp(5e-4, 0.5)                    # 0.5 ms … 500 ms
        a = torch.exp(-self.dt / tau)                                # [N]
        j = torch.arange(self.kernel_frames, device=a.device, dtype=a.dtype)
        k = (1.0 - a[:, None]) * a[:, None] ** j                     # [N, K] newest→oldest
        k = k / k.sum(dim=-1, keepdim=True)                          # fix truncation
        return k.flip(-1).unsqueeze(1)                               # conv kernel, causal

    def forward(self, energy_padded: torch.Tensor) -> torch.Tensor:
        # energy_padded: [B, 1, K-1+T] (caller prepends K-1 context frames)
        return F.conv1d(energy_padded, self.kernels())               # [B, N, T]


class KnobToDSPParams(nn.Module):
    """Normalized knobs → physical (threshold dB, ratio, knee dB, τ_att s, τ_rel s).

    The analytic denormalization inverts `splits.normalize_setting_params`, so
    at init the DSP prior runs at the *exact* front-panel values. A zero-init
    MLP adds knob-dependent corrections (threshold additive in dB — its bias
    also absorbs the detector-level/dBu calibration offset; ratio and time
    constants multiplicative via exp, so they stay positive).
    """

    def __init__(self, hidden: int = 16, knee_db_init: float = 6.0):
        super().__init__()
        self.idx = {k: i for i, k in enumerate(PARAM_ORDER)}
        self.mlp = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, 5))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        self.knee_base = nn.Parameter(torch.tensor(float(knee_db_init)))

    def _denorm(self, p: torch.Tensor, key: str) -> torch.Tensor:
        lo, hi = DIFFSSL_PARAM_RANGES[key]
        return p[:, self.idx[key]] * (hi - lo) + lo

    def forward(self, p: torch.Tensor) -> dict[str, torch.Tensor]:
        # p: [B, 4] normalized knobs -> dict of [B] physical DSP params
        d = self.mlp(p)                                              # [B, 5], zero at init
        return {
            "threshold_db": self._denorm(p, "threshold") + 10.0 * d[:, 0],
            "att_s": (self._denorm(p, "attack") * 1e-3).clamp(min=1e-4) * torch.exp(d[:, 1]),
            "rel_s": self._denorm(p, "release").clamp(min=5e-3) * torch.exp(d[:, 2]),
            "ratio": self._denorm(p, "ratio").clamp(min=1.2) * torch.exp(d[:, 3]),
            "knee_db": F.softplus(self.knee_base + d[:, 4]) + 0.1,
        }


def gain_computer(
    level_db: torch.Tensor,
    threshold_db: torch.Tensor,
    ratio: torch.Tensor,
    knee_db: torch.Tensor,
) -> torch.Tensor:
    """Soft-knee static compression gain (Giannoulis et al. 2012, eq. 4).

    level_db: [B, 1, T]; knob tensors: [B]. Returns gain in dB (≤ 0).
    """
    t = threshold_db[:, None, None]
    w = knee_db[:, None, None]
    slope = (1.0 / ratio[:, None, None]) - 1.0                       # ≤ 0
    o = level_db - t                                                 # overshoot
    knee = slope * (o + w / 2).pow(2) / (2 * w)
    g = torch.where(o >= w / 2, slope * o, knee)
    return torch.where(o <= -w / 2, torch.zeros_like(o), g)


class DSPPriorGRLSTM(nn.Module):
    """Frame-rate GR predictor: DSP compressor prior + zero-init LSTM correction."""

    def __init__(
        self,
        hop_size: int = 256,
        sample_rate: int = 44100,
        detector_taus_ms: tuple[float, ...] = (12.0, 2.0, 40.0, 150.0),
        detector_kernel_frames: int = 256,
        hidden_size: int = 32,
        num_controls: int = 4,
        delta_max_db: float = 6.0,
        knee_db_init: float = 6.0,
        knob_hidden: int = 16,
    ):
        super().__init__()
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.delta_max_db = delta_max_db
        self.dt = hop_size / sample_rate

        self.detector = OnePoleDetectorBank(
            taus_ms=detector_taus_ms,
            hop_size=hop_size,
            sample_rate=sample_rate,
            kernel_frames=detector_kernel_frames,
        )
        self.knob_map = KnobToDSPParams(hidden=knob_hidden, knee_db_init=knee_db_init)
        self.lstm = nn.LSTM(
            self.detector.num_detectors + 1 + num_controls,
            hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _ballistics(
        self,
        g_static: torch.Tensor,
        att_s: torch.Tensor,
        rel_s: torch.Tensor,
        y0: torch.Tensor,
    ) -> torch.Tensor:
        # g_static: [B,1,T] dB; att/rel: [B] seconds; y0: [B,1,1] dB
        alpha_a = (1.0 - torch.exp(-self.dt / att_s))[:, None, None]
        alpha_r = (1.0 - torch.exp(-self.dt / rel_s))[:, None, None]
        y, ys = y0, []
        for t in range(g_static.shape[-1]):
            g = g_static[..., t : t + 1]
            a = torch.where(g < y, alpha_a, alpha_r)   # deeper reduction -> attack
            y = y + a * (g - y)
            ys.append(y)
        return torch.cat(ys, dim=-1)

    def forward(
        self,
        dry: torch.Tensor,
        params: torch.Tensor,
        state: tuple | None = None,
        return_state: bool = False,
        return_parts: bool = False,
    ):
        # dry: [B, 1, L]   params: [B, 4]
        # state: (lstm_state, ballistics y [B,1,1], energy tail [B,1,K-1]) or None
        energy = F.avg_pool1d(dry * dry, self.hop_size)              # [B, 1, T]
        K = self.detector.kernel_frames
        if state is None:
            lstm_state = None
            y0 = energy.new_zeros(energy.shape[0], 1, 1)             # 0 dB = no reduction
            tail = energy.new_zeros(energy.shape[0], 1, K - 1)       # silence context
        else:
            lstm_state, y0, tail = state

        energy_padded = torch.cat((tail, energy), dim=-1)
        envs_db = 10.0 * torch.log10(self.detector(energy_padded) + 1e-10)
        envs_db = envs_db.clamp(min=ENV_DB_FLOOR)                    # [B, N, T]

        dsp = self.knob_map(params)
        g_static = gain_computer(
            envs_db[:, :1], dsp["threshold_db"], dsp["ratio"], dsp["knee_db"]
        )
        gr_dsp = self._ballistics(g_static, dsp["att_s"], dsp["rel_s"], y0)

        envs_norm = (envs_db - ENV_DB_FLOOR) / -ENV_DB_FLOOR         # ~[0, 1]
        knobs = params.unsqueeze(-1).expand(-1, -1, gr_dsp.shape[-1])
        x = torch.cat((envs_norm, normalize_gr_01(gr_dsp), knobs), dim=1)
        h, lstm_state = self.lstm(x.transpose(1, 2), lstm_state)
        delta = self.delta_max_db * torch.tanh(self.head(h)).transpose(1, 2)
        gr = gr_dsp + delta                                          # [B, 1, T] dB

        out = (gr,)
        if return_parts:
            out += (gr_dsp, delta)
        if return_state:
            new_state = (
                lstm_state,
                gr_dsp[..., -1:].detach(),
                energy_padded[..., -(K - 1):].detach(),
            )
            out += (new_state,)
        return out[0] if len(out) == 1 else out

    def to_db(self, gr: torch.Tensor, sample_len: int | None = None) -> torch.Tensor:
        """Frame-rate GR (already dB) → optional sample-rate interpolation.

        Keeps the eval-notebook interface of the previous models. Clamped to
        the range `06_output/amplitude_match.py` expects downstream.
        """
        db = gr.clamp(GR_DB_MIN, GR_DB_MAX)
        if sample_len is not None:
            db = F.interpolate(db, size=sample_len, mode="linear", align_corners=False)
        return db

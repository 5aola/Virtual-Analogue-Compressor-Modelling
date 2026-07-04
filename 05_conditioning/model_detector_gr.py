"""Blackbox detector-LSTM GR predictor on level-domain features.

Replaces the bins recipe (`model.StatefulCondLSTMGRBins`) after the measured
val→test generalization gap (0.267 → 0.554 dB MAE): the raw-waveform conv
frontend can encode song-specific timbre, and the bins/LDS/CFG machinery
patched training-dynamics symptoms of that representation problem.

Exactly one structural constraint is kept — the frontend lives in the
*detector family* (rectify → smooth → dB), which is phase- and timbre-blind
by construction. Everything after the frontend is blackbox: the GR target is
itself a synthetic quantity extracted from the dataset (RMS-1024 wet/dry
ratio), so no compressor gain law is assumed, and the head regresses GR in
dB directly (no bins, no decode step).

    dry ─ x² ─ frame energy (hop 256) ─ one-pole bank (8 fixed log-spaced τ) ─ dB
    envelopes → LSTM → FiLM(knobs, poly order 3) → GLU → Linear → gr  [B, 1, T] dB

Two v2 revisions over the 20260702 knob-concat run:

- **Fixed, wider detector bank.** With learnable τ the bank collapsed onto
  the label's own RMS-1024 window (init (2, 12, 40, 150) ms → learned
  (14.6, 15.4, 19.1, 74.8) ms), leaving no envelope at the release timescale
  (0.4–0.8 s). The default bank is now 8 FIXED log-spaced τ spanning
  2–800 ms (`learnable` stays available on `OnePoleDetectorBank` for the
  greybox ablation).
- **FiLM+GLU conditioning replaces plain knob concat.** The Comparative
  Study (EURASIP 2025) conditions every model — including CL1B with the
  same 4 knobs (threshold/ratio/attack/release) — through a single FiLM+GLU
  layer after the recurrent block; plain concat is the Optical-DRC
  *baseline*, the weakest variant in that literature. FiLM here is
  polynomial (`h^order · g + b`, Conditioning-Methods SMC 2024: cubic beats
  linear for knob-space interpolation; odd order preserves sign), followed
  by a GLU (`out · softsign(gate)`). Knobs no longer enter the LSTM input.

The detector one-poles are exact exponential smoothers implemented as causal
depthwise convs with kernels built from τ each forward (differentiable in τ
when learnable, no scan) — the model is fully parallel.

Stateless by design: `forward` takes and returns explicit state
(lstm_state, detector context tail), so training uses fresh state per crop
(cold-start-realistic) and eval streams whole songs chunked with exact
equivalence to one full forward. No module-held recurrent state.

`model_dsp_prior.DSPPriorGRLSTM` is the greybox ablation of this model (same
frontend, plus a physical gain-computer prior).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gr_target import GR_DB_MAX, GR_DB_MIN

ENV_DB_FLOOR = -80.0  # envelope dB clamp / input normalisation floor


class OnePoleDetectorBank(nn.Module):
    """N exact one-pole energy smoothers with fixed or learnable time constants.

    Implemented as a causal depthwise conv over frame energy: for coefficient
    a = exp(-Δt/τ) the impulse response (1-a)·a^j is truncated at
    `kernel_frames` and renormalized to unit DC gain, so the conv equals the
    IIR one-pole up to the truncation tail. Differentiable in τ when
    `learnable`, no scan.

    `learnable=False` freezes the bank: learnable τ collapse onto the GR
    label's RMS-1024 analysis window (~23 ms) and discard slow context —
    measured in run 20260702_174039. A fixed log-spaced bank keeps detectors
    at the release timescale.
    """

    def __init__(
        self,
        taus_ms: tuple[float, ...] = (12.0, 2.0, 40.0, 150.0),
        hop_size: int = 256,
        sample_rate: int = 44100,
        kernel_frames: int = 256,
        learnable: bool = True,
    ):
        super().__init__()
        self.num_detectors = len(taus_ms)
        self.kernel_frames = kernel_frames
        self.dt = hop_size / sample_rate
        log_tau = torch.log(torch.tensor(taus_ms, dtype=torch.float32) / 1000.0)
        if learnable:
            self.log_tau = nn.Parameter(log_tau)
        else:
            self.register_buffer("log_tau", log_tau)

    @property
    def taus_ms(self) -> torch.Tensor:
        return self.log_tau.detach().exp() * 1000.0

    def kernels(self) -> torch.Tensor:
        tau = self.log_tau.exp().clamp(5e-4, 1.0)                    # 0.5 ms … 1 s
        a = torch.exp(-self.dt / tau)                                # [N]
        j = torch.arange(self.kernel_frames, device=a.device, dtype=a.dtype)
        k = (1.0 - a[:, None]) * a[:, None] ** j                     # [N, K] newest→oldest
        k = k / k.sum(dim=-1, keepdim=True)                          # fix truncation
        return k.flip(-1).unsqueeze(1)                               # conv kernel, causal

    def forward(self, energy_padded: torch.Tensor) -> torch.Tensor:
        # energy_padded: [B, 1, K-1+T] (caller prepends K-1 context frames)
        return F.conv1d(energy_padded, self.kernels())               # [B, N, T]


class DetectorGRLSTM(nn.Module):
    """Frame-rate blackbox GR predictor: fixed detector bank → LSTM → FiLM+GLU."""

    def __init__(
        self,
        hop_size: int = 256,
        sample_rate: int = 44100,
        detector_taus_ms: tuple[float, ...] = (2.0, 5.0, 11.0, 26.0, 61.0, 144.0, 340.0, 800.0),
        detector_kernel_frames: int = 512,
        detector_learnable: bool = False,
        hidden_size: int = 32,
        num_controls: int = 4,
        film_order: int = 3,
    ):
        super().__init__()
        assert film_order >= 1 and film_order % 2 == 1, "odd film_order preserves feature sign"
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.film_order = film_order

        self.detector = OnePoleDetectorBank(
            taus_ms=detector_taus_ms,
            hop_size=hop_size,
            sample_rate=sample_rate,
            kernel_frames=detector_kernel_frames,
            learnable=detector_learnable,
        )
        self.lstm = nn.LSTM(
            self.detector.num_detectors,
            hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.film = nn.Linear(num_controls, 2 * hidden_size)   # knobs → (g, b)
        self.glu = nn.Linear(hidden_size, 2 * hidden_size)     # → (out, gate)
        self.head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        dry: torch.Tensor,
        params: torch.Tensor,
        state: tuple | None = None,
        return_state: bool = False,
    ):
        # dry: [B, 1, L]   params: [B, 4]
        # state: (lstm_state, energy tail [B, 1, K-1]) or None
        energy = F.avg_pool1d(dry * dry, self.hop_size)              # [B, 1, T]
        K = self.detector.kernel_frames
        if state is None:
            lstm_state = None
            tail = energy.new_zeros(energy.shape[0], 1, K - 1)       # silence context
        else:
            lstm_state, tail = state

        energy_padded = torch.cat((tail, energy), dim=-1)
        envs_db = 10.0 * torch.log10(self.detector(energy_padded) + 1e-10)
        envs_db = envs_db.clamp(min=ENV_DB_FLOOR)                    # [B, N, T]

        envs_norm = (envs_db - ENV_DB_FLOOR) / -ENV_DB_FLOOR         # ~[0, 1]
        h, lstm_state = self.lstm(envs_norm.transpose(1, 2), lstm_state)

        g, b = self.film(params).unsqueeze(1).chunk(2, dim=-1)      # [B, 1, H] each
        h = h.pow(self.film_order) * g + b                           # polynomial FiLM
        out, gate = self.glu(h).chunk(2, dim=-1)
        h = out * F.softsign(gate)                                   # GLU
        gr = self.head(h).transpose(1, 2)                            # [B, 1, T] dB

        if return_state:
            return gr, (lstm_state, energy_padded[..., -(K - 1):].detach())
        return gr

    def to_db(self, gr: torch.Tensor, sample_len: int | None = None) -> torch.Tensor:
        """Frame-rate GR (already dB) → optional sample-rate interpolation.

        Keeps the eval-notebook interface of the previous models. Clamped to
        the range `06_output/amplitude_match.py` expects downstream.
        """
        db = gr.clamp(GR_DB_MIN, GR_DB_MAX)
        if sample_len is not None:
            db = F.interpolate(db, size=sample_len, mode="linear", align_corners=False)
        return db

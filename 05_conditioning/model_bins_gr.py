"""Bins-head GR predictor — the output-head ablation (O1) of `DetectorGRLSTM`.

This is a deliberate ablation twin of `model_detector_gr.DetectorGRLSTM`.
The frontend (fixed 8-τ one-pole detector bank), LSTM, polynomial-FiLM+GLU
conditioning, split policy, crop regime and streaming/state contract are all
byte-for-byte the same recipe. The ONLY thing that changes is the output head:

    DetectorGRLSTM :  … ─ FiLM ─ GLU ─ Linear(32→1)      → gr in dB (regression)
    BinsGRLSTM     :  … ─ FiLM ─ GLU ─ Linear(32→71)     → CREPE-style logits over
                                                            [GR_DB_MIN, GR_DB_MAX], 0.5 dB bins

Why it exists: the predecessor bins model (`model.StatefulCondLSTMGRBins`)
showed a 0.267 → 0.554 dB val→test MAE gap, but it changed FOUR things at
once relative to the current base — a learned raw-waveform conv frontend,
knob-concat + TFiLM conditioning, the discretized head, AND the LDS/CFG/
cold-start training machinery — so nothing can be attributed to the head.
This model isolates the head: detector frontend and FiLM+GLU identical to the
base, trained by `system_bins_gr.BinsGRSystem` with the *clean* bins loss
(BCE on Gaussian soft targets + the base recipe's Huber/Δ terms on the
differentiable soft-argmax decode) — **no LDS depth reweighting, no
conditioning dropout, no cold-start mixing**. If this cell matches the
regression base, bins were never the problem (or the benefit); if it degrades,
the discretized head costs accuracy on its own.

Decode: training uses the differentiable softmax-expectation decode; eval
(`to_db`) uses the robust local-average decode around the argmax bin, exactly
like the predecessor, so the comparison to the old bins numbers stays fair.

State contract matches `DetectorGRLSTM`: `forward(dry, params, state,
return_state)` with `state = (lstm_state, energy tail)`. Cold start = zeros
= silence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gr_target import GR_DB_MAX, GR_DB_MIN, NUM_BINS, logits_to_local_avg_db
from model_detector_gr import ENV_DB_FLOOR, OnePoleDetectorBank


class BinsGRLSTM(nn.Module):
    """Frame-rate GR predictor: fixed detector bank → LSTM → FiLM+GLU → bins logits."""

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
        num_bins: int = NUM_BINS,
    ):
        super().__init__()
        assert film_order >= 1 and film_order % 2 == 1, "odd film_order preserves feature sign"
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.film_order = film_order
        self.num_bins = num_bins

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
        self.head = nn.Linear(hidden_size, num_bins)           # the ONLY change vs DetectorGRLSTM

    def forward(
        self,
        dry: torch.Tensor,
        params: torch.Tensor,
        state: tuple | None = None,
        return_state: bool = False,
    ):
        # dry: [B, 1, L]   params: [B, 4]   ->   logits: [B, num_bins, T]
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
        logits = self.head(h).transpose(1, 2)                        # [B, num_bins, T]

        if return_state:
            return logits, (lstm_state, energy_padded[..., -(K - 1):].detach())
        return logits

    def to_db(self, logits: torch.Tensor, sample_len: int | None = None) -> torch.Tensor:
        """Bins logits → dB via the robust local-average decode (+ optional upsample).

        Same eval decode as the predecessor bins model; clamped to the range
        `06_output/amplitude_match.py` expects downstream.
        """
        db = logits_to_local_avg_db(logits).clamp(GR_DB_MIN, GR_DB_MAX)
        if sample_len is not None:
            db = F.interpolate(db, size=sample_len, mode="linear", align_corners=False)
        return db

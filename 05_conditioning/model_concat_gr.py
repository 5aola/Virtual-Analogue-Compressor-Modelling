"""Knob-concat GR predictor — the conditioning-axis ablation (C1) of `DetectorGRLSTM`.

This is a deliberate ablation twin of `model_detector_gr.DetectorGRLSTM`.
The frontend (fixed 8-τ one-pole detector bank), hidden size, loss/system
(`system_detector_gr.DetectorGRSystem`), split policy, crop regime and
streaming/state contract are all byte-for-byte the same recipe. The ONLY
thing that changes is the conditioning mechanism:

    DetectorGRLSTM :  envelopes(8) ─────────────→ LSTM(8→32)  ─ FiLM(knobs, poly 3) ─ GLU ─ Linear ─ gr   (~7.8k params)
    ConcatGRLSTM   :  [envelopes(8) ‖ knobs(4)] → LSTM(12→32) ─ Linear ─ gr                               (~5.9k params)

Why it exists: plain knob concat at the recurrent input is the *baseline*
conditioning of the Optical-DRC literature (JAES 2025), while a single
polynomial-FiLM+GLU layer after the recurrent block is the Comparative-Study
recipe (EURASIP 2025) that every model there uses — including CL1B with the
same 4 knobs. This run measures that axis on our task and split instead of
only citing it. The earlier knob-concat run (`lstm_gr_20260702_174039`) is
NOT usable as this cell of the ablation: its split was contaminated
(test_ground_truth pairs inside train/val) and its detector bank was
learnable; this model reruns concat under the honest v2 regime.

Design note: no FiLM/GLU stand-in is inserted after the LSTM — concat-at-input
IS the published baseline (recurrent core → linear head), so this variant
carries slightly fewer parameters than the FiLM twin (~5.9k vs ~7.8k). That
is a property of the recipe, noted rather than compensated; the knobs instead
gain direct access to the recurrent dynamics, which the FiLM variant denies.

State contract matches `DetectorGRLSTM`: `forward(dry, params, state,
return_state)` with `state = (lstm_state, energy tail)`. Cold start = zeros
= silence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gr_target import GR_DB_MAX, GR_DB_MIN
from model_detector_gr import ENV_DB_FLOOR, OnePoleDetectorBank


class ConcatGRLSTM(nn.Module):
    """Frame-rate blackbox GR predictor: fixed detector bank → [envs ‖ knobs] → LSTM → Linear."""

    def __init__(
        self,
        hop_size: int = 256,
        sample_rate: int = 44100,
        detector_taus_ms: tuple[float, ...] = (2.0, 5.0, 11.0, 26.0, 61.0, 144.0, 340.0, 800.0),
        detector_kernel_frames: int = 512,
        detector_learnable: bool = False,
        hidden_size: int = 32,
        num_controls: int = 4,
    ):
        super().__init__()
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.num_controls = num_controls

        self.detector = OnePoleDetectorBank(
            taus_ms=detector_taus_ms,
            hop_size=hop_size,
            sample_rate=sample_rate,
            kernel_frames=detector_kernel_frames,
            learnable=detector_learnable,
        )
        self.lstm = nn.LSTM(
            self.detector.num_detectors + num_controls,
            hidden_size,
            num_layers=1,
            batch_first=True,
        )
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
        x = envs_norm.transpose(1, 2)                                # [B, T, N]
        knobs = params.unsqueeze(1).expand(-1, x.shape[1], -1)       # [B, T, 4]
        h, lstm_state = self.lstm(torch.cat((x, knobs), dim=-1), lstm_state)
        gr = self.head(h).transpose(1, 2)                            # [B, 1, T] dB

        if return_state:
            return gr, (lstm_state, energy_padded[..., -(K - 1):].detach())
        return gr

    def to_db(self, gr: torch.Tensor, sample_len: int | None = None) -> torch.Tensor:
        """Frame-rate GR (already dB) → optional sample-rate interpolation.

        Identical to `DetectorGRLSTM.to_db` — clamps to the range
        `06_output/amplitude_match.py` expects and (optionally) linearly
        interpolates the frame-rate curve up to sample rate.
        """
        db = gr.clamp(GR_DB_MIN, GR_DB_MAX)
        if sample_len is not None:
            db = F.interpolate(db, size=sample_len, mode="linear", align_corners=False)
        return db

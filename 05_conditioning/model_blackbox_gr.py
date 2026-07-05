"""Blackbox-frontend GR predictor — the learned-frontend control for `DetectorGRLSTM`.

This is the deliberate ablation twin of `model_detector_gr.DetectorGRLSTM`.
Everything downstream of the frontend is byte-for-byte the same recipe
(single-layer LSTM → polynomial FiLM(knobs) → GLU → Linear → gr in dB, same
hidden size, same conditioning, same streaming/state contract, same loss via
`system_detector_gr.DetectorGRSystem`, same split and crop regime). The ONLY
thing that changes is the frontend:

    DetectorGRLSTM  :  dry ─ x² ─ pool256 ─ 8 FIXED one-pole τ ─ dB ─ norm  (0 learned params, timbre-blind)
    BlackboxGRLSTM  :  dry ─ learned Conv frame-encoder ─ dilated temporal Conv stack   (learned, timbre-CAPABLE)

Why it exists: the detector frontend imposes exactly one inductive bias — the
level-detector family (rectify → smooth → dB) is phase- and timbre-blind, so
song identity cannot leak into the GR estimate through timbre. The predecessor
bins model used a *learned* raw-waveform conv frontend and showed a 0.267 → 0.554 dB
val→test MAE gap, diagnosed as that exact leak. This model reinstates a learned
frontend (with none of the bins/LDS/CFG machinery, so the only difference from
the detector run is representational) to *measure* the gap directly under the
clean external-test split, instead of arguing it. If this model reproduces the
val→test degradation and the detector model doesn't, the structural frontend is
justified empirically; if it matches the detector, the prior is unnecessary.

Frontend design (kept minimal and, if anything, generous to the blackbox):

- **Learned frame encoder.** `Conv1d(1 → enc_channels, kernel=hop, stride=hop)`
  — a learned, hop-aligned replacement for `x² → avg_pool`. No squaring: the
  model must discover that level (not phase) is what the target depends on. With
  stride == kernel each frame is a function of its own `hop` samples only, so
  there is no cross-chunk dependency and streaming needs no input-sample tail.
  256 samples/frame is ample to encode within-frame spectral shape → this is the
  timbre channel the detector frontend removes by construction.
- **Dilated causal temporal stack.** A few `Conv1d(enc → enc, kernel=k, dilation=d)`
  at frame rate build multi-frame context so the blackbox can form multi-scale
  level estimates comparable to the detector bank's 2–800 ms smoothers. Causal
  via carried left-context (no future frames), padding done by the state tail so
  chunked streaming is exact vs a full forward.
- **1×1 projection** to `frontend_channels` (default matches the detector bank
  width, so the LSTM and everything after it are *identical* in shape and param
  count to `DetectorGRLSTM`).

State contract matches `DetectorGRLSTM`: `forward(dry, params, state, return_state)`
with `state = (lstm_state, frame_context)`, `frame_context` = last
`temporal_context` frame-encoder frames (detached). Cold start = zeros = silence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gr_target import GR_DB_MAX, GR_DB_MIN


class BlackboxGRLSTM(nn.Module):
    """Learned-frontend GR predictor: Conv frame-encoder → dilated temporal Conv → LSTM → FiLM+GLU.

    Drop-in twin of `DetectorGRLSTM` (identical interface, downstream stack, and
    `hop_size`) whose frontend is fully learned instead of a fixed detector bank.
    """

    def __init__(
        self,
        hop_size: int = 256,
        sample_rate: int = 44100,
        enc_channels: int = 16,
        frontend_channels: int = 8,          # == DetectorGRLSTM's detector-bank width → identical LSTM
        temporal_kernel: int = 4,
        temporal_dilations: tuple[int, ...] = (1, 4, 16, 64),  # RF 256 frames ≈ 1.49 s, brackets 2–800 ms
        hidden_size: int = 32,
        num_controls: int = 4,
        film_order: int = 3,
    ):
        super().__init__()
        assert film_order >= 1 and film_order % 2 == 1, "odd film_order preserves feature sign"
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.film_order = film_order

        # -- learned frontend (the ONLY difference vs DetectorGRLSTM) --
        self.frame_encoder = nn.Conv1d(1, enc_channels, kernel_size=hop_size, stride=hop_size)
        self.temporal = nn.ModuleList(
            nn.Conv1d(enc_channels, enc_channels, kernel_size=temporal_kernel, dilation=d)
            for d in temporal_dilations
        )
        # left-context frames the causal temporal stack consumes (== its RF - 1)
        self.temporal_context = sum((temporal_kernel - 1) * d for d in temporal_dilations)
        self.proj = nn.Conv1d(enc_channels, frontend_channels, kernel_size=1)
        self.act = nn.GELU()

        # -- downstream: identical to DetectorGRLSTM --
        self.lstm = nn.LSTM(frontend_channels, hidden_size, num_layers=1, batch_first=True)
        self.film = nn.Linear(num_controls, 2 * hidden_size)   # knobs → (g, b)
        self.glu = nn.Linear(hidden_size, 2 * hidden_size)     # → (out, gate)
        self.head = nn.Linear(hidden_size, 1)

    def _frontend(self, dry: torch.Tensor, ctx: torch.Tensor):
        """dry [B,1,L] + prior frame context [B,C,ctx] → features [B,front,T], new context.

        Frame encoder is hop-aligned (stride == kernel) so each frame depends
        only on its own hop samples — no input-sample tail needed. The temporal
        stack's causal left-context comes entirely from `ctx` (zeros at cold
        start), padding=0 in the convs, so output length == number of frames T
        and chunked streaming is exact.
        """
        f = self.act(self.frame_encoder(dry))          # [B, C, T]
        fp = torch.cat((ctx, f), dim=-1)               # [B, C, temporal_context + T]
        new_ctx = fp[..., -self.temporal_context:].detach() if self.temporal_context else ctx
        for conv in self.temporal:
            fp = self.act(conv(fp))                     # each: length -= (k-1)*dilation; sum == temporal_context
        return self.proj(fp), new_ctx                  # [B, frontend_channels, T]

    def forward(
        self,
        dry: torch.Tensor,
        params: torch.Tensor,
        state: tuple | None = None,
        return_state: bool = False,
    ):
        # dry: [B, 1, L]   params: [B, 4]
        # state: (lstm_state, frame_context [B, C, temporal_context]) or None
        if state is None:
            lstm_state = None
            ctx = dry.new_zeros(dry.shape[0], self.frame_encoder.out_channels, self.temporal_context)
        else:
            lstm_state, ctx = state

        z, new_ctx = self._frontend(dry, ctx)          # [B, frontend_channels, T]
        h, lstm_state = self.lstm(z.transpose(1, 2), lstm_state)

        g, b = self.film(params).unsqueeze(1).chunk(2, dim=-1)   # [B, 1, H] each
        h = h.pow(self.film_order) * g + b                        # polynomial FiLM
        out, gate = self.glu(h).chunk(2, dim=-1)
        h = out * F.softsign(gate)                                # GLU
        gr = self.head(h).transpose(1, 2)                         # [B, 1, T] dB

        if return_state:
            return gr, (lstm_state, new_ctx)
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

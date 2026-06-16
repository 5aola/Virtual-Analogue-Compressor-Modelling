"""Sample-rate LSTM **output transformer** for the amplitude-matched grey box.

Architecture — a faithful port of the Optical-DRC ``create_model_LSTM``
(Simionato et al., JAES 2025; see ``02b_sota_training``) with conditioning
removed and a single output layer, plus the PReLU conv front-end from the
``03_initial_GR_pred`` compact stateful model:

    amp-matched [B,1,S+w-1]
      → Conv1d(1→C, k=window, s=1) → PReLU      windowed encoder (sample rate)
        → LSTM(C→H)   ── state carried chunk→chunk (stateful TBPTT) ──┐
          → Linear(H→C2)                                              │
            → LSTM(C2→H) ── state carried chunk→chunk ────────────────┘
              → Linear(H→1)  → (output combiner)  → wet [B,1,S]

The **discretized 61-bin CREPE head of the GR model is dropped** — this is an
audio→audio regressor, so the head is a single ``Linear(H→1)`` plus an output
combiner. Because the input is already amplitude-matched (its RMS tracks the
wet to ≈2 %; the wet−input residual is only ~8 % RMS), the default combiner is
an **additive residual** ``out = current_input + Δ``: at init Δ≈0 so the model
starts at the strong amplitude-matched baseline and only learns the nonlinear
coloration. ``residual_gain`` (Optical-DRC native, ``out = current × g``) and
``direct`` are available for ablation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_ACTIVATIONS = {"tanh": nn.Tanh, "none": nn.Identity}


class OutputTransformerLSTM(nn.Module):
    def __init__(
        self,
        window: int = 64,
        encoder_channels: int = 8,
        hidden_size: int = 16,
        mid_channels: int = 8,
        num_lstm_layers: int = 2,
        output_mode: str = "residual_add",
        out_activation: str = "none",
    ):
        super().__init__()
        if output_mode not in ("residual_add", "residual_gain", "direct"):
            raise ValueError(f"unknown output_mode: {output_mode}")
        if num_lstm_layers not in (1, 2):
            raise ValueError("num_lstm_layers must be 1 or 2")
        self.window = window
        self.output_mode = output_mode
        self.num_lstm_layers = num_lstm_layers

        # stride=1 windowed conv: each step sees `window` samples (no downsampling)
        self.proj = nn.Conv1d(1, encoder_channels, kernel_size=window, stride=1)
        self.act = nn.PReLU(encoder_channels)
        self.lstm1 = nn.LSTM(encoder_channels, hidden_size, batch_first=True)
        if num_lstm_layers == 2:
            self.dense_mid = nn.Linear(hidden_size, mid_channels)
            self.lstm2 = nn.LSTM(mid_channels, hidden_size, batch_first=True)
        self.dense_out = nn.Linear(hidden_size, 1)
        self.out_activation = _ACTIVATIONS[out_activation]()
        self._init_output_baseline()

    def _init_output_baseline(self) -> None:
        """Start the residual modes exactly at the amplitude-matched baseline.

        Zero-init the output layer so at step 0 the correction vanishes and the
        model emits its input unchanged (``residual_add`` → out = current input;
        ``residual_gain`` → gain = 1). The net then only has to *learn the ~8 %
        coloration residual* on top of the validated matched signal, instead of
        also unlearning random output-layer junk (which otherwise corrupts
        silent regions — see the inflated untrained ESR). Assumes the default
        ``out_activation="none"`` for an exact identity; with ``tanh`` it is
        approximate. ``direct`` keeps the default init (no baseline prior)."""
        with torch.no_grad():
            if self.output_mode == "residual_add":
                nn.init.zeros_(self.dense_out.weight)
                nn.init.zeros_(self.dense_out.bias)
            elif self.output_mode == "residual_gain":
                nn.init.zeros_(self.dense_out.weight)
                nn.init.ones_(self.dense_out.bias)

    def forward(self, x: torch.Tensor, state=None, return_state: bool = False):
        # x: [B, 1, S + window - 1]  (amplitude-matched, with left context)
        cur = x[:, :, self.window - 1 :]            # [B, 1, S] aligned current sample
        h = self.act(self.proj(x)).transpose(1, 2)  # [B, S, C]
        s1, s2 = (None, None) if state is None else state
        h, s1 = self.lstm1(h, s1)                   # [B, S, H]
        if self.num_lstm_layers == 2:
            h = self.dense_mid(h)                   # [B, S, mid]
            h, s2 = self.lstm2(h, s2)               # [B, S, H]
        y = self.out_activation(self.dense_out(h)).transpose(1, 2)  # [B, 1, S]

        if self.output_mode == "residual_add":
            out = cur + y
        elif self.output_mode == "residual_gain":
            out = cur * y
        else:  # direct
            out = y
        return (out, (s1, s2)) if return_state else out

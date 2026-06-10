"""TFiLM-conditioned stateful LSTM for frame-rate GR prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


from nablafx.processors.components import TFiLM

from gr_target import NUM_BINS, denormalize_gr_01, logits_to_local_avg_db


class StatefulTFiLMLSTMGR(nn.Module):
    """Compact frame-rate GR LSTM with nablafx TFiLM conditioning.

    Architecture mirrors the Optical-DRC / nablafx recipe at frame rate:

        Conv stride=hop  →  LSTM  →  Linear  →  TFiLM(params)  →  1-ch head

    TFiLM block-wise modulates the LSTM features using the four compressor
    knobs (threshold, attack, release, ratio) normalised to [0, 1].
    Block size 8 frames = 8 × hop / sr ≈ 46 ms modulation rate.
  """

    def __init__(
        self,
        hop_size: int = 256,
        encoder_channels: int = 8,
        hidden_size: int = 24,
        tfilm_channels: int = 8,
        cond_dim: int = 4,
        tfilm_block_size: int = 8,
        tfilm_num_layers: int = 1,
    ):
        super().__init__()
        self.hop_size = hop_size
        self.tfilm_block_size = tfilm_block_size

        self.proj = nn.Conv1d(
            1, encoder_channels, kernel_size=hop_size, stride=hop_size
        )
        self.act = nn.PReLU(encoder_channels)
        self.lstm = nn.LSTM(
            encoder_channels, hidden_size, num_layers=1, batch_first=True
        )
        self.dense_mid = nn.Linear(hidden_size, tfilm_channels)
        self.tfilm = TFiLM(
            nfeatures=tfilm_channels,
            cond_dim=cond_dim,
            block_size=tfilm_block_size,
            num_layers=tfilm_num_layers,
        )
        self.head = nn.Conv1d(tfilm_channels, 1, kernel_size=1)

    def reset_cond_states(self) -> None:
        self.tfilm.reset_state()

    def forward(
        self,
        dry: torch.Tensor,
        params: torch.Tensor,
        state=None,
        return_state: bool = False,
    ):
        # dry: [B, 1, L]   params: [B, 4]
        x = self.act(self.proj(dry))  # [B, C, T]
        x = x.transpose(1, 2)  # [B, T, C]
        x, state = self.lstm(x, state)
        x = self.dense_mid(x).transpose(1, 2)  # [B, tfilm_ch, T]
        x = self.tfilm(x, params)
        gr_01 = torch.sigmoid(self.head(x))  # [B, 1, T] in (0, 1)
        if return_state:
            return gr_01, state
        return gr_01

    def to_db(self, gr_01: torch.Tensor, sample_len: int | None = None) -> torch.Tensor:
        db = denormalize_gr_01(gr_01)
        if sample_len is not None:
            db = F.interpolate(db, size=sample_len, mode="linear", align_corners=False)
        return db


class StatefulTFiLMLSTMGRBins(StatefulTFiLMLSTMGR):
    """Discretized variant: same trunk, head emits NUM_BINS CREPE-style logits
    over [GR_DB_MIN, GR_DB_MAX] (0.5 dB resolution) instead of a sigmoid scalar."""

    def __init__(
        self,
        hop_size: int = 256,
        encoder_channels: int = 8,
        hidden_size: int = 24,
        tfilm_channels: int = 8,
        cond_dim: int = 4,
        tfilm_block_size: int = 8,
        tfilm_num_layers: int = 1,
        num_bins: int = NUM_BINS,
    ):
        super().__init__(
            hop_size=hop_size,
            encoder_channels=encoder_channels,
            hidden_size=hidden_size,
            tfilm_channels=tfilm_channels,
            cond_dim=cond_dim,
            tfilm_block_size=tfilm_block_size,
            tfilm_num_layers=tfilm_num_layers,
        )
        self.num_bins = num_bins
        self.head = nn.Conv1d(tfilm_channels, num_bins, kernel_size=1)

    def forward(
        self,
        dry: torch.Tensor,
        params: torch.Tensor,
        state=None,
        return_state: bool = False,
    ):
        # dry: [B, 1, L]   params: [B, 4]   ->   logits: [B, num_bins, T]
        x = self.act(self.proj(dry))
        x = x.transpose(1, 2)
        x, state = self.lstm(x, state)
        x = self.dense_mid(x).transpose(1, 2)
        x = self.tfilm(x, params)
        logits = self.head(x)
        if return_state:
            return logits, state
        return logits

    def to_db(self, logits: torch.Tensor, sample_len: int | None = None) -> torch.Tensor:
        db = logits_to_local_avg_db(logits)
        if sample_len is not None:
            db = F.interpolate(db, size=sample_len, mode="linear", align_corners=False)
        return db

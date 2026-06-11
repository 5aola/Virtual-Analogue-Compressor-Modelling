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


class KnobEmbedding(nn.Module):
    """Diffusion-style scalar embedding: per-knob sinusoidal Fourier features
    (counters MLP spectral bias on raw [0,1] scalars, Tancik et al. 2020)
    followed by a small MLP."""

    def __init__(self, num_knobs: int = 4, num_freqs: int = 8, embed_dim: int = 16):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(num_freqs)) * torch.pi)
        self.mlp = nn.Sequential(
            nn.Linear(num_knobs * num_freqs * 2, embed_dim),
            nn.PReLU(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        # params: [B, num_knobs] in [0, 1]  ->  [B, embed_dim]
        x = params.unsqueeze(-1) * self.freqs
        return self.mlp(torch.cat((torch.sin(x), torch.cos(x)), dim=-1).flatten(1))


class StatefulCondLSTMGRBins(nn.Module):
    """Bins-head GR LSTM with the conditioning entering the recurrent core.

        knobs → Fourier embed ─┬─ concat → LSTM → Linear ─ TFiLM(embed) → bins
        dry → Conv stride=hop ─┘                            (optional)

    The knob embedding is concatenated to the LSTM input at every frame
    (nablafx cond_type="fixed"), so the recurrent dynamics are knob-aware;
    TFiLM (flagged for the where-to-condition ablation) adds time-varying
    modulation on top. `cond_drop` swaps rows to a learned null embedding —
    conditioning dropout for CFG-style use.
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
        num_bins: int = NUM_BINS,
        knob_freqs: int = 8,
        knob_embed_dim: int = 16,
        use_tfilm: bool = True,
    ):
        super().__init__()
        self.hop_size = hop_size
        self.num_bins = num_bins
        self.tfilm_block_size = tfilm_block_size
        self.use_tfilm = use_tfilm

        self.knob_embed = KnobEmbedding(cond_dim, knob_freqs, knob_embed_dim)
        self.null_embed = nn.Parameter(torch.zeros(knob_embed_dim))
        self.proj = nn.Conv1d(1, encoder_channels, kernel_size=hop_size, stride=hop_size)
        self.act = nn.PReLU(encoder_channels)
        self.lstm = nn.LSTM(
            encoder_channels + knob_embed_dim, hidden_size, num_layers=1, batch_first=True
        )
        self.dense_mid = nn.Linear(hidden_size, tfilm_channels)
        if use_tfilm:
            self.tfilm = TFiLM(
                nfeatures=tfilm_channels,
                cond_dim=knob_embed_dim,
                block_size=tfilm_block_size,
                num_layers=tfilm_num_layers,
            )
        self.head = nn.Conv1d(tfilm_channels, num_bins, kernel_size=1)

    def reset_cond_states(self) -> None:
        if self.use_tfilm:
            self.tfilm.reset_state()

    def forward(
        self,
        dry: torch.Tensor,
        params: torch.Tensor,
        state=None,
        return_state: bool = False,
        cond_drop: torch.Tensor | None = None,
    ):
        # dry: [B, 1, L]   params: [B, 4]   cond_drop: bool [B] or None
        e = self.knob_embed(params)
        if cond_drop is not None:
            e = torch.where(cond_drop[:, None], self.null_embed.expand_as(e), e)
        x = self.act(self.proj(dry)).transpose(1, 2)            # [B, T, C]
        ec = e.unsqueeze(1).expand(-1, x.shape[1], -1)          # [B, T, E]
        x, state = self.lstm(torch.cat((x, ec), dim=-1), state)
        x = self.dense_mid(x).transpose(1, 2)                   # [B, tfilm_ch, T]
        if self.use_tfilm:
            x = self.tfilm(x, e)
        logits = self.head(x)                                   # [B, num_bins, T]
        if return_state:
            return logits, state
        return logits

    def to_db(self, logits: torch.Tensor, sample_len: int | None = None) -> torch.Tensor:
        db = logits_to_local_avg_db(logits)
        if sample_len is not None:
            db = F.interpolate(db, size=sample_len, mode="linear", align_corners=False)
        return db

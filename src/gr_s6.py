"""S6-based frame-rate models for gain-reduction estimation.

The S6 layer is a PyTorch port of
``external/A-Comparative-Study-State-Based-main/Code/S6.py`` so it can be used
inside the existing PyTorch/Lightning GR pipeline.
"""

from __future__ import annotations

import math
from typing import Any

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dsp_torch import GR_DB_MAX, GR_DB_MIN

GR_NUM_BINS = 151
GR_BIN_RESOLUTION = 0.2

_SCAN_CHUNK_SIZE = 128


def gr_bin_centers(
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return the fixed CREPE-style GR bin centers in dB."""
    return torch.linspace(
        GR_DB_MIN,
        GR_DB_MAX,
        GR_NUM_BINS,
        device=device,
        dtype=dtype,
    )


def gr_db_to_soft_target(gr_db: torch.Tensor, sigma_bins: float = 2.0) -> torch.Tensor:
    """Map continuous GR in dB to Gaussian-blurred bin targets.

    Args:
        gr_db: Tensor shaped ``[B, 1, T]``.
        sigma_bins: Gaussian spread in bin units.

    Returns:
        Tensor shaped ``[B, GR_NUM_BINS, T]`` with values in ``[0, 1]``.
    """
    centers = gr_bin_centers(device=gr_db.device, dtype=gr_db.dtype)
    diff = gr_db - centers.view(1, -1, 1)
    sigma_db = sigma_bins * GR_BIN_RESOLUTION
    return torch.exp(-0.5 * (diff / sigma_db) ** 2)


def logits_to_local_avg_db(logits: torch.Tensor, window: int = 5) -> torch.Tensor:
    """Convert GR-bin logits to continuous dB via local weighted averaging."""
    probs = torch.sigmoid(logits)
    centers = gr_bin_centers(device=logits.device, dtype=logits.dtype)

    peak = probs.argmax(dim=1)
    batch, num_bins, frames = probs.shape

    bin_idx = torch.arange(num_bins, device=logits.device).view(1, num_bins, 1)
    bin_idx = bin_idx.expand(batch, num_bins, frames)
    lo = (peak.unsqueeze(1) - window).clamp(min=0)
    hi = (peak.unsqueeze(1) + window + 1).clamp(max=num_bins)
    mask = (bin_idx >= lo) & (bin_idx < hi)

    masked_probs = probs * mask
    weighted = (masked_probs * centers.view(1, -1, 1)).sum(dim=1, keepdim=True)
    total = masked_probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return weighted / total


def _selective_scan_chunk(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    last_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Selective SSM scan over a single chunk (short sequence, numerically safe)."""
    d_a = torch.einsum("bld,dn->bldn", delta, a)
    d_b_u = torch.einsum("bld,bld,bln->bldn", delta, u, b)

    zero = torch.zeros_like(d_a[:, :1])
    d_a_cumsum = torch.cat([zero, d_a[:, 1:]], dim=1)
    d_a_cumsum = torch.exp(torch.cumsum(d_a_cumsum, dim=1))

    x = d_b_u / (d_a_cumsum + 1e-12)
    x = torch.cumsum(x, dim=1) * d_a_cumsum

    if last_state is not None:
        d_a_cumsum_l = torch.exp(torch.cumsum(d_a, dim=1))
        x = x + d_a_cumsum_l * last_state.unsqueeze(1)

    next_state = x[:, -1]
    y = torch.einsum("bldn,bln->bld", x, c)
    return y + u * d.view(1, 1, -1), next_state


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    last_state: torch.Tensor | None = None,
    stateful: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked selective SSM scan to avoid numerical underflow on long sequences."""
    seq_len = u.shape[1]
    chunk_size = _SCAN_CHUNK_SIZE

    if seq_len <= chunk_size:
        state = last_state if stateful else None
        return _selective_scan_chunk(u, delta, a, b, c, d, state)

    state = last_state if stateful else None
    outputs = []

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        y_chunk, state = _selective_scan_chunk(
            u[:, start:end],
            delta[:, start:end],
            a,
            b[:, start:end],
            c[:, start:end],
            d,
            state,
        )
        outputs.append(y_chunk)

    return torch.cat(outputs, dim=1), state


class S6Layer(nn.Module):
    """Selective SSM layer ported from the comparative-study S6 code."""

    def __init__(
        self,
        model_input_dims: int,
        model_states: int,
        stateful: bool = False,
    ):
        super().__init__()
        self.model_input_dims = model_input_dims
        self.model_states = model_states
        self.stateful = stateful
        self.delta_t_rank = math.ceil(model_input_dims / 2)

        self.x_projection = nn.Linear(
            model_input_dims,
            self.delta_t_rank + 2 * model_states,
            bias=False,
        )
        self.delta_t_projection = nn.Linear(self.delta_t_rank, model_input_dims)

        a = torch.arange(1, model_states + 1, dtype=torch.float32)
        a = a.view(1, model_states).repeat(model_input_dims, 1)
        self.a_log = nn.Parameter(torch.log(a))
        self.d = nn.Parameter(torch.ones(model_input_dims))

        self.out_projection = nn.Linear(model_input_dims, model_input_dims)

        self.register_buffer("state", torch.empty(0), persistent=False)

    def reset_states(self) -> None:
        self.state = torch.empty(0, device=self.state.device)

    def _last_state(self, x: torch.Tensor) -> torch.Tensor | None:
        if not self.stateful:
            return None
        expected = (x.shape[0], self.model_input_dims, self.model_states)
        if tuple(self.state.shape) != expected or self.state.device != x.device:
            self.state = torch.zeros(expected, device=x.device, dtype=x.dtype)
        return self.state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process ``x`` shaped ``[B, L, D]`` and return the same shape."""
        a = -torch.exp(self.a_log.float()).to(dtype=x.dtype)
        d = self.d.to(dtype=x.dtype)

        x_dbl = self.x_projection(x)
        delta, b, c = torch.split(
            x_dbl,
            [self.delta_t_rank, self.model_states, self.model_states],
            dim=-1,
        )
        delta = F.softplus(self.delta_t_projection(delta))

        y, next_state = selective_scan(
            x,
            delta,
            a,
            b,
            c,
            d,
            last_state=self._last_state(x),
            stateful=self.stateful,
        )
        if self.stateful:
            self.state = next_state.detach()
        return self.out_projection(y)


class FrameRateS6GR(nn.Module):
    """Frame-rate S6 model for discretized gain-reduction prediction."""

    def __init__(
        self,
        hop_size: int = 256,
        encoder_channels: int = 32,
        model_dim: int = 64,
        state_dim: int = 32,
        num_s6_layers: int = 2,
        dropout: float = 0.0,
        num_bins: int = GR_NUM_BINS,
        stateful: bool = False,
        use_layer_norm: bool = True,
        head_bottleneck: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.hop_size = hop_size
        self.num_bins = num_bins
        self.rf = hop_size * 4
        self.use_layer_norm = use_layer_norm

        kernel_size = hop_size * 2
        self.encoder = nn.Sequential(
            nn.Conv1d(1, encoder_channels, kernel_size, stride=hop_size, padding=kernel_size // 2),
            nn.PReLU(encoder_channels),
            nn.Conv1d(encoder_channels, encoder_channels, 3, padding=1),
            nn.PReLU(encoder_channels),
        )
        self.input_projection = nn.Linear(encoder_channels, model_dim)

        self.s6_layers = nn.ModuleList(
            [
                S6Layer(
                    model_input_dims=model_dim,
                    model_states=state_dim,
                    stateful=stateful,
                )
                for _ in range(num_s6_layers)
            ]
        )
        if use_layer_norm:
            self.norms = nn.ModuleList(
                [nn.LayerNorm(model_dim) for _ in range(num_s6_layers)]
            )
        self.post_s6 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(model_dim, model_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(num_s6_layers)
            ]
        )

        head_hidden = max(1, model_dim // 2) if head_bottleneck else model_dim
        self.head = nn.Sequential(
            nn.Linear(model_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_bins),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits at frame rate: ``[B, GR_NUM_BINS, T / hop]``."""
        feat = self.encoder(x).transpose(1, 2)
        h = self.input_projection(feat)
        if self.use_layer_norm:
            for s6_layer, norm, post_layer in zip(self.s6_layers, self.norms, self.post_s6):
                h = h + post_layer(s6_layer(norm(h)))
        else:
            for s6_layer, post_layer in zip(self.s6_layers, self.post_s6):
                h = h + post_layer(s6_layer(h))
        logits = self.head(h)
        return logits.transpose(1, 2)

    def to_db(self, logits: torch.Tensor, samples: int | None = None) -> torch.Tensor:
        """Convert logits to continuous GR dB, optionally upsampled to samples."""
        db = logits_to_local_avg_db(logits)
        if samples is not None:
            db = F.interpolate(db, size=samples, mode="linear", align_corners=False)
        return db

    def reset_states(self) -> None:
        for layer in self.s6_layers:
            layer.reset_states()


class DiscretizedGRPredictionSystem(pl.LightningModule):
    """Lightning system for CREPE-style discretized GR estimation."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        sigma_bins: float = 2.0,
        warmup_frames: int = 4,
        lr_patience: int = 20,
        min_lr: float = 1e-6,
        local_avg_window: int = 5,
        grad_clip_norm: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.sigma_bins = sigma_bins
        self.warmup_frames = warmup_frames
        self.lr_patience = lr_patience
        self.min_lr = min_lr
        self.local_avg_window = local_avg_window
        self.grad_clip_norm = grad_clip_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_norm)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], mode: str) -> torch.Tensor:
        dry, gr_db = batch
        gr_db = gr_db.clamp(GR_DB_MIN, GR_DB_MAX)

        if hasattr(self.model, "reset_states"):
            self.model.reset_states()
        logits = self(dry)

        frames = logits.shape[-1]
        gr_db_frames = F.adaptive_avg_pool1d(gr_db, frames)
        soft_target = gr_db_to_soft_target(gr_db_frames, sigma_bins=self.sigma_bins)

        warmup = min(self.warmup_frames, max(0, frames - 1))
        if warmup:
            logits = logits[..., warmup:]
            soft_target = soft_target[..., warmup:]
            gr_db_frames = gr_db_frames[..., warmup:]

        loss = F.binary_cross_entropy_with_logits(logits, soft_target)

        with torch.no_grad():
            pred_db = logits_to_local_avg_db(logits, window=self.local_avg_window)
            mae_db = F.l1_loss(pred_db, gr_db_frames)

        self.log(f"loss/{mode}", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"mae_db/{mode}", mae_db, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self) -> dict[str, Any]:
        ssm_params = []
        other_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "a_log" in name or "delta_t_projection" in name:
                ssm_params.append(p)
            else:
                other_params.append(p)

        opt = torch.optim.AdamW(
            [
                {"params": other_params, "lr": self.lr},
                {"params": ssm_params, "lr": self.lr * 0.1},
            ],
            weight_decay=1e-2,
        )

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=self.lr_patience, min_lr=self.min_lr,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "loss/val"},
        }


def build_frame_rate_s6_from_hparams(hparams: dict[str, Any]) -> FrameRateS6GR:
    """Rebuild ``FrameRateS6GR`` from the notebook's saved ``hparams.json``.

    Automatically detects old-style checkpoints (no LayerNorm, bottlenecked head)
    via the ``use_layer_norm`` and ``head_bottleneck`` fields, falling back to
    legacy defaults when ``fixed_out_projection`` is present in the config.
    """
    cfg = hparams.get("s6", hparams)
    is_legacy = "fixed_out_projection" in cfg and "use_layer_norm" not in cfg
    return FrameRateS6GR(
        hop_size=cfg.get("hop_size", 256),
        encoder_channels=cfg.get("encoder_channels", 32),
        model_dim=cfg.get("model_dim", 64),
        state_dim=cfg.get("state_dim", 32),
        num_s6_layers=cfg.get("num_s6_layers", 2),
        dropout=cfg.get("dropout", 0.0),
        num_bins=hparams.get("discretization", {}).get("num_bins", GR_NUM_BINS),
        stateful=cfg.get("stateful", False),
        use_layer_norm=cfg.get("use_layer_norm", not is_legacy),
        head_bottleneck=cfg.get("head_bottleneck", is_legacy),
    )


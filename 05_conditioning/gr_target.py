"""GR target normalisation for the scalar [0, 1] sigmoid regression head.

Limits derived from the exported Diff-SSL gr_curves (100 files = 10 songs x
10 settings): observed min = -28.252 dB, max = +4.213 dB, no NaN/Inf.
[-30, +5] covers that with margin so the sigmoid never has to saturate at
0/1 (0 dB GR maps to 0.857, the observed max to 0.977). See NORMALISATION.md.
"""

import torch

GR_DB_MIN = -30.0
GR_DB_MAX = 5.0


def normalize_gr_01(gr_db: torch.Tensor) -> torch.Tensor:
    """Map GR in dB from [GR_DB_MIN, GR_DB_MAX] to [0, 1]."""
    return (gr_db - GR_DB_MIN) / (GR_DB_MAX - GR_DB_MIN)


def denormalize_gr_01(gr_01: torch.Tensor) -> torch.Tensor:
    """Map [0, 1] back to dB in [GR_DB_MIN, GR_DB_MAX]."""
    return gr_01 * (GR_DB_MAX - GR_DB_MIN) + GR_DB_MIN


# ── CREPE-style discretization (03_initial_GR_pred recipe over the new range) ──

NUM_BINS = 71
BIN_RESOLUTION_DB = (GR_DB_MAX - GR_DB_MIN) / (NUM_BINS - 1)   # 0.5 dB


def bin_centers(like: torch.Tensor) -> torch.Tensor:
    return torch.linspace(
        GR_DB_MIN, GR_DB_MAX, NUM_BINS, device=like.device, dtype=like.dtype
    )


def gr_db_to_soft_target(gr_db: torch.Tensor, sigma_bins: float = 2.0) -> torch.Tensor:
    """Gaussian-blurred soft targets over GR bins. [B,1,T] dB -> [B,NUM_BINS,T]."""
    centers = bin_centers(gr_db)
    sigma_db = sigma_bins * BIN_RESOLUTION_DB
    return torch.exp(-0.5 * ((gr_db - centers[None, :, None]) / sigma_db) ** 2)


def logits_to_expected_db(logits: torch.Tensor) -> torch.Tensor:
    """Differentiable soft-argmax decode: softmax-weighted bin centers. [B,1,T] dB."""
    w = torch.softmax(logits, dim=1)
    return (w * bin_centers(logits)[None, :, None]).sum(dim=1, keepdim=True)


def logits_to_local_avg_db(logits: torch.Tensor, window: int = 5) -> torch.Tensor:
    """Robust eval decode: weighted average of sigmoid probs around the argmax bin."""
    probs = torch.sigmoid(logits)
    centers = bin_centers(logits)
    peak = probs.argmax(dim=1)
    B, N, T = probs.shape
    bin_idx = torch.arange(N, device=logits.device).view(1, N, 1).expand(B, N, T)
    lo = (peak.unsqueeze(1) - window).clamp(min=0)
    hi = (peak.unsqueeze(1) + window + 1).clamp(max=N)
    mask = (bin_idx >= lo) & (bin_idx < hi)
    masked = probs * mask
    weighted = (masked * centers[None, :, None]).sum(dim=1, keepdim=True)
    return weighted / masked.sum(dim=1, keepdim=True).clamp(min=1e-8)

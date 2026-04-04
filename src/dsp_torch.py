"""
PyTorch-based DSP utilities: RMS envelopes, gain reduction, GR normalisation.

Intended for training pipelines where the computation graph must stay on GPU.
"""

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Gain-reduction normalisation constants
# ---------------------------------------------------------------------------

GR_DB_MIN = -30.0
GR_DB_MAX = 0.0
RMS_WINDOW = 1024


# ---------------------------------------------------------------------------
# RMS helpers
# ---------------------------------------------------------------------------


def windowed_rms(signal: torch.Tensor, window_size: int) -> torch.Tensor:
    """Sample-rate windowed RMS via 1-D convolution.

    Args:
        signal: ``[1, T]`` mono audio.
        window_size: RMS analysis window in samples.

    Returns:
        ``[1, T]`` RMS envelope (linear amplitude).
    """
    sq = (signal.unsqueeze(0)) ** 2
    kernel = torch.ones(1, 1, window_size, device=signal.device) / window_size
    pad = window_size - 1
    rms_sq = F.conv1d(sq, kernel, padding=pad)
    rms_sq = rms_sq[..., : signal.shape[-1]]
    return torch.sqrt(rms_sq.squeeze(0).clamp(min=1e-10))


# ---------------------------------------------------------------------------
# Gain reduction
# ---------------------------------------------------------------------------


def gain_reduction_db(
    dry: torch.Tensor, wet: torch.Tensor, window_size: int = RMS_WINDOW
) -> torch.Tensor:
    """GR in dB = RMS_dB(wet) − RMS_dB(dry).  Negative means compression.

    Args:
        dry: ``[1, T]`` dry (input) audio.
        wet: ``[1, T]`` wet (compressed output) audio.
        window_size: RMS window size.

    Returns:
        ``[1, T]`` gain-reduction signal in dB.
    """
    dry_rms = windowed_rms(dry, window_size)
    wet_rms = windowed_rms(wet, window_size)
    dry_db = 20 * torch.log10(dry_rms)
    wet_db = 20 * torch.log10(wet_rms)
    return wet_db - dry_db


# ---------------------------------------------------------------------------
# GR normalisation (for training targets)
# ---------------------------------------------------------------------------


def normalize_gr(gr_db: torch.Tensor) -> torch.Tensor:
    """Map ``[GR_DB_MIN, GR_DB_MAX]`` → ``[-1, 1]``."""
    return (gr_db - GR_DB_MIN) / (GR_DB_MAX - GR_DB_MIN) * 2 - 1


def denormalize_gr(gr_norm: torch.Tensor) -> torch.Tensor:
    """Map ``[-1, 1]`` → ``[GR_DB_MIN, GR_DB_MAX]``."""
    return (gr_norm + 1) / 2 * (GR_DB_MAX - GR_DB_MIN) + GR_DB_MIN

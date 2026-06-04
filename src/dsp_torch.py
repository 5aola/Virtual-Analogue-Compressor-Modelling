"""
PyTorch-based DSP utilities: RMS envelopes, gain reduction, GR normalisation,
and compressor-parameter normalisation.

Intended for training pipelines where the computation graph must stay on GPU.
"""

import torch
import torch.nn.functional as F

from src.dsp import PARAM_ORDER, PARAM_RANGES_LOCAL

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
        signal: ``[T]``, ``[C, T]``, or ``[B, C, T]`` audio.
        window_size: RMS analysis window in samples.

    Returns:
        RMS envelope (linear amplitude) with the same rank as ``signal``.
    """
    orig_ndim = signal.ndim
    if orig_ndim == 1:
        sig = signal.view(1, 1, -1)
    elif orig_ndim == 2:
        sig = signal.unsqueeze(0)
    elif orig_ndim == 3:
        sig = signal
    else:
        raise ValueError(f"windowed_rms expects 1/2/3-D input, got {orig_ndim}-D")

    batch, channels, frames = sig.shape
    sq = sig.reshape(batch * channels, 1, frames) ** 2
    kernel = (
        torch.ones(1, 1, window_size, device=sig.device, dtype=sq.dtype)
        / window_size
    )
    pad = window_size - 1
    rms_sq = F.conv1d(sq, kernel, padding=pad)[..., :frames]
    rms = torch.sqrt(rms_sq.clamp(min=1e-10)).reshape(batch, channels, frames)

    if orig_ndim == 1:
        return rms.view(-1)
    if orig_ndim == 2:
        return rms.squeeze(0)
    return rms


# ---------------------------------------------------------------------------
# Gain reduction
# ---------------------------------------------------------------------------


def gain_reduction_db(
    dry: torch.Tensor, wet: torch.Tensor, window_size: int = RMS_WINDOW
) -> torch.Tensor:
    """GR in dB = RMS_dB(wet) − RMS_dB(dry).  Negative means compression.

    Args:
        dry: ``[T]``, ``[C, T]``, or ``[B, C, T]`` dry input audio.
        wet: matching wet compressed output audio.
        window_size: RMS window size.

    Returns:
        Gain-reduction signal in dB with the same rank as the inputs.
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


def compute_gr_target_norm(
    dry: torch.Tensor,
    wet: torch.Tensor,
    rms_window: int = RMS_WINDOW,
) -> torch.Tensor:
    """Exact GR target used by training/evaluation: dB GR → normalised clamp."""
    gr_db = gain_reduction_db(dry.float(), wet.float(), rms_window)
    return normalize_gr(gr_db).clamp(-1.0, 1.0)


def gr_norm_to_db(gr_norm: torch.Tensor) -> torch.Tensor:
    """Convert a normalised GR curve back to dB for plotting/evaluation."""
    return denormalize_gr(gr_norm)


# ---------------------------------------------------------------------------
# Compressor-parameter normalisation
# ---------------------------------------------------------------------------


def normalize_params(
    threshold: float,
    attack: float,
    release: float,
    ratio: float,
    ranges: dict | None = None,
) -> torch.Tensor:
    """Map raw compressor parameters to ``[0, 1]`` using *ranges*.

    Default ranges are :data:`PARAM_RANGES_LOCAL`.
    """
    if ranges is None:
        ranges = PARAM_RANGES_LOCAL
    raw = torch.tensor([threshold, attack, release, ratio], dtype=torch.float32)
    for i, key in enumerate(PARAM_ORDER):
        lo, hi = ranges[key]
        raw[i] = (raw[i] - lo) / (hi - lo)
    return raw.clamp(0, 1)

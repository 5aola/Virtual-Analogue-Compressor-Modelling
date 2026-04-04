"""
NumPy-based DSP utilities: RMS envelopes, gain reduction, level helpers,
compressor-parameter parsing and normalisation.
"""

import re

import numpy as np
import pandas as pd
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Compressor parameter constants
# ---------------------------------------------------------------------------

PARAM_ORDER = ["threshold", "attack", "release", "ratio"]

PARAM_RANGES_LOCAL = {
    "threshold": (-20.0, 0.0),
    "attack": (0.1, 30.0),
    "release": (0.1, 1.6),
    "ratio": (2.0, 10.0),
}

PARAM_RANGES_NABLAFX = {
    "threshold": (-20.0, 20.0),
    "attack": (0.0, 30.0),
    "release": (0.0, 1.6),
    "ratio": (0.0, 10.0),
}


def parse_settings_from_folder_name(folder_name: str) -> dict:
    """Parse compressor settings from a folder name using individual regex
    patterns.  Returns dict with keys: threshold, attack, release, ratio
    (values may be ``None``).
    """
    settings: dict = {
        "threshold": None,
        "attack": None,
        "release": None,
        "ratio": None,
    }
    patterns = {
        "threshold": r"threshold_(-?\d+(?:\.\d+)?)",
        "attack": r"attack_(-?\d+(?:\.\d+)?)",
        "release": r"release_(-?\d+(?:\.\d+)?)",
        "ratio": r"ratio_(-?\d+(?:\.\d+)?)",
    }
    for param, pattern in patterns.items():
        match = re.search(pattern, folder_name, re.IGNORECASE)
        if match:
            value = match.group(1)
            settings[param] = float(value) if "." in value else int(value)
    return settings


# ---------------------------------------------------------------------------
# Level helpers
# ---------------------------------------------------------------------------


def to_dB(signal, ref=1.0, eps: float = 1e-10):
    """
    Convert an amplitude-like quantity to dB, with numerical safeguards.

    Clamps the input to at least `eps` (after scaling by `ref`) to avoid
    taking log10 of zero or negative values, which would otherwise produce
    -inf or NaNs and raise runtime warnings.
    """
    signal = np.asarray(signal)
    scaled = signal / ref
    scaled = np.maximum(scaled, eps)
    return 20 * np.log10(scaled)


def to_amplitude(signal, ref=1.0):
    return 10 ** (signal / 20) * ref


# ---------------------------------------------------------------------------
# RMS helpers
# ---------------------------------------------------------------------------


def window_rms(signal, window_size, in_dB=False):
    """Sample-aligned RMS via convolution (efficient)."""
    window_size = int(window_size)
    signal = np.square(signal)
    window = np.ones(window_size) / float(window_size)
    rms = np.sqrt(np.convolve(signal, window, "full"))
    rms = rms[: len(signal)]
    if in_dB:
        rms = to_dB(rms)
    return rms


def calculate_rms(a: NDArray):
    """Given a numpy array, return its RMS power level (last axis)."""
    return np.sqrt(np.mean(np.square(a), axis=-1))


def rms_to_db(rms):
    return 20 * np.log10(np.maximum(rms, 1e-10))


def moving_rms(a: NDArray, window_size: int) -> NDArray:
    """
    Frame-based RMS in dB using convolve-based window_rms then decimation.
    Output length: (a.shape[-1] - window_size) // hop_size + 1 with hop_size = window_size // 2.
    """
    if 0 in a.shape:
        raise ValueError("Cannot input empty array")
    window_size = int(window_size)
    hop_size = window_size // 2
    n_frames = (a.shape[-1] - window_size) // hop_size + 1

    if a.ndim == 1:
        rms = window_rms(a, window_size, in_dB=True)
        return rms[hop_size * np.arange(n_frames)]

    # Multi-dim: apply along last axis
    flat = a.reshape(-1, a.shape[-1])
    out_flat = np.zeros((flat.shape[0], n_frames), dtype=a.dtype)
    for i in range(flat.shape[0]):
        rms = window_rms(flat[i], window_size, in_dB=True)
        out_flat[i] = rms[hop_size * np.arange(n_frames)]
    return out_flat.reshape(a.shape[:-1] + (n_frames,))


# ---------------------------------------------------------------------------
# Peak envelope
# ---------------------------------------------------------------------------


def window_peak(signal, window_size, in_dB=False):
    window_size = int(window_size)
    signal = np.abs(signal)
    peak = (
        pd.Series(signal)
        .rolling(window=window_size, min_periods=1, center=True)
        .max()
        .values
    )
    if in_dB:
        peak = to_dB(peak)
    return peak


# ---------------------------------------------------------------------------
# Gain reduction
# ---------------------------------------------------------------------------


def calc_gain_reduction(signal1, signal2, window_size):
    """Compute gain reduction in dB between two signals using windowed RMS."""
    env1 = window_rms(signal1, window_size, in_dB=True)
    env2 = window_rms(signal2, window_size, in_dB=True)
    return env2 - env1


def estimate_attack_release_time(
    x: NDArray,
    y: NDArray,
    sr: int = 44100,
    window_size: int = 256,
) -> tuple[float, float]:
    """
    Estimate attack and release steepness of a compressor from input (x) and output (y).

    Returns (attack_steepness, release_steepness) in dB per analysis frame.
    """
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-D arrays")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length")

    x_rms_db = moving_rms(x, window_size)
    y_rms_db = moving_rms(y, window_size)
    gain_reduction_db = y_rms_db - x_rms_db

    if gain_reduction_db.shape[0] < 2:
        return 0.0, 0.0

    gr_diff = np.diff(gain_reduction_db)
    attack_steepness = float(np.min(gr_diff))
    release_steepness = float(np.max(gr_diff))
    return attack_steepness, release_steepness

"""
Merged analytics and compressor-estimation utilities.
Uses convolve-based RMS where applicable for efficiency.
"""

import os

import essentia.standard as es
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import stft

# Constants
SSL_DSET_PATH = "/Volumes/Production Tools/coding_projs/THESIS/data_preprocesses/data/Diff-SSL-G-Comp"
SR = 44100
window_size = 1024
overlap = 0.75


# ---------------------------------------------------------------------------
# Audio I/O and stats
# ---------------------------------------------------------------------------


def collect_audio_files(root_dir):
    audio_files = []
    for file in os.listdir(root_dir):
        if file.endswith(".wav"):
            audio_files.append(os.path.join(root_dir, file))
    return audio_files


def load_audio(path, sr=SR, cut=(None, None)):
    """
    Load an audio file using Essentia and return its time vector and audio samples.

    Example:
        # Load the whole audio file
        t, audio = load_audio("path/to/file.wav")

        # Load audio between 2.0s and 5.5s
        t, audio = load_audio("path/to/file.wav", cut=(2.0, 5.5))

        # Load audio from 10.0s until the end of the file
        t, audio = load_audio("path/to/file.wav", cut=(10.0, None))
    """
    audio = es.MonoLoader(filename=path, sampleRate=sr)()
    t = np.arange(len(audio)) / sr
    if cut[0] is not None:
        audio = audio[int(cut[0] * sr) : min(int(cut[1] * sr), len(audio))]
        t = t[int(cut[0] * sr) : min(int(cut[1] * sr), len(t))]
    return t, audio


def get_audio_stats(audio, sr=SR):
    _, _, ebu_integrated, loudness_range = es.LoudnessEBUR128(
        hopSize=1024 / sr, startAtZero=True
    )(audio)
    return ebu_integrated, loudness_range


# ---------------------------------------------------------------------------
# RMS and level helpers (convolve-based where applicable)
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
# Peak envelope and gain reduction
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


def calc_gain_reduction(signal1, signal2, window_size):
    """Compute gain reduction in dB between two signals using windowed RMS."""
    env1 = window_rms(signal1, window_size, in_dB=True)
    env2 = window_rms(signal2, window_size, in_dB=True)
    return env2 - env1


# ---------------------------------------------------------------------------
# Gain curve estimation and plotting
# ---------------------------------------------------------------------------


def regression(x: NDArray, y: NDArray):
    """Perform polynomial regression on the given x and y data."""
    if len(x) != len(y):
        raise ValueError("x and y must be of the same length")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    coeffs = np.polyfit(x, y, 6)
    poly_eq = np.poly1d(coeffs)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = poly_eq(x_fit)
    return x_fit, y_fit


def estimate_gain_curve(x, y, window_size=1024, lims=(-80, 0)):
    """Estimate and plot the gain curve of a signal (input vs output RMS in dB)."""
    x_rms = moving_rms(x, window_size)
    y_rms = moving_rms(y, window_size)
    fig, ax = plt.subplots()
    ax.plot(lims, lims, "k--", linewidth=0.8)
    ax.scatter(x_rms, y_rms, s=2)
    ax.set_xlabel("X RMS [dB]")
    ax.set_ylabel("Y RMS [dB]")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title("Gain Curve")
    ax.grid()
    plt.show()
    return fig, ax


def compare_gain_curves(x, y_pred, y_target, window_size=4410, lims=(-80, 0)):
    """Plot predicted and target gain curves on the same axes. Returns (fig, ax)."""
    x_rms = moving_rms(x, window_size)
    pred_rms = moving_rms(y_pred, window_size)
    target_rms = moving_rms(y_target, window_size)
    x_fit_pred, y_fit_pred = regression(x_rms, pred_rms)
    x_fit_tgt, y_fit_tgt = regression(x_rms, target_rms)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        x_rms, target_rms, s=2, alpha=1, color="tab:green", label="Target (scatter)"
    )
    ax.scatter(
        x_rms, pred_rms, s=2, alpha=1, color="tab:blue", label="Predicted (scatter)"
    )
    ax.plot(lims, lims, "k--", linewidth=0.8)
    ax.set_xlabel("Input RMS [dB]")
    ax.set_ylabel("Output RMS [dB]")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig, ax


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


# ---------------------------------------------------------------------------
# FFT / transfer (from _my_analytic_funcs)
# ---------------------------------------------------------------------------


def auto_fft(signal, fs=SR, window_size=window_size):
    fft_size = int(window_size / 2 + 1)
    fft = np.fft.fft(signal, fft_size)
    f = np.fft.fftfreq(fft_size, 1 / fs)
    return f, fft


def H1(signal1, signal2, sr=SR, window_size=window_size, overlap=overlap):
    auto_spectr = np.zeros(int(window_size / 2 + 1), dtype="complex128")
    cross_spectr = np.zeros(int(window_size / 2 + 1), dtype="complex128")
    for index in range(
        0,
        len(signal1) - window_size,
        int(window_size * (1 - overlap)),
    ):
        f, signal1_f = auto_fft(signal1[index : (index + window_size)], sr)
        f, signal2_f = auto_fft(signal2[index : (index + window_size)], sr)
        auto_spectr += np.abs(signal1_f) ** 2
        cross_spectr += signal2_f * np.conj(signal1_f)
    h1 = cross_spectr / auto_spectr
    return f, h1


def FRAC(m1s1, m1s2, m2s1, m2s2, sr=SR, window_size=window_size, overlap=overlap):
    _, h1 = H1(m1s1, m1s2, sr, window_size, overlap)
    _, h2 = H1(m2s1, m2s2, sr, window_size, overlap)
    return np.abs((np.conj(h1) @ h2)) ** 2 / np.abs(
        (np.conj(h1) @ h1) * (np.conj(h2) @ h2)
    )


def time_varying_transfer(
    x,
    y,
    fs,
    n_fft=2048,
    hop_length=None,
    window="hann",
    eps=1e-10,
):
    """
    Estimate time-varying transfer function using complex STFT ratio with coherence weighting.

    Returns
    -------
    f : frequency bins
    t : time frames
    H : complex transfer estimate (f x t)
    gain_db : magnitude gain in dB
    coherence : magnitude-squared coherence
    weighted_gain_db : coherence-weighted gain in dB
    """
    if hop_length is None:
        hop_length = n_fft // 4

    f, t, X = stft(
        x,
        fs=fs,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        padded=False,
        boundary=None,
    )
    _, _, Y = stft(
        y,
        fs=fs,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        padded=False,
        boundary=None,
    )

    Sxx = np.abs(X) ** 2
    Syy = np.abs(Y) ** 2
    Sxy = Y * np.conj(X)
    H = Sxy / (Sxx + eps)
    gain = np.abs(H)
    gain_db = 20 * np.log10(gain + eps)
    coherence = (np.abs(Sxy) ** 2) / (Sxx * Syy + eps)
    weighted_gain = gain * coherence
    weighted_gain_db = 20 * np.log10(weighted_gain + eps)
    return f, t, H, gain_db, coherence, weighted_gain_db

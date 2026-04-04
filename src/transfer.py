"""
FFT and transfer-function utilities for compressor analysis.
"""

import numpy as np
from scipy.signal import stft

SR = 44100
_WINDOW_SIZE = 1024
_OVERLAP = 0.75


def auto_fft(signal, fs=SR, window_size=_WINDOW_SIZE):
    fft_size = int(window_size / 2 + 1)
    fft = np.fft.fft(signal, fft_size)
    f = np.fft.fftfreq(fft_size, 1 / fs)
    return f, fft


def H1(signal1, signal2, sr=SR, window_size=_WINDOW_SIZE, overlap=_OVERLAP):
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


def FRAC(m1s1, m1s2, m2s1, m2s2, sr=SR, window_size=_WINDOW_SIZE, overlap=_OVERLAP):
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

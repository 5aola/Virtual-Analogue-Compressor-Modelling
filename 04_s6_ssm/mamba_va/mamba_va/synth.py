"""A tiny synthetic optical-compressor generator.

This is NOT a research-grade emulation -- it exists so the codebase runs and
trains end-to-end with zero external data (used by the smoke test and as a
sanity target).  It implements a feed-forward compressor with a peak detector
and asymmetric, *program-dependent* release, which gives the model the kind of
nonlinear, long-memory behaviour it is meant to learn.
"""

from __future__ import annotations

import numpy as np


def _one_pole(coeff):
    return coeff


def compress(x, sr=48000, threshold_db=-24.0, ratio=4.0,
             attack_ms=5.0, release_ms=200.0, makeup_db=0.0):
    """Apply a simple optical-style compressor to a mono signal `x`."""
    x = np.asarray(x, dtype=np.float64)
    eps = 1e-9
    a_att = np.exp(-1.0 / (sr * attack_ms * 1e-3))
    a_rel = np.exp(-1.0 / (sr * release_ms * 1e-3))

    env = 0.0
    g_smooth = 0.0
    y = np.zeros_like(x)
    thr = threshold_db
    for n in range(len(x)):
        # peak detector with asymmetric, program-dependent timing
        rect = abs(x[n])
        if rect > env:
            env = a_att * env + (1 - a_att) * rect
        else:
            # release slows down the louder we have been (program dependence)
            slow = a_rel + (1 - a_rel) * min(env * 4.0, 0.9)
            env = slow * env + (1 - slow) * rect
        level_db = 20.0 * np.log10(env + eps)
        over = level_db - thr
        gain_db = 0.0 if over <= 0 else -over * (1.0 - 1.0 / ratio)
        # smooth the gain a touch (optical inertia)
        g_smooth = 0.7 * g_smooth + 0.3 * gain_db
        y[n] = x[n] * (10.0 ** ((g_smooth + makeup_db) / 20.0))
    return y.astype(np.float32)


def make_test_signal(seconds=4.0, sr=48000, seed=0):
    """A mix of tones, sweeps and noise bursts with strong dynamics."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    sig = np.zeros(n, dtype=np.float64)
    # decaying tone bursts at random times (transients -> tests attack/release)
    for _ in range(int(seconds * 4)):
        f = rng.uniform(80, 4000)
        start = rng.integers(0, n - 1)
        dur = int(rng.uniform(0.05, 0.5) * sr)
        idx = slice(start, min(start + dur, n))
        env = np.exp(-np.arange(min(dur, n - start)) / (0.15 * sr))
        sig[idx] += np.sin(2 * np.pi * f * t[idx]) * env * rng.uniform(0.3, 1.0)
    # a sweep
    sweep = np.sin(2 * np.pi * (50 + (8000 - 50) * (t / seconds) ** 2) * t) * 0.3
    sig += sweep
    sig += 0.02 * rng.standard_normal(n)
    sig /= max(1e-6, np.max(np.abs(sig)))
    return (0.9 * sig).astype(np.float32)


def make_dataset(seconds=8.0, sr=48000, params=None, seed=0):
    """Return (dry, wet, params_vector) for one recording.

    params order matches the default CompSSM: [threshold, ratio, attack, release]
    normalised to [0, 1].
    """
    if params is None:
        params = dict(threshold_db=-24.0, ratio=4.0, attack_ms=5.0, release_ms=200.0)
    dry = make_test_signal(seconds, sr, seed)
    wet = compress(dry, sr=sr, **params)
    # normalise the four controls to [0,1] over plausible ranges
    pvec = np.array([
        (params["threshold_db"] + 60.0) / 60.0,
        (params["ratio"] - 1.0) / 19.0,
        (np.log10(params["attack_ms"]) - np.log10(0.1)) / (np.log10(500) - np.log10(0.1)),
        (np.log10(params["release_ms"]) - np.log10(5)) / (np.log10(10000) - np.log10(5)),
    ], dtype=np.float32).clip(0, 1)
    return dry, wet, pvec

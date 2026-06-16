"""Amplitude matching: apply an exported gain-reduction curve to the dry input.

This is the front-end of the grey-box decomposition used in `06_conditioning`:

    dry ──(× exported GR gain envelope)──► amplitude-matched input ──► [LSTM] ──► wet

The exported GR curves (`gr_curves/<setting>/<song>.pt`, key ``gr_db``) were
computed sample-for-sample as

    gr_db[i] = 20·log10(wet_rms[i]) − 20·log10(dry_rms[i])

with a **causal** 1024-sample trailing-RMS window (see
``03_initial_GR_pred/gr_dataset.py``). Both ``dry`` and ``gr_db`` therefore live
on the **same sample grid with the same causal convention** — index *i* of the
GR curve aligns with index *i* of the dry/wet audio, so **no time shift** is
applied here. Multiplying

    amp_matched[i] = dry[i] · 10**(gr_db[i] / 20)

rescales the dry waveform so its short-time RMS envelope matches the wet's
(verified on real data: ≈2 % relative envelope error, corr(amp, wet)=0.997 vs
corr(dry, wet)=0.989). The compressor's *level/dynamics* are thus reproduced by
this known envelope; the LSTM downstream only has to learn the residual
nonlinear **coloration** (harmonics, transient micro-shaping) — empirically just
~8 % RMS of the signal.

dB scaling note: ``gr_db`` is an **amplitude** ratio in dB, so the inverse is
``10**(gr/20)`` (NOT ``/10``, which would be a power ratio). In near-silence the
RMS ratio is ill-conditioned (tiny ÷ tiny), so ``gr_db`` is clamped to
``[GR_DB_MIN, GR_DB_MAX]`` before exponentiation to keep the gain bounded
(≈[0.03, 1.78]×) — this is the only guard the matched signal needs.
"""

from __future__ import annotations

import torch

# Project-wide GR range (matches 05_conditioning/gr_target.py): the exported
# curves over all 10 settings span ≈[-28.3, +4.2] dB, so [-30, +5] clamps only
# the ill-conditioned silence tails, never real compression.
GR_DB_MIN = -30.0
GR_DB_MAX = 5.0


def gr_db_to_gain(gr_db: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    """Convert a gain-reduction curve in dB to a linear amplitude gain.

    Args:
        gr_db: gain reduction in dB, any shape. Negative = attenuation.
        clamp: clamp to ``[GR_DB_MIN, GR_DB_MAX]`` first (guards silence blowup).

    Returns:
        Linear gain tensor (same shape), ``10**(gr_db/20)``.
    """
    if clamp:
        gr_db = gr_db.clamp(GR_DB_MIN, GR_DB_MAX)
    return torch.pow(10.0, gr_db / 20.0)


def amplitude_match(
    dry: torch.Tensor, gr_db: torch.Tensor, clamp: bool = True
) -> torch.Tensor:
    """Apply the exported GR envelope to the dry signal (amplitude matching).

    ``dry`` and ``gr_db`` must be sample-aligned and equal length on the time
    axis (the caller is expected to have trimmed both to ``min`` length).

    Args:
        dry: dry audio ``[..., T]``.
        gr_db: gain reduction in dB ``[..., T]``, same trailing shape as ``dry``.
        clamp: clamp ``gr_db`` to ``[GR_DB_MIN, GR_DB_MAX]`` before applying.

    Returns:
        Amplitude-matched input ``[..., T]`` whose short-time RMS tracks the wet.
    """
    return dry * gr_db_to_gain(gr_db, clamp=clamp)

"""Adaptive (asymmetric) level detector -- the nonlinear, signal-dependent
memory that a linear SSM cannot represent.

A real optical compressor's gain reduction is driven by the *level* of the
side-chain signal smoothed by a cell whose time constant differs for rising
(attack) vs falling (release) levels, and whose release is program-dependent
(it lengthens after sustained loud passages).  Classic DSP models this with a
peak/RMS detector followed by a branching smoother

    if  x_t > env_{t-1}:  coeff = alpha_attack
    else:                 coeff = alpha_release
    env_t = coeff * env_{t-1} + (1 - coeff) * x_t

This recurrence is *nonlinear in the state* (the coefficient depends on the
comparison ``x_t > env_{t-1}``), which is exactly the kind of realization that
Shoukry (2008) shows a faithful nonlinear system requires, and exactly what
Mamba/S6's linear h-update lacks.

Three design decisions, learned the hard way (v0.1 trained worse than a
constant gain):

1.  **Time constants live in log-tau space.**  ``coeff = exp(-1/(sr*tau))``
    with ``tau = exp(log_tau)`` seconds.  A one-pole coefficient at audio rate
    must sit within ~1e-4 of 1.0 to hold a 0.4 s release; sigmoid-logit
    parameterisation initialises 2-3 orders of magnitude too fast and has to
    crawl into its own saturated tail to get there.  In log-tau space the init
    directly spans the range real devices use (0.1 ms .. 1.5 s) and a gradient
    step moves *tau multiplicatively*, which is the natural geometry for time
    constants.

2.  **The level is in (normalized) dB.**  Compressor gain laws are
    (piecewise-)linear in dB level, and the ~40 dB range that matters spans
    only a ~2:1 numeric range after |.| -- let alone after a softplus.  See
    :func:`level_db_norm`; the same signal feeds the SSM selectivity input and
    the gain readout.

3.  **The branch is gated on a *pilot* envelope, not on the band's own
    state.**  ``rising_t = sigmoid(k * (x_t - pilot_t))`` where the pilot is a
    short *linear* one-pole of the level.  The smoothing coefficient then
    depends only on the *input* history, so each band becomes a time-varying
    **linear** recurrence ``env_t = c_t*env_{t-1} + (1-c_t)*x_t`` that the
    parallel scan solves directly -- no Python per-sample loop (v0.1 spent
    ~2 s per training step in that loop).  The composite map level->env is
    still nonlinear-with-memory (the coefficient is a nonlinear functional of
    the signal history), which is what the realization argument needs.

The detector output is used two ways: concatenated into the block features,
and (more importantly) fed to the SSM as the *selectivity signal* that sets the
SSM's own time constants -- coupling the linear long-memory SSM to a nonlinear,
level-dependent clock.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from .scan import scan_parallel, scan_sequential

# Level front end shared by the detector, the SSM selectivity signal and the
# gain readout: |u| -> dB with a -80 dBFS floor, scaled so 0 dBFS -> +1 and
# the floor -> -1.  One normalized unit == 40 dB.
LEVEL_EPS = 1e-4
LEVEL_DB_SCALE = 40.0


def level_db_norm(u: torch.Tensor) -> torch.Tensor:
    """Map a waveform (or magnitude) to normalized-dB level.

    ``(20*log10(|u| + 1e-4) + 40) / 40`` -- 0 dBFS -> +1, -40 dB -> 0,
    the -80 dB floor -> -1.  Smooth, monotone, and gives the network the
    decibel scale on which compressor gain laws are linear.
    """
    db = 20.0 * torch.log10(u.abs() + LEVEL_EPS)
    return (db + LEVEL_DB_SCALE) / LEVEL_DB_SCALE


def coeff_from_log_tau(log_tau: torch.Tensor, sr: float) -> torch.Tensor:
    """One-pole smoothing coefficient for a time constant of exp(log_tau) s.

    coeff = exp(-1 / (sr * tau)); close to 1 -> slow, close to 0 -> fast.
    """
    tau = torch.exp(log_tau)
    return torch.exp(-1.0 / (sr * tau))


class AdaptiveLevelDetector(nn.Module):
    def __init__(self, n_bands: int = 4, sr: float = 44100.0,
                 sharpness: float = 60.0,
                 attack_range: tuple[float, float] = (1e-4, 3e-2),
                 release_range: tuple[float, float] = (2e-2, 1.5),
                 pilot_tau: float = 5e-3):
        """
        Args:
            n_bands: number of parallel detectors with distinct, learned attack
                     and release time constants.
            sr: audio sample rate the time constants are defined against.
                A trained detector is sample-rate-specific (as is the SSM's
                Delta); retrain or rescale log-taus to run at another rate.
            sharpness: gate steepness in 1/normalized-dB units (1 unit = 40 dB),
                     i.e. 60 -> the attack/release transition spans ~+-2 dB.
            attack_range / release_range: log-spaced per-band init for the
                     attack / release time constants, in **seconds**.  Defaults
                     cover the ranges hardware compressors use (0.1-30 ms
                     attack, 20 ms-1.5 s release).
            pilot_tau: init time constant of the linear pilot envelope the
                     rising/falling gate compares against.
        """
        super().__init__()
        self.n_bands = n_bands
        # buffer (not parameter): travels with the state_dict, so a loaded
        # checkpoint keeps the rate its time constants were trained against.
        self.register_buffer("sr", torch.tensor(float(sr)))
        self.sharpness = sharpness

        att = torch.linspace(math.log(attack_range[0]), math.log(attack_range[1]), n_bands)
        rel = torch.linspace(math.log(release_range[0]), math.log(release_range[1]), n_bands)
        self.attack_log_tau = nn.Parameter(att)
        self.release_log_tau = nn.Parameter(rel)
        self.pilot_log_tau = nn.Parameter(torch.tensor([math.log(pilot_tau)]))
        # learnable affine on the (normalized-dB) level, init = identity
        self.in_gain = nn.Parameter(torch.ones(1))
        self.in_bias = nn.Parameter(torch.zeros(1))

    @property
    def out_dim(self) -> int:
        return self.n_bands

    def time_constants(self):
        """Learned (attack_s, release_s, pilot_s) -- for logging/inspection."""
        return (torch.exp(self.attack_log_tau.detach()),
                torch.exp(self.release_log_tau.detach()),
                torch.exp(self.pilot_log_tau.detach()))

    def forward(self, level: torch.Tensor, state=None, parallel: bool = True):
        """
        Args:
            level: (B, L) normalized-dB level from :func:`level_db_norm`.
            state: (pilot, env) from a previous call -- pilot (B, 1),
                   env (B, n_bands) -- or None.
            parallel: solve the recurrences with the parallel scan (training)
                   vs the sequential one (streaming).  Identical results.

        Returns:
            env: (B, L, n_bands) detector envelopes (normalized-dB units).
            state: (pilot_last, env_last) for chunk-to-chunk carry.
        """
        B, L = level.shape
        scan = scan_parallel if parallel else scan_sequential
        x = self.in_gain * level + self.in_bias                  # (B, L)

        pilot0, env0 = state if state is not None else (None, None)

        # Pass 1: linear pilot envelope (the "recent average" the gate uses).
        a_p = coeff_from_log_tau(self.pilot_log_tau, self.sr)    # (1,)
        a_p_seq = a_p.view(1, 1, 1).expand(B, L, 1)
        b_p_seq = (1.0 - a_p) * x.unsqueeze(-1)                  # (B, L, 1)
        pilot, pilot_last = scan(a_p_seq, b_p_seq, pilot0)

        # Pass 2: gate on input vs pilot -> per-band time-varying coefficient.
        rising = torch.sigmoid(self.sharpness * (x.unsqueeze(-1) - pilot))
        a_att = coeff_from_log_tau(self.attack_log_tau, self.sr)   # (n_bands,)
        a_rel = coeff_from_log_tau(self.release_log_tau, self.sr)  # (n_bands,)
        coeff = rising * a_att + (1.0 - rising) * a_rel          # (B, L, n_bands)
        drive = (1.0 - coeff) * x.unsqueeze(-1)
        env, env_last = scan(coeff, drive, env0)

        return env, (pilot_last, env_last)

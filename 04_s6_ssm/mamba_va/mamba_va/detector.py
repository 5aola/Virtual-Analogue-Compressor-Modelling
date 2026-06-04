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
Mamba/S6's linear h-update lacks.  We make it (a) differentiable, by replacing
the hard branch with a soft gate, (b) learnable, with per-channel attack and
release time constants, and (c) multi-band, by running several detectors with
different learned time constants in parallel so the network can compose the
multiple release stages seen in devices like the LA-2A.

The detector output is used two ways: concatenated into the block features,
and (more importantly) fed to the SSM as the *selectivity signal* that sets the
SSM's own time constants -- coupling the linear long-memory SSM to a nonlinear,
level-dependent clock.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _coeff_from_tau(tau_logits: torch.Tensor) -> torch.Tensor:
    """Map unconstrained logits to a smoothing coefficient in (0, 1).

    coeff close to 1 -> long time constant (slow); close to 0 -> fast.
    """
    return torch.sigmoid(tau_logits)


class AdaptiveLevelDetector(nn.Module):
    def __init__(self, n_bands: int = 4, sharpness: float = 8.0):
        """
        Args:
            n_bands: number of parallel detectors with distinct, learned attack
                     and release time constants.
            sharpness: steepness of the soft attack/release branch. Higher ->
                     closer to the hard ``x > env`` switch.
        """
        super().__init__()
        self.n_bands = n_bands
        self.sharpness = sharpness

        # Initialise attack fast (small coeff) and release slow (large coeff),
        # spread across bands so they cover short to long time constants.
        spread = torch.linspace(-2.0, 2.0, n_bands)
        self.attack_logit = nn.Parameter(-2.0 + spread)     # fast-ish
        self.release_logit = nn.Parameter(2.0 + spread)     # slow-ish
        # learnable input shaping: detector sees a soft-rectified, scaled level
        self.in_gain = nn.Parameter(torch.ones(1))
        self.in_bias = nn.Parameter(torch.zeros(1))

    @property
    def out_dim(self) -> int:
        return self.n_bands

    def forward(self, level: torch.Tensor, state: torch.Tensor | None = None):
        """
        Args:
            level: (B, L) a non-negative-ish level/side-chain signal
                   (e.g. |input| or input energy).
            state: (B, n_bands) previous envelopes for streaming/TBPTT, or None.

        Returns:
            env: (B, L, n_bands) detector envelopes.
            env_last: (B, n_bands) final envelopes (for chunk carry).
        """
        B, L = level.shape
        x = F.softplus(self.in_gain * level + self.in_bias)  # smooth, >= 0
        a_att = _coeff_from_tau(self.attack_logit)           # (n_bands,)
        a_rel = _coeff_from_tau(self.release_logit)          # (n_bands,)

        env = torch.zeros(B, self.n_bands, device=level.device, dtype=level.dtype) \
            if state is None else state
        outs = []
        for t in range(L):
            xt = x[:, t : t + 1]                             # (B, 1)
            rising = torch.sigmoid(self.sharpness * (xt - env))  # soft x>env
            coeff = rising * a_att + (1.0 - rising) * a_rel       # (B, n_bands)
            env = coeff * env + (1.0 - coeff) * xt
            outs.append(env)
        env_seq = torch.stack(outs, dim=1)                  # (B, L, n_bands)
        return env_seq, env

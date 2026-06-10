"""Selective (input-dependent) diagonal state-space layer.

This is the S6 core (Gu & Dao, 2023), with one change motivated by the
compressor-modelling problem: the per-channel time constant ``Delta_t`` is a
function not only of the current sample but also of an *external selectivity
signal* ``s_t`` -- in practice the output of the adaptive level detector
(see ``detector.py``).  Because the detector itself has signal-dependent
attack/release memory, this makes the effective decay ``a_t = exp(Delta_t A)``
depend on the *history* of the signal level, not just its instantaneous value.
That is the mechanism by which the hidden state acquires the program-dependent,
asymmetric (fast-attack / slow-release) memory that a real optical cell has and
that a plain linear SSM cannot represent.

State recurrence (per channel d, state n):

    a_t[d,n] = exp(Delta_t[d] * A[d,n])          # A < 0  ->  a in (0,1)
    b_t[d,n] = Delta_t[d] * B_t[n] * x_t[d]       # ZOH-style input drive
    h_t[d,n] = a_t[d,n] h_{t-1}[d,n] + b_t[d,n]
    y_t[d]   = sum_n C_t[n] h_t[d,n] + D[d] x_t[d]

``B_t``, ``C_t`` and ``Delta_t`` are produced by linear projections of the
input (and the selectivity signal), i.e. they are *selective*.  ``A`` and ``D``
are learned and input-independent, as in Mamba.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .scan import scan_sequential, scan_parallel


class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int | None = None,
                 sel_dim: int = 0, dt_min: float = 1e-5, dt_max: float = 1e-2):
        """
        Args:
            d_model: number of channels D.
            d_state: SSM order N per channel.
            dt_rank: rank of the low-rank Delta projection (defaults to ceil(D/16)).
            sel_dim: width of the external selectivity signal fed into Delta
                     (0 disables it and recovers vanilla S6).
            dt_min/dt_max: initialisation range for the Delta bias.  Delta is
                     *per sample*, so with A in -[1..N] the slowest/fastest
                     pole time constants are 1/(dt_min*1) .. 1/(dt_max*N)
                     samples.  The defaults give ~0.14 ms .. 2.3 s at
                     44.1 kHz -- covering a compressor's attack *and* its
                     program-dependent release.  (Mamba's token-sequence
                     defaults of 1e-3/0.1 cap the memory at ~23 ms of audio,
                     which is why v0.1 could not learn the 0.4 s release.)
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.sel_dim = sel_dim
        self.dt_rank = dt_rank or max(1, math.ceil(d_model / 16))

        # selective projections: produce dt_rank (for Delta) + 2*N (for B, C)
        self.x_proj = nn.Linear(d_model + sel_dim, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        # Delta bias init so softplus(bias) lands in [dt_min, dt_max] (Mamba init)
        dt = torch.exp(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=min(dt_min, 1e-4))
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A: (D, N), real, negative. Parameterised as -exp(A_log) (HiPPO-style init).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor, sel: torch.Tensor | None = None,
                state: torch.Tensor | None = None, parallel: bool = True):
        """
        Args:
            x:    (B, L, D) input sequence.
            sel:  (B, L, sel_dim) external selectivity signal, or None.
            state:(B, D, N) initial hidden state for streaming/TBPTT, or None.
            parallel: use the parallel scan (training) vs sequential (streaming).

        Returns:
            y:        (B, L, D)
            h_last:   (B, D, N) final state (detached-ready for TBPTT carry)
        """
        B, L, D = x.shape
        N = self.d_state

        proj_in = x if (sel is None or self.sel_dim == 0) else torch.cat([x, sel], dim=-1)
        x_dbl = self.x_proj(proj_in)                       # (B, L, dt_rank + 2N)
        delta, Bc, Cc = torch.split(x_dbl, [self.dt_rank, N, N], dim=-1)
        delta = F.softplus(self.dt_proj(delta))            # (B, L, D)  > 0

        A = -torch.exp(self.A_log)                         # (D, N)  < 0
        # discretise.  b is formed as (delta*x) ⊗ B so the only (B,L,D,N)
        # tensors autograd retains are the decay a and the scan output h --
        # the adjoint-backward scan needs nothing else (see scan.py).
        a = torch.exp(delta.unsqueeze(-1) * A)             # (B, L, D, N) in (0,1)
        b = (delta * x).unsqueeze(-1) * Bc.unsqueeze(2)    # (B, L, D, N)

        a_flat = a.reshape(B, L, D * N)
        b_flat = b.reshape(B, L, D * N)
        h0 = None if state is None else state.reshape(B, D * N)

        scan = scan_parallel if parallel else scan_sequential
        h_flat, h_last = scan(a_flat, b_flat, h0)
        h = h_flat.reshape(B, L, D, N)

        y = torch.einsum("bldn,bln->bld", h, Cc) + x * self.D
        return y, h_last.reshape(B, D, N)

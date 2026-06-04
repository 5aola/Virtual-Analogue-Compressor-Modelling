"""First-order linear recurrence solvers (the SSM "scan").

The selective SSM reduces, per (channel, state) element, to the time-varying
linear recurrence

        h_t = a_t * h_{t-1} + b_t ,      a_t in (0, 1],  b_t in R

where ``a_t`` is the (discretized, input-dependent) decay and ``b_t`` is the
(discretized, input-dependent) drive.  Two solvers are provided:

* ``scan_sequential`` -- an explicit time loop.  Exact, O(L) memory, and the
  natural form for *streaming, sample-by-sample inference* (no windowing).
  It can also resume from a previous hidden state, which is what makes
  truncated back-prop-through-time and real-time operation possible.

* ``scan_parallel`` -- a Hillis-Steele associative scan using the monoid
  (a, b) o (a', b') = (a*a', a'*b + b').  Runs in O(L log L) work but is fully
  parallel over time, so training on long chunks is fast.  It stays entirely
  in the linear domain, so unlike the cumulative-product-in-log-space trick it
  does not overflow for strong decays.

Both return identical results (up to float error); ``tests/test_scan.py``
checks this.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def scan_sequential(a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor | None = None):
    """Reference / streaming solver.

    Args:
        a: decays, shape (B, L, ...). Values in (0, 1].
        b: drives, shape (B, L, ...).
        h0: optional initial state, shape (B, ...). Defaults to zeros.

    Returns:
        h: states, shape (B, L, ...).
        h_last: final state, shape (B, ...), for chunk-to-chunk carry.
    """
    L = a.shape[1]
    h = torch.zeros_like(a[:, 0]) if h0 is None else h0
    out = []
    for t in range(L):
        h = a[:, t] * h + b[:, t]
        out.append(h)
    h_stacked = torch.stack(out, dim=1)
    return h_stacked, h


def scan_parallel(a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor | None = None):
    """Parallel associative (inclusive) scan of the same recurrence.

    Args / returns: identical signature to :func:`scan_sequential`.

    The initial state ``h0`` is folded in by augmenting the drive of the first
    step with ``a_0 * h0`` (so that ``h_0 = a_0 * h0 + b_0``).
    """
    a = a.clone()
    b = b.clone()
    L = a.shape[1]

    if h0 is not None:
        # h_0 should equal a_0 * h0 + b_0
        b[:, 0] = b[:, 0] + a[:, 0] * h0

    # Hillis-Steele inclusive scan with the (a, b) monoid.
    shift = 1
    # build a (1, L, 1, ...) broadcastable position index once
    idx_shape = [1, L] + [1] * (a.dim() - 2)
    pos = torch.arange(L, device=a.device).view(idx_shape)
    while shift < L:
        # pad along time dim (dim=1) by `shift` at the front, drop the tail
        pad = [0, 0] * (a.dim() - 2) + [shift, 0]  # F.pad pads last dim first
        a_prev = F.pad(a, pad)[:, :L]
        b_prev = F.pad(b, pad)[:, :L]
        mask = pos >= shift
        new_b = torch.where(mask, a * b_prev + b, b)
        new_a = torch.where(mask, a * a_prev, a)
        a, b = new_a, new_b
        shift *= 2

    return b, b[:, -1]

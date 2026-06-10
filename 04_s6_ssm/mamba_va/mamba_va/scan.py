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

* ``scan_parallel`` -- a **two-level chunked** associative scan using the
  monoid (a, b) o (a', b') = (a*a', a'*b + b').  The sequence is split into
  chunks of ``chunk`` steps; a Hillis-Steele scan runs *within* all chunks in
  parallel (log2(chunk) full-size rounds), a second tiny scan combines the
  per-chunk carries (log2(L/chunk) rounds at 1/chunk the size), and one fused
  multiply-add broadcasts the carries back in.  Versus a flat Hillis-Steele
  scan this cuts the number of full-size rounds from log2(L) (e.g. 12 at
  L=4096) to log2(chunk)+1 (7 at chunk=64) -- the scan is memory-bound, so
  wall time and backward-graph size drop proportionally.  It stays entirely
  in the linear domain, so unlike the cumulative-product-in-log-space trick it
  does not overflow for strong decays (the long-release regime we care about).

Both return identical results (up to float error); ``tests/test_scan.py``
checks this for many lengths, with and without an initial state.
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


def _hillis_steele(a: torch.Tensor, b: torch.Tensor):
    """Flat inclusive scan along dim=1 with the (a, b) monoid.

    Returns the cumulative (A, B) pairs: ``B[:, t]`` is the state at t from a
    zero initial state, ``A[:, t]`` is the cumulative decay prod(a[0..t]).
    """
    L = a.shape[1]
    idx_shape = [1, L] + [1] * (a.dim() - 2)
    pos = torch.arange(L, device=a.device).view(idx_shape)
    shift = 1
    while shift < L:
        pad = [0, 0] * (a.dim() - 2) + [shift, 0]  # F.pad pads last dim first
        a_prev = F.pad(a, pad)[:, :L]
        b_prev = F.pad(b, pad)[:, :L]
        mask = pos >= shift
        b = torch.where(mask, a * b_prev + b, b)
        a = torch.where(mask, a * a_prev, a)
        shift *= 2
    return a, b


def scan_parallel(a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor | None = None,
                  chunk: int = 64):
    """Parallel associative (inclusive) scan of the same recurrence.

    Args / returns: identical to :func:`scan_sequential`, plus ``chunk`` --
    the within-chunk scan width of the two-level decomposition (power of two;
    lengths that are not a multiple are padded with the identity element
    a=1, b=0 and trimmed afterwards).

    The initial state ``h0`` is folded in by augmenting the drive of the first
    step with ``a_0 * h0`` (so that ``h_0 = a_0 * h0 + b_0``).
    """
    B, L = a.shape[0], a.shape[1]
    rest = a.shape[2:]

    if h0 is not None:
        # h_0 should equal a_0 * h0 + b_0.  Clone b before the in-place write;
        # a is never mutated, so it needs no copy (saves a full-size alloc).
        b = b.clone()
        b[:, 0] = b[:, 0] + a[:, 0] * h0

    if L <= 2 * chunk:
        _, h = _hillis_steele(a, b)
        return h, h[:, -1]

    n_chunks = (L + chunk - 1) // chunk
    pad_n = n_chunks * chunk - L
    if pad_n:
        a = torch.cat([a, a.new_ones(B, pad_n, *rest)], dim=1)
        b = torch.cat([b, b.new_zeros(B, pad_n, *rest)], dim=1)

    # level 1: scan within all chunks at once (folded into the batch dim)
    A_w, B_w = _hillis_steele(a.reshape(B * n_chunks, chunk, *rest),
                              b.reshape(B * n_chunks, chunk, *rest))
    A_w = A_w.reshape(B, n_chunks, chunk, *rest)
    B_w = B_w.reshape(B, n_chunks, chunk, *rest)

    # level 2: scan the per-chunk summaries (total decay, end state)
    _, B_c = _hillis_steele(A_w[:, :, -1], B_w[:, :, -1])      # (B, n_chunks, ...)

    # broadcast the carry entering each chunk (zero for the first chunk)
    carry = torch.cat([torch.zeros_like(B_c[:, :1]), B_c[:, :-1]], dim=1)
    h = B_w + A_w * carry.unsqueeze(2)
    h = h.reshape(B, n_chunks * chunk, *rest)[:, :L]
    return h, h[:, -1]

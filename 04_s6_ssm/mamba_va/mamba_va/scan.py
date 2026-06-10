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
  per-chunk carries, and one fused multiply-add broadcasts the carries back
  in.  It stays entirely in the linear domain, so unlike the cumulative-
  product-in-log-space trick it does not overflow for strong decays (the
  long-release regime we care about).

``scan_parallel`` is wrapped in a ``torch.autograd.Function`` with an
**analytical adjoint backward**: for the recurrence above,

        lambda_t = g_t + a_{t+1} * lambda_{t+1}        (g_t = dL/dh_t)
        dL/db_t  = lambda_t
        dL/da_t  = lambda_t * h_{t-1}
        dL/dh0   = lambda_0 * a_0

i.e. the backward pass is *the same scan run in reverse* plus two elementwise
products.  Only ``a``, the outputs ``h`` and (if given) ``h0`` are kept for
backward -- the scan's internal rounds are never retained by autograd.  This
matters: backprop *through* the Hillis-Steele rounds retains O(log L)
full-size tensors per layer (multiple GiB at audio lengths) and was the cause
of CUDA OOM at L=16k; the adjoint form needs none of it and is faster than
checkpoint-recompute as well.

``tests/test_scan.py`` checks parallel == sequential for many lengths (with
and without an initial state) and that the analytical gradients match
autograd-through-the-sequential-loop, plus a float64 ``gradcheck``.
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


def _hillis_steele(a: torch.Tensor, b: torch.Tensor, reverse: bool = False):
    """Flat inclusive scan along dim=1 with the (a, b) monoid.

    Forward (reverse=False): ``B[:, t]`` is the state at t from a zero initial
    state, ``A[:, t]`` is the cumulative decay prod(a[0..t]).

    Reverse (reverse=True) solves the *adjoint* recurrence
    ``lam_t = a_t * lam_{t+1} + b_t`` from a zero state at the right end --
    used by the analytical backward.  Implemented natively (partner element
    taken from t+shift, padding at the tail) so no flip copies are needed.
    """
    L = a.shape[1]
    idx_shape = [1, L] + [1] * (a.dim() - 2)
    pos = torch.arange(L, device=a.device).view(idx_shape)
    shift = 1
    while shift < L:
        if reverse:
            pad = [0, 0] * (a.dim() - 2) + [0, shift]  # F.pad pads last dim first
            a_prev = F.pad(a, pad)[:, shift:]
            b_prev = F.pad(b, pad)[:, shift:]
            mask = pos <= (L - 1 - shift)
        else:
            pad = [0, 0] * (a.dim() - 2) + [shift, 0]
            a_prev = F.pad(a, pad)[:, :L]
            b_prev = F.pad(b, pad)[:, :L]
            mask = pos >= shift
        b = torch.where(mask, a * b_prev + b, b)
        a = torch.where(mask, a * a_prev, a)
        shift *= 2
    return a, b


def _scan_inclusive(a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor | None,
                    chunk: int, reverse: bool = False):
    """Pure (autograd-agnostic) chunked inclusive scan.  See scan_parallel.

    reverse=True solves ``h_t = a_t * h_{t+1} + b_t`` (zero state past the
    end; ``h0`` unsupported) and returns (h, h[:, 0]).
    """
    B, L = a.shape[0], a.shape[1]
    rest = a.shape[2:]

    if h0 is not None:
        assert not reverse, "h0 is not supported for the reverse scan"
        # h_0 should equal a_0 * h0 + b_0.  Clone b before the in-place write;
        # a is never mutated, so it needs no copy.
        b = b.clone()
        b[:, 0] = b[:, 0] + a[:, 0] * h0

    if L <= 2 * chunk:
        _, h = _hillis_steele(a, b, reverse)
        return h, h[:, 0 if reverse else -1]

    n_chunks = (L + chunk - 1) // chunk
    pad_n = n_chunks * chunk - L
    if pad_n:
        # pad with the identity element (a=1, b=0) at the tail; for the
        # reverse scan the zero state past the end makes the tail contribute
        # nothing either way, and the pad region is trimmed below.
        a = torch.cat([a, a.new_ones(B, pad_n, *rest)], dim=1)
        b = torch.cat([b, b.new_zeros(B, pad_n, *rest)], dim=1)

    # level 1: scan within all chunks at once (folded into the batch dim)
    A_w, B_w = _hillis_steele(a.reshape(B * n_chunks, chunk, *rest),
                              b.reshape(B * n_chunks, chunk, *rest), reverse)
    A_w = A_w.reshape(B, n_chunks, chunk, *rest)
    B_w = B_w.reshape(B, n_chunks, chunk, *rest)

    # level 2: scan the per-chunk summaries (total decay, boundary state)
    edge = 0 if reverse else -1
    _, B_c = _hillis_steele(A_w[:, :, edge], B_w[:, :, edge], reverse)

    # broadcast the carry entering each chunk (zero for the boundary chunk)
    if reverse:
        carry = torch.cat([B_c[:, 1:], torch.zeros_like(B_c[:, :1])], dim=1)
    else:
        carry = torch.cat([torch.zeros_like(B_c[:, :1]), B_c[:, :-1]], dim=1)
    h = B_w + A_w * carry.unsqueeze(2)
    h = h.reshape(B, n_chunks * chunk, *rest)[:, :L]
    return h, h[:, 0 if reverse else -1]


class _ScanParallelFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, h0, chunk):
        # Function.forward runs with grad disabled; the scan's internal
        # rounds therefore leave no autograd graph behind.
        h, h_last = _scan_inclusive(a, b, h0, chunk)
        h_last = h_last.clone()  # outputs of a Function must not alias each other
        if h0 is None:
            ctx.save_for_backward(a, h)
        else:
            ctx.save_for_backward(a, h, h0)
        ctx.has_h0 = h0 is not None
        ctx.chunk = chunk
        ctx.set_materialize_grads(False)
        return h, h_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_h, grad_hlast):
        saved = ctx.saved_tensors
        a, h = saved[0], saved[1]
        h0 = saved[2] if ctx.has_h0 else None
        if grad_h is None and grad_hlast is None:
            return None, None, None, None

        if grad_h is None:
            g = torch.zeros_like(h)
        elif grad_hlast is not None:
            g = grad_h.clone()
        else:
            g = grad_h
        if grad_hlast is not None:
            g[:, -1] = g[:, -1] + grad_hlast

        # adjoint scan, reversed in time: lambda_t = g_t + a_{t+1} lambda_{t+1}
        a_next = torch.cat([a[:, 1:], torch.ones_like(a[:, :1])], dim=1)
        lam, lam0 = _scan_inclusive(a_next, g, None, ctx.chunk, reverse=True)
        del a_next, g

        grad_b = lam
        grad_a = None
        if ctx.needs_input_grad[0]:
            # grad_a_t = lam_t * h_{t-1}; assembled by slices to avoid a
            # full-size shifted copy of h.
            grad_a = torch.empty_like(a)
            torch.mul(lam[:, 1:], h[:, :-1], out=grad_a[:, 1:])
            if h0 is None:
                grad_a[:, 0] = 0.0
            else:
                torch.mul(lam0, h0, out=grad_a[:, 0])
        grad_h0 = None
        if ctx.has_h0 and ctx.needs_input_grad[2]:
            grad_h0 = lam0 * a[:, 0]
        return grad_a, grad_b, grad_h0, None


def scan_parallel(a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor | None = None,
                  chunk: int = 64):
    """Parallel associative (inclusive) scan of the same recurrence.

    Args / returns: identical to :func:`scan_sequential`, plus ``chunk`` --
    the within-chunk scan width of the two-level decomposition (power of two;
    lengths that are not a multiple are padded with the identity element
    a=1, b=0 and trimmed afterwards).

    The initial state ``h0`` is folded in by augmenting the drive of the first
    step with ``a_0 * h0`` (so that ``h_0 = a_0 * h0 + b_0``).  Gradients are
    computed analytically via the adjoint recurrence (see module docstring),
    so training memory is O(L) regardless of length.
    """
    needs_grad = torch.is_grad_enabled() and (
        a.requires_grad or b.requires_grad
        or (h0 is not None and h0.requires_grad))
    if needs_grad:
        return _ScanParallelFn.apply(a, b, h0, chunk)
    return _scan_inclusive(a, b, h0, chunk)

"""Parallel scan must equal the sequential reference, including state carry."""

import torch
from mamba_va.scan import scan_sequential, scan_parallel


def test_parallel_equals_sequential():
    torch.manual_seed(0)
    B, L, D = 3, 257, 5
    a = torch.rand(B, L, D) * 0.999 + 1e-4      # decays in (0,1)
    b = torch.randn(B, L, D)
    h_seq, last_seq = scan_sequential(a, b)
    h_par, last_par = scan_parallel(a, b)
    assert torch.allclose(h_seq, h_par, atol=1e-4), (h_seq - h_par).abs().max()
    assert torch.allclose(last_seq, last_par, atol=1e-4)


def test_state_carry_matches_full_pass():
    """Running two chunks with carried state == one full pass."""
    torch.manual_seed(1)
    B, L, D = 2, 200, 4
    a = torch.rand(B, L, D) * 0.99 + 1e-3
    b = torch.randn(B, L, D)
    full, _ = scan_sequential(a, b)

    cut = 73
    h1, last1 = scan_parallel(a[:, :cut], b[:, :cut])
    h2, _ = scan_parallel(a[:, cut:], b[:, cut:], h0=last1)
    joined = torch.cat([h1, h2], dim=1)
    assert torch.allclose(full, joined, atol=1e-4), (full - joined).abs().max()


def test_chunked_scan_many_lengths():
    """The two-level chunked path must be exact for all length/chunk
    relations: shorter than a chunk, exact multiples, ragged tails."""
    torch.manual_seed(2)
    for L in [1, 5, 64, 100, 128, 129, 257, 640, 1000, 4096]:
        for with_h0 in (False, True):
            a = torch.rand(2, L, 3) * 0.999 + 5e-4
            b = torch.randn(2, L, 3)
            h0 = torch.randn(2, 3) if with_h0 else None
            h_seq, last_seq = scan_sequential(a, b, h0)
            h_par, last_par = scan_parallel(a, b, h0)
            err = (h_seq - h_par).abs().max()
            assert torch.allclose(h_seq, h_par, atol=1e-4), (L, with_h0, err)
            assert torch.allclose(last_seq, last_par, atol=1e-4), (L, with_h0)


def test_chunked_scan_slow_decays():
    """Decays within 1e-5 of 1.0 (multi-second release regime) must not
    lose accuracy in the chunked recombination."""
    torch.manual_seed(3)
    B, L, D = 2, 2048, 4
    a = 1.0 - torch.rand(B, L, D) * 1e-5
    b = torch.randn(B, L, D) * 1e-3
    h_seq, _ = scan_sequential(a, b)
    h_par, _ = scan_parallel(a, b)
    assert torch.allclose(h_seq, h_par, atol=1e-4), (h_seq - h_par).abs().max()


def test_adjoint_gradients_match_sequential():
    """The analytical adjoint backward of scan_parallel must equal autograd
    through the sequential loop, for both outputs, all inputs, ragged L."""
    torch.manual_seed(4)
    for L in [1, 5, 64, 100, 257]:
        B, D = 2, 3
        a0 = torch.rand(B, L, D) * 0.99 + 5e-3
        b0 = torch.randn(B, L, D)
        h00 = torch.randn(B, D)
        w = torch.randn(B, L, D)     # upstream grads for h
        v = torch.randn(B, D)        # upstream grads for h_last

        grads = {}
        for name, scan in (("seq", scan_sequential), ("par", scan_parallel)):
            a = a0.clone().requires_grad_(True)
            b = b0.clone().requires_grad_(True)
            h0 = h00.clone().requires_grad_(True)
            h, h_last = scan(a, b, h0)
            loss = (h * w).sum() + (h_last * v).sum()
            grads[name] = torch.autograd.grad(loss, (a, b, h0))
        for ga, gb in zip(grads["seq"], grads["par"]):
            err = (ga - gb).abs().max()
            assert torch.allclose(ga, gb, atol=1e-3), (L, err)


def test_gradcheck_float64():
    """Numerical gradcheck of the custom Function (chunked path, L > 2*chunk)."""
    torch.manual_seed(5)
    B, L, D, chunk = 2, 9, 3, 4
    a = (torch.rand(B, L, D, dtype=torch.float64) * 0.9 + 0.05).requires_grad_(True)
    b = torch.randn(B, L, D, dtype=torch.float64, requires_grad=True)
    h0 = torch.randn(B, D, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a_, b_, h0_: scan_parallel(a_, b_, h0_, chunk=chunk), (a, b, h0))


def test_no_graph_blowup():
    """Backward must not retain the scan's internal rounds: peak autograd
    memory stays within a small multiple of the I/O tensors (proxy check:
    backward succeeds and grads are finite at a notebook-like size)."""
    a = (torch.rand(2, 16384, 48) * 0.5 + 0.49).requires_grad_(True)
    b = torch.randn(2, 16384, 48, requires_grad=True)
    h, h_last = scan_parallel(a, b)
    (h.square().mean() + h_last.sum()).backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


if __name__ == "__main__":
    test_parallel_equals_sequential()
    test_state_carry_matches_full_pass()
    test_chunked_scan_many_lengths()
    test_chunked_scan_slow_decays()
    test_adjoint_gradients_match_sequential()
    test_gradcheck_float64()
    test_no_graph_blowup()
    print("scan tests passed")

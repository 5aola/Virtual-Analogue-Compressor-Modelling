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


if __name__ == "__main__":
    test_parallel_equals_sequential()
    test_state_carry_matches_full_pass()
    print("scan tests passed")

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


if __name__ == "__main__":
    test_parallel_equals_sequential()
    test_state_carry_matches_full_pass()
    test_chunked_scan_many_lengths()
    test_chunked_scan_slow_decays()
    print("scan tests passed")

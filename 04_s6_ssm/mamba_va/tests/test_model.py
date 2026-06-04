"""Model-level invariants: shape, causality, streaming==parallel, gradients."""

import torch
from mamba_va import CompSSM


def _model():
    torch.manual_seed(0)
    return CompSSM(n_params=4, d_model=12, d_state=8, n_layers=2, n_bands=3).eval()


def test_shapes_and_grad():
    m = _model().train()
    u = torch.randn(2, 300) * 0.3
    p = torch.rand(2, 4)
    y, _ = m(u, p, parallel=True)
    assert y.shape == (2, 300)
    y.pow(2).mean().backward()
    g = [pm.grad for pm in m.parameters() if pm.grad is not None]
    assert len(g) > 0 and all(torch.isfinite(x).all() for x in g)


def test_causality():
    """Changing a future sample must not change the current output."""
    m = _model()
    u = torch.randn(1, 200) * 0.3
    p = torch.rand(1, 4)
    with torch.no_grad():
        y1, _ = m(u, p, parallel=False)
        u2 = u.clone()
        u2[0, 150:] += 5.0           # perturb the future
        y2, _ = m(u2, p, parallel=False)
    assert torch.allclose(y1[0, :150], y2[0, :150], atol=1e-5)


def test_streaming_matches_parallel():
    """Chunked streaming (sequential, state carry) == single parallel pass."""
    m = _model()
    u = torch.randn(2, 257) * 0.3
    p = torch.rand(2, 4)
    with torch.no_grad():
        y_full, _ = m(u, p, parallel=True)
        state = None
        outs = []
        for s in range(0, u.shape[1], 64):
            yc, state = m(u[:, s : s + 64], p, state, parallel=False)
            outs.append(yc)
        y_stream = torch.cat(outs, dim=1)
    assert torch.allclose(y_full, y_stream, atol=1e-4), (y_full - y_stream).abs().max()


if __name__ == "__main__":
    test_shapes_and_grad()
    test_causality()
    test_streaming_matches_parallel()
    print("model tests passed")

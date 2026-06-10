"""Detector invariants: parallel == sequential == chunked streaming, learned
time constants land where the init says, and gradients reach every parameter."""

import math
import torch
from mamba_va.detector import (AdaptiveLevelDetector, coeff_from_log_tau,
                               level_db_norm)


def _detector():
    torch.manual_seed(0)
    return AdaptiveLevelDetector(n_bands=4, sr=44100.0)


def test_parallel_equals_sequential():
    det = _detector()
    level = level_db_norm(torch.randn(3, 500) * 0.3)
    with torch.no_grad():
        env_p, (pp, ep) = det(level, parallel=True)
        env_s, (ps, es) = det(level, parallel=False)
    assert torch.allclose(env_p, env_s, atol=1e-5), (env_p - env_s).abs().max()
    assert torch.allclose(pp, ps, atol=1e-5)
    assert torch.allclose(ep, es, atol=1e-5)


def test_streaming_matches_full_pass():
    """Chunked calls with carried (pilot, env) state == one full pass."""
    det = _detector()
    level = level_db_norm(torch.randn(2, 333) * 0.3)
    with torch.no_grad():
        env_full, _ = det(level, parallel=True)
        state, outs = None, []
        for s in range(0, level.shape[1], 50):
            env_c, state = det(level[:, s : s + 50], state, parallel=False)
            outs.append(env_c)
        env_stream = torch.cat(outs, dim=1)
    assert torch.allclose(env_full, env_stream, atol=1e-5), \
        (env_full - env_stream).abs().max()


def test_time_constant_mapping():
    """coeff_from_log_tau must place a 0.4 s release at the right coefficient
    -- the regime the sigmoid-logit parameterisation could not reach."""
    sr = 44100.0
    coeff = coeff_from_log_tau(torch.tensor(math.log(0.4)), sr)
    assert torch.allclose(coeff, torch.tensor(math.exp(-1.0 / (sr * 0.4))))
    # one-pole with that coeff decays to 1/e in ~0.4 s
    n = int(0.4 * sr)
    assert abs(coeff.item() ** n - math.exp(-1)) < 1e-3
    # init ranges actually span ms..s
    det = _detector()
    att, rel, pilot = det.time_constants()
    assert att.min() >= 1e-4 - 1e-9 and att.max() <= 3e-2 + 1e-9
    assert rel.min() >= 2e-2 - 1e-9 and rel.max() >= 1.0  # slowest band > 1 s


def test_gradients_reach_all_params():
    det = _detector()
    level = level_db_norm(torch.randn(2, 256) * 0.3)
    env, _ = det(level, parallel=True)
    env.pow(2).mean().backward()
    for name, p in det.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, f"no gradient signal in {name}"


if __name__ == "__main__":
    test_parallel_equals_sequential()
    test_streaming_matches_full_pass()
    test_time_constant_mapping()
    test_gradients_reach_all_params()
    print("detector tests passed")

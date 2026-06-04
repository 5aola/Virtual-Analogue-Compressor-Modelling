"""Small shared helpers."""

from __future__ import annotations

import torch


def detach_state(state):
    """Detach every tensor in a (possibly nested) streaming-state structure,
    so it can be carried across TBPTT chunks without growing the graph."""
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, dict):
        return {k: detach_state(v) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return type(state)(detach_state(v) for v in state)
    return state


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

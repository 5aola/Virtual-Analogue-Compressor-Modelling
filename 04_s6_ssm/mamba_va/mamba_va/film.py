"""FiLM conditioning on the device control parameters.

The device parameters (threshold, ratio, attack, release, peak-reduction,
limit/compress switch, ...) are static per recording.  Following Simionato &
Fasciani we modulate the network features with Feature-wise Linear Modulation:
``y = gamma(p) * x + beta(p)``.  A small MLP maps the parameter vector ``p`` to
per-channel scale and shift.  Because ``p`` is constant over a chunk this adds
negligible cost and lets one model cover the whole parameter space.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, n_params: int, d_model: int, hidden: int = 32):
        super().__init__()
        self.n_params = n_params
        self.net = nn.Sequential(
            nn.Linear(n_params, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * d_model),
        )

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      (B, L, D)
            params: (B, n_params) static control parameters in [0, 1].
        """
        gamma, beta = self.net(params).chunk(2, dim=-1)     # (B, D) each
        return (1.0 + gamma).unsqueeze(1) * x + beta.unsqueeze(1)

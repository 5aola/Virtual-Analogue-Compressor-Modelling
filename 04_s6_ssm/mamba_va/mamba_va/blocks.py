"""The CompSSM block: a Mamba-style gated block whose SSM is clocked by the
adaptive level detector.

Layout (residual around the whole thing):

    u  --RMSNorm--> in_proj --> [x_ssm | gate]
    x_ssm --causal depthwise conv--> SiLU --> SelectiveSSM(sel=detector_env)
        --> y
    y = y * SiLU(gate)
    out = out_proj(y)
    return u + out

The only structural departure from vanilla Mamba is the ``sel`` argument: the
detector envelope is concatenated into the SSM's selective projection so the
SSM time constants follow the (nonlinear, asymmetric) signal level.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssm import SelectiveSSM


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise causal conv that can carry its (k-1)-sample tail across chunks
    so that chunked / streaming inference is exactly equal to a single pass."""

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              groups=channels, padding=0, bias=True)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        # x: (B, L, C)
        B, L, C = x.shape
        xt = x.transpose(1, 2)                              # (B, C, L)
        if state is None:
            pad = torch.zeros(B, C, self.kernel_size - 1, device=x.device, dtype=x.dtype)
        else:
            pad = state                                     # (B, C, k-1)
        xt = torch.cat([pad, xt], dim=-1)
        new_state = xt[:, :, -(self.kernel_size - 1):] if self.kernel_size > 1 else None
        y = self.conv(xt)                                   # (B, C, L)
        return y.transpose(1, 2), new_state


class CompSSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2,
                 conv_kernel: int = 4, sel_dim: int = 0,
                 dt_min: float = 1e-5, dt_max: float = 1e-2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv = CausalDepthwiseConv1d(self.d_inner, conv_kernel)
        self.ssm = SelectiveSSM(self.d_inner, d_state=d_state, sel_dim=sel_dim,
                                dt_min=dt_min, dt_max=dt_max)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u, sel=None, state=None, parallel=True):
        """state: dict with keys 'conv', 'ssm' or None."""
        state = state or {}
        x = self.norm(u)
        x, gate = self.in_proj(x).chunk(2, dim=-1)          # (B,L,d_inner) each
        x, conv_state = self.conv(x, state.get("conv"))
        x = F.silu(x)
        y, ssm_state = self.ssm(x, sel=sel, state=state.get("ssm"), parallel=parallel)
        y = y * F.silu(gate)
        out = self.out_proj(y)
        return u + out, {"conv": conv_state, "ssm": ssm_state}

"""CompSSM -- the full model.

A streaming, sample-by-sample (no windowing / no tokenization) selective
state-space model for nonlinear, time-variant audio effects.

Signal path:

    u_t (scalar)
      |-- level_db_norm(u) --> AdaptiveLevelDetector --> env_t  (nonlinear memory)
      |-- input_proj(u) --> x_t  (B,L,D)
    x_t = FiLM(x_t, params)                      # device-parameter conditioning
    x_t, env_t --> [CompSSMBlock x n_layers]     # SSM clocked by detector
      --> features
    g_t = GainComputer(features, env, params)    # nonlinear gain in dB
    y_t = u_t * 10**(g_t / 20)                    # multiplicative gain (compressor prior)

Why multiplicative?  A compressor *applies a time-varying gain* to the input.
Predicting that gain (rather than the raw waveform) bakes the structure of the
device into the model: it removes the need for the 64-sample look-ahead window
Riccardo used to suppress boundary artifacts, it makes silence map to silence
exactly, and it concentrates the network's capacity on the thing that is
actually hard -- the time-varying, level-dependent gain.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import CompSSMBlock
from .detector import AdaptiveLevelDetector, level_db_norm
from .film import FiLM


class GainComputer(nn.Module):
    """Maps block features (+ detector env + params) to a gain in decibels,
    bounded to [-max_db, +max_db]."""

    def __init__(self, d_model: int, det_dim: int, n_params: int,
                 hidden: int = 32, max_db: float = 48.0):
        super().__init__()
        self.max_db = max_db
        self.net = nn.Sequential(
            nn.Linear(d_model + det_dim + n_params, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat, env, params):
        B, L, _ = feat.shape
        p = params.unsqueeze(1).expand(B, L, params.shape[-1])
        z = torch.cat([feat, env, p], dim=-1)
        g = torch.tanh(self.net(z).squeeze(-1)) * self.max_db   # (B, L) dB
        return g


class CompSSM(nn.Module):
    def __init__(self, n_params: int = 4, d_model: int = 24, d_state: int = 16,
                 n_layers: int = 3, expand: int = 2, conv_kernel: int = 4,
                 n_bands: int = 4, max_db: float = 48.0, sr: float = 44100.0,
                 dt_min: float = 1e-5, dt_max: float = 1e-2):
        super().__init__()
        self.n_params = n_params
        self.sr = float(sr)
        self.detector = AdaptiveLevelDetector(n_bands=n_bands, sr=sr)
        det_dim = self.detector.out_dim

        self.input_proj = nn.Linear(1, d_model)
        self.film = FiLM(n_params, d_model) if n_params > 0 else None

        self.blocks = nn.ModuleList([
            CompSSMBlock(d_model, d_state=d_state, expand=expand,
                         conv_kernel=conv_kernel, sel_dim=det_dim,
                         dt_min=dt_min, dt_max=dt_max)
            for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.gain = GainComputer(d_model, det_dim, n_params, max_db=max_db)

    def init_state(self):
        return {"det": None, "blocks": [None] * len(self.blocks)}

    def forward(self, u, params=None, state=None, parallel=True, return_gain=False):
        """
        Args:
            u:      (B, L) or (B, L, 1) the dry input samples.
            params: (B, n_params) device controls in [0, 1] (or None if n_params==0).
            state:  streaming/TBPTT state dict from a previous call, or None.
            parallel: parallel scan (training) vs sequential (streaming).
            return_gain: also return the predicted gain (dB) per sample.

        Returns:
            y:        (B, L) processed output.
            state:    updated state dict.
            [gain_db: (B, L)] if return_gain.
        """
        if u.dim() == 3:
            u = u.squeeze(-1)
        B, L = u.shape
        state = state or self.init_state()
        if params is None:
            params = torch.zeros(B, self.n_params, device=u.device, dtype=u.dtype)

        level = level_db_norm(u)                                 # normalized dB
        env, det_last = self.detector(level, state.get("det"),
                                      parallel=parallel)         # (B,L,n_bands)

        x = self.input_proj(u.unsqueeze(-1))
        if self.film is not None:
            x = self.film(x, params)

        new_block_states = []
        for blk, st in zip(self.blocks, state["blocks"]):
            x, st_new = blk(x, sel=env, state=st, parallel=parallel)
            new_block_states.append(st_new)

        feat = self.norm_f(x)
        g_db = self.gain(feat, env, params)                      # (B, L)
        y = u * torch.pow(10.0, g_db / 20.0)

        new_state = {"det": det_last, "blocks": new_block_states}
        if return_gain:
            return y, new_state, g_db
        return y, new_state

    @torch.no_grad()
    def render(self, u, params=None, chunk: int = 16384):
        """Stream a long signal through the model in chunks, carrying state.
        Equivalent to a single causal pass; use for inference on full files."""
        self.eval()
        if u.dim() == 1:
            u = u.unsqueeze(0)
        B, L = u.shape
        state = self.init_state()
        outs = []
        for s in range(0, L, chunk):
            y, state = self.forward(u[:, s : s + chunk], params, state, parallel=False)
            outs.append(y)
        return torch.cat(outs, dim=1)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

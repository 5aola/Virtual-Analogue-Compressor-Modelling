"""Losses for time-domain audio-effect modelling.

* ESR  -- error-to-signal ratio (the standard VA-modelling metric).
* Pre-emphasised ESR -- ESR after a first-order high-pass, which weights the
  transient/onset error that Simionato & Fasciani identify as the hardest part
  of compressor modelling (fast attack).
* Multi-resolution STFT -- spectral convergence + log-magnitude over several
  window sizes, to penalise the high-frequency / spectral mismatch they report.

``CombinedLoss`` mixes them; tune the weights in the config.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def esr(y_hat, y, eps=1e-8):
    num = torch.sum((y - y_hat) ** 2, dim=-1)
    den = torch.sum(y ** 2, dim=-1) + eps
    return (num / den).mean()


def pre_emphasis(x, coeff=0.95):
    # first-order high-pass: y[n] = x[n] - coeff*x[n-1]
    x0 = F.pad(x, (1, 0))[..., :-1]
    return x - coeff * x0


def dc_loss(y_hat, y):
    return (y_hat.mean(-1) - y.mean(-1)).abs().mean()


class MRSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(512, 1024, 2048), hop_ratio=0.25, eps=1e-7):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_ratio = hop_ratio
        self.eps = eps

    def _stft_mag(self, x, n_fft):
        win = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
        hop = max(1, int(n_fft * self.hop_ratio))
        X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                       window=win, return_complex=True, center=True)
        return X.abs()

    def forward(self, y_hat, y):
        L = y.shape[-1]
        # only use FFT sizes that fit the (chunk) length; keep at least one
        sizes = [n for n in self.fft_sizes if n < L] or [min(self.fft_sizes[0], 2 ** int(L).bit_length() // 2 or 2)]
        total = 0.0
        for n in sizes:
            S_hat = self._stft_mag(y_hat, n)
            S = self._stft_mag(y, n)
            sc = torch.norm(S - S_hat, p="fro") / (torch.norm(S, p="fro") + self.eps)
            mag = F.l1_loss(torch.log(S_hat + self.eps), torch.log(S + self.eps))
            total = total + sc + mag
        return total / len(sizes)


class CombinedLoss(nn.Module):
    def __init__(self, w_esr=1.0, w_preemph=0.5, w_stft=0.5, w_dc=0.1,
                 preemph_coeff=0.95, fft_sizes=(512, 1024, 2048)):
        super().__init__()
        self.w_esr = w_esr
        self.w_preemph = w_preemph
        self.w_stft = w_stft
        self.w_dc = w_dc
        self.preemph_coeff = preemph_coeff
        self.mrstft = MRSTFTLoss(fft_sizes=fft_sizes)

    def forward(self, y_hat, y):
        loss = self.w_esr * esr(y_hat, y)
        if self.w_preemph:
            loss = loss + self.w_preemph * esr(
                pre_emphasis(y_hat, self.preemph_coeff),
                pre_emphasis(y, self.preemph_coeff),
            )
        if self.w_dc:
            loss = loss + self.w_dc * dc_loss(y_hat, y)
        if self.w_stft:
            loss = loss + self.w_stft * self.mrstft(y_hat, y)
        return loss

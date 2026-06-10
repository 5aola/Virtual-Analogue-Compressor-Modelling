"""GR target normalisation for the scalar [0, 1] sigmoid regression head.

Limits derived from the exported Diff-SSL gr_curves (100 files = 10 songs x
10 settings): observed min = -28.252 dB, max = +4.213 dB, no NaN/Inf.
[-30, +5] covers that with margin so the sigmoid never has to saturate at
0/1 (0 dB GR maps to 0.857, the observed max to 0.977). See NORMALISATION.md.
"""

import torch

GR_DB_MIN = -30.0
GR_DB_MAX = 5.0


def normalize_gr_01(gr_db: torch.Tensor) -> torch.Tensor:
    """Map GR in dB from [GR_DB_MIN, GR_DB_MAX] to [0, 1]."""
    return (gr_db - GR_DB_MIN) / (GR_DB_MAX - GR_DB_MIN)


def denormalize_gr_01(gr_01: torch.Tensor) -> torch.Tensor:
    """Map [0, 1] back to dB in [GR_DB_MIN, GR_DB_MAX]."""
    return gr_01 * (GR_DB_MAX - GR_DB_MIN) + GR_DB_MIN

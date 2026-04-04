"""
SOTA analysis loss module — re-exports from ``src.losses``.

Backward-compatible: ``import loss`` or ``from loss import ...``
continue to work.
"""

from src.losses import (  # noqa: F401
    compute_all_losses,
    esr,
    evaluate_to_dataframe,
    m_stfte,
    mae,
    mse,
    rmse,
    spectral_flux_error,
)

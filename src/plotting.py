"""
Gain-curve estimation and comparison plots.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.dsp import moving_rms


def regression(x, y):
    """Perform polynomial regression on the given x and y data."""
    if len(x) != len(y):
        raise ValueError("x and y must be of the same length")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    coeffs = np.polyfit(x, y, 6)
    poly_eq = np.poly1d(coeffs)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = poly_eq(x_fit)
    return x_fit, y_fit


def estimate_gain_curve(x, y, window_size=1024, lims=(-80, 0)):
    """Estimate and plot the gain curve of a signal (input vs output RMS in dB)."""
    x_rms = moving_rms(x, window_size)
    y_rms = moving_rms(y, window_size)
    fig, ax = plt.subplots()
    ax.plot(lims, lims, "k--", linewidth=0.8)
    ax.scatter(x_rms, y_rms, s=2)
    ax.set_xlabel("X RMS [dB]")
    ax.set_ylabel("Y RMS [dB]")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title("Gain Curve")
    ax.grid()
    plt.show()
    return fig, ax


def compare_gain_curves(x, y_pred, y_target, window_size=4410, lims=(-80, 0)):
    """Plot predicted and target gain curves on the same axes. Returns (fig, ax)."""
    x_rms = moving_rms(x, window_size)
    pred_rms = moving_rms(y_pred, window_size)
    target_rms = moving_rms(y_target, window_size)
    x_fit_pred, y_fit_pred = regression(x_rms, pred_rms)
    x_fit_tgt, y_fit_tgt = regression(x_rms, target_rms)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        x_rms, target_rms, s=2, alpha=1, color="tab:green", label="Target (scatter)"
    )
    ax.scatter(
        x_rms, pred_rms, s=2, alpha=1, color="tab:blue", label="Predicted (scatter)"
    )
    ax.plot(lims, lims, "k--", linewidth=0.8)
    ax.set_xlabel("Input RMS [dB]")
    ax.set_ylabel("Output RMS [dB]")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig, ax

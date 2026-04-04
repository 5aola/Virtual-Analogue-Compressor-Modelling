"""
Evaluate steepness (attack/release) for all ground-truth compressor settings
and plot how they correlate with threshold, attack, release, and ratio.

Uses the same dataset pairs as eval_all_settings.py (one input WAV, one
target WAV per setting folder). No model inference — only GT targets.

Usage
-----
    uv run python plot_steepness_vs_params.py
    uv run python plot_steepness_vs_params.py --output_dir eval_output/steepness_plots
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from B_analysis.utils import estimate_attack_release_time
from B_analysis.eval_all_settings import (
    GT_ROOT,
    INPUT_ROOT,
    INPUT_NAME,
    discover_settings,
)
from eval_nabla_models import SAMPLE_RATE, load_wav


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt_root", default=GT_ROOT)
    parser.add_argument("--input_wav", default=os.path.join(INPUT_ROOT, INPUT_NAME))
    parser.add_argument("--output_dir", default="eval_output/steepness_vs_params")
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--sr", type=int, default=None)
    args = parser.parse_args()

    sr = args.sr or SAMPLE_RATE
    os.makedirs(args.output_dir, exist_ok=True)

    # Load input once
    print("Loading input:", args.input_wav)
    x = load_wav(args.input_wav, sr)

    settings = discover_settings(args.gt_root)
    if not settings:
        raise SystemExit(f"No settings found under {args.gt_root!r}")
    print(f"Found {len(settings)} settings")

    # Collect per-setting: params + steepness
    thresholds = []
    attacks = []
    releases = []
    ratios = []
    attack_steepness = []
    release_steepness = []

    for i, s in enumerate(settings):
        y = load_wav(s["target_path"], sr)
        min_len = min(len(x), len(y))
        x_trim = x[:min_len].astype(np.float32)
        y_trim = y[:min_len].astype(np.float32)

        atk_stp, rel_stp = estimate_attack_release_time(
            x_trim, y_trim, sr=sr, window_size=args.window_size
        )

        thresholds.append(s["threshold"])
        attacks.append(s["attack"])
        releases.append(s["release"])
        ratios.append(s["ratio"])
        attack_steepness.append(atk_stp)
        release_steepness.append(rel_stp)

    thresholds = np.array(thresholds)
    attacks = np.array(attacks)
    releases = np.array(releases)
    ratios = np.array(ratios)
    attack_steepness = np.array(attack_steepness)
    release_steepness = np.array(release_steepness)

    # Figure: 2 rows x 4 columns — steepness vs each parameter
    param_names = ["threshold (dB)", "attack (ms)", "release (s)", "ratio"]
    param_arrays = [thresholds, attacks, releases, ratios]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex="col")

    for col, (pname, pvals) in enumerate(zip(param_names, param_arrays)):
        # Row 0: attack steepness vs parameter
        ax_atk = axes[0, col]
        ax_atk.scatter(
            pvals,
            attack_steepness,
            alpha=0.8,
            s=40,
            c="C0",
            edgecolors="k",
            linewidths=0.5,
        )
        ax_atk.set_ylabel("Attack steepness (dB/frame)")
        ax_atk.set_title(pname)
        ax_atk.grid(True, alpha=0.3)
        # Trend line
        if len(pvals) > 1:
            z = np.polyfit(pvals, attack_steepness, 1)
            x_line = np.linspace(pvals.min(), pvals.max(), 50)
            ax_atk.plot(x_line, np.poly1d(z)(x_line), "C0--", alpha=0.8, linewidth=1.5)

        # Row 1: release steepness vs parameter
        ax_rel = axes[1, col]
        ax_rel.scatter(
            pvals,
            release_steepness,
            alpha=0.8,
            s=40,
            c="C1",
            edgecolors="k",
            linewidths=0.5,
        )
        ax_rel.set_ylabel("Release steepness (dB/frame)")
        ax_rel.set_xlabel(pname)
        ax_rel.grid(True, alpha=0.3)
        if len(pvals) > 1:
            z = np.polyfit(pvals, release_steepness, 1)
            x_line = np.linspace(pvals.min(), pvals.max(), 50)
            ax_rel.plot(x_line, np.poly1d(z)(x_line), "C1--", alpha=0.8, linewidth=1.5)

    axes[0, 0].set_xlabel("")
    for col in range(1, 4):
        axes[0, col].set_xlabel("")
    fig.suptitle(
        "Steepness vs compressor parameters (all settings, ground-truth targets)",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()

    out_path = os.path.join(args.output_dir, "steepness_vs_params.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)

    # Optional: one combined scatter with color = one parameter (e.g. ratio)
    fig2, (ax_a, ax_r) = plt.subplots(1, 2, figsize=(10, 4))
    sc_a = ax_a.scatter(
        attacks, releases, c=attack_steepness, s=60, alpha=0.9, cmap="viridis"
    )
    ax_a.set_xlabel("Attack (ms)")
    ax_a.set_ylabel("Release (s)")
    ax_a.set_title("Attack steepness (color)")
    plt.colorbar(sc_a, ax=ax_a, label="Attack steepness (dB/frame)")
    ax_a.grid(True, alpha=0.3)

    sc_r = ax_r.scatter(
        attacks, releases, c=release_steepness, s=60, alpha=0.9, cmap="plasma"
    )
    ax_r.set_xlabel("Attack (ms)")
    ax_r.set_ylabel("Release (s)")
    ax_r.set_title("Release steepness (color)")
    plt.colorbar(sc_r, ax=ax_r, label="Release steepness (dB/frame)")
    ax_r.grid(True, alpha=0.3)

    fig2.suptitle(
        "Steepness on attack/release plane (color = steepness)", fontsize=12, y=1.02
    )
    plt.tight_layout()
    out_path2 = os.path.join(args.output_dir, "steepness_attack_release_plane.png")
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved:", out_path2)


if __name__ == "__main__":
    main()

"""
Batch-evaluate a trained compressor model on ALL ground-truth settings
for a single input WAV (default: Electrvm).

Loads the model once, then iterates over every setting folder found in
the processed_ground_truth directory, generating a gain-comparison plot
for each setting.

Usage
-----
    uv run python B_sota_analysis/eval_all_settings.py \
        --ckpt "external/nablafx-for-diffssl-compressor/experiments/TCN_L_TF/dafx/brqmbvok/checkpoints/last.ckpt" \
        --device mps

    external/nablafx-for-diffssl-compressor/experiments/S4_L_TF/dafx/h8sxc679/checkpoints/last.ckpt
    external/nablafx-for-diffssl-compressor/experiments/TCN_L_TF/dafx/brqmbvok/checkpoints/last.ckpt
"""

import argparse
import csv
import math
import os
import re

import matplotlib

matplotlib.use("Agg")

import auraloss
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from src.dsp import estimate_attack_release_time, moving_rms
from B_sota_analysis.eval_nabla_models import (
    SAMPLE_RATE,
    load_model,
    load_wav,
    normalize_params,
    run_model,
)

GT_ROOT = "/Volumes/Saola's Drive/thesis/data/Diff-SSL-G-Comp/processed_ground_truth"
INPUT_ROOT = "/Volumes/Saola's Drive/thesis/data/Diff-SSL-G-Comp/processed_normalized"
INPUT_NAME = "Electrvm_UnmasteredWAV.wav"
TARGET_NAME = "Electrvm-exported.wav"

SETTING_RE = re.compile(
    r"^threshold_(?P<threshold>-?\d+(?:\.\d+)?)"
    r"_attack_(?P<attack>\d+(?:\.\d+)?)"
    r"_release_(?P<release>\d+(?:\.\d+)?)"
    r"_ratio_(?P<ratio>\d+(?:\.\d+)?)$"
)


def discover_settings(gt_root: str) -> list[dict]:
    """Scan *gt_root* for setting folders and return parsed parameter dicts."""
    settings = []
    for name in sorted(os.listdir(gt_root)):
        full = os.path.join(gt_root, name)
        if not os.path.isdir(full):
            continue
        m = SETTING_RE.match(name)
        if m is None:
            print(f"  [skip] {name!r} — does not match pattern")
            continue
        target = os.path.join(full, TARGET_NAME)
        if not os.path.isfile(target):
            print(f"  [skip] {name!r} — no {TARGET_NAME}")
            continue
        settings.append(
            {
                "folder": name,
                "target_path": target,
                "threshold": float(m.group("threshold")),
                "attack": float(m.group("attack")),
                "release": float(m.group("release")),
                "ratio": float(m.group("ratio")),
            }
        )
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample_rate", type=int, default=None)
    parser.add_argument("--output_dir", default="B_sota_analysis/eval_output/all_settings")
    parser.add_argument("--gt_root", default=GT_ROOT)
    parser.add_argument("--input_wav", default=os.path.join(INPUT_ROOT, INPUT_NAME))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- derive a tag from model name + track name for output files ---
    # e.g. "nablafx-diffssl/experiments/S4_L_TF/dafx/…" → "S4_L_TF"
    ckpt_parts = args.ckpt.replace("\\", "/").split("/")
    try:
        exp_idx = ckpt_parts.index("experiments")
        model_name = ckpt_parts[exp_idx + 1]
    except (ValueError, IndexError):
        model_name = "model"
    # e.g. "Electrvm_UnmasteredWAV.wav" → "Electrvm"
    track_name = os.path.splitext(os.path.basename(args.input_wav))[0].split("_")[0]
    eval_tag = f"{model_name}-{track_name}"

    # --- load model once ---
    model, cfg_sr, param_ranges = load_model(args.ckpt, device=args.device)
    sr = args.sample_rate or cfg_sr or SAMPLE_RATE
    print(f"[Eval] sample rate = {sr}")

    # --- load input once ---
    print("\n--- Loading input audio ---")
    x_np = load_wav(args.input_wav, sr)
    x_t = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0)  # (1, 1, T)

    # --- initialize loss functions ---
    l1_loss_fn = torch.nn.L1Loss()
    mrstft_loss_fn = auraloss.freq.MultiResolutionSTFTLoss()
    print("Initialized L1 and Multi-Resolution STFT loss functions")

    # --- discover settings ---
    print("\n--- Discovering settings ---")
    settings = discover_settings(args.gt_root)
    print(f"  Found {len(settings)} setting(s)\n")

    # --- initialize results storage ---
    results = []

    # --- prepare combined figure ---
    n = len(settings)
    ncols = min(n, 5)
    nrows = math.ceil(n / ncols)
    fig_all, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 5 * nrows),
        squeeze=False,
    )
    lims = (-80, 0)

    for i, s in enumerate(settings):
        label = s["folder"]
        print(f"\n{'=' * 70}")
        print(f"[{i + 1}/{n}] {label}")
        print(
            f"  threshold={s['threshold']} dB  attack={s['attack']} ms  "
            f"release={s['release']} s  ratio={s['ratio']}"
        )

        controls = normalize_params(
            s["threshold"],
            s["attack"],
            s["release"],
            s["ratio"],
            ranges=param_ranges,
        ).unsqueeze(0)
        print(f"  normalized controls: {controls.squeeze().tolist()}")

        # --- run model ---
        y_t = run_model(model, x_t, controls)
        y_np = y_t.squeeze().numpy()

        pred_rms_db = 20 * np.log10(max(np.sqrt(np.mean(y_np**2)), 1e-10))
        print(f"  pred rms={pred_rms_db:.1f} dB, peak={float(np.abs(y_np).max()):.4f}")

        # --- save output wav ---
        out_wav = os.path.join(args.output_dir, f"{label}_output.wav")
        sf.write(out_wav, y_np, sr)

        # --- load target ---
        tgt_np = load_wav(s["target_path"], sr)

        # --- align lengths ---
        min_len = min(len(x_np), len(y_np), len(tgt_np))
        x_trim = x_np[:min_len]
        y_trim = y_np[:min_len]
        tgt_trim = tgt_np[:min_len]

        # --- compute loss metrics ---
        # Convert to tensors for loss computation
        y_tensor = torch.from_numpy(y_trim).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        tgt_tensor = torch.from_numpy(tgt_trim).unsqueeze(0).unsqueeze(0)  # (1, 1, T)

        # L1 loss (time-domain)
        l1_loss = l1_loss_fn(y_tensor, tgt_tensor).item()

        # Multi-Resolution STFT loss (frequency-domain)
        mrstft_loss = mrstft_loss_fn(y_tensor, tgt_tensor).item()

        print(f"  L1 loss (time-domain): {l1_loss:.6f}")
        print(f"  MRSTFT loss (frequency-domain): {mrstft_loss:.6f}")

        # --- estimate attack / release steepness ---
        pred_attack_stp, pred_release_stp = estimate_attack_release_time(
            x_trim,
            y_trim,
            sr=sr,
        )
        tgt_attack_stp, tgt_release_stp = estimate_attack_release_time(
            x_trim,
            tgt_trim,
            sr=sr,
        )
        print(
            "  Predicted  — attack steepness: "
            f"{pred_attack_stp:.4f} dB/frame, "
            f"release steepness: {pred_release_stp:.4f} dB/frame"
        )
        print(
            "  Target     — attack steepness: "
            f"{tgt_attack_stp:.4f} dB/frame, "
            f"release steepness: {tgt_release_stp:.4f} dB/frame"
        )

        # --- store results ---
        results.append(
            {
                "setting": label,
                "threshold": s["threshold"],
                "attack": s["attack"],
                "release": s["release"],
                "ratio": s["ratio"],
                "l1_loss": l1_loss,
                "mrstft_loss": mrstft_loss,
                "pred_attack_stp": pred_attack_stp,
                "pred_release_stp": pred_release_stp,
                "tgt_attack_stp": tgt_attack_stp,
                "tgt_release_stp": tgt_release_stp,
            }
        )

        # --- compute RMS envelopes ---
        x_rms = moving_rms(x_trim, 44100)
        pred_rms = moving_rms(y_trim, 44100)
        target_rms = moving_rms(tgt_trim, 44100)

        # --- plot on combined figure ---
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        ax.scatter(x_rms, target_rms, s=1, alpha=0.8, color="tab:green", label="Target")
        ax.scatter(x_rms, pred_rms, s=1, alpha=0.8, color="tab:blue", label="Predicted")
        ax.plot(lims, lims, "k--", linewidth=0.8)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(
            f"thr={s['threshold']:.0f}  atk={s['attack']:.0f}  "
            f"rel={s['release']:.1f}  rat={s['ratio']:.0f}",
            fontsize=9,
        )
        ax.set_xlabel("Input RMS [dB]", fontsize=8)
        ax.set_ylabel("Output RMS [dB]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper left", fontsize=7, markerscale=3)

        print(f"  Added to combined plot (row={row}, col={col})")

    # hide unused subplots
    for j in range(n, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row][col].set_visible(False)

    fig_all.suptitle(
        f"Gain Curves — All Compressor Settings ({eval_tag})", fontsize=14, y=1.01
    )
    fig_all.tight_layout()
    combined_path = os.path.join(
        args.output_dir, f"all_settings_gain_curves_{eval_tag}.png"
    )
    fig_all.savefig(combined_path, dpi=200, bbox_inches="tight")
    plt.close(fig_all)
    print(f"\n  Combined plot saved: {combined_path}")

    # --- save loss results to CSV ---
    csv_path = os.path.join(args.output_dir, f"loss_results_{eval_tag}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "setting",
                "threshold",
                "attack",
                "release",
                "ratio",
                "l1_loss",
                "mrstft_loss",
                "pred_attack_stp",
                "pred_release_stp",
                "tgt_attack_stp",
                "tgt_release_stp",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Loss results saved to: {csv_path}")

    # --- print summary statistics ---
    l1_losses = [r["l1_loss"] for r in results]
    mrstft_losses = [r["mrstft_loss"] for r in results]
    print(
        f"\n  L1 Loss - Mean: {np.mean(l1_losses):.6f}, Std: {np.std(l1_losses):.6f}, "
        f"Min: {np.min(l1_losses):.6f}, Max: {np.max(l1_losses):.6f}"
    )
    print(
        f"  MRSTFT Loss - Mean: {np.mean(mrstft_losses):.6f}, Std: {np.std(mrstft_losses):.6f}, "
        f"Min: {np.min(mrstft_losses):.6f}, Max: {np.max(mrstft_losses):.6f}"
    )

    print(f"\n{'=' * 70}")
    print(f"Done — {n} settings evaluated.")
    print(f"Results in {args.output_dir}/")


if __name__ == "__main__":
    main()

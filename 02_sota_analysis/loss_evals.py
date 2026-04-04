#!/usr/bin/env python
"""
Compute multiple losses between unmastered↔target, model-output↔target,
and model-output↔unmastered audio.

Uses infer_compressor for model loading, param normalization, and inference.

Example:
uv run 02_sota_analysis/loss_evals.py \
  "/Volumes/Saola's Drive/thesis/data/Diff-SSL-G-Comp/processed_normalized/Air_UnmasteredWAV.wav" \
  "/Volumes/Saola's Drive/thesis/data/Diff-SSL-G-Comp/processed_ground_truth/threshold_-12_attack_1_release_0.1_ratio_2/Air-exported.wav" \
  --model TCN_S_TF

uv run 02_sota_analysis/loss_evals.py \
  "/Volumes/Saola's Drive/thesis/data/Diff-SSL-G-Comp/processed_normalized/LivingLie_UnmasteredWAV.wav" \
  "/Volumes/Saola's Drive/thesis/data/Diff-SSL-G-Comp/processed_ground_truth/threshold_-12_attack_10_release_0.4_ratio_10/LivingLie-exported.wav" \
  --model S4_S_TF
  """

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "nablafx")
)  # local nablafx checkout for infer_compressor

from dataset import extract_params_from_target_path
from infer_compressor import (
    discover_models,
    load_audio,
    load_model,
    run_inference,
)
from src.dsp import calc_gain_reduction, to_amplitude, PARAM_RANGES_NABLAFX
from src.dsp_torch import normalize_params
from src import losses as loss


def calc_losses(
    unmastered: str,
    target: str,
    model: str = "TCN_S_TF",
    use_last: bool = False,
    device: str = "cpu",
    threshold: float = None,
    attack: float = None,
    release: float = None,
    ratio: float = None,
):
    """
    Compute multiple losses between unmastered↔target, model-output↔target,
    and model-output↔unmastered audio.

    Returns:
        results (list of tuples): A list containing comparison labels and loss dicts.
    """
    # --- resolve params: explicit args take priority, otherwise extract from target path ---
    if all(v is not None for v in [threshold, attack, release, ratio]):
        raw_params = {
            "threshold": threshold,
            "attack": attack,
            "release": release,
            "ratio": ratio,
        }
    else:
        raw_params = extract_params_from_target_path(target)
        print(f"Extracted params from target path: {raw_params}")

    controls = normalize_params(**raw_params, ranges=PARAM_RANGES_NABLAFX)
    print(
        f"Params: threshold={raw_params['threshold']}, attack={raw_params['attack']}, "
        f"release={raw_params['release']}, ratio={raw_params['ratio']}"
    )
    print(f"Normalized: {controls.tolist()}")

    # --- load model ---
    models_dict = discover_models()
    system, config = load_model(
        model, models_dict, use_best=not use_last, device=device
    )

    # --- load audio ---
    sr = config.get("data", {}).get("init_args", {}).get("sample_rate", 44100)
    x = load_audio(unmastered, sample_rate=sr)  # (1, T)
    y = load_audio(target, sample_rate=sr)  # (1, T)

    # --- inference ---
    pred = run_inference(system, x, controls, device=device)  # (1, T)

    # --- match lengths ---
    min_len = min(x.size(-1), y.size(-1), pred.size(-1))
    x = x[..., :min_len].unsqueeze(0)  # (1, 1, T)
    y = y[..., :min_len].unsqueeze(0)
    pred = pred[..., :min_len].unsqueeze(0)

    # --- compute input × gain-reduction (analytic reconstruction) ---
    # gain_reduction returns dB difference (output − input) per sample
    gr_db = calc_gain_reduction(
        x.squeeze().numpy(), y.squeeze().numpy(), window_size=64
    )
    gr_linear = to_amplitude(gr_db)
    # The gain-reduction array may differ slightly in length; trim to match
    x_np = x.squeeze().numpy()
    gr_len = min(len(gr_linear), len(x_np))
    reconstructed = x_np[:gr_len] * gr_linear[:gr_len]
    # Pad or trim to min_len so shapes match
    if len(reconstructed) < min_len:
        reconstructed = np.pad(reconstructed, (0, min_len - len(reconstructed)))
    else:
        reconstructed = reconstructed[:min_len]
    x_times_gr = torch.from_numpy(reconstructed).float().unsqueeze(0).unsqueeze(0)

    pairs = [
        (f"{model} Output → Target", pred, y),
        (f"{model} Output → Input", pred, x),
        ("Input → Target", x, y),
        ("Input × GainReduction → Target", x_times_gr, y),
    ]

    # Compute losses for all pairs
    results = []
    for label, a, b in pairs:
        results.append((label, loss.compute_all_losses(a, b)))

    # Print as a table: rows = comparisons, columns = losses
    loss_names = list(results[0][1].keys())
    label_width = max(len(r[0]) for r in results)
    col_width = 10

    # Header
    header = f"{'Comparison':<{label_width}}  " + "  ".join(
        f"{n:>{col_width}}" for n in loss_names
    )
    print()
    print(header)
    print("─" * len(header))

    # Rows
    for label, losses in results:
        vals = "  ".join(f"{losses[n]:>{col_width}.6f}" for n in loss_names)
        print(f"{label:<{label_width}}  {vals}")
    print()

    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute multiple losses between "
            "unmastered↔target, model-output↔target, and model-output↔unmastered audio."
        )
    )
    parser.add_argument("unmastered", type=str, help="Path to unmastered (input) wav")
    parser.add_argument("target", type=str, help="Path to target (reference) wav")
    parser.add_argument(
        "--model", type=str, default="TCN_S_TF", help="Model name (default: TCN_S_TF)"
    )
    parser.add_argument("--use_last", action="store_true", help="Use last.ckpt")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda, mps)")

    # Compressor params — optional if target path contains them
    parser.add_argument("--threshold", type=float, help="dB  (-20 … 0)")
    parser.add_argument("--attack", type=float, help="ms   (0.1 … 30)")
    parser.add_argument("--release", type=float, help="s    (0.1 … 1.6)")
    parser.add_argument("--ratio", type=float, help="ratio (2 … 10)")
    args = parser.parse_args()

    calc_losses(
        unmastered=args.unmastered,
        target=args.target,
        model=args.model,
        use_last=args.use_last,
        device=args.device,
        threshold=args.threshold,
        attack=args.attack,
        release=args.release,
        ratio=args.ratio,
    )


if __name__ == "__main__":
    main()

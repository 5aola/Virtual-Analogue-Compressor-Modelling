"""
Run a trained TCN-TFiLM compressor model on a WAV file and plot
the resulting gain-reduction curve.

Usage
-----
    python eval_transfer.py \
        --ckpt path/to/checkpoint.ckpt \
        --wav  song.wav \
        --threshold -12 --attack 10 --release 0.4 --ratio 4
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import yaml

from src.audio_io import load_wav
from src.dsp import (
    PARAM_RANGES_LOCAL,
    PARAM_RANGES_NABLAFX,
    calculate_rms,
    rms_to_db,
)
from src.dsp_torch import normalize_params
from src.plotting import compare_gain_curves, estimate_gain_curve

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "external", "nablafx-for-diffssl-compressor"
    ),
)  # local fork overrides pip-installed nablafx

from nablafx.models import BlackBoxModel
from nablafx.tcn import TCN
from nablafx.s4 import S4

PROCESSOR_MAP = {
    "nablafx.tcn.TCN": TCN,
    "nablafx.s4.S4": S4,
}

SAMPLE_RATE = 48000


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _find_experiment_config(ckpt_path: str) -> dict | None:
    """Walk up from the checkpoint path looking for config.yaml."""
    d = os.path.dirname(os.path.abspath(ckpt_path))
    for _ in range(6):
        candidate = os.path.join(d, "config.yaml")
        if os.path.isfile(candidate):
            with open(candidate) as f:
                return yaml.safe_load(f)
        d = os.path.dirname(d)
    return None


def load_model(ckpt_path: str, device: str = "cpu"):
    """Load a nablafx BlackBoxSystem checkpoint, falling back to
    workspace-root CompressorSystem.

    Returns ``(model, sample_rate | None, param_ranges)``.
    *param_ranges* is the dict to use with :func:`normalize_params`.
    """

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", {})

    cfg = _find_experiment_config(ckpt_path)
    if cfg is not None:
        try:
            model_args = cfg["model"]["init_args"]
            proc_cfg = model_args["model"]["init_args"]["processor"]
            proc_class_path = proc_cfg["class_path"]
            proc_init_args = proc_cfg["init_args"]

            ProcessorClass = PROCESSOR_MAP.get(proc_class_path)
            if ProcessorClass is None:
                raise ValueError(
                    f"Unknown processor class_path: {proc_class_path!r}. "
                    f"Known: {list(PROCESSOR_MAP)}"
                )
            processor = ProcessorClass(**proc_init_args)
            model = BlackBoxModel(processor=processor)

            bb_sd = {
                k.removeprefix("model."): v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }
            model.load_state_dict(bb_sd, strict=True)
            model.eval().to(device)

            cfg_sr = cfg.get("data", {}).get("init_args", {}).get("sample_rate")
            proc_name = proc_class_path.rsplit(".", 1)[-1]
            print(
                f"[Model] nablafx BlackBoxModel({proc_name}) — cond_type={proc_init_args.get('cond_type')}"
            )
            print(f"[Model] num_controls = {model.num_controls}")
            n = sum(p.numel() for p in model.parameters())
            print(f"[Model] parameters: {n:,}")
            print("[Model] param normalization: NABLAFX ranges")
            return model, cfg_sr, PARAM_RANGES_NABLAFX
        except Exception as e:
            print(f"[Model] config-based loading failed: {e}")

    try:
        from main import CompressorSystem

        system = CompressorSystem.load_from_checkpoint(ckpt_path, map_location=device)
        system.eval().to(device)
        print("[Model] Loaded CompressorSystem (workspace root)")
        n = sum(p.numel() for p in system.parameters())
        print(f"[Model] parameters: {n:,}")
        print("[Model] param normalization: LOCAL ranges")
        return system, None, PARAM_RANGES_LOCAL
    except Exception as e:
        print(f"[Model] CompressorSystem loading failed: {e}")

    raise RuntimeError(f"Could not load checkpoint: {ckpt_path}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_model(
    model,
    x: torch.Tensor,
    controls: torch.Tensor | None = None,
    chunk_samples: int = SAMPLE_RATE * 10,
) -> torch.Tensor:
    """Run model on (B, 1, T) input, chunking long signals to save memory.
    Resets LSTM / recurrent states once before the first chunk, then lets
    the hidden state carry through subsequent chunks.
    """
    device = next(model.parameters()).device
    x = x.to(device)
    if controls is not None:
        controls = controls.to(device)

    if hasattr(model, "reset_states"):
        model.reset_states()

    T = x.shape[-1]

    if T <= chunk_samples:
        return model(x, controls).cpu()

    chunks = []
    for start in range(0, T, chunk_samples):
        end = min(start + chunk_samples, T)
        chunks.append(model(x[:, :, start:end], controls).cpu())
    return torch.cat(chunks, dim=-1)


# ---------------------------------------------------------------------------
# WAV processing
# ---------------------------------------------------------------------------


def process_wav(
    model,
    wav_path: str,
    controls,
    output_dir: str,
    sr: int,
    target_path: str | None = None,
):
    """Load WAV, run through model, save output and gain curve.
    If *target_path* is given, also plot the target gain curve for comparison."""

    # ---- 1. Preprocess input (same pipeline as training) --------------------
    print("\n--- Preprocessing ---")
    x_np = load_wav(wav_path, sr)

    # ---- 2. Run model -------------------------------------------------------
    print("\n--- Inference ---")
    x_t = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    y_t = run_model(model, x_t, controls)
    y_np = y_t.squeeze().numpy()

    pred_peak = float(np.abs(y_np).max())
    pred_rms_db = float(rms_to_db(calculate_rms(y_np)))
    print(f"  [pred] frames={len(y_np)}, peak={pred_peak:.4f}, rms={pred_rms_db:.1f} dB")

    stem = os.path.splitext(os.path.basename(wav_path))[0]
    out_wav = os.path.join(output_dir, f"{stem}_output.wav")
    sf.write(out_wav, y_np, sr)
    print(f"  Saved: {out_wav}")

    # ---- 3. Preprocess target (same pipeline) -------------------------------
    if target_path is not None:
        tgt_np = load_wav(target_path, sr)

        # ---- 4. Align lengths -----------------------------------------------
        min_len = min(len(x_np), len(y_np), len(tgt_np))
        x_np = x_np[:min_len]
        y_np = y_np[:min_len]
        tgt_np = tgt_np[:min_len]
        print(f"\n  Aligned to {min_len} samples ({min_len / sr:.1f} s)")

        # ---- 5. Generate comparison gain curves -----------------------------
        print("\n--- Gain curves (post-preprocess) ---")
        fig, ax = compare_gain_curves(x_np, y_np, tgt_np)
        ax.set_title(f"Predicted vs Target — {stem}")
        fig_path = os.path.join(output_dir, f"{stem}_gain_compare.png")
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Compare: {fig_path}")
    else:
        fig, ax = estimate_gain_curve(x_np, y_np)
        ax.set_title(f"Gain Curve — {stem}")
        fig_path = os.path.join(output_dir, f"{stem}_gain_curve.png")
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        print(f"  Plot  : {fig_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Process a WAV through a trained compressor model and plot gain curve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ckpt", required=True, help="Path to .ckpt file")
    p.add_argument("--wav", required=True, help="Input WAV file")
    p.add_argument(
        "--target", default=None, help="Target (ground truth) WAV for comparison"
    )
    p.add_argument("--device", default="cpu", help="Device (cpu, cuda, mps)")
    p.add_argument("--sample_rate", type=int, default=None)
    p.add_argument("--output_dir", default="02_sota_analysis/eval_output")

    p.add_argument(
        "--controls",
        type=float,
        nargs="+",
        default=None,
        help="Pre-normalized controls (0-1)",
    )
    p.add_argument("--threshold", type=float, default=None, help="dB  (-20 … 0)")
    p.add_argument("--attack", type=float, default=None, help="ms   (0.1 … 30)")
    p.add_argument("--release", type=float, default=None, help="s    (0.1 … 1.6)")
    p.add_argument("--ratio", type=float, default=None, help="ratio (2 … 10)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, cfg_sr, param_ranges = load_model(args.ckpt, device=args.device)

    global SAMPLE_RATE
    if args.sample_rate is not None:
        SAMPLE_RATE = args.sample_rate
    elif cfg_sr is not None:
        SAMPLE_RATE = cfg_sr
    print(f"[Eval] sample rate = {SAMPLE_RATE}")

    # --- resolve controls ---------------------------------------------------
    n_ctrl = getattr(model, "num_controls", 0)

    has_named = any(getattr(args, k) is not None for k in PARAM_ORDER)
    if has_named:
        thr = args.threshold if args.threshold is not None else -12.0
        atk = args.attack if args.attack is not None else 10.0
        rel = args.release if args.release is not None else 0.4
        rat = args.ratio if args.ratio is not None else 4.0
        controls = normalize_params(thr, atk, rel, rat, ranges=param_ranges).unsqueeze(
            0
        )
        print(
            f"[Params] threshold={thr} dB, attack={atk} ms, release={rel} s, ratio={rat}"
        )
        print(f"[Params] normalized: {controls.squeeze().tolist()}")
    elif args.controls is not None:
        controls = torch.tensor(args.controls, dtype=torch.float32).unsqueeze(0)
        print(f"[Params] raw normalized: {args.controls}")
    elif n_ctrl > 0:
        controls = 0.5 * torch.ones(1, n_ctrl)
        print(f"[Params] defaulting to 0.5 × {n_ctrl}")
    else:
        controls = None

    process_wav(
        model, args.wav, controls, args.output_dir, SAMPLE_RATE, target_path=args.target
    )
    print(f"\nDone — results in {args.output_dir}/")


if __name__ == "__main__":
    main()
    """
  uv run python 02_sota_analysis/eval_nabla_models.py \
  --ckpt "external/nablafx-for-diffssl-compressor/experiments/TCN_L_TF/dafx/brqmbvok/checkpoints/last.ckpt" \
  --wav "/Volumes/Saola's Drive/thesis/data/Diff-SSL-G-Comp/processed_normalized/Electrvm_UnmasteredWAV.wav" \
  --target "/Volumes/Saola's Drive/thesis/data/Diff-SSL-G-Comp/processed_ground_truth/threshold_-12_attack_10_release_0.4_ratio_10/Electrvm-exported.wav" \
  --threshold -12 --attack 10 --release 0.4 --ratio 10 \
  --device mps

  uv run python 02_sota_analysis/eval_all_settings.py \
  --ckpt "external/nablafx-for-diffssl-compressor/experiments/TCN_L_TF/dafx/brqmbvok/checkpoints/last.ckpt" \
  --device mps

    """

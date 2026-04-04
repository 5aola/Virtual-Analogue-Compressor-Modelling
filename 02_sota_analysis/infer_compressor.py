"""
Inference script for trained compressor models.

Two modes:
  1. free   – provide a WAV + freely chosen compressor settings
  2. replicate – provide a WAV + target WAV path (params extracted from path)

Examples:
  # List available models
  uv run 02_sota_analysis/infer_compressor.py --list_models

  # Free-parameter inference
  python infer_compressor.py --model GCN_L_TVF --mode free \
    --input_wav input.wav --threshold -12 --attack 1 --release 0.1 --ratio 2

  # Replicate target
  python infer_compressor.py --model GCN_L_TVF --mode replicate \
    --input_wav input.wav \
    --target_wav '.../threshold_-12_attack_1_release_0.1_ratio_2/Air-exported.wav'
"""

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "nablafx")
)  # local nablafx checkout

import nablafx.processors  # noqa: F401  # for _instantiate_class
import torch
import yaml
from nablafx.core import BlackBoxSystem, BlackBoxSystemWithTBPTT, GreyBoxSystem

from src.audio_io import load_wav_tensor
from src.dsp import PARAM_RANGES_NABLAFX
from src.dsp_torch import normalize_params
from dataset import extract_params_from_target_path

EXPERIMENTS_DIR = Path(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "external",
        "nablafx-for-diffssl-compressor",
        "experiments",
    )
).resolve()

# Map system class paths to their classes
SYSTEM_CLASS_MAP = {
    "nablafx.core.BlackBoxSystem": BlackBoxSystem,
    "nablafx.core.BlackBoxSystemWithTBPTT": BlackBoxSystemWithTBPTT,
    "nablafx.core.GreyBoxSystem": GreyBoxSystem,
}

def load_audio(filepath: str, sample_rate: int = 44100) -> torch.Tensor:
    """Load, resample to target rate, and convert to mono. Returns (1, T) tensor."""
    return load_wav_tensor(filepath, sample_rate)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


def _ckpt_progress_key(ckpt_path: Path) -> tuple[int, int]:
    """
    Heuristic for how "far" training progressed for a checkpoint, based on filename.
    Treats last.ckpt as the longest-trained version, otherwise parses epoch/step.
    """
    name = ckpt_path.name
    if name == "last.ckpt":
        # Prefer last.ckpt when deciding which version is "longer trained"
        return (10**9, 10**9)

    epoch_match = re.search(r"epoch=(\d+)", name)
    step_match = re.search(r"step=(\d+)", name)
    epoch = int(epoch_match.group(1)) if epoch_match else -1
    step = int(step_match.group(1)) if step_match else -1
    return (epoch, step)


def discover_models() -> dict:
    """
    Scan experiments/ for valid model directories (non-subset, with checkpoints).
    Returns {model_name: {"config": path, "best_ckpt": path, "last_ckpt": path}}.
    """
    models = {}
    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        if "subset" in name.lower():
            continue
        config_path = exp_dir / "config.yaml"
        if not config_path.exists():
            continue

        ckpt_dirs = list(exp_dir.glob("dafx/*/checkpoints"))
        if not ckpt_dirs:
            continue
        ckpts = list(ckpt_dirs[0].glob("*.ckpt"))
        if not ckpts:
            continue

        best_ckpt = None
        best_progress = (-1, -1)
        last_ckpt = None
        for c in ckpts:
            if c.name == "last.ckpt":
                last_ckpt = c

            progress = _ckpt_progress_key(c)
            if progress > best_progress:
                best_progress = progress
                best_ckpt = c

        if best_ckpt is None:
            best_ckpt = last_ckpt

        models[name] = {
            "config": config_path,
            "best_ckpt": best_ckpt,
            "last_ckpt": last_ckpt,
        }
    return models


def get_cond_type(config: dict) -> str:
    """Extract the conditioning type from the config."""
    model_cfg = config["model"]["init_args"]["model"]["init_args"]
    if "processor" in model_cfg:
        return model_cfg["processor"]["init_args"].get("cond_type", "none")
    return "greybox"


def patch_class_path(class_path: str) -> str:
    """Map class paths from old nablafx structure to new nablafx structure."""
    if class_path.startswith("nablafx.system."):
        return class_path.replace("nablafx.system.", "nablafx.core.")
    elif class_path.startswith("nablafx.models."):
        return class_path.replace("nablafx.models.", "nablafx.core.")
    elif class_path.startswith("nablafx.loss."):
        return class_path.replace("nablafx.loss.", "nablafx.evaluation.")
    elif class_path.startswith("nablafx.") and any(
        p in class_path for p in [".gcn.", ".tcn.", ".lstm.", ".s4.", ".siren."]
    ):
        parts = class_path.split(".")
        return f"nablafx.processors.{parts[-1]}"
    return class_path


def _instantiate_class(class_path: str, init_args: dict):
    """Instantiate a class from its dotted path and init args (like LightningCLI does)."""
    import importlib

    class_path = patch_class_path(class_path)
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    # Recursively instantiate any nested class_path/init_args dicts
    resolved_args = {}
    for key, val in init_args.items():
        if isinstance(val, dict) and "class_path" in val:
            resolved_args[key] = _instantiate_class(
                val["class_path"], val.get("init_args", {})
            )
        elif isinstance(val, list):
            resolved_args[key] = [
                _instantiate_class(item["class_path"], item.get("init_args", {}))
                if isinstance(item, dict) and "class_path" in item
                else item
                for item in val
            ]
        else:
            resolved_args[key] = val

    return cls(**resolved_args)


def load_model(
    model_name: str, models_dict: dict, use_best: bool = True, device: str = "cpu"
):
    """
    Load a trained model from its config + checkpoint.
    Manually instantiates the system from config.yaml and loads the state_dict,
    because the system classes don't call save_hyperparameters().
    """
    if model_name not in models_dict:
        available = ", ".join(sorted(models_dict.keys()))
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")

    info = models_dict[model_name]
    ckpt_path = info["best_ckpt"] if use_best else info["last_ckpt"]

    with open(info["config"]) as f:
        config = yaml.safe_load(f)

    # We don't need the actual loss function for inference.
    # Replace it with a dummy to avoid instantiation errors if the old loss class doesn't exist.
    if "loss" in config["model"]["init_args"]:
        config["model"]["init_args"]["loss"] = {"class_path": "torch.nn.MSELoss"}

    class_path = patch_class_path(config["model"]["class_path"])
    if class_path not in SYSTEM_CLASS_MAP:
        raise ValueError(f"Unknown system class: {class_path}")

    print(f"Loading model: {model_name}")
    print(f"  System class: {class_path.split('.')[-1]}")
    print(f"  Cond type:    {get_cond_type(config)}")
    print(f"  Checkpoint:   {ckpt_path}")

    # Instantiate system from config (model + loss + other args)
    system = _instantiate_class(class_path, config["model"]["init_args"])

    # Load checkpoint state_dict
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    system.load_state_dict(ckpt["state_dict"])
    system.eval()
    system.to(device)

    return system, config


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_inference(
    system, input_audio: torch.Tensor, controls: torch.Tensor, device: str = "cpu"
) -> torch.Tensor:
    """
    Run inference on an audio tensor.

    Args:
        system: loaded Lightning system
        input_audio: (1, num_samples) audio tensor
        controls: (4,) normalized parameter tensor
        device: torch device

    Returns:
        (1, num_samples) output audio tensor
    """
    system.model.reset_states()

    x = input_audio.unsqueeze(0).to(device)  # (1, 1, T)
    c = controls.unsqueeze(0).to(device)  # (1, 4)

    if system.model.num_controls > 0:
        y = system.model(x, c)
    else:
        y = system.model(x)

    return y.squeeze(0).cpu()  # (1, T)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on trained compressor models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--list_models", action="store_true", help="List available models and exit"
    )
    parser.add_argument("--model", type=str, help="Model name (e.g. GCN_L_TVF)")
    parser.add_argument(
        "--use_last",
        action="store_true",
        help="Use last.ckpt instead of best checkpoint",
    )
    parser.add_argument("--mode", choices=["free", "replicate"], help="Inference mode")
    parser.add_argument("--input_wav", type=str, help="Path to input WAV file")
    parser.add_argument(
        "--output_wav", type=str, default="output.wav", help="Path to output WAV file"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu, cuda, mps)"
    )

    # Free mode params
    parser.add_argument(
        "--threshold", type=float, help="Threshold in dB (raw, e.g. -12)"
    )
    parser.add_argument("--attack", type=float, help="Attack in ms (raw, e.g. 1)")
    parser.add_argument(
        "--release", type=float, help="Release in seconds (raw, e.g. 0.1)"
    )
    parser.add_argument("--ratio", type=float, help="Ratio (raw, e.g. 2)")

    # Replicate mode
    parser.add_argument(
        "--target_wav", type=str, help="Path to target WAV (params extracted from path)"
    )

    args = parser.parse_args()

    models_dict = discover_models()

    if args.list_models:
        print(f"\nAvailable models ({len(models_dict)}):\n")
        for name, info in sorted(models_dict.items()):
            with open(info["config"]) as f:
                config = yaml.safe_load(f)
            cond = get_cond_type(config)
            sys_cls = config["model"]["class_path"].split(".")[-1]
            best_name = info["best_ckpt"].name if info["best_ckpt"] else "N/A"
            # Count parameters from checkpoint state_dict
            ckpt = torch.load(info["best_ckpt"], map_location="cpu", weights_only=False)
            num_params = sum(
                p.numel() for p in ckpt["state_dict"].values() if p.ndim > 0
            )
            if num_params >= 1_000_000:
                params_str = f"{num_params / 1_000_000:.2f}M"
            elif num_params >= 1_000:
                params_str = f"{num_params / 1_000:.1f}K"
            else:
                params_str = str(num_params)
            print(
                f"  {name:<20s}  cond={cond:<10s}  system={sys_cls:<25s}  params={params_str:>8s}  best={best_name}"
            )
        print()
        return

    # Validate args
    if not args.model:
        parser.error("--model is required (use --list_models to see options)")
    if not args.mode:
        parser.error("--mode is required (free or replicate)")
    if not args.input_wav:
        parser.error("--input_wav is required")

    # Get parameters
    if args.mode == "free":
        if (
            args.threshold is None
            or args.attack is None
            or args.release is None
            or args.ratio is None
        ):
            parser.error(
                "--mode free requires --threshold, --attack, --release, --ratio"
            )
        raw_params = {
            "threshold": args.threshold,
            "attack": args.attack,
            "release": args.release,
            "ratio": args.ratio,
        }
    elif args.mode == "replicate":
        if not args.target_wav:
            parser.error("--mode replicate requires --target_wav")
        raw_params = extract_params_from_target_path(args.target_wav)
        print(f"Extracted params from target path: {raw_params}")

    controls = normalize_params(**raw_params, ranges=PARAM_RANGES_NABLAFX)
    print(
        f"Raw params:        threshold={raw_params['threshold']}, attack={raw_params['attack']}, "
        f"release={raw_params['release']}, ratio={raw_params['ratio']}"
    )
    print(f"Normalized params: {controls.tolist()}")

    # Load model
    system, config = load_model(
        args.model, models_dict, use_best=not args.use_last, device=args.device
    )

    # Load audio (resamples + converts to mono)
    target_sr = config.get("data", {}).get("init_args", {}).get("sample_rate", 44100)
    input_audio = load_audio(args.input_wav, sample_rate=target_sr)
    print(f"Input audio: {args.input_wav}")
    print(
        f"  Sample rate: {target_sr}, Duration: {input_audio.shape[-1] / target_sr:.2f}s"
    )

    # Run inference
    print(f"\nRunning inference with {args.model}...")
    output_audio = run_inference(system, input_audio, controls, device=args.device)

    # Save output
    torchaudio.save(args.output_wav, output_audio, target_sr)
    print(f"Output saved to: {args.output_wav}")
    print(f"  Duration: {output_audio.shape[-1] / target_sr:.2f}s")


if __name__ == "__main__":
    main()

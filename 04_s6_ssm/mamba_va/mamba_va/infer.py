"""Render audio through a trained CompSSM checkpoint (streaming, causal).

    python -m mamba_va.infer --ckpt runs/cl1b/best.pt \
        --input dry.wav --output wet_pred.wav \
        --params 0.4 0.3 0.1 0.6
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

from .model import CompSSM
from .data import load_wav

try:
    import soundfile as sf
except Exception:
    sf = None


def load_model(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device)
    a = ck["args"]
    model = CompSSM(
        n_params=ck["n_params"], d_model=a["d_model"], d_state=a["d_state"],
        n_layers=a["n_layers"], expand=a["expand"], conv_kernel=a["conv_kernel"],
        n_bands=a["n_bands"], max_db=a["max_db"],
        sr=ck.get("sr", 44100.0),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--params", type=float, nargs="*", default=[])
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    model = load_model(args.ckpt, args.device)
    x, sr = load_wav(args.input)
    u = torch.tensor(x, device=args.device).unsqueeze(0)
    params = torch.tensor([args.params], dtype=torch.float32, device=args.device) \
        if args.params else None
    y = model.render(u, params, chunk=args.chunk).squeeze(0).cpu().numpy()
    if sf is None:
        raise RuntimeError("soundfile is required to write WAV files")
    sf.write(args.output, y, sr)
    print(f"wrote {args.output} ({len(y)} samples @ {sr} Hz)")


if __name__ == "__main__":
    main()

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.dsp import estimate_attack_release_time
from dataset import GT_ROOT, INPUT_ROOT, INPUT_NAME, TARGET_NAME, discover_settings
from eval_nabla_models import SAMPLE_RATE, load_wav


def main():
    """
    Small sanity-check script for `estimate_attack_release_time`.

    It:
    - loads the default input track
    - discovers the first available ground-truth setting
    - loads the corresponding target WAV
    - trims them to the same length
    - prints attack/release steepness for target vs input.
    """
    sr = SAMPLE_RATE

    x_path = os.path.join(INPUT_ROOT, INPUT_NAME)
    x = load_wav(x_path, sr)

    settings = discover_settings(GT_ROOT)
    if not settings:
        raise RuntimeError(f"No settings found under {GT_ROOT!r}")

    s0 = settings[0]
    tgt_path = os.path.join(s0["path"], TARGET_NAME)
    y = load_wav(tgt_path, sr)

    min_len = min(len(x), len(y))
    x_trim = x[:min_len]
    y_trim = y[:min_len]

    atk_stp, rel_stp = estimate_attack_release_time(
        x_trim.astype(np.float32),
        y_trim.astype(np.float32),
        sr=sr,
        window_size=64,
    )

    print(f"Setting: {s0['folder_name']}")
    print(
        f"Ground-truth params — "
        f"attack={s0['attack']} ms, release={s0['release']} s, ratio={s0['ratio']}"
    )
    print(
        "Estimated steepness — "
        f"attack:  {atk_stp:.4f} dB/frame, "
        f"release: {rel_stp:.4f} dB/frame"
    )


if __name__ == "__main__":
    main()

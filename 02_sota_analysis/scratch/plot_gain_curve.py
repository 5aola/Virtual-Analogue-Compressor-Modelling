import numpy as np

from src.audio_io import load_wav
from src.plotting import estimate_gain_curve


def main():
    dry_path = "test-samples/LivingLie_UnmasteredWAV-dry.wav"
    wet_path = "test-samples/LivingLie_UnmasteredWAV-40thresh-fast-lin-scale.wav"

    x = load_wav(dry_path)
    y = load_wav(wet_path)

    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    estimate_gain_curve(x, y, window_size=44100, lims=(-80, 0))


if __name__ == "__main__":
    main()

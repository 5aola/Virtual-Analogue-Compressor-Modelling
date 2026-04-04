import numpy as np
import soundfile as sf

from src.plotting import estimate_gain_curve


def load_mono_wav(path: str):
    """Load WAV as mono float32 numpy array."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    # average channels to mono if needed
    audio_mono = audio.mean(axis=1)
    return audio_mono, sr


def main():
    dry_path = "test-samples/LivingLie_UnmasteredWAV-dry.wav"
    wet_path = "test-samples/LivingLie_UnmasteredWAV-40thresh-fast-lin-scale.wav"

    x, sr_x = load_mono_wav(dry_path)
    y, sr_y = load_mono_wav(wet_path)

    if sr_x != sr_y:
        raise ValueError(f"Sample rates differ: {sr_x} vs {sr_y}")

    # Make sure they are the same length
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    # Uses your existing estimate_gain_curve (moving_rms + regression + scatter)
    estimate_gain_curve(x, y, window_size=44100, lims=(-80, 0))


if __name__ == "__main__":
    main()

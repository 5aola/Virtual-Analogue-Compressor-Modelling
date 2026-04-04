import matplotlib

matplotlib.use("Agg")
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from src.transfer import time_varying_transfer

# Load audio files (first 30s for speed)
dry_path = "test-samples/LivingLie_UnmasteredWAV-dry.wav"
wet_path = "test-samples/LivingLie_UnmasteredWAV (threshold-24dB_attack-0.3ms_release-0.8s_ratio-4).wav"

info = sf.info(dry_path)
fs = info.samplerate
max_samples = 30 * fs  # 30 seconds

x, _ = sf.read(dry_path, stop=max_samples, always_2d=True)
y, _ = sf.read(wet_path, stop=max_samples, always_2d=True)

x = x[:, 0]
y = y[:, 0]
min_len = min(len(x), len(y))
x = x[:min_len]
y = y[:min_len]

print(f"Sample rate: {fs}")
print(f"Signal length: {min_len} samples ({min_len / fs:.2f}s)")

# Run analysis (larger hop for speed)
f, t, H, gain_db, coherence, weighted_gain_db = time_varying_transfer(
    x, y, fs, n_fft=2048, hop_length=1024
)

print(f"Frequency bins: {len(f)}")
print(f"Time frames: {len(t)}")
print(f"Gain dB range: [{gain_db.min():.1f}, {gain_db.max():.1f}]")
print(f"Coherence range: [{coherence.min():.4f}, {coherence.max():.4f}]")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

im0 = axes[0].pcolormesh(
    t, f, gain_db, shading="gouraud", vmin=-30, vmax=10, cmap="RdBu_r"
)
axes[0].set_ylabel("Frequency (Hz)")
axes[0].set_title("Transfer Function Gain (dB)")
axes[0].set_ylim(0, 16000)
fig.colorbar(im0, ax=axes[0], label="dB")

im1 = axes[1].pcolormesh(
    t, f, coherence, shading="gouraud", vmin=0, vmax=1, cmap="inferno"
)
axes[1].set_ylabel("Frequency (Hz)")
axes[1].set_title("Magnitude-Squared Coherence")
axes[1].set_ylim(0, 16000)
fig.colorbar(im1, ax=axes[1], label="Coherence")

im2 = axes[2].pcolormesh(
    t, f, weighted_gain_db, shading="gouraud", vmin=-30, vmax=10, cmap="RdBu_r"
)
axes[2].set_ylabel("Frequency (Hz)")
axes[2].set_xlabel("Time (s)")
axes[2].set_title("Coherence-Weighted Gain (dB)")
axes[2].set_ylim(0, 16000)
fig.colorbar(im2, ax=axes[2], label="dB")

plt.suptitle(
    "Time-Varying Transfer Function: Dry → Compressed (first 30s)", fontsize=14, y=1.01
)
plt.tight_layout()
plt.savefig("test_transfer_plot.png", dpi=150, bbox_inches="tight")
print("Plot saved to test_transfer_plot.png")

import os
import sys

import numpy as np
import essentia
import essentia.standard as es
from scipy.signal import hilbert
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import B_sota_analysis.utils as u
from src import losses as loss

SSL_DSET_PATH = "/Volumes/Saola's Drive/thesis/data/Diff-SSL-G-Comp"
DRY_PATH = '/processed_normalized'
WET_PATH = '/processed_ground_truth'

path_dry = SSL_DSET_PATH+DRY_PATH+'/NosPalpitants_UnmasteredWAV.wav'
path_wet = SSL_DSET_PATH+WET_PATH+'/threshold_-12_attack_10_release_0.4_ratio_10/NosPalpitants-exported.wav'

SR = 44100


path_dry = "/Volumes/Saola's Drive/thesis/data/LA2A/all/input_158_.wav"
path_wet = "/Volumes/Saola's Drive/thesis/data/LA2A/all/target_158_LA2A_3c__0__100.wav"
import pickle
from pathlib import Path
CL1B_TEST_PATH = "/Volumes/Saola's Drive/thesis/data/CL1B/data_train.pickle"
def load_cl1b_test(path: str | Path = CL1B_TEST_PATH):
    """
    Load the CL1B test dataset pickle and return the raw Python object
    (dict/list/etc., depending on how it was saved).
    """
    path = Path(path)
    with path.open("rb") as f:
        data = pickle.load(f)
    return data

cl1b_data = load_cl1b_test()

sig_dry = cl1b_data['inp'][6]
sig_wet = cl1b_data['tar'][6]


sig_dry = cl1b_data['inp'][6]
sig_wet = cl1b_data['tar'][6]
# Load audio
#t, sig_dry = u.load_audio(path_dry)
#t, sig_wet = u.load_audio(path_wet)

# compute difference
diff = sig_wet / sig_dry
diff_dB = u.to_dB(np.abs(diff))

# Hilbert
analytic_signal_dry = hilbert(sig_dry)
env_dry_dB = u.to_dB(np.abs(analytic_signal_dry))
analytic_signal_wet = hilbert(sig_wet)
env_wet_dB = u.to_dB(np.abs(analytic_signal_wet))
gr_dB_hilbert = env_wet_dB - env_dry_dB
gr_hilbert = u.to_amplitude(gr_dB_hilbert)

# Recreate signal using Hilbert GR
recreated_sig_hilbert = sig_dry * gr_hilbert

# Calculate RMS models for the different window sizes
eval_windows = [64, 1024, 4096]
gr_eval = {}
for win_size in eval_windows:
    gr_eval[f"{win_size}"] = u.calc_gain_reduction(sig_dry, sig_wet, window_size=win_size)
# PLOTTING

t_start = 61
t_length = 10

# restrict all signals to the plotted time window
mask = (t >= t_start) & (t <= t_start + t_length)

t_cut = t[mask]
sig_dry_cut = sig_dry[mask]
analytic_signal_dry_cut = analytic_signal_dry[mask]
rms_env = u.window_rms(sig_dry, eval_windows[1], in_dB=False)
rms_env_cut = rms_env[mask]
diff_dB_cut = diff_dB[mask]
gr_dB_hilbert_cut = gr_dB_hilbert[mask]
gr_eval_cut = {name: gr[mask] for name, gr in gr_eval.items()}
sig_wet_cut = sig_wet[mask]
recreated_sig_hilbert_cut = recreated_sig_hilbert[mask]

fig, ax = plt.subplots(nrows=3, sharex='all', tight_layout=True, figsize=(10, 10))
ax[0].set_title("Hilbert Envelope vs RMS for Gain Reduction and Reconstruction")
ax[0].set_ylabel("Amplitude")
ax[0].plot(t_cut, sig_dry_cut, label='DRY')
ax[0].plot(t_cut, np.abs(analytic_signal_dry_cut), label='Hilbert Env')
ax[0].plot(t_cut, rms_env_cut, label='RMS Env (Medium: 1024)')
ax[0].legend()

ax[1].set_title("Gain Reduction [dB]")
ax[1].set_ylabel("Gain Reduction [dB]")
ax[1].plot(t_cut, diff_dB_cut, label='Difference (Ground Truth)', alpha=0.5)
ax[1].plot(t_cut, gr_dB_hilbert_cut, label='Hilbert GR')
for name, gr in gr_eval_cut.items():
    ax[1].plot(t_cut, gr, label=f'RMS GR ({name})', linestyle='--')
ax[1].grid()
ax[1].legend()

ax[2].set_title("Reconstruction")
ax[2].set_ylabel("Amplitude")
ax[2].set_xlabel("Time [s]")
ax[2].plot(t_cut, sig_wet_cut, label='WET (Ground Truth)', alpha=0.5)
ax[2].plot(t_cut, recreated_sig_hilbert_cut, label='Recreated - Hilbert')
ax[2].legend()

plt.show()

s_wet = torch.as_tensor(sig_wet, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
s_hilbert = torch.as_tensor(recreated_sig_hilbert, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

results = [
    ("Hilbert Envelope", loss.compute_all_losses(s_hilbert, s_wet))
]

# Add RMS results for all window sizes
for name, gr_dB in gr_eval.items():
    gr_linear = u.to_amplitude(gr_dB)
    recreated_rms = sig_dry * gr_linear
    s_rms = torch.as_tensor(recreated_rms, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    results.append((f"RMS ({name})", loss.compute_all_losses(s_rms, s_wet)))

# Add average gain reduction result (using largest window: 4096)
avg_gr = u.to_amplitude(np.mean(gr_eval["4096"]))
recreated_avg = sig_dry * avg_gr
s_avg = torch.as_tensor(recreated_avg, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
results.append(("Avg RMS (4096)", loss.compute_all_losses(s_avg, s_wet)))

# Print comparison table
loss_names  = list(results[0][1].keys())
label_width = max(len(r[0]) for r in results) + 2
col_width   = 10

header = f"\n{'Comparison':<{label_width}}  " + "  ".join(f"{n:>{col_width}}" for n in loss_names)
print(header)
print("─" * len(header))
for label, losses in results:
    vals = "  ".join(f"{losses[n]:>{col_width}.6f}" for n in loss_names)
    print(f"{label:<{label_width}}  {vals}")
 
s_wet = torch.as_tensor(sig_wet, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
s_hilbert = torch.as_tensor(recreated_sig_hilbert, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

results = [
    ("Hilbert Envelope", loss.compute_all_losses(s_hilbert, s_wet))
]

# Add RMS results for all window sizes
for name, gr_dB in gr_eval.items():
    gr_linear = u.to_amplitude(gr_dB)
    recreated_rms = sig_dry * gr_linear
    s_rms = torch.as_tensor(recreated_rms, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    results.append((f"RMS ({name})", loss.compute_all_losses(s_rms, s_wet)))

# Add average gain reduction result (using largest window: 4096)
avg_gr = u.to_amplitude(np.mean(gr_eval["4096"]))
recreated_avg = sig_dry * avg_gr
s_avg = torch.as_tensor(recreated_avg, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
results.append(("Avg RMS (4096)", loss.compute_all_losses(s_avg, s_wet)))

# Print comparison table
loss_names  = list(results[0][1].keys())
label_width = max(len(r[0]) for r in results) + 2
col_width   = 10

header = f"\n{'Comparison':<{label_width}}  " + "  ".join(f"{n:>{col_width}}" for n in loss_names)
print(header)
print("─" * len(header))
for label, losses in results:
    vals = "  ".join(f"{losses[n]:>{col_width}.6f}" for n in loss_names)
    print(f"{label:<{label_width}}  {vals}")
 
from B_sota_analysis.loss_evals import calc_losses
results = calc_losses(
    unmastered=path_dry,
    target=path_wet,
    model="TCN_S_TF",
    device="cpu"
)

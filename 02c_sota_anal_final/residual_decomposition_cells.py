"""§2b — Decomposing the gain-matched residual into delay artifact vs linear
coloration vs nonlinearity.

Run this AFTER the Summary cell of ``eval_output_transformer_nonlinearity.ipynb``
with a single notebook cell::

    %run -i residual_decomposition_cells.py

The ``-i`` flag executes in the notebook namespace, so this script reuses the
already-built ``results`` dict plus the helpers/constants defined earlier
(``pair_paths``, ``_stft``, ``_read_dry_wet_segment``, ``_pair_num_frames``,
``gr_db_to_gain``, ``GR_DB_MIN/MAX``, ``SAMPLE_RATE``, ``FREQS``, ``chunk_frames``,
``N_FFT``, ``EPS``, ``fmask``, ``band``, ``F_MIN``, ``F_MAX``, ``short_label``,
``SETTINGS_BY_SEV``, ``HARDEST``). It adds ``results[setting]["aligned"]`` and
renders the §2b plot + table.

Why this matters
----------------
The §2 residual ``r = wet - g*dry`` lumps together three different things; the
rising, *setting-independent* HF tail in the §2 plot is the tell that most of it
is not coloration:

  (a) a constant (fractional-sample) delay between exported wet and dry — a pure
      delay leaves a residual rising ~ sin^2(pi f tau), identical across settings,
      that a zero-phase broadband gain g(t) cannot correct;
  (b) a fixed linear coloration (frequency response the broadband gain misses) —
      removable by the best per-bin LTI filter H1 = Sxy/Sxx;
  (c) genuinely non-LTI energy (nonlinear + time-varying gain) = 1 - gamma^2 —
      the only part a learned colour model can reproduce. This is the ceiling.

§2b.1 estimates the per-pair delay, realigns wet, and re-accumulates S_rr.
§2b.2 overlays (a) no-align, (b) aligned, (c) best-LTI floor so the gaps read off:
(a)->(b) = delay artifact, (b)->(c) = fixed linear coloration, (c) = learnable.
Note 1 - gamma^2 is already immune to any LTI delay/EQ, so it needs no alignment;
the alignment pass only attributes how much of the §2 gap is delay vs frequency
response.
"""

import gc

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# ── §2b.1 — sub-sample alignment of wet to the gain-matched signal ───────────

def _rfft_delay(x, tau):
    """Shift x by tau samples via an FFT phase ramp (positive tau = delay)."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n)
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * tau), n).astype(np.float32)


def _gcc_phat_lag(sig, ref, max_lag=64, upsample=32):
    """Sub-sample lag (samples) maximising the PHAT cross-correlation of sig vs ref."""
    n = len(sig) + len(ref)
    nfft = 1 << int(np.ceil(np.log2(n)))
    R = np.fft.rfft(sig, nfft) * np.conj(np.fft.rfft(ref, nfft))
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, nfft * upsample)            # sinc-interpolated xcorr
    m = int(max_lag * upsample)
    cc = np.concatenate((cc[-m:], cc[: m + 1]))
    return (np.argmax(cc) - m) / upsample


def _load_gr_mono(gr_path):
    gr = torch.load(gr_path, weights_only=False, map_location="cpu")["gr_db"].float()
    if gr.ndim == 1:
        gr = gr.unsqueeze(0)
    if gr.shape[0] > 1:
        gr = gr.mean(dim=0, keepdim=True)
    return gr


def estimate_pair_delay(song, setting, probe_sec=40.0, max_lag=64):
    """Signed delay d (samples) so that _rfft_delay(wet, d) best matches m = g*dry."""
    dry_path, wet_path, gr_path = pair_paths(song, setting)
    gr = _load_gr_mono(gr_path)
    total = min(_pair_num_frames(dry_path, wet_path, SAMPLE_RATE), gr.shape[-1])
    stop = min(total, int(probe_sec * SAMPLE_RATE))
    dry, wet = _read_dry_wet_segment(dry_path, wet_path, 0, stop, SAMPLE_RATE)
    L = min(dry.shape[-1], wet.shape[-1])
    dry, wet = dry[..., :L], wet[..., :L]
    m = (dry * gr_db_to_gain(gr[..., :L].clamp(GR_DB_MIN, GR_DB_MAX), clamp=False)).squeeze(0).numpy()
    w = wet.squeeze(0).numpy()
    tau = _gcc_phat_lag(w, m, max_lag=max_lag)
    # GCC sign convention is easy to flip; keep whichever shift of wet best matches
    # m, which makes the estimate convention-proof.
    cand = sorted({0.0, tau, -tau}, key=lambda d: float(np.mean((_rfft_delay(w, d) - m) ** 2)))
    return cand[0]


@torch.no_grad()
def accumulate_aligned(setting, songs):
    """Re-accumulate S_rr (and the matching S_yy) with wet realigned to the dry grid."""
    Srr = np.zeros(len(FREQS))
    Syy = np.zeros(len(FREQS))
    delays = []
    for song in songs:
        d = estimate_pair_delay(song, setting)
        delays.append(d)
        dry_path, wet_path, gr_path = pair_paths(song, setting)
        gr = _load_gr_mono(gr_path)
        total = min(_pair_num_frames(dry_path, wet_path, SAMPLE_RATE), gr.shape[-1])
        for o in range(0, total, chunk_frames):
            stop = min(o + chunk_frames, total)
            dry, wet = _read_dry_wet_segment(dry_path, wet_path, o, stop, SAMPLE_RATE)
            L = min(dry.shape[-1], wet.shape[-1])
            if L < N_FFT:
                continue
            dry, wet = dry[..., :L], wet[..., :L]
            grc = gr[..., o:o + L].clamp(GR_DB_MIN, GR_DB_MAX)
            m = (dry * gr_db_to_gain(grc, clamp=False)).squeeze(0).numpy()
            w = _rfft_delay(wet.squeeze(0).numpy(), d)
            M = _stft(m)
            Wl = _stft(w)
            Syy += np.sum(np.abs(Wl) ** 2, axis=1)
            Srr += np.sum(np.abs(Wl - M) ** 2, axis=1)
        gc.collect()
    return Srr, Syy, float(np.mean(delays))


print("Estimating sub-sample delay + re-accumulating aligned residual per setting...")
for si, setting in enumerate(SETTINGS_BY_SEV, start=1):
    songs = results[setting]["songs"]
    Srr_a, Syy_a, d_mean = accumulate_aligned(setting, songs)
    results[setting]["aligned"] = {"Srr": Srr_a, "Syy": Syy_a, "delay": d_mean}
    print(f"[{si:2d}/{len(SETTINGS_BY_SEV)}] {short_label(setting):34s} "
          f"delay = {d_mean:+7.3f} samples  ({d_mean / SAMPLE_RATE * 1e3:+.4f} ms)")


# ── §2b.2 — three-way decomposition of the residual ─────────────────────────
band_lf = (FREQS >= 20) & (FREQS < 500)

acc = results[HARDEST]["acc"]
al = results[HARDEST]["aligned"]
no_align = 10 * np.log10(acc.distortion_ratio() + EPS)
aligned = 10 * np.log10(al["Srr"] / (al["Syy"] + EPS) + EPS)
best_lti = 10 * np.log10(acc.linear_unexplained() + EPS)

fig, ax = plt.subplots(figsize=(11, 5))
ax.semilogx(FREQS[fmask], no_align[fmask], lw=1.3, color="#9467bd",
            label=r"(a) gain-matched, no alignment  $S_{rr}/S_{yy}$")
ax.semilogx(FREQS[fmask], aligned[fmask], lw=1.3, color="#1f77b4",
            label=r"(b) gain-matched, sub-sample aligned")
ax.semilogx(FREQS[fmask], best_lti[fmask], lw=1.6, color="#d62728",
            label=r"(c) best per-bin LTI floor  $1-\gamma^2$")
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel(r"residual / output  (dB)")
ax.set_title(f"§2b  Residual decomposition -- hardest setting ({short_label(HARDEST)}, "
             f"delay {al['delay']:+.2f} samp)\n"
             "(a)->(b): delay artifact    (b)->(c): fixed linear coloration    "
             "(c): nonlinear + time-varying (learnable)")
ax.set_xlim(F_MIN, F_MAX)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.show()


def _band_db(num, den, mask):
    return 10 * np.log10(np.sum(num[mask]) / (np.sum(den[mask]) + EPS) + EPS)


rows = []
for setting in SETTINGS_BY_SEV:
    a = results[setting]["acc"]
    al = results[setting]["aligned"]
    nlti = a.linear_unexplained() * a.Syy            # non-LTI residual power per bin
    for bname, bmask in [("LF<500", band_lf), ("200-8k", band)]:
        rows.append({
            "Setting": short_label(setting), "Band": bname,
            "delay samp": round(al["delay"], 3),
            "(a) no-align dB": _band_db(a.Srr, a.Syy, bmask),
            "(b) aligned dB": _band_db(al["Srr"], al["Syy"], bmask),
            "(c) best-LTI dB": _band_db(nlti, a.Syy, bmask),
        })
decomp_df = pd.DataFrame(rows)
print("Residual decomposition (dB residual/output). (c) is the learnable nonlinear ceiling.")
try:
    display(decomp_df.round(2))          # noqa: F821  (IPython builtin under %run -i)
except NameError:
    print(decomp_df.round(2).to_string(index=False))

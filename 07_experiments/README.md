# 07_experiments — inference-only evaluation experiments

Academic-style analysis experiments on the two trained stages — **no training
anywhere**. Every notebook loads existing checkpoints, runs locally on CPU, and
reads directly from the thesis data drive (same conventions as the 05/06 eval
notebooks). Training-side follow-ups (loss ablations, retraining the coloration
head, Δg on/off ablations) are deliberately out of scope here.

**Models under test**

| stage | model | run | headline |
|---|---|---|---|
| 1 — gain computer | `DetectorGRLSTM` (05_conditioning, 5.4 k) | `lstm_gr_20260702_174039_lstm_detector_gr` | test GR MAE 0.268 dB |
| 2 — gain application | `GainPriorDiffSSLLSTM` (06_output, 8.3 k) | `gain_prior_20260702_085618_diffssl_lstm32_gain_prior` | test GR MAE 0.159 dB (oracle GR) |
| 2 (variant) | `GainPriorWSDiffSSLLSTM` (waveshaper coloration) | `gain_prior_ws_20260703_*` | set `SELECTED_GAIN_PRIOR_RUN` where supported |

## Notebooks

| notebook | question | method |
|---|---|---|
| `01_gr_source_ablation` | **What if the GR input is something else?** What is the honest (leakage-free) end-to-end number? | Same checkpoint streamed with 8 GR sources (oracle / detector cascade / bypasses / constants / re-windowed conventions), full val+test splits, 9-column metric engine, error-budget decomposition |
| `02_gr_robustness` | How does GR-input error propagate — and does the network absorb it? | Controlled corruption sweeps (noise, bias, scale, lag, smoothing, quantisation) through model vs raw-multiply paths; first-order (11.5 %/dB) check; absorbed-fraction table |
| `03_detector_characterization` | What gain law did the detector actually learn? | Bench-test methodology (Eichas & Zölzer-style): static compression curves, threshold/ratio extraction, knob-interpolation sweeps at unseen values, step-response ballistics, timbre/phase-invariance + level-equivariance audit, learned-τ readout |
| `04_gain_stage_coloration` | Is anything beyond the time-varying gain being modelled? | Residual accounting (target vs model residual, gain/color decomposition), THD & harmonic-series bench test, waveshaper Chebyshev/Fourier analysis (ws run), Δg-vs-depth statistics |
| `05_transient_coldstart` | Does Δg really re-sharpen the label's 23 ms RMS smear at attacks? How long is cold-start warmup really? | Onset-triggered averaging of envelope error / Δg / detector GR error across hundreds of events; cold-vs-warm convergence curves for both stages |
| `06_sidechain_demo` | Does the factorisation deliver the stated end-goal? | GR predicted from a *different* signal (music / synthetic kick pattern) applied to the programme; behavioural evidence + audio exports |
| `07_realtime_benchmark` | Can the cascade run live? | RTF + per-buffer latency (mean/p99 vs budget, 128–4096 samples, 1 CPU thread), chunked-vs-full exactness verification, algorithmic-latency accounting |

Suggested order: 01 → 02 form the system-level story; 03 → 04 → 05 are the
per-stage interpretability story; 06 → 07 are the deployment story (capability +
feasibility). Measured feasibility headline (M-series CPU, single thread, eager
PyTorch): cascade **3.3× realtime**, p99 per-buffer cost ≤ 47 % of budget at every
buffer size 128–4096, chunked streaming numerically exact, intrinsic latency
5.8 ms (one detector frame, zero lookahead).

## Running

```bash
uv sync                       # repo root, once
cd 07_experiments
uv run jupyter lab            # open notebooks from this directory
```

- `exp_common.py` bootstraps sys.path for the sibling folders (06_output takes
  precedence over 05_conditioning for colliding module names, 05 is appended for
  `model_detector_gr`/`gr_target` — same trick as the 06 eval notebooks) and
  provides: checkpoint loaders, deployment-exact stateful streaming, the canonical
  9-column metric engine (ported from `02b`, verified against
  `nablafx.evaluation` there), probe-signal generators, onset detection, and
  harmonic-analysis helpers.
- Requires the local `./nablafx/` clone (TVFiLMCond) and the data drive mounted at
  `/Volumes/Saola's Drive/AllCode/thesis/data/`.
- Approximate runtimes (M-series CPU): 03 ≈ 2 min · 07 ≈ 3–5 min ·
  04, 05, 06 ≈ 5–10 min · 02 ≈ 10 min · 01 ≈ 30–60 min full splits (set
  `MAX_PAIRS` to smoke-test).
- Figures/CSVs/WAVs land in `07_experiments/eval_output/`.

## Cross-references

- `06_output/MODEL_GAIN_PRIOR.md` — §5.2 open coloration question (→ 04), §5.3
  oracle-leakage caveat (→ 01, 02), §2.2 Δg attack-smear claim (→ 05).
- `05_conditioning/MODEL_DETECTOR_GR.md` — §2.1 timbre-blindness claim (→ 03),
  §6.2 substitutable interface (→ 01, 06).

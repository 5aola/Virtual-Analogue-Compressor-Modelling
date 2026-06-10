# mamba_va — a streaming selective SSM for analog compressor modeling

A small PyTorch codebase for capturing nonlinear, time-variant audio effects
(analog dynamic-range compressors: LA-2A, CL 1B, …) from raw WAV input/output
pairs. It is an **improvement on Mamba / S6** designed specifically for this
problem. The two headline changes:

1. **No tokenization.** The model consumes audio one scalar sample at a time.
   No 64-sample windows, no FFT feature vectors, no framing latency. Training
   uses a parallel scan; inference streams sample-by-sample.
2. **A nonlinear state.** A learnable, asymmetric attack/release level detector
   gives the model the *nonlinear* internal memory that the math of system
   realization says a faithful compressor model requires (see `DESIGN.md`).

Since v0.2 every time constant in the model (detector attack/release bands,
SSM ``Delta`` init) is parameterised and initialised in **physical units at the
audio sample rate** -- spanning ~0.1 ms to ~2 s -- the level is processed on a
**normalized dB scale** (compressor gain laws are linear in dB), and the
detector is solved by the **parallel scan** (two linear passes + an
input-dependent gate) instead of a per-sample Python loop, which makes it
~100x faster to train and exactly streamable.

## Why this is different from a vanilla S6 / Riccardo's Optical-DRC model

| | Riccardo S6 / Optical-DRC | mamba_va |
|---|---|---|
| Input | 64-sample window as a feature vector `(B,1,64)` | scalar stream `(B,L)` |
| FFT features | yes | none |
| State recurrence | linear in state | linear SSM **+** nonlinear detector |
| Nonlinearity | only outside the SSM | inside the state (detector) and at the gain readout |
| Streaming | windowed (latency) | true sample-by-sample, state-exact |
| Output | gain × input | gain (dB) × input |

The result is a causal, streamable model whose hidden variables track the
program-dependent envelope (the slow, signal-dependent release that makes
optical compressors hard) instead of re-deriving it from a short window every
step.

## Install

```bash
pip install -r requirements.txt   # torch, numpy, soundfile
# or, from this folder:
PYTHONPATH=. python3 -c "import mamba_va; print(mamba_va.__version__)"
```

## Quick start

Smoke-train on built-in synthetic data (no files needed):

```bash
PYTHONPATH=. python3 -m mamba_va.train --synthetic --synth_sr 8000 \
    --synth_seconds 1.0 --epochs 20 --batch_size 3 --seq_len 512 \
    --d_model 16 --d_state 8 --n_layers 2 --n_bands 3 --out runs/synth
```

Train on your own WAV pairs (manifest CSV with columns `dry,wet` plus one
numeric column per device parameter, e.g. `threshold,ratio,attack,release`):

```bash
PYTHONPATH=. python3 -m mamba_va.train --manifest data/cl1b/manifest.csv \
    --d_model 24 --n_layers 3 --d_state 16 --n_params 4 \
    --seq_len 2048 --batch_size 8 --epochs 200 --out runs/cl1b
```

Render audio through a trained model (streaming):

```bash
PYTHONPATH=. python3 -m mamba_va.infer --ckpt runs/cl1b/best.pt \
    --in input.wav --out output.wav --params 0.3 0.5 0.1 0.7
```

## Tests

```bash
PYTHONPATH=. python3 tests/test_scan.py     # parallel scan == sequential, state carry, chunked lengths
PYTHONPATH=. python3 tests/test_detector.py # detector: parallel==sequential==streaming, tau mapping
PYTHONPATH=. python3 tests/test_model.py    # shapes, causality, streaming==parallel, grads
PYTHONPATH=. python3 tests/smoke_train.py   # a few steps reduce ESR on synthetic data
```

## Repository layout

```
mamba_va/
  scan.py       chunked two-level parallel scan and sequential associative scan
  ssm.py        SelectiveSSM — input-dependent Δ,B,C; detector-driven selectivity
  detector.py   AdaptiveLevelDetector — log-τ asymmetric attack/release memory,
                dB-domain level front end, solved by the parallel scan
  blocks.py     RMSNorm, causal depthwise conv, Mamba-style gated block
  film.py       FiLM conditioning on device parameters
  model.py      CompSSM — full model + streaming render()
  losses.py     pooled ESR, pre-emphasis ESR, multi-resolution STFT, DC; CombinedLoss
  data.py       WAV pair dataset, time-split, TBPTT loader
  synth.py      synthetic compressor for smoke tests / no-data training
  train.py      CLI training entry point
  infer.py      CLI streaming inference
  utils.py      state detach, param counting
tests/          scan / detector / model / smoke-train tests
```

See `DESIGN.md` for the theory, the audio-specific challenges, and the
tokenization analysis.

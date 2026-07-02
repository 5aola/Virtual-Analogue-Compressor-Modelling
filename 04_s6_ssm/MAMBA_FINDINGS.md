# CompSSM / mamba_va — Findings & Recommendations (2026-07-02)

Code-level review of `04_s6_ssm/mamba_va` and the `train_mamba_va_nocond.ipynb`
runs, prompted by "the model didn't learn anything". Verdict up front: **the
architecture is not disproven — the runs were too small to test it**, the loss
readout hid what progress there was, and one structural choice (unconstrained
per-sample gain) works against it. The scan/detector engineering is sound.

## 1. What the runs actually show

Best run (`mamba_va_20260610_110429`, L4, 21.8k params):

- Combined val loss fell **1.34 → 1.24** over ~24 epochs, then plateaued;
  early stopping ended it around epoch 33.
- Sanity floor from the notebook itself: a single **constant gain** (~−10 dB)
  achieves **ESR ≈ 0.075–0.090** on the val songs. Combined val loss ≈ 1.24 is
  *not* directly comparable to that (see §3), so "didn't learn anything" was
  never actually established — read `esr/val` from
  `mamba_va_runs/<run>/csv/metrics.csv` before drawing conclusions.
- A unity-gain (identity) model on this setting (threshold −4, ratio 10,
  gain ≈ −13…−3 dB) would have ESR ≈ 4+. If `esr/val` is near or below the
  constant-gain floor, the model learned the average gain but not dynamics;
  that is a *training-signal* problem, not proof of a broken architecture.

## 2. Root cause #1 — data budget (the dominant factor)

Per-epoch audio throughput, from the notebook's own config:

```
12 walks × 4 segments × 8 chunks × 16384 samples ≈ 2.4 min of audio / epoch
× ~33 epochs before early stop                    ≈ 80 min of audio TOTAL
```

The 02b LSTM32TVC baseline sees **325 min/epoch × 100 epochs** — roughly
**400×** more data. No compressor model in the reference projects converges on
80 minutes of audio. The loss was still improving when early stopping fired.

The GPU was at **15 % memory** (3.6 GB of 23.7 GB) and 765 ms/step. Fixes, in
order of leverage:

1. Raise `BATCH_SIZE` to 12–16 and `WALKS_PER_EPOCH` to ~48 (target ≥ 30 min
   audio/epoch; memory scales ~linearly, there is 6× headroom).
2. Raise `EARLY_STOP_PATIENCE` — with a fixed epoch budget and cosine LR
   (as in 02b) instead of plateau-LR + early stop.
3. The 765 ms/step is scan-bound in fp32; larger batches amortize it (the
   parallel scan is O(L log chunk) regardless of B).

## 3. Root cause #2 — the loss hid the signal

`CombinedLoss = ESR + 0.5·preemph-ESR + 0.5·MR-STFT + 0.1·DC`. The MR-STFT
**log-magnitude L1** term sits near ~1.0 even for perceptually decent matches
(it is dominated by low-energy bins), so the combined number is a poor progress
indicator: a model can halve its ESR while the combined loss moves 5 %.

- Log each component separately (they are cheap to split in `CombinedLoss`).
- Monitor `esr/val` against two printed floors: constant-gain ESR (~0.08) and
  identity ESR (~4). "Learning dynamics" == clearly below the first.

## 4. Root cause #3 — unconstrained per-sample gain

`GainComputer` emits an independent gain value **every sample** from
sample-rate features (`model.py:120`). Real compressor gain is bandlimited
(attack ≥ 1 ms, release ≥ 100 ms); an unconstrained per-sample gain produces
modulation noise/aliasing that the STFT loss punishes, and the easiest way for
the optimizer to suppress that noise is to flatten the gain — i.e. converge to
quasi-static gain. Exactly the observed failure mode. Fixes:

1. **Predict the gain at block rate** (64–128 samples) and upsample with
   linear interpolation (the TFiLM pattern), or
2. keep per-sample prediction but pass `g_db` through a **one-pole smoother
   with learnable τ** (the detector's `coeff_from_log_tau` machinery already
   exists and the scan can solve it), and
3. keep `y = u·10^(g/20)` — the multiplicative prior is right.

## 5. The convergence opportunity — GR supervision

`CompSSM` already predicts an explicit gain curve and exposes it
(`forward(..., return_gain=True)`). The exported oracle GR curves
(`Diff-SSL-G-Comp/gr_curves/<setting>/<song>.pt`, key `gr_db`, 1024-sample
causal RMS, dB) are **dense per-sample supervision for exactly this quantity**:

```
loss += λ · L1( g_db , oracle_gr_db )        # λ ≈ 0.1–0.5, both in dB
```

This gives the model the training signal the waveform losses fail to deliver
(the envelope is a tiny fraction of waveform-loss gradient), costs nothing at
inference, and unifies the 04 (SSM) and 06 (gain-prior) threads: CompSSM *is*
architecturally the gain-prior model with a learned detector. This is the
single highest-value change if CompSSM development continues.

## 6. What was checked and is NOT the problem

- **`scan.py`** — the two-level Hillis-Steele scan with analytical adjoint
  backward is correct by construction and covered by tests
  (`tests/test_scan.py`: parallel == sequential, gradcheck in float64). Not a
  gradient bug.
- **Detector parameterisation** (`detector.py`) — log-τ time constants,
  normalized-dB level, pilot-gated branch: all sensible; init ranges cover
  0.1 ms–1.5 s.
- **SSM Δt range** (`ssm.py`, `dt_min=1e-5`, `dt_max=1e-2`) — spans ~0.1 ms to
  ~2.3 s memory at 44.1 kHz; the v0.1 "23 ms cap" issue is fixed.

## 7. Recommended validation ladder (cheapest first)

1. **Overfit test**: 30 s of one song, no split, ~2k steps. Target ESR < 0.01.
   If this fails, there is a real architecture/optimization bug — stop and fix
   before any full run. If it passes, the problem is (and was) data budget.
2. Add GR supervision (§5) + block-rate gain (§4); rerun the overfit test.
3. Full dataset at ≥ 30 min audio/epoch (§2) with component-wise loss logging
   (§3), fixed 100-epoch cosine budget.
4. Only then compare against the 02b baseline and the 06 gain-prior model.

## 8. Thesis risk management

CompSSM is a from-scratch architecture; the SSM chapter should not depend on
it. The nablafx fork ships **38 reference experiments** (TCN/GCN/S4/LSTM +
FiLM/TFiLM/TVFiLM) on this exact dataset, and the Comparative-Study repo has
S4D/S6/LRU recipes with checkpoints. Use one of those as the SSM
representative for the headline comparison table; position CompSSM as an
exploratory contribution gated on the §7 ladder.

---
*Sources: `04_s6_ssm/mamba_va/mamba_va/{model,detector,ssm,scan,losses}.py`,
`04_s6_ssm/train_mamba_va_nocond.ipynb` (run `mamba_va_20260610_110429`),
`04_s6_ssm/eval_mamba_va.ipynb`, 02b/06_output eval CSVs.*

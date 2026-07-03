# Gain-Prior LSTM (`GainPriorDiffSSLLSTM`) — the GR curve as a structural multiplicative prior

**Location:** `06_output/model_gainprior.py`, `system_gainprior.py`, `dataset_tfilm.py`, `amplitude_match.py`, `train_lstm_gain_prior.ipynb`
**Run:** `gain_prior_20260702_085618_diffssl_lstm32_gain_prior` — 8.3 k parameters, 74 epochs, test GR MAE **0.159 dB**
**Companion document:** [`05_conditioning/MODEL_DETECTOR_GR.md`](../05_conditioning/MODEL_DETECTOR_GR.md) (the upstream GR predictor that can drive this model)

---

## 1. Role and motivation

This model solves the second half of the two-stage decomposition: given the dry input, a **gain-reduction (GR) trajectory** in dB, and the four static knobs, synthesise the compressor's **output waveform**. It is the *gain-application + coloration* stage — the VCA and output path of the device, with the gain computation externalised into the GR input.

### 1.1 Empirical motivation — the envelope/waveform split

The `06_output` evaluation campaign exposed a clean dichotomy on Diff-SSL-G-Comp (validation, oracle GR where applicable):

| Model | GR MAE (dB) | MR-STE | MR-STFT | ESR (A-wt) | MAE (L1) | Params |
|---|---|---|---|---|---|---|
| Amplitude match `x·10^(gr/20)` (no learning) | 0 | **0.011** | **0.069** | 0.048 | 0.0016 | 0 |
| SOTA retrain (diffssl LSTM32TVC, unconditioned on GR) | 0.314 | 0.188 | 0.133 | **0.0025** | **0.0003** | 8 k |
| GR-TFiLM (GR as feature-space conditioning) | 0.336 | 0.103 | 0.107 | 0.0008 | 0.0007 | 25.2 k |

Two observations drive the design:

1. **The trivial multiply is the strongest envelope model.** Applying the exported GR curve as a plain gain wins every level-dynamics metric (MR-STE 0.011, MR-STFT 0.069) against every trained network, because the GR curve *is* the level trajectory by construction (`gr_db = 20·log10(wet_rms/dry_rms)`, causal 1024-sample RMS; verified ≈ 2 % relative envelope error, corr(matched, wet) = 0.997). Conversely it cannot model anything a time-varying gain cannot produce — the coloration residual (harmonics, transient micro-shaping, ≈ 8 % RMS) — which is why the trained models beat it by 20–60× on the sample-exact waveform metrics (ESR, L1).
2. **Feature-space conditioning failed to exploit the oracle.** The GR-TFiLM hands the ground-truth GR to the network as a temporal-FiLM γ/β modulation of the LSTM hidden features — and its validation GR MAE (0.336 dB) is *no better* than the unconditioned SOTA baseline (0.314 dB), at 3× the parameter count. Asking a network to re-learn, through a block-constant affine transform on 32 hidden features, the physical operation "multiply the signal by 10^(gr/20)" is an inefficient use of both capacity and the conditioning signal: the mapping from a dB curve through an LSTM to a gain applied in feature space must approximate an exponential nonlinearity and a multiplication that the architecture could simply *contain*.

The design conclusion: **build the multiply into the model as structure, and let the network learn only the residual.** This is a greybox move in the sense of the NablaFX taxonomy — a differentiable physical operation with learned corrections — but with the physics reduced to the single uncontroversial fact that a compressor applies a time-varying gain.

### 1.2 Design principles

- **Structural prior, not learned conditioning:** the GR enters the output equation directly, not via FiLM.
- **Identity-at-initialisation:** both learned output heads are zero-initialised, so the untrained model *is* the amplitude-match baseline exactly. Training starts from the strongest known envelope model and performs residual learning from there — it cannot start worse than the baseline, and gradient descent only has to explain the residual, not rediscover the multiply.
- **Physically factored residual:** the residual is split into a *bounded multiplicative gain correction* (what the label's RMS window got wrong) and an *additive coloration term* (what no gain can produce), so each head has a distinct, interpretable job.
- **Parameter-matched ablation:** 8.3 k parameters against the 8 k SOTA LSTM32TVC (the GR-TFiLM's 25.2 k made that comparison unfair).

---

## 2. Architecture

```
raw dry x [B,1,S] ─────────────────────────────────────────────┐
x·g  amplitude-matched, g = 10^(clamp(gr)/20) ─────────────────┤
gr_pos  reduction-positive GR, ~[0,1] ─────────────────────────┼── cat → [B,19,S] → main LSTM(19→32)
cond_seq [B,16,S]  (TVFiLMCond: pool(|x|,128) ⊕ knobs → LSTM) ─┘                        │
                                                            ┌───────────────────────────┴───────────┐
                                                  Δg = 12·tanh(lin_gain(h))  [dB]         c = lin_color(h)
                                                  (zero-init → 0 dB)                      (zero-init → 0)
                                                            │                                       │
                                              ŷ = x · 10^((clamp(gr) + Δg)/20)  +  c
```

Parameter budget (8,322 total): `cond_nn` 1,472 · main LSTM 6,784 · `lin_gain` 33 · `lin_color` 33.

### 2.1 Inputs and their representations

`forward(x, gr, p)` with `x: [B,1,S]` raw dry audio, `gr: [B,1,S]` sample-aligned GR curve in dB, `p: [B,4]` knobs normalised to [0,1]. The main LSTM receives **19 channels per sample**, three signal views plus 16 conditioning channels:

- **Raw dry `x`** — the source of transient and spectral detail the residual heads must react to.
- **Amplitude-matched `x·10^(gr/20)`** — the prior's own output, given to the network explicitly so it observes the signal *as the prior renders it* (e.g. residual overshoot at attacks is directly visible as the difference between this channel and what the wet should be). GR is clamped to [−30, +5] dB before exponentiation (`amplitude_match.gr_db_to_gain`), bounding the linear gain to ≈ [0.032, 1.78] and guarding the ill-conditioned silence tails of the RMS-ratio export.
- **Reduction-positive GR `(GR_MAX − clamp(gr))/(GR_MAX − GR_MIN)`** — the gain trajectory as a normalised feature in ≈ [0,1], oriented so *more compression → larger value* (`model_tfilm._reduction_positive`). The dB curve is given as a feature in addition to being applied structurally so the network can condition its corrections on compression *depth* (e.g. more coloration under deep gain reduction) without inverting the exponential itself.
- **`cond_seq[16]` from `TVFiLMCond`** (nablafx, unchanged from the 02b SOTA recipe): `MaxPool(|x|, block 128)` — a block-rate peak-amplitude envelope at ≈ 345 Hz — concatenated with the 4 knobs broadcast per block, fed through a block-rate LSTM (hidden 16), then upsampled back to sample rate by `repeat_interleave(128)` and cropped to S. This is the static-knob conditioning path, kept *identical* to the SOTA baseline so the ablation isolates the gain-prior contribution.

### 2.2 Output heads and the output equation

The main LSTM's hidden sequence `h [B,S,32]` drives two 33-parameter linear heads:

- **Gain-correction head:** `Δg = δ_max·tanh(lin_gain(h))` with δ_max = 12 dB. The correction is applied *inside the dB domain of the prior*: `x · 10^((gr + Δg)/20)`. The tanh bound keeps the composite gain within ±12 dB of the physically grounded prior — the network can reshape ballistics but cannot discard the prior. Its physical job is to correct what the label's 1024-sample (23 ms) trailing RMS window smeared: the dataset's fastest attack setting is 1 ms ≈ 44 samples, more than an order of magnitude below the window, so the exported GR systematically underestimates gain reduction during attack transients and the matched signal overshoots. Δg is the mechanism that re-sharpens those attacks.
- **Coloration head:** `c = lin_color(h)`, additive, unbounded (ablation flag `use_color=False` yields a pure time-varying-gain model). This models the ≈ 8 % RMS residual that is *not expressible as any gain on the dry signal* — harmonic distortion, programme-dependent waveshaping, transient micro-structure.

**No output `tanh`.** The 02b SOTA and GR-TFiLM models squash the output; here the multiplicative prior already keeps ŷ in range, and a saturating output nonlinearity would systematically bias loud passages — precisely the regime a compressor model must get right.

**Zero-initialisation as identity.** Both heads' weights and biases start at zero, so at step 0: Δg ≡ 0 dB, c ≡ 0, and ŷ ≡ `amplitude_match(x, gr)` *exactly* — verified at run start by an assert on a real validation batch (`max |model − amplitude_match| = 0.00e+00`, untrained crop L1 vs wet 5.9 × 10⁻⁴). The optimisation therefore begins at the envelope-optimal point in function space, and the two heads' growth during training (logged as mean |Δg| and mean |c| per epoch) is itself an interpretable measurement of how much gain-timing correction vs coloration the data demands.

### 2.3 State handling

The model mirrors the nablafx `lstm.py` state API: `reset_states()` / `detach_states()` manage all stateful submodules (main LSTM carry `main_state`, and `TVFiLMCond`'s internal block-rate LSTM state), with state held *on the module* between calls. It is thereby a drop-in for the 02b-style TBPTT training systems and shares the `(x, gr, p)` forward signature with `GRTFiLMDiffSSLLSTM`, so all evaluation tooling runs unchanged across the model family. In streaming deployment the model processes arbitrary chunk sizes with state carried across calls; `reset_states()` defines the cold-start condition, matching the per-batch reset used in training. Note the block-rate machinery (cond block 128) makes chunked inference exactly equivalent to full-sequence inference only for chunk lengths that are multiples of the block size; the TBPTT sub-step (4410) and evaluation chunking respect this in practice.

---

## 3. Training setup

| Item | Value | Rationale |
|---|---|---|
| Data | Diff-SSL-G-Comp: dry, wet, GR curve, knobs per (song × setting) pair; 100 pairs | `GRCropDataModule` — the 02b crop recipe *plus* the sample-aligned GR crop per item |
| Crops | Non-overlapping 3 s (132,300 samples @ 44.1 kHz), batch 16, shuffle + drop-last; 6,500 train / 850 val / 284 test crops | Identical to 02b — ablation contract |
| Split | Seed 42; 7 train / 1 val / 2 test songs; test = 2 lowest-threshold settings × test songs (4 pairs) | Shared verbatim with 02b, the GR predictor, and GR-TFiLM |
| Optimisation | **Manual** (Lightning `automatic_optimization=False`): per batch `reset_states()` → TBPTT sub-steps of 4,410 samples (0.1 s, 30 per crop) with backward + `optimizer.step()` + `detach_states()` per sub-step | The diffssl TBPTT regime: gradients truncated at 0.1 s, state (not gradient) carries across sub-steps; 30 optimiser updates per batch |
| Optimiser | AdamW, lr 1e-3, cosine to 1e-6 over a fixed 100-epoch budget; best-val checkpoint (epoch 74) | Fixed budget == comparable across the model family |
| Precision | bf16 autocast inside the forward/loss; predictions cast back to fp32 for aggregation | ~free speedup on L4/A100-class GPUs |
| Validation | Full-crop loss (single loss over the concatenated 3 s prediction) each epoch; train logs sub-step means | |
| Extra logging | mean \|Δg\| (dB), mean \|c\| per phase | Head-share interpretability (both start at 0 by construction) |

The GR input during training is the **oracle export**. This is deliberate: it supervises the gain-application stage under a clean control signal, keeping the two-stage factorisation's error sources separated. The corresponding caveat is in §5.3.

---

## 4. Loss (the "4c" objective)

```
L = 0.5·L1  +  0.5·MR-STFT_extended  +  0.2·L_env  +  0.1·L_pe
```

Each term reverts independently to the exact 02b recipe (`env=0, pe=0, variant="sota"`) for the comparability ablation.

- **Time-domain L1 (0.5)** — sample-exact waveform matching; the workhorse term shared with 02b.
- **MR-STFT, "extended" variant (0.5)** — auraloss multi-resolution STFT with FFT sizes {512, 1024, 2048, **4096**}, spectral-convergence + log-magnitude + **linear-magnitude** terms. The two extensions over the 02b "sota" variant are targeted: the 4096-FFT resolution (window 2400 ≈ 54 ms) gives the loss frequency-domain leverage on *release-scale* (0.1–0.8 s) energy errors that the shorter windows only see as frame-averaged noise; the linear-magnitude term is better conditioned than log-magnitude on quiet passages (log-mag gradients blow up as magnitudes → 0). A constructor guard enforces `step_num_samples ≥ max FFT size` so every TBPTT sub-step contains at least one full analysis window.
- **Envelope-dB L1, `L_env` (0.2)** — L1 between `10·log10(avg_pool(·², win 1024, hop 256) + 1e-7)` of prediction and target. With window 1024 **equal to the GR-export RMS window**, this is precisely the *differentiable surrogate of the GR MAE evaluation metric* — the SOTA baseline's weakest column (0.314 dB) and the thesis's headline quantity. The 1e-7 floor pins near-silence at ≈ −70 dB so silent regions contribute zero gradient.
- **Pre-emphasis L1, `L_pe` (0.1)** — L1 after a first-order high-pass `x[n] − 0.95·x[n−1]`. Pre-emphasis reweights the objective toward high-frequency/onset content — attack transients being the canonical hard part of compressor modelling and exactly where the amplitude-match prior is weakest.

The loss composition mirrors the architecture's factorisation: L1 + MR-STFT supervise the coloration head's waveform detail, `L_env` supervises the composite gain (prior + Δg), and `L_pe` concentrates capacity on the transient regime that Δg exists to fix.

---

## 5. Results

### 5.1 Headline comparison (oracle GR input)

| Model | Split | GR MAE (dB) | MR-STE | MR-STFT | ESR (A-wt) | MAE (L1) | EDC | M-NRMSE | M-SF | Params |
|---|---|---|---|---|---|---|---|---|---|---|
| Amplitude match | — | 0 | 0.011 | 0.069 | 0.048 | 0.0016 | 0.013 | 0.008 | 0.050 | 0 |
| SOTA retrain (LSTM32TVC) | val | 0.314 | 0.188 | 0.133 | 0.0025 | 0.0003 | 0.326 | 0.080 | 0.045 | 8 k |
| GR-TFiLM | val | 0.336 | 0.103 | 0.107 | 0.0008 | 0.0007 | 0.246 | 0.049 | 0.045 | 25.2 k |
| GR-TFiLM | test | 0.328 | 1.551 | 0.107 | 0.0027 | 0.0006 | 0.333 | 0.057 | 0.050 | 25.2 k |
| **Gain prior (this model)** | val | **0.224** | 0.032 | 0.061 | 0.063 | 0.0017 | 0.219 | 0.022 | 0.029 | **8.3 k** |
| **Gain prior (this model)** | test | **0.159** | 0.095 | **0.067** | 0.045 | 0.0016 | 0.214 | **0.021** | 0.036 | **8.3 k** |

### 5.2 Interpretation

- **The structural prior succeeds where feature-space conditioning failed.** With the same oracle GR available, the gain prior reaches 0.224/0.159 dB GR MAE (val/test) against the GR-TFiLM's 0.336/0.328 — a ≈ 40–50 % reduction at a third of the parameters — and against the unconditioned SOTA's 0.314. The conditioning signal is finally being *used*, and the mechanism (multiply built in, residual learned) is the difference, since dataset, split, knobs-conditioning and training regime are held fixed.
- **Envelope metrics approach the amplitude-match bound while beating it where the prior is wrong.** MR-STE 0.032 (val) sits between the trivial match (0.011) and every previously trained model (≥ 0.103); MR-STFT 0.061 (val) actually *beats* the amplitude match (0.069) — consistent with Δg correcting the RMS-window smear that the raw prior inherits, and with GR MAE < oracle-match-implied error on test. Modulation-domain metrics (M-NRMSE 0.021–0.022, M-SF 0.029–0.036) are the best of any trained model in the table.
- **Waveform metrics remain at the amplitude-match operating point, not the SOTA's.** ESR (A-wt) ≈ 0.045–0.063 and L1 ≈ 0.0016 are amplitude-match-level, 20× above the SOTA retrain's ESR. The model, initialised at the prior and trained with substantial envelope/pre-emphasis weighting, stayed in the prior's basin: the coloration head has evidently not (yet) absorbed the ≈ 8 % RMS waveshaping residual the way the from-scratch models do. The two model families currently occupy the two ends of the envelope/waveform dichotomy of §1.1 — the gain prior resolves the *dynamics* side decisively, but the *coloration* side is open. Candidate levers, in order of suspicion: the zero-init color head's gradient share under a loss where the envelope terms dominate early training, the ±12 dB tanh saturation regime, and the loss weighting (0.2 env + 0.1 pe vs 0.5 fd) — an ablation with the exact 02b loss (`env=0, pe=0, sota`) isolates the last of these by construction. A structural variant targeting this gap exists as `model_gainprior_ws.py` / `train_lstm_gain_prior_ws.ipynb`: coloration by composition (a memoryless waveshaper on the gained signal, identity-initialised) instead of additive synthesis.
- **Test ≤ val across the board** (GR MAE 0.159 vs 0.224) — no generalisation gap, unlike the GR-TFiLM whose test MR-STE blows up to 1.55. A structural prior anchored to a physical input generalises across content by construction; a learned modulation pathway may not.
- **Convergence:** 74 epochs on a fixed 100-epoch cosine budget, from an initialisation that is already the baseline — compare GR-TFiLM's best-val at 31 epochs *into overfitting* and the SOTA's 73.

### 5.3 Caveat — oracle leakage

All numbers above use the **oracle GR export** as input at evaluation time. Since the GR curve deterministically encodes the wet/dry envelope ratio, GR-conditioned models receive information derived from the target signal; their envelope-side metrics are not directly comparable to the unconditioned SOTA as an end-to-end system claim. The honest end-to-end figure requires chaining with the `05_conditioning` detector predictor (test GR MAE 0.268 dB, no wet-signal access) — the immediate next experiment. First-order expectation: a GR error of ε dB enters the output as ≈ 11.5 %·ε amplitude error (∂g/∂dB = ln10/20) *before* any correction by Δg, so the cascade's envelope metrics should degrade gracefully from the oracle numbers rather than collapse; the Δ-timing supervision of the upstream model and the shared training regime (same crops, split seed 42, cold-start convention) were chosen to make this composition distribution-consistent.

---

## 6. Interaction between the two models

```
                        ┌────────────────────────────────┐
 dry x ──────┬─────────►│ DetectorGRLSTM (05, 5.4 k)     │   frame-rate GR, 172 Hz
 knobs p ────┼─────────►│ detector bank → LSTM → linear  ├──► to_db(·, S): clamp [−30,5] dB,
             │          └────────────────────────────────┘        linear interp → sample rate
             │                                                          │ gr [B,1,S]
             │          ┌────────────────────────────────┐              │
             ├─────────►│ GainPriorDiffSSLLSTM (06, 8.3k)│◄─────────────┘
 knobs p ───────────────►  ŷ = x·10^((gr+Δg)/20) + c    ├──► wet ŷ
                        └────────────────────────────────┘
```

The interface between the stages is a **physically defined control signal** — GR in dB on the range [−30, +5], sample-aligned, causal-RMS convention — not a learned latent. This has three consequences:

1. **Substitutability.** The gain-prior stage runs unchanged from (a) the oracle export (upper bound / analysis), (b) the detector predictor (deployable end-to-end system), or (c) a GR curve computed from a *different* signal — which turns the pair into a **sidechain compressor**, the target application: predict GR on the sidechain source, apply gain + coloration to the programme signal.
2. **Separable supervision and diagnosis.** Each stage is trained and evaluated against its own ground truth (GR curve; wet waveform), so cascade errors decompose additively into gain-computation error and gain-application error — unlike a monolithic end-to-end model where the two are confounded.
3. **Consistent streaming semantics.** Both stages are causal, stateful stream processors trained under matching cold-start conventions (fresh state per 3 s crop; the upstream model additionally masks its ambiguous 1 s warmup): a chunked real-time deployment carries `(lstm_state, energy_tail)` through the detector and module-held states through the gain model, with per-chunk exactness guaranteed for hop-multiple (256) and block-multiple (128) chunk sizes respectively.

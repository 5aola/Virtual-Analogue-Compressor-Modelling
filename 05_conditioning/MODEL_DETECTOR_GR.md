# Detector-LSTM Gain-Reduction Predictor (`DetectorGRLSTM`)

**Location:** `05_conditioning/model_detector_gr.py`, `system_detector_gr.py`, `dataset_crops.py`, `train_lstm_detector_gr.ipynb`
**Run:** `lstm_gr_20260702_174039_lstm_detector_gr` — 5.4 k parameters, 49 epochs, test GR MAE **0.268 dB**
**Companion document:** [`06_output/MODEL_GAIN_PRIOR.md`](../06_output/MODEL_GAIN_PRIOR.md) (the downstream gain-application model this predictor feeds)

---

## 1. Role and problem formulation

The thesis decomposes end-to-end virtual-analogue compressor modelling into two separately supervised stages:

```
                 ┌─────────────────────────┐  gr_db [B,1,T_frames]  ┌───────────────────────┐
 dry x [B,1,S] ──┤ DetectorGRLSTM (this)   ├── to_db(·, S) ────────►│ GainPriorDiffSSLLSTM  ├──► wet ŷ [B,1,S]
 knobs p [B,4] ──┤ "what gain trajectory?" │   (clamp+interp)       │ "apply gain + colour" │
                 └─────────────────────────┘                        └───────────────────────┘
```

This model solves the first sub-problem: given only the **dry input** and the **four static control parameters** (threshold, attack, release, ratio), predict the compressor's **time-varying gain reduction (GR) trajectory** in dB. It is a *sidechain/gain-computer surrogate* — it never sees or produces audio output; it produces a control signal.

### 1.1 The GR target

The supervision target is a synthetic quantity exported from the Diff-SSL-G-Comp dataset (`03_initial_GR_pred/gr_dataset.py`):

```
gr_db[i] = 20·log10( wet_rms[i] / dry_rms[i] )
```

where both RMS envelopes use a **causal 1024-sample trailing window** (≈ 23.2 ms at 44.1 kHz). The curve is sample-aligned to the dry/wet audio (index *i* of the GR curve corresponds to index *i* of the waveforms; no lag compensation is applied anywhere in the pipeline, because both signals share the same causal windowing convention). Over all 100 exported curves (10 songs × 10 settings) the observed range is [−28.25, +4.21] dB; the project-wide clamp range [−30, +5] dB (`gr_target.GR_DB_MIN/MAX`) covers this with margin and only ever clips ill-conditioned silence tails, never real compression.

Two properties of this target matter for the model design:

1. **It is level-domain by construction.** The wet/dry RMS ratio is invariant to the phase and (largely) to the spectral fine structure of the input — only its short-time energy trajectory matters. A model whose input representation preserves timbre therefore carries *more* information than the target function depends on, and that excess capacity is a memorisation channel.
2. **It embeds one specific smoothing scale (1024 samples).** The analogue unit's true internal detector time constants are unknown and setting-dependent; the label's 23 ms window is an analysis choice, not a physical fact. The model's own envelope analysis should therefore not be hard-wired to that scale.

### 1.2 Why this model replaced the bins recipe

The predecessor (`StatefulCondLSTMGRBins`, run `lstm_gr_20260611_132730`, 11.7 k params) used a raw-waveform convolutional frontend followed by a 71-bin CREPE-style classification head with Gaussian soft targets, BCE loss, label-distribution smoothing (LDS) reweighting, classifier-free-guidance-style conditioning dropout, Fourier knob embeddings, TFiLM, and a stateful truncated-BPTT training loop with explicit cold-start crop mixing. Its measured behaviour:

| Split | GR MAE (dB) |
|---|---|
| Validation (held-out song, all settings) | 0.267 |
| Test (held-out songs, lowest-threshold settings) | **0.554** |

The 2.1× val→test degradation is a generalisation gap, and the diagnosis was **representational**: a learned conv frontend over the raw waveform can encode song-specific timbre, which correlates with the GR target within the training distribution but not across songs. The surrounding machinery (bins, LDS, conditioning dropout, cold-start mixing) had each been added to stabilise training dynamics — i.e. it was patching *symptoms* of the representation problem rather than the problem itself. The present model removes all of it and fixes the representation instead.

---

## 2. Architecture

```
dry x [B,1,S] ── x² ── frame energy, avg-pool hop 256 ──► e [B,1,T]        (T = ⌊S/256⌋, 172.3 Hz)
                                                            │
                              OnePoleDetectorBank (4 learnable τ, causal FIR) 
                                                            │
                                  10·log10(·+1e-10), clamp ≥ −80 dB
                                  normalise (env+80)/80 → ~[0,1]           envs [B,4,T]
                                                            │
knobs p [B,4] ── broadcast over T ──────────────────────────┤
                                                            ▼
                                          concat → [B,8,T] → LSTM(8→32) → Linear(32→1)
                                                            │
                                                     gr [B,1,T]  (dB, unbounded)
```

Total: **5,413 parameters** — LSTM 5,376, head 33, detector log-τ 4.

### 2.1 The detector frontend — the single structural constraint

Exactly **one** inductive bias is imposed: the frontend belongs to the *detector family* of classical dynamics processors — rectify → smooth → convert to dB. This family is **phase-blind and timbre-blind by construction**: squaring destroys phase, and pooling + one-pole smoothing at millisecond-to-hundreds-of-milliseconds scales destroys spectral fine structure. Whatever the network downstream does, it *cannot* condition on song identity through timbre, because that information is removed before any learned parameters see the signal. This is the direct structural fix for the bins model's failure mode.

Crucially, the constraint is a *family*, not a fixed operator:

- **Frame energy.** `avg_pool1d(x², 256)` computes mean-square energy per non-overlapping 256-sample frame (hop 5.8 ms, frame rate 172.3 Hz). This decimates the sequence 256× before the recurrent core, making a 3 s crop only 516 LSTM steps.
- **One-pole detector bank with learnable time constants.** Four exponential smoothers with time constants τ initialised at (12, 2, 40, 150) ms — a spread bracketing the 23 ms label window from both sides plus a fast transient tracker and a slow programme-level tracker. The τ are stored as `log τ` parameters (positivity for free, multiplicative learning dynamics) and clamped to [0.5 ms, 500 ms]. Because the analogue unit's internal energy computation is unknown, the bank *spans* the RMS-window family rather than predefining a member of it; where the τ settle after training is the model's one interpretable diagnostic (inspected in notebook cell 10).
- **Exact FIR realisation, no scan.** Each one-pole IIR `y[t] = (1−a)·e[t] + a·y[t−1]` with `a = exp(−Δt/τ)` is materialised as a **causal depthwise convolution** whose kernel is the truncated impulse response `(1−a)·aʲ`, j = 0…255, renormalised to unit DC gain to correct the truncation tail. The kernels are rebuilt from `log τ` on every forward pass, so the smoothing is differentiable in τ while the whole frontend remains fully parallel (a single `conv1d`) — no sequential scan, no recurrence in the frontend. Kernel length 256 frames gives each detector ≈ 1.49 s of causal context, comfortably longer than the slowest release setting (0.8 s).
- **dB conversion and normalisation.** `10·log10(env + 1e-10)`, clamped at −80 dB, then affinely mapped to ≈ [0, 1]. Working in dB linearises the target relationship (GR is itself a dB quantity, and compressor gain laws are piecewise-linear in the log-log domain), and the clamp bounds the input dynamic range seen by the LSTM.

### 2.2 Conditioning — plain knob concatenation

The four control parameters, normalised to [0, 1] against the physical ranges in `splits.DIFFSSL_PARAM_RANGES` (threshold [−20, 20] dB, attack [0, 30] ms, release [0, 1.6] s, ratio [0, 10]; order per `src.dsp.PARAM_ORDER`), are broadcast along the time axis and concatenated to the four detector envelopes at every frame. No embedding, no FiLM, no hypernetwork. This follows the empirical result of the Comparative-Study reference (EURASIP JASM 2025): for D = 4 static parameters, input concatenation is not measurably worse than learned conditioning mechanisms, and it costs zero parameters. The knobs enter the LSTM's input affine transform, which is formally equivalent to a per-setting learned input bias — sufficient expressivity for static conditioning.

### 2.3 Recurrent core and head — everything else is blackbox

A single-layer LSTM (input 8, hidden 32) followed by `Linear(32→1)` regresses GR in dB **directly** — no bins, no sigmoid range squashing, no decode step. Two considerations justify the blackbox choice here:

- **No compressor gain law is assumed.** The target is itself a synthetic analysis product (RMS-ratio), not the device's physical gain signal; imposing a threshold/ratio/knee gain-computer structure would fit the model to an assumption twice removed from the hardware. (The greybox ablation with a physical gain-computer prior exists separately as `model_dsp_prior.DSPPriorGRLSTM` / `system_dsp_prior.py` — same frontend, plus a DSP ballistics prior and a prior-anchor loss term.)
- **Direct dB regression removes the train/eval mismatch** of the bins recipe, where training optimised BCE over soft bin targets but evaluation decoded via local averaging around the argmax — two different objective geometries. Here the training loss and the evaluation metric (GR MAE in dB) are the same quantity up to the Huber transition.

The LSTM's role, given detector envelopes as input, is learnable **ballistics and gain-law composition**: mapping multi-scale level estimates + knob settings to the attack/release-shaped gain trajectory, including programme-dependent behaviour that a fixed one-pole attack/release model cannot express.

---

## 3. Streaming semantics and state handling

The module is **stateless by design**: no recurrent state is held on `self`. `forward(dry, params, state, return_state)` takes and returns an explicit state tuple:

```python
state = (lstm_state, energy_tail)   # energy_tail: [B, 1, K−1] frames, detached
```

- `lstm_state` is the standard `(h, c)` LSTM carry.
- `energy_tail` is the last K−1 = 255 frame-energy values, prepended to the next chunk's energy sequence so the causal FIR detector sees seamless context across chunk boundaries.

Consequences:

1. **Chunked streaming is exact.** For chunk sizes that are multiples of the hop (the eval notebook uses 10 s chunks rounded to hop multiples), running N chunks with state carry produces bit-identical output to one full-sequence forward — verified deployment path in `train_lstm_detector_gr.ipynb` cell 9 and `eval_lstm_detector_gr.ipynb`. There is no receptive-field re-priming cost per chunk, unlike TCN-style models.
2. **Cold start is well defined.** `state=None` means silence context (`energy_tail = 0`) and zero LSTM state — exactly the condition at the head of every training crop, so train and deployment distributions match at stream start.
3. **Latency/causality.** Every operator is causal; the intrinsic latency is one frame (256 samples ≈ 5.8 ms), the granularity at which GR estimates are emitted.

The interface method `to_db(gr, sample_len)` clamps predictions to [−30, +5] dB and linearly interpolates the 172 Hz frame-rate curve to sample rate — producing exactly the signal format that `06_output/amplitude_match.py` and the gain-prior model consume (§6).

---

## 4. Training setup

| Item | Value | Rationale |
|---|---|---|
| Data | Diff-SSL-G-Comp, 10 songs × 10 settings, 44.1 kHz mono | Only hardware SSL G-Bus dataset with dense setting coverage |
| Crops | Non-overlapping 3 s (132,300 samples), batch 16, shuffle + drop-last | Identical regime to the downstream gain-prior model (`06_output/dataset_tfilm.py` recipe), so the two stages train under the same distribution |
| State | Fresh (`None`) every batch | Matches deployment cold start; replaces the predecessor's stateful-TBPTT + cold-start crop-mixing machinery entirely |
| Split | Seed 42, song-level: 7 train / 1 val / 2 test songs | Song-level held-out prevents content leakage |
| Test condition | 2 held-out songs × 2 **lowest-threshold** settings (4 pairs) | Deliberately the hardest regime: deepest compression, unseen content |
| Optimiser | AdamW, lr 1e-3 | |
| Schedule | Cosine to 1e-6 over a fixed 150-epoch budget; best-val checkpoint selected (epoch 49) | Fixed budget = comparable across runs; no early-stopping hyperparameter |
| Grad clip | 1.0 (norm) | |
| Loggers | TensorBoard + CSV; top-3 + last checkpoints on `loss/val` | |

The split is *shared verbatim* (same seed, same manifest builder `splits.build_split_manifest`) with the SOTA retrain (02b), the GR-TFiLM and the gain-prior notebooks — every model in the comparison table sees identical train/val/test content. Validation covers **all 10 settings** on the held-out song; test isolates song generalisation at the most severe compression settings. Note the test settings do appear (with train songs) in training — the test axis is content generalisation, not setting extrapolation.

Because the frontend's memory is windowed (1.49 s) and the LSTM's useful horizon is on the order of one release time (≤ 0.8 s), 3 s random crops with a warmup mask (below) cover all state-handling requirements; no gradient flows across crop boundaries and none is needed.

---

## 5. Loss

```
L = Huber_β=1dB( gr̂, gr )  +  1.0 · Huber_β=1dB( Δgr̂, Δgr )
```

with all terms **masked** and normalised by the valid-frame count:

- **Level term.** Smooth-L1 (Huber) in dB with β = 1 dB: quadratic below 1 dB error (fine convergence near the target), linear above (robustness to the target's own artefacts — the RMS-ratio label is noisy at energy transitions). The target is aligned to the prediction grid by `avg_pool1d(gr_db, 256)` after trimming to whole frames, so prediction and label live on exactly the same 172 Hz grid.
- **Δ (first-difference) term**, weight 1.0. Huber between consecutive-frame differences of prediction and target. The level term alone under-penalises *timing* errors — a prediction that reaches the correct GR depth a few frames late scores well on level MAE but sounds wrong (attack/release ballistics are the perceptually critical aspect of compression). The Δ term is a frame-rate derivative matching that directly supervises the attack and release slopes.
- **Dry-energy floor mask.** Frames whose dry level `10·log10(mean(x²))` is below −60 dB are excluded. In near-silence the label is a ratio of two vanishing RMS values — numerically ill-conditioned noise, not a learning signal.
- **Warmup mask.** The first 1.0 s (172 frames) of every crop is excluded from the loss. The model starts each crop from silence context and empty state, but the *label* at the crop head reflects true pre-history (the compressor's release tail from before the crop boundary). That prefix is irreducibly ambiguous given the model's inputs; 1.0 s ≥ the longest release setting (0.8 s) guarantees the masked region covers the ambiguity. Without this mask the model would be penalised for information it cannot have, biasing it toward hedged mid-range predictions at stream starts.

The logged `mae_db` metric (masked GR MAE) is the headline evaluation quantity and is directly comparable across the run table.

---

## 6. Results and interaction with the gain-prior model

### 6.1 Headline numbers

| Model | Params | Epochs | Val GR MAE (dB) | Test GR MAE (dB) |
|---|---|---|---|---|
| Bins predictor (`lstm_condlstm_bins_lds_cdrop_coldstart`) | 11.7 k | 479 | 0.267 | 0.554 |
| **Detector predictor (`lstm_detector_gr`, this model)** | **5.4 k** | **49** | — | **0.268** |

The detector model's **test** MAE (0.268 dB) matches the bins model's **validation** MAE (0.267 dB) — i.e. the val→test generalisation gap that motivated the redesign is closed, at 46 % of the parameter count and roughly a tenth of the training epochs. This is consistent with the representational hypothesis: removing the timbre-capable frontend removed the overfitting channel, and with it the need for the six auxiliary regularisation mechanisms.

Secondary audio-domain metrics for a GR predictor (obtained by applying the predicted curve to the dry signal, i.e. amplitude-matching with predicted rather than oracle GR) on test: MR-STE 0.123, MR-STFT 0.153, EDC 0.097, M-NRMSE 0.076 — all substantially better than the bins model's test values (0.271, 0.242, 0.108, 0.141 respectively).

### 6.2 Interface to the downstream model

The two-stage system composes as follows (see the companion document for the second stage):

1. `DetectorGRLSTM(dry, knobs)` → frame-rate GR at 172 Hz;
2. `to_db(gr, S)` → clamp to [−30, +5] dB, linear interpolation to sample rate — the exact format of the exported oracle curves;
3. `GainPriorDiffSSLLSTM(dry, gr, knobs)` → wet audio, with the GR entering as a *structural multiplicative prior* rather than a learned conditioning feature.

Because the GR interface is an explicit physical quantity (dB gain on a defined range) rather than a learned latent, the two models are **independently replaceable**: the gain-prior stage runs identically from the oracle export, from this predictor, or from a GR curve computed on a *different* signal — which is the mechanism by which the system generalises to **sidechain compression**, the stated end-goal of the pipeline. Conversely, the predictor is useful standalone as a gain-reduction meter / analysis tool.

Both stages were trained under the same crop regime, split seed, and cold-start convention specifically so that cascade evaluation (predicted-GR → gain-prior, no oracle anywhere) is distribution-consistent. Error composition is benign to first order: a GR error of ε dB passes through the downstream multiplicative prior as an amplitude error of ≈ 11.5 %·ε (since ∂/∂dB of 10^(g/20) = ln10/20 ≈ 0.115 per dB), and the downstream model's bounded correction head (±12 dB) has the capacity to absorb systematic components of it. The measured 0.268 dB test MAE therefore bounds the cascade's added envelope error at roughly 3 % RMS — of the same order as the coloration residual the second stage models anyway. Quantifying the actual cascade degradation (oracle-GR vs predicted-GR input to the gain-prior model) is the immediate next evaluation.

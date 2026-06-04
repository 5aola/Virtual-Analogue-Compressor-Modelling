# Design notes: improving on Mamba for analog compressor modeling

This document records *why* the model is built the way it is. It answers the
three research questions: (1) what math problem makes this hard and what a
selective SSM can/can't do about it, (2) the main challenges of modeling audio
effects and their long-term dependencies, and (3) tokenization — whether it is
necessary, how we eliminate it, and how the hidden state captures long
nonlinear behavior instead.

## 1. The math: why a *linear* state is not enough

A compressor is a nonlinear, time-variant input/output system. In discrete
time its behavior can be written as an input/output map

    y[k] = g( y[k-1..k-m], u[k..k-m] ).

Shoukry's realization result (Shoukry, 2008) is the relevant theorem: such a
map admits an *observable*, order-`m` state-space realization

    x[k+1] = f( x[k], u[k] ),   y[k] = h(x[k]) = x_1[k]

if and only if the I/O map satisfies a difference-equation observability
condition (DIOM). The crucial point for us is that **the transition `f` is
generically nonlinear**. There is no faithful realization of a nonlinear system
with a transition that is linear in the state.

Mamba / S6 is exactly a *linear-in-state* recurrence:

    h_t = Ā_t h_{t-1} + B̄_t x_t,   y_t = C_t h_t.

The "selectivity" (input-dependent Δ, B, C) makes the *coefficients*
time-varying, but the recurrence in `h` is still linear, and the readout is
linear. So a plain S6 — and Riccardo's Optical-DRC simplification of it, which
puts all nonlinearity *outside* the SSM — is, by the theorem above, structurally
unable to realize the compressor's state transition. It compensates by feeding a
64-sample window so the network can re-derive level information every step; that
is a workaround for a missing nonlinear state, at the cost of latency and
redundancy.

**Our fix is to put a small, physically-grounded nonlinear element *into* the
state.** The level detector (below) is a nonlinear recurrence; its envelope is
fed into the selective SSM as the selectivity signal, so the SSM's effective
decay becomes a function of signal history, and the gain readout is a nonlinear
MLP. The model now has the nonlinear `f` and nonlinear `h` the realization
theorem demands, while keeping the linear SSM's parallelizable, long-memory
core.

## 2. Challenges of audio-effect modeling and long-term dependencies

**Sample rates and sequence length.** Audio is 44.1–48 kHz. A 5-second clip is
~240k samples. Any architecture has to handle very long sequences without an
O(L²) cost — this is why an SSM (linear recurrence, O(L) and parallelizable) is
a better fit than attention.

**Program-dependent release.** Optical compressors have release times that
depend on recent signal history and can run to several seconds. That is a
genuine *long-term dependency*: the gain now depends on what the signal did
hundreds of milliseconds — tens of thousands of samples — ago. The detector's
slow release coefficient is the mechanism that holds this memory.

**Transients are the hard part.** Simionato & Fasciani report fast-attack
behavior and the onset/transient error as the hardest to capture, and the
spectral (high-frequency) mismatch as the most audible failure. We target both
directly in the loss: a pre-emphasis ESR (a first-order high-pass before ESR)
weights transient/onset error, and a multi-resolution STFT term penalizes
spectral mismatch across several window sizes.

**Streaming and causality.** A usable effect must run in real time, sample by
sample, with no look-ahead. The model is strictly causal (verified by
`test_causality`) and its chunked streaming output is bit-for-bit equal to a
single parallel pass (verified by `test_streaming_matches_parallel`), so the
same weights train in parallel and deploy as a streaming processor.

**Numerical stability of the scan.** The parallel scan runs in the *linear*
domain using the associative monoid `(a,b)∘(a',b') = (a·a', a'·b + b')`, a
Hillis-Steele inclusive scan. This avoids the log-space `cumprod` trick that
overflows when decays approach 1 (precisely the long-release regime we care
about). `test_scan` checks parallel == sequential including carried state.

## 3. Tokenization: not necessary — and how the state replaces it

**Is it necessary? No.** Tokenization in the Optical-DRC model means slicing the
waveform into overlapping 64-sample windows and presenting each as a feature
vector (plus FFT magnitudes). It exists only to hand the network enough local
context to estimate level, because the model has no persistent nonlinear state
of its own. It costs framing latency, introduces redundancy (adjacent windows
overlap heavily), and breaks true sample-rate-agnostic streaming.

**How we eliminate it.** The model's input is a scalar stream `u ∈ (B, L)`. Each
sample is projected to `d_model` and processed directly; there is no windowing,
no FFT front end, no frame hop. Local context that a window used to provide is
supplied instead by (a) a short *causal* depthwise convolution (a few samples,
state-carried across chunks for exact streaming) and (b) the SSM's recurrent
state.

**How the hidden variables capture longer nonlinear behavior.** Three
cooperating stores of state, all carried across chunks and exact under
streaming:

- **AdaptiveLevelDetector** — a multi-band leaky integrator with *separate,
  learnable attack and release coefficients* and a soft gate
  `rising = σ(sharpness·(x_t − env))` selecting between them:
  `coeff = rising·a_att + (1−rising)·a_rel`, `env = coeff·env + (1−coeff)·x_t`.
  This is a nonlinear recurrence (the coefficient depends on the state vs.
  input comparison), and its slow release branch is what holds the
  program-dependent, multi-second memory. This is the nonlinear `f` from §1.
- **Selective SSM** — the detector envelope is fed in as the selectivity signal,
  so the discretized decay `ā = exp(Δ·A)` becomes a function of the envelope,
  i.e. of signal history. Diagonal `A` with HiPPO-style init gives the linear
  long-memory backbone; the detector makes its time constants signal-dependent.
- **Nonlinear gain readout** — a small MLP maps the SSM state + detector
  envelope + device params to a gain in dB, and the output is
  `y_t = u_t · 10^(g_db/20)`. This multiplicative form is the correct prior for
  a compressor (it acts by scaling the input) and is the nonlinear `h` from §1.

Device parameters (threshold, ratio, attack, release) condition the network via
FiLM, so one model covers the whole control surface rather than one setting.

## Architecture summary

```
u (B,L) ──► |·| ──► AdaptiveLevelDetector ──► env (B,L,n_bands)
   │                                              │  (selectivity)
   └─► input_proj ─► FiLM(params) ─► [CompSSMBlock × n_layers] ─► RMSNorm ─► GainComputer ─► g_db
                                         (conv → SiLU → SelectiveSSM(sel=env) → gate)         │
                                                                                              ▼
                                                                          y = u · 10^(g_db/20)
```

Every recurrent component exposes its state, so `model.render(u, params)` streams
arbitrarily long audio in chunks with results identical to a full parallel pass.

## References

- A. Shoukry, *Nonlinear State-Space Realization of I/O Maps* (M.A.Sc. thesis,
  2008) — the realization theorem motivating a nonlinear state.
- A. Gu & T. Dao, *Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces* (2023) — the S6 backbone.
- R. Simionato & S. Fasciani, *Optical Dynamic Range Compression with Selective
  State-Space Models* — the prior approach this work improves on.

# Normalisation reference — 05_conditioning

Single source of truth for how conditioning parameters and GR targets are
normalised in this experiment. **Eval code must import these constants from
`splits.py` / `gr_target.py`** — do not use `src.dsp_torch.normalize_params`
(different ranges, see below).

## Conditioning parameters

Order (`src.dsp.PARAM_ORDER`): `[threshold, attack, release, ratio]`.
Physical values are parsed from the Diff-SSL setting folder names and mapped
linearly to `[0, 1]`:

```
x01 = (x - lo) / (hi - lo)
```

Ranges (`splits.DIFFSSL_PARAM_RANGES`):

| param     | lo    | hi   | unit |
|-----------|-------|------|------|
| threshold | -20.0 | 20.0 | dB   |
| attack    | 0.0   | 30.0 | ms   |
| release   | 0.0   | 1.6  | s    |
| ratio     | 0.0   | 10.0 | —    |

### Why not the nablafx-fork ranges

`external/nablafx-for-diffssl-compressor` (`SSLGCompDataset.PARAM_RANGES`) and
`src.dsp.PARAM_RANGES_LOCAL` use `threshold (-20, 0)`, `attack (0.1, 30)`,
`release (0.1, 1.6)`, `ratio (2, 10)`. The actual dataset contains thresholds
up to **+12 dB** — the fork's range would clip the settings with threshold
0/4/8/12 all to 1.0, collapsing four distinct settings onto identical
conditioning values. The ranges above cover the full dataset without clipping.

### Normalised vectors for the 10 dataset settings

| setting | thr | atk | rel | ratio |
|---|---|---|---|---|
| `threshold_-12_attack_10_release_0.4_ratio_10` | 0.200 | 0.333 | 0.250 | 1.000 |
| `threshold_-12_attack_1_release_0.1_ratio_2` | 0.200 | 0.033 | 0.062 | 0.200 |
| `threshold_-4_attack_10_release_0.1_ratio_2` | 0.400 | 0.333 | 0.062 | 0.200 |
| `threshold_-4_attack_1_release_0.4_ratio_10` | 0.400 | 0.033 | 0.250 | 1.000 |
| `threshold_-8_attack_30_release_0.8_ratio_4` | 0.300 | 1.000 | 0.500 | 0.400 |
| `threshold_0_attack_3_release_0.8_ratio_4` | 0.500 | 0.100 | 0.500 | 0.400 |
| `threshold_12_attack_3_release_0.8_ratio_2` | 0.800 | 0.100 | 0.500 | 0.200 |
| `threshold_4_attack_10_release_0.1_ratio_10` | 0.600 | 0.333 | 0.062 | 1.000 |
| `threshold_8_attack_1_release_0.1_ratio_10` | 0.700 | 0.033 | 0.062 | 1.000 |
| `threshold_8_attack_30_release_0.4_ratio_2` | 0.700 | 1.000 | 0.250 | 0.200 |

## GR target

GR curves (`gr_curves/<setting>/<song>.pt`, key `gr_db`) are sample-rate
`wet_dB − dry_dB` from windowed RMS (`RMS_WINDOW = 1024`, see
`03_initial_GR_pred/gr_dataset.py`).

Measured over all 100 exported curves (10 songs × 10 settings, 2026-06-10):

- **min = −28.252 dB, max = +4.213 dB**, no NaN/Inf
- every setting spans nearly the same extremes (≈ −28.2 / +4.1 dB), i.e. the
  extremes are transient RMS-ratio spikes (attack overshoot, fades), not
  steady-state compression

Limits (`gr_target.py`), chosen to cover the data with margin so the sigmoid
head never saturates at 0/1:

```
GR_DB_MIN = -30.0
GR_DB_MAX = +5.0

gr01 = (gr_db - GR_DB_MIN) / (GR_DB_MAX - GR_DB_MIN)   # 0 dB GR -> 0.857
```

The frame-rate target is `adaptive_avg_pool1d(gr01, T_frames)` (hop = 256
⇒ ≈ 172 Hz). Loss is MSE in `[0, 1]`; the logged `mae_db` is the `[0, 1]` L1
scaled by `GR_DB_MAX − GR_DB_MIN` = 35 dB.

Note: `03_initial_GR_pred` used `[−30, 0]` (`src.dsp_torch.GR_DB_MIN/MAX`)
with a discretized 61-bin head — dB metrics are comparable, the target
normalisation is not.

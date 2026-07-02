# Diff-SSL-G-Comp — Compressor Settings Inventory

Full list of the **220** recorded parameter combinations in the [`amphion/SolidStateBusComp`](https://huggingface.co/datasets/amphion/SolidStateBusComp) dataset (SSL G-series bus compressor), and which are currently downloaded locally under `data/Diff-SSL-G-Comp/processed_ground_truth/`.

- **Total settings:** 220
- **Downloaded:** 10
- **Missing:** 210
- ✅ = present locally &nbsp;&nbsp; ⬜ = not downloaded

Parameters: **threshold** (dB), **attack** (ms), **release** (s, `auto` = program-dependent), **ratio** (:1).

## Downloaded settings (10)

| Threshold | Attack | Release | Ratio | Folder |
|---|---|---|---|---|
| -12 | 10 | 0.4 | 10 | `threshold_-12_attack_10_release_0.4_ratio_10` |
| -12 | 1 | 0.1 | 2 | `threshold_-12_attack_1_release_0.1_ratio_2` |
| -4 | 10 | 0.1 | 2 | `threshold_-4_attack_10_release_0.1_ratio_2` |
| -4 | 1 | 0.4 | 10 | `threshold_-4_attack_1_release_0.4_ratio_10` |
| -8 | 30 | 0.8 | 4 | `threshold_-8_attack_30_release_0.8_ratio_4` |
| 0 | 3 | 0.8 | 4 | `threshold_0_attack_3_release_0.8_ratio_4` |
| 12 | 3 | 0.8 | 2 | `threshold_12_attack_3_release_0.8_ratio_2` |
| 4 | 10 | 0.1 | 10 | `threshold_4_attack_10_release_0.1_ratio_10` |
| 8 | 1 | 0.1 | 10 | `threshold_8_attack_1_release_0.1_ratio_10` |
| 8 | 30 | 0.4 | 2 | `threshold_8_attack_30_release_0.4_ratio_2` |

## All 220 settings (grouped by threshold)

### Threshold -12 dB — 12 settings (2 downloaded)

| ✓ | Attack (ms) | Release (s) | Ratio | Folder |
|---|---|---|---|---|
| ✅ | 1 | 0.1 | 2 | `threshold_-12_attack_1_release_0.1_ratio_2` |
| ⬜ | 1 | 0.8 | 2 | `threshold_-12_attack_1_release_0.8_ratio_2` |
| ⬜ | 3 | 0.4 | 10 | `threshold_-12_attack_3_release_0.4_ratio_10` |
| ⬜ | 3 | auto | 4 | `threshold_-12_attack_3_release_auto_ratio_4` |
| ⬜ | 10 | 0.1 | 2 | `threshold_-12_attack_10_release_0.1_ratio_2` |
| ⬜ | 10 | 0.4 | 4 | `threshold_-12_attack_10_release_0.4_ratio_4` |
| ✅ | 10 | 0.4 | 10 | `threshold_-12_attack_10_release_0.4_ratio_10` |
| ⬜ | 10 | auto | 10 | `threshold_-12_attack_10_release_auto_ratio_10` |
| ⬜ | 30 | 0.4 | 2 | `threshold_-12_attack_30_release_0.4_ratio_2` |
| ⬜ | 30 | 0.4 | 4 | `threshold_-12_attack_30_release_0.4_ratio_4` |
| ⬜ | 30 | 0.4 | 10 | `threshold_-12_attack_30_release_0.4_ratio_10` |
| ⬜ | 30 | auto | 10 | `threshold_-12_attack_30_release_auto_ratio_10` |

### Threshold -8 dB — 15 settings (1 downloaded)

| ✓ | Attack (ms) | Release (s) | Ratio | Folder |
|---|---|---|---|---|
| ⬜ | 1 | 0.1 | 2 | `threshold_-8_attack_1_release_0.1_ratio_2` |
| ⬜ | 1 | 0.4 | 4 | `threshold_-8_attack_1_release_0.4_ratio_4` |
| ⬜ | 1 | auto | 10 | `threshold_-8_attack_1_release_auto_ratio_10` |
| ⬜ | 3 | 0.1 | 4 | `threshold_-8_attack_3_release_0.1_ratio_4` |
| ⬜ | 3 | 0.4 | 4 | `threshold_-8_attack_3_release_0.4_ratio_4` |
| ⬜ | 3 | 0.4 | 10 | `threshold_-8_attack_3_release_0.4_ratio_10` |
| ⬜ | 3 | auto | 10 | `threshold_-8_attack_3_release_auto_ratio_10` |
| ⬜ | 10 | 0.1 | 4 | `threshold_-8_attack_10_release_0.1_ratio_4` |
| ⬜ | 10 | 0.8 | 2 | `threshold_-8_attack_10_release_0.8_ratio_2` |
| ⬜ | 10 | 0.8 | 10 | `threshold_-8_attack_10_release_0.8_ratio_10` |
| ⬜ | 10 | auto | 2 | `threshold_-8_attack_10_release_auto_ratio_2` |
| ⬜ | 30 | 0.1 | 4 | `threshold_-8_attack_30_release_0.1_ratio_4` |
| ⬜ | 30 | 0.4 | 2 | `threshold_-8_attack_30_release_0.4_ratio_2` |
| ✅ | 30 | 0.8 | 4 | `threshold_-8_attack_30_release_0.8_ratio_4` |
| ⬜ | 30 | auto | 10 | `threshold_-8_attack_30_release_auto_ratio_10` |

### Threshold -4 dB — 48 settings (2 downloaded)

| ✓ | Attack (ms) | Release (s) | Ratio | Folder |
|---|---|---|---|---|
| ⬜ | 1 | 0.1 | 2 | `threshold_-4_attack_1_release_0.1_ratio_2` |
| ⬜ | 1 | 0.1 | 4 | `threshold_-4_attack_1_release_0.1_ratio_4` |
| ⬜ | 1 | 0.1 | 10 | `threshold_-4_attack_1_release_0.1_ratio_10` |
| ⬜ | 1 | 0.4 | 2 | `threshold_-4_attack_1_release_0.4_ratio_2` |
| ⬜ | 1 | 0.4 | 4 | `threshold_-4_attack_1_release_0.4_ratio_4` |
| ✅ | 1 | 0.4 | 10 | `threshold_-4_attack_1_release_0.4_ratio_10` |
| ⬜ | 1 | 0.8 | 2 | `threshold_-4_attack_1_release_0.8_ratio_2` |
| ⬜ | 1 | 0.8 | 4 | `threshold_-4_attack_1_release_0.8_ratio_4` |
| ⬜ | 1 | 0.8 | 10 | `threshold_-4_attack_1_release_0.8_ratio_10` |
| ⬜ | 1 | auto | 2 | `threshold_-4_attack_1_release_auto_ratio_2` |
| ⬜ | 1 | auto | 4 | `threshold_-4_attack_1_release_auto_ratio_4` |
| ⬜ | 1 | auto | 10 | `threshold_-4_attack_1_release_auto_ratio_10` |
| ⬜ | 3 | 0.1 | 2 | `threshold_-4_attack_3_release_0.1_ratio_2` |
| ⬜ | 3 | 0.1 | 4 | `threshold_-4_attack_3_release_0.1_ratio_4` |
| ⬜ | 3 | 0.1 | 10 | `threshold_-4_attack_3_release_0.1_ratio_10` |
| ⬜ | 3 | 0.4 | 2 | `threshold_-4_attack_3_release_0.4_ratio_2` |
| ⬜ | 3 | 0.4 | 4 | `threshold_-4_attack_3_release_0.4_ratio_4` |
| ⬜ | 3 | 0.4 | 10 | `threshold_-4_attack_3_release_0.4_ratio_10` |
| ⬜ | 3 | 0.8 | 2 | `threshold_-4_attack_3_release_0.8_ratio_2` |
| ⬜ | 3 | 0.8 | 4 | `threshold_-4_attack_3_release_0.8_ratio_4` |
| ⬜ | 3 | 0.8 | 10 | `threshold_-4_attack_3_release_0.8_ratio_10` |
| ⬜ | 3 | auto | 2 | `threshold_-4_attack_3_release_auto_ratio_2` |
| ⬜ | 3 | auto | 4 | `threshold_-4_attack_3_release_auto_ratio_4` |
| ⬜ | 3 | auto | 10 | `threshold_-4_attack_3_release_auto_ratio_10` |
| ✅ | 10 | 0.1 | 2 | `threshold_-4_attack_10_release_0.1_ratio_2` |
| ⬜ | 10 | 0.1 | 4 | `threshold_-4_attack_10_release_0.1_ratio_4` |
| ⬜ | 10 | 0.1 | 10 | `threshold_-4_attack_10_release_0.1_ratio_10` |
| ⬜ | 10 | 0.4 | 2 | `threshold_-4_attack_10_release_0.4_ratio_2` |
| ⬜ | 10 | 0.4 | 4 | `threshold_-4_attack_10_release_0.4_ratio_4` |
| ⬜ | 10 | 0.4 | 10 | `threshold_-4_attack_10_release_0.4_ratio_10` |
| ⬜ | 10 | 0.8 | 2 | `threshold_-4_attack_10_release_0.8_ratio_2` |
| ⬜ | 10 | 0.8 | 4 | `threshold_-4_attack_10_release_0.8_ratio_4` |
| ⬜ | 10 | 0.8 | 10 | `threshold_-4_attack_10_release_0.8_ratio_10` |
| ⬜ | 10 | auto | 2 | `threshold_-4_attack_10_release_auto_ratio_2` |
| ⬜ | 10 | auto | 4 | `threshold_-4_attack_10_release_auto_ratio_4` |
| ⬜ | 10 | auto | 10 | `threshold_-4_attack_10_release_auto_ratio_10` |
| ⬜ | 30 | 0.1 | 2 | `threshold_-4_attack_30_release_0.1_ratio_2` |
| ⬜ | 30 | 0.1 | 4 | `threshold_-4_attack_30_release_0.1_ratio_4` |
| ⬜ | 30 | 0.1 | 10 | `threshold_-4_attack_30_release_0.1_ratio_10` |
| ⬜ | 30 | 0.4 | 2 | `threshold_-4_attack_30_release_0.4_ratio_2` |
| ⬜ | 30 | 0.4 | 4 | `threshold_-4_attack_30_release_0.4_ratio_4` |
| ⬜ | 30 | 0.4 | 10 | `threshold_-4_attack_30_release_0.4_ratio_10` |
| ⬜ | 30 | 0.8 | 2 | `threshold_-4_attack_30_release_0.8_ratio_2` |
| ⬜ | 30 | 0.8 | 4 | `threshold_-4_attack_30_release_0.8_ratio_4` |
| ⬜ | 30 | 0.8 | 10 | `threshold_-4_attack_30_release_0.8_ratio_10` |
| ⬜ | 30 | auto | 2 | `threshold_-4_attack_30_release_auto_ratio_2` |
| ⬜ | 30 | auto | 4 | `threshold_-4_attack_30_release_auto_ratio_4` |
| ⬜ | 30 | auto | 10 | `threshold_-4_attack_30_release_auto_ratio_10` |

### Threshold 0 dB — 48 settings (1 downloaded)

| ✓ | Attack (ms) | Release (s) | Ratio | Folder |
|---|---|---|---|---|
| ⬜ | 1 | 0.1 | 2 | `threshold_0_attack_1_release_0.1_ratio_2` |
| ⬜ | 1 | 0.1 | 4 | `threshold_0_attack_1_release_0.1_ratio_4` |
| ⬜ | 1 | 0.1 | 10 | `threshold_0_attack_1_release_0.1_ratio_10` |
| ⬜ | 1 | 0.4 | 2 | `threshold_0_attack_1_release_0.4_ratio_2` |
| ⬜ | 1 | 0.4 | 4 | `threshold_0_attack_1_release_0.4_ratio_4` |
| ⬜ | 1 | 0.4 | 10 | `threshold_0_attack_1_release_0.4_ratio_10` |
| ⬜ | 1 | 0.8 | 2 | `threshold_0_attack_1_release_0.8_ratio_2` |
| ⬜ | 1 | 0.8 | 4 | `threshold_0_attack_1_release_0.8_ratio_4` |
| ⬜ | 1 | 0.8 | 10 | `threshold_0_attack_1_release_0.8_ratio_10` |
| ⬜ | 1 | auto | 2 | `threshold_0_attack_1_release_auto_ratio_2` |
| ⬜ | 1 | auto | 4 | `threshold_0_attack_1_release_auto_ratio_4` |
| ⬜ | 1 | auto | 10 | `threshold_0_attack_1_release_auto_ratio_10` |
| ⬜ | 3 | 0.1 | 2 | `threshold_0_attack_3_release_0.1_ratio_2` |
| ⬜ | 3 | 0.1 | 4 | `threshold_0_attack_3_release_0.1_ratio_4` |
| ⬜ | 3 | 0.1 | 10 | `threshold_0_attack_3_release_0.1_ratio_10` |
| ⬜ | 3 | 0.4 | 2 | `threshold_0_attack_3_release_0.4_ratio_2` |
| ⬜ | 3 | 0.4 | 4 | `threshold_0_attack_3_release_0.4_ratio_4` |
| ⬜ | 3 | 0.4 | 10 | `threshold_0_attack_3_release_0.4_ratio_10` |
| ⬜ | 3 | 0.8 | 2 | `threshold_0_attack_3_release_0.8_ratio_2` |
| ✅ | 3 | 0.8 | 4 | `threshold_0_attack_3_release_0.8_ratio_4` |
| ⬜ | 3 | 0.8 | 10 | `threshold_0_attack_3_release_0.8_ratio_10` |
| ⬜ | 3 | auto | 2 | `threshold_0_attack_3_release_auto_ratio_2` |
| ⬜ | 3 | auto | 4 | `threshold_0_attack_3_release_auto_ratio_4` |
| ⬜ | 3 | auto | 10 | `threshold_0_attack_3_release_auto_ratio_10` |
| ⬜ | 10 | 0.1 | 2 | `threshold_0_attack_10_release_0.1_ratio_2` |
| ⬜ | 10 | 0.1 | 4 | `threshold_0_attack_10_release_0.1_ratio_4` |
| ⬜ | 10 | 0.1 | 10 | `threshold_0_attack_10_release_0.1_ratio_10` |
| ⬜ | 10 | 0.4 | 2 | `threshold_0_attack_10_release_0.4_ratio_2` |
| ⬜ | 10 | 0.4 | 4 | `threshold_0_attack_10_release_0.4_ratio_4` |
| ⬜ | 10 | 0.4 | 10 | `threshold_0_attack_10_release_0.4_ratio_10` |
| ⬜ | 10 | 0.8 | 2 | `threshold_0_attack_10_release_0.8_ratio_2` |
| ⬜ | 10 | 0.8 | 4 | `threshold_0_attack_10_release_0.8_ratio_4` |
| ⬜ | 10 | 0.8 | 10 | `threshold_0_attack_10_release_0.8_ratio_10` |
| ⬜ | 10 | auto | 2 | `threshold_0_attack_10_release_auto_ratio_2` |
| ⬜ | 10 | auto | 4 | `threshold_0_attack_10_release_auto_ratio_4` |
| ⬜ | 10 | auto | 10 | `threshold_0_attack_10_release_auto_ratio_10` |
| ⬜ | 30 | 0.1 | 2 | `threshold_0_attack_30_release_0.1_ratio_2` |
| ⬜ | 30 | 0.1 | 4 | `threshold_0_attack_30_release_0.1_ratio_4` |
| ⬜ | 30 | 0.1 | 10 | `threshold_0_attack_30_release_0.1_ratio_10` |
| ⬜ | 30 | 0.4 | 2 | `threshold_0_attack_30_release_0.4_ratio_2` |
| ⬜ | 30 | 0.4 | 4 | `threshold_0_attack_30_release_0.4_ratio_4` |
| ⬜ | 30 | 0.4 | 10 | `threshold_0_attack_30_release_0.4_ratio_10` |
| ⬜ | 30 | 0.8 | 2 | `threshold_0_attack_30_release_0.8_ratio_2` |
| ⬜ | 30 | 0.8 | 4 | `threshold_0_attack_30_release_0.8_ratio_4` |
| ⬜ | 30 | 0.8 | 10 | `threshold_0_attack_30_release_0.8_ratio_10` |
| ⬜ | 30 | auto | 2 | `threshold_0_attack_30_release_auto_ratio_2` |
| ⬜ | 30 | auto | 4 | `threshold_0_attack_30_release_auto_ratio_4` |
| ⬜ | 30 | auto | 10 | `threshold_0_attack_30_release_auto_ratio_10` |

### Threshold 4 dB — 48 settings (1 downloaded)

| ✓ | Attack (ms) | Release (s) | Ratio | Folder |
|---|---|---|---|---|
| ⬜ | 1 | 0.1 | 2 | `threshold_4_attack_1_release_0.1_ratio_2` |
| ⬜ | 1 | 0.1 | 4 | `threshold_4_attack_1_release_0.1_ratio_4` |
| ⬜ | 1 | 0.1 | 10 | `threshold_4_attack_1_release_0.1_ratio_10` |
| ⬜ | 1 | 0.4 | 2 | `threshold_4_attack_1_release_0.4_ratio_2` |
| ⬜ | 1 | 0.4 | 4 | `threshold_4_attack_1_release_0.4_ratio_4` |
| ⬜ | 1 | 0.4 | 10 | `threshold_4_attack_1_release_0.4_ratio_10` |
| ⬜ | 1 | 0.8 | 2 | `threshold_4_attack_1_release_0.8_ratio_2` |
| ⬜ | 1 | 0.8 | 4 | `threshold_4_attack_1_release_0.8_ratio_4` |
| ⬜ | 1 | 0.8 | 10 | `threshold_4_attack_1_release_0.8_ratio_10` |
| ⬜ | 1 | auto | 2 | `threshold_4_attack_1_release_auto_ratio_2` |
| ⬜ | 1 | auto | 4 | `threshold_4_attack_1_release_auto_ratio_4` |
| ⬜ | 1 | auto | 10 | `threshold_4_attack_1_release_auto_ratio_10` |
| ⬜ | 3 | 0.1 | 2 | `threshold_4_attack_3_release_0.1_ratio_2` |
| ⬜ | 3 | 0.1 | 4 | `threshold_4_attack_3_release_0.1_ratio_4` |
| ⬜ | 3 | 0.1 | 10 | `threshold_4_attack_3_release_0.1_ratio_10` |
| ⬜ | 3 | 0.4 | 2 | `threshold_4_attack_3_release_0.4_ratio_2` |
| ⬜ | 3 | 0.4 | 4 | `threshold_4_attack_3_release_0.4_ratio_4` |
| ⬜ | 3 | 0.4 | 10 | `threshold_4_attack_3_release_0.4_ratio_10` |
| ⬜ | 3 | 0.8 | 2 | `threshold_4_attack_3_release_0.8_ratio_2` |
| ⬜ | 3 | 0.8 | 4 | `threshold_4_attack_3_release_0.8_ratio_4` |
| ⬜ | 3 | 0.8 | 10 | `threshold_4_attack_3_release_0.8_ratio_10` |
| ⬜ | 3 | auto | 2 | `threshold_4_attack_3_release_auto_ratio_2` |
| ⬜ | 3 | auto | 4 | `threshold_4_attack_3_release_auto_ratio_4` |
| ⬜ | 3 | auto | 10 | `threshold_4_attack_3_release_auto_ratio_10` |
| ⬜ | 10 | 0.1 | 2 | `threshold_4_attack_10_release_0.1_ratio_2` |
| ⬜ | 10 | 0.1 | 4 | `threshold_4_attack_10_release_0.1_ratio_4` |
| ✅ | 10 | 0.1 | 10 | `threshold_4_attack_10_release_0.1_ratio_10` |
| ⬜ | 10 | 0.4 | 2 | `threshold_4_attack_10_release_0.4_ratio_2` |
| ⬜ | 10 | 0.4 | 4 | `threshold_4_attack_10_release_0.4_ratio_4` |
| ⬜ | 10 | 0.4 | 10 | `threshold_4_attack_10_release_0.4_ratio_10` |
| ⬜ | 10 | 0.8 | 2 | `threshold_4_attack_10_release_0.8_ratio_2` |
| ⬜ | 10 | 0.8 | 4 | `threshold_4_attack_10_release_0.8_ratio_4` |
| ⬜ | 10 | 0.8 | 10 | `threshold_4_attack_10_release_0.8_ratio_10` |
| ⬜ | 10 | auto | 2 | `threshold_4_attack_10_release_auto_ratio_2` |
| ⬜ | 10 | auto | 4 | `threshold_4_attack_10_release_auto_ratio_4` |
| ⬜ | 10 | auto | 10 | `threshold_4_attack_10_release_auto_ratio_10` |
| ⬜ | 30 | 0.1 | 2 | `threshold_4_attack_30_release_0.1_ratio_2` |
| ⬜ | 30 | 0.1 | 4 | `threshold_4_attack_30_release_0.1_ratio_4` |
| ⬜ | 30 | 0.1 | 10 | `threshold_4_attack_30_release_0.1_ratio_10` |
| ⬜ | 30 | 0.4 | 2 | `threshold_4_attack_30_release_0.4_ratio_2` |
| ⬜ | 30 | 0.4 | 4 | `threshold_4_attack_30_release_0.4_ratio_4` |
| ⬜ | 30 | 0.4 | 10 | `threshold_4_attack_30_release_0.4_ratio_10` |
| ⬜ | 30 | 0.8 | 2 | `threshold_4_attack_30_release_0.8_ratio_2` |
| ⬜ | 30 | 0.8 | 4 | `threshold_4_attack_30_release_0.8_ratio_4` |
| ⬜ | 30 | 0.8 | 10 | `threshold_4_attack_30_release_0.8_ratio_10` |
| ⬜ | 30 | auto | 2 | `threshold_4_attack_30_release_auto_ratio_2` |
| ⬜ | 30 | auto | 4 | `threshold_4_attack_30_release_auto_ratio_4` |
| ⬜ | 30 | auto | 10 | `threshold_4_attack_30_release_auto_ratio_10` |

### Threshold 8 dB — 33 settings (2 downloaded)

| ✓ | Attack (ms) | Release (s) | Ratio | Folder |
|---|---|---|---|---|
| ⬜ | 1 | 0.1 | 2 | `threshold_8_attack_1_release_0.1_ratio_2` |
| ⬜ | 1 | 0.1 | 4 | `threshold_8_attack_1_release_0.1_ratio_4` |
| ✅ | 1 | 0.1 | 10 | `threshold_8_attack_1_release_0.1_ratio_10` |
| ⬜ | 1 | 0.4 | 2 | `threshold_8_attack_1_release_0.4_ratio_2` |
| ⬜ | 1 | 0.4 | 4 | `threshold_8_attack_1_release_0.4_ratio_4` |
| ⬜ | 1 | 0.8 | 2 | `threshold_8_attack_1_release_0.8_ratio_2` |
| ⬜ | 1 | 0.8 | 4 | `threshold_8_attack_1_release_0.8_ratio_4` |
| ⬜ | 1 | auto | 2 | `threshold_8_attack_1_release_auto_ratio_2` |
| ⬜ | 1 | auto | 4 | `threshold_8_attack_1_release_auto_ratio_4` |
| ⬜ | 3 | 0.1 | 2 | `threshold_8_attack_3_release_0.1_ratio_2` |
| ⬜ | 3 | 0.1 | 4 | `threshold_8_attack_3_release_0.1_ratio_4` |
| ⬜ | 3 | 0.4 | 2 | `threshold_8_attack_3_release_0.4_ratio_2` |
| ⬜ | 3 | 0.4 | 4 | `threshold_8_attack_3_release_0.4_ratio_4` |
| ⬜ | 3 | 0.8 | 2 | `threshold_8_attack_3_release_0.8_ratio_2` |
| ⬜ | 3 | 0.8 | 4 | `threshold_8_attack_3_release_0.8_ratio_4` |
| ⬜ | 3 | auto | 2 | `threshold_8_attack_3_release_auto_ratio_2` |
| ⬜ | 3 | auto | 4 | `threshold_8_attack_3_release_auto_ratio_4` |
| ⬜ | 10 | 0.1 | 2 | `threshold_8_attack_10_release_0.1_ratio_2` |
| ⬜ | 10 | 0.1 | 4 | `threshold_8_attack_10_release_0.1_ratio_4` |
| ⬜ | 10 | 0.4 | 2 | `threshold_8_attack_10_release_0.4_ratio_2` |
| ⬜ | 10 | 0.4 | 4 | `threshold_8_attack_10_release_0.4_ratio_4` |
| ⬜ | 10 | 0.8 | 2 | `threshold_8_attack_10_release_0.8_ratio_2` |
| ⬜ | 10 | 0.8 | 4 | `threshold_8_attack_10_release_0.8_ratio_4` |
| ⬜ | 10 | auto | 2 | `threshold_8_attack_10_release_auto_ratio_2` |
| ⬜ | 10 | auto | 4 | `threshold_8_attack_10_release_auto_ratio_4` |
| ⬜ | 30 | 0.1 | 2 | `threshold_8_attack_30_release_0.1_ratio_2` |
| ⬜ | 30 | 0.1 | 4 | `threshold_8_attack_30_release_0.1_ratio_4` |
| ✅ | 30 | 0.4 | 2 | `threshold_8_attack_30_release_0.4_ratio_2` |
| ⬜ | 30 | 0.4 | 4 | `threshold_8_attack_30_release_0.4_ratio_4` |
| ⬜ | 30 | 0.8 | 2 | `threshold_8_attack_30_release_0.8_ratio_2` |
| ⬜ | 30 | 0.8 | 4 | `threshold_8_attack_30_release_0.8_ratio_4` |
| ⬜ | 30 | auto | 2 | `threshold_8_attack_30_release_auto_ratio_2` |
| ⬜ | 30 | auto | 4 | `threshold_8_attack_30_release_auto_ratio_4` |

### Threshold 12 dB — 16 settings (1 downloaded)

| ✓ | Attack (ms) | Release (s) | Ratio | Folder |
|---|---|---|---|---|
| ⬜ | 1 | 0.1 | 2 | `threshold_12_attack_1_release_0.1_ratio_2` |
| ⬜ | 1 | 0.4 | 2 | `threshold_12_attack_1_release_0.4_ratio_2` |
| ⬜ | 1 | 0.8 | 2 | `threshold_12_attack_1_release_0.8_ratio_2` |
| ⬜ | 1 | auto | 2 | `threshold_12_attack_1_release_auto_ratio_2` |
| ⬜ | 3 | 0.1 | 2 | `threshold_12_attack_3_release_0.1_ratio_2` |
| ⬜ | 3 | 0.4 | 2 | `threshold_12_attack_3_release_0.4_ratio_2` |
| ✅ | 3 | 0.8 | 2 | `threshold_12_attack_3_release_0.8_ratio_2` |
| ⬜ | 3 | auto | 2 | `threshold_12_attack_3_release_auto_ratio_2` |
| ⬜ | 10 | 0.1 | 2 | `threshold_12_attack_10_release_0.1_ratio_2` |
| ⬜ | 10 | 0.4 | 2 | `threshold_12_attack_10_release_0.4_ratio_2` |
| ⬜ | 10 | 0.8 | 2 | `threshold_12_attack_10_release_0.8_ratio_2` |
| ⬜ | 10 | auto | 2 | `threshold_12_attack_10_release_auto_ratio_2` |
| ⬜ | 30 | 0.1 | 2 | `threshold_12_attack_30_release_0.1_ratio_2` |
| ⬜ | 30 | 0.4 | 2 | `threshold_12_attack_30_release_0.4_ratio_2` |
| ⬜ | 30 | 0.8 | 2 | `threshold_12_attack_30_release_0.8_ratio_2` |
| ⬜ | 30 | auto | 2 | `threshold_12_attack_30_release_auto_ratio_2` |

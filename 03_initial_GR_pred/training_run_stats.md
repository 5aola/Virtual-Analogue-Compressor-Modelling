# Training Run Stats

Generated from the four training notebooks plus saved run folders under `/Volumes/Saola's Drive/AllCode/thesis/data/gr_pred_runs`.

Sources:

- `train_tcn_simple_nocond.ipynb`
- `train_gcn_nocond.ipynb`
- `train_lstm_nocond.ipynb`
- `train_lstm_nocond_discretized.ipynb`
- each run's `hparams.json` and `csv/metrics.csv`
- `eval_audio_metrics_validation.csv` when present

Notes:

- TCN and GCN training cells ended with `KeyboardInterrupt`, so their notebooks did not print wall-clock training time.
- The GCN notebook resumed `gcn_gr_20260427_104716_gcn_split_fix` from `last.ckpt`; Lightning warned that the existing CSV log directory would be deleted. The saved `csv/metrics.csv` therefore only covers the resumed segment, epochs 61-63.
- Checkpoint inventories in evaluation notebooks sometimes list a different `eval_ckpt` filename than the training notebook's final `best_cb.best_model_path`. Loss and epoch statistics below use `csv/metrics.csv`; checkpoint paths use the training notebook output when it exists.

## Shared Dataset

- Dataset: `Diff-SSL-G-Comp`
- Colab dataset root: `/content/drive/Othercomputers/MacBook Air/data/Diff-SSL-G-Comp`
- Local synced run root: `/Volumes/Saola's Drive/AllCode/thesis/data/gr_pred_runs`
- Setting: `threshold_-4_attack_1_release_0.4_ratio_10`
- Raw file counts printed by notebooks: 175 dry WAV files, 10 wet WAV files for the selected setting
- Training target: gain reduction from dry/wet RMS ratio
- Sample rate: 44100 Hz
- RMS window: 1024 samples
- GR target range: -30.0 dB to 0.0 dB
- Precomputed GR curves: `.pt` files from `gr_dataset.py`; cached to Colab local SSD at `/content/Diff-SSL-G-Comp`
- Train/validation split seed where song split is used: 42

TCN dataset split:

- Chunking: 3.0 s chunks, `sample_length=132300`, `sample_stride=132300`
- Dataset output: 10 songs, 877 chunks
- Split: 701 training chunks, 176 validation chunks

GCN and LSTM dataset split:

- Chunking: 10.0 s crops, `sample_length=441000`, `sample_stride=110250` (2.5 s)
- Dataset output: 10 songs, 1033 anchored crops
- Train: 849 crops from 8 songs: `Air`, `AncoraQui`, `BackroomInTulsa`, `Ecstasy`, `Electrvm`, `LivingLie`, `OpenFire`, `SongForJohn`
- Validation: 184 crops from 2 songs: `Borderline`, `NosPalpitants`
- Discretized LSTM cache log: 445 MB cached GR curves

## Run Overview

### `tcn_gr_20260427_094122_with_recommendations_h100`

Notebook: `train_tcn_simple_nocond.ipynb`

Model and objective:

- Model type: causal NablAFx TCN
- Conditioning: none
- Output target: normalized GR curve from `src.dsp_torch.compute_gr_target_norm`
- Loss: `l1+diff_l1`
- Difference loss weight: 0.1

Hyperparameters:

- Batch size: 16
- Learning rate: 0.001
- Max epochs: 1000
- Early stopping patience: 100 epochs
- ReduceLROnPlateau patience: 20 epochs
- Trainer precision: `16-mixed`
- Gradient clip value: 1.0
- Logged LR values: 0.001, 0.0005, 0.00025, 0.000125, 0.0000625, 0.00003125, 0.000015625, 0.00001

Architecture:

- TCN blocks: 10
- Stack size: 10
- Kernel size: 5
- Dilation growth: 3
- Channel width: 32
- Causal: true
- Batch norm: false
- Bias: true
- Activation: PReLU
- Receptive field: 118097 samples, 2.6779 s
- Parameters: 46913

Dataset:

- Sample length: 132300 samples, 3.0 s
- Sample stride: 132300 samples, non-overlapping
- Split: 701 train chunks, 176 validation chunks

Training output and run length:

- CSV epochs covered: 0-427
- Epoch count from CSV: 428
- Max logged step: 18404
- Best validation loss from CSV: 0.0248658 at epoch 344, step 14834
- Last validation loss from CSV: 0.0248783 at epoch 427, step 18403
- Best training loss from CSV: 0.0202938 at epoch 416, step 17930
- Last training loss from CSV: 0.0205291 at epoch 427, step 18403
- Evaluation notebook checkpoint inventory: 89 checkpoints, `best-426-18361.ckpt`, displayed `best_val=0.024866`
- Notebook status: interrupted by `KeyboardInterrupt`; no wall-clock time printed

Saved outputs observed locally:

- `hparams.json`
- `csv/metrics.csv`
- `csv/hparams.yaml`
- `tb/hparams.yaml`

### `gcn_gr_20260427_104716_gcn_split_fix`

Notebook: `train_gcn_nocond.ipynb`

Model and objective:

- Model type: causal NablAFx GCN
- Conditioning: none
- Output target: normalized GR curve from `src.dsp_torch.compute_gr_target_norm`
- Loss: `warmup_masked_l1+diff_l1+second_diff_l1`
- Difference loss weight: 1.0
- Second-difference smoothness weight: 0.2
- Warmup mask: 118097 samples

Hyperparameters:

- Batch size: 8
- Learning rate: 0.001
- Max epochs: 1000
- Early stopping patience: 100 epochs
- ReduceLROnPlateau patience: 20 epochs
- Trainer precision: `16-mixed`
- Gradient clip value: 1.0
- Logged LR values in saved resumed CSV: 0.0005

Architecture:

- GCN blocks: 10
- Stack size: 10
- Kernel size: 5
- Dilation growth: 3
- Channel width: 32
- Causal: true
- Batch norm: false
- Bias: true
- Residual: false
- Direct path: false
- Receptive field: 118097 samples, 2.6779 s
- Parameters: 104001

Dataset:

- Sample length: 441000 samples, 10.0 s
- Sample stride: 110250 samples, 2.5 s
- Split unit: song
- Split: 849 train crops from 8 songs, 184 validation crops from 2 songs

Training output and run length:

- Run mode in notebook: resumed from `/content/drive/Othercomputers/MacBook Air/data/gr_pred_runs/gcn_gr_20260427_104716_gcn_split_fix/checkpoints/last.ckpt`
- Saved CSV epochs covered: 61-63
- Epoch count from saved CSV: 3 resumed epochs
- Max logged step in saved CSV: 6784
- Best validation loss from saved CSV: 0.0414249 at epoch 62, step 6677
- Last validation loss from saved CSV: 0.0470726 at epoch 63, step 6783
- Best training loss from saved CSV: 0.0233989 at epoch 62, step 6677
- Last training loss from saved CSV: 0.0235774 at epoch 63, step 6783
- Evaluation notebook checkpoint inventory: 39 checkpoints, `best-060-6466.ckpt`, displayed `epoch=63`, `best_val=0.041425`
- Notebook status: interrupted by `KeyboardInterrupt`; no wall-clock time printed

Validation audio metrics:

- Split: validation
- Songs: 2
- Chunks: 48
- Duration: 471.9694 s
- MAE (L1): 0.0016800
- MSE (L2): 0.000006455
- ESR: 0.0301545
- MR-STFT: 0.0819903
- EDC: 0.1883580
- M_NRMSE: 0.0998244
- M_SF: 0.1348891

Saved outputs observed locally:

- `hparams.json`
- `csv/metrics.csv`
- `csv/hparams.yaml`
- `tb/hparams.yaml`
- `eval_audio_metrics_validation.csv`

### `lstm_gr_20260507_190047_lstm_v2`

Notebook: `train_lstm_nocond.ipynb`

Model and objective:

- Model type: frame-rate LSTM
- Conditioning: none
- Output target: GR directly in dB
- Loss: `warmup_masked_l1_db+diff_l1_db`
- Difference loss weight: 0.3
- Warmup mask: 1024 samples

Hyperparameters:

- Batch size: 16
- Learning rate: 0.001
- Max epochs: 1000
- Early stopping patience: 50 epochs
- ReduceLROnPlateau patience: 20 epochs
- Trainer precision: `16-mixed`
- Gradient clip value: 1.0
- Logged LR values: 0.001, 0.0005, 0.00025

Architecture:

- Hop size: 256 samples
- Frame rate: 172.265625 Hz
- Sequence length: 1722 frames per 10 s crop
- Encoder channels: 32
- Hidden size: 64
- LSTM layers: 2
- Dropout: 0.0
- Parameters: 80065

Dataset:

- Sample length: 441000 samples, 10.0 s
- Sample stride: 110250 samples, 2.5 s
- Split unit: song
- Split: 849 train crops from 8 songs, 184 validation crops from 2 songs

Training output and run length:

- CSV epochs covered: 0-75
- Epoch count from CSV: 76
- Max logged step: 4027
- Best validation loss from CSV: 0.5930500 at epoch 25, step 1377
- Last validation loss from CSV: 0.7052317 at epoch 75, step 4027
- Best training loss from CSV: 0.3800789 at epoch 69, step 3709
- Last training loss from CSV: 0.3814560 at epoch 75, step 4027
- Early stopping: stopped after 50 records without validation improvement
- Total training time printed by notebook: 10.2 min
- Training notebook best checkpoint: `/content/drive/Othercomputers/MacBook Air/data/gr_pred_runs/lstm_gr_20260507_190047_lstm_v2/checkpoints/best-025-1378.ckpt`
- Training notebook last checkpoint: `/content/drive/Othercomputers/MacBook Air/data/gr_pred_runs/lstm_gr_20260507_190047_lstm_v2/checkpoints/last.ckpt`
- Evaluation notebook checkpoint inventory: 19 checkpoints, displayed `epoch=75`, `best_val=0.593050`

Validation audio metrics:

- Split: validation
- Songs: 2
- Chunks: 48
- Duration: 471.9694 s
- MAE (L1): 0.0016921
- MSE (L2): 0.000006621
- ESR: 0.0310604
- MR-STFT: 0.0828489
- EDC: 0.2072578
- M_NRMSE: 0.1037046
- M_SF: 0.1343402

Saved outputs observed locally:

- `hparams.json`
- `csv/metrics.csv`
- `csv/hparams.yaml`
- `tb/hparams.yaml`
- `eval_audio_metrics_validation.csv`

### `lstm_gr_20260520_121432_lstm_v3_discretized`

Notebook: `train_lstm_nocond_discretized.ipynb`

Model and objective:

- Model type: frame-rate LSTM with discretized GR output
- Conditioning: none
- Output target: logits over GR bins at frame rate
- Discretization: 151 bins, 0.2 dB resolution, range -30.0 dB to 0.0 dB
- Loss: `bce_gaussian_soft_targets`
- Gaussian target width: `sigma_bins=2.0` (0.4 dB)
- Warmup: 4 frames
- Inference smoothing: local weighted average with window 5 bins

Hyperparameters:

- Batch size: 16
- Learning rate: 0.001
- Max epochs: 1000
- Early stopping patience: 50 epochs
- ReduceLROnPlateau patience: 20 epochs
- Trainer precision: `16-mixed`
- Gradient clip value: 1.0
- Logged LR values: 0.001, 0.0005, 0.00025

Architecture:

- Hop size: 256 samples
- Frame rate: 172.265625 Hz
- Sequence length: 1722 frames per 10 s crop
- Encoder channels: 32
- Hidden size: 64
- LSTM layers: 2
- Dropout: 0.0
- Output bins: 151
- Parameters: 85015

Dataset:

- Sample length: 441000 samples, 10.0 s
- Sample stride: 110250 samples, 2.5 s
- Split unit: song
- Split: 849 train crops from 8 songs, 184 validation crops from 2 songs

Training output and run length:

- CSV epochs covered: 0-135
- Epoch count from CSV: 136
- Max logged step: 7207
- Best validation loss from CSV: 0.0756354 at epoch 85, step 4557
- Last validation loss from CSV: 0.0803715 at epoch 135, step 7207
- Best validation MAE in dB from CSV: 0.6728483 at epoch 117, step 6253
- Last validation MAE in dB from CSV: 0.6977484 at epoch 135, step 7207
- Best training loss from CSV: 0.0538748 at epoch 135, step 7207
- Last training loss from CSV: 0.0538748 at epoch 135, step 7207
- Best training MAE in dB from CSV: 0.3541010 at epoch 135, step 7207
- Early stopping: stopped after 50 records without validation improvement
- Total training time printed by notebook: 17.3 min
- Training notebook best checkpoint: `/content/drive/Othercomputers/MacBook Air/data/gr_pred_runs/lstm_gr_20260520_121432_lstm_v3_discretized/checkpoints/best-085-4558.ckpt`
- Training notebook last checkpoint: `/content/drive/Othercomputers/MacBook Air/data/gr_pred_runs/lstm_gr_20260520_121432_lstm_v3_discretized/checkpoints/last.ckpt`
- Evaluation notebook checkpoint inventory: 31 checkpoints, displayed `epoch=135`, `best_val=0.075635`

Validation audio metrics:

- Split: validation
- Songs: 2
- Chunks: 48
- Duration: 471.9694 s
- MAE (L1): 0.0017904
- MSE (L2): 0.000007490
- ESR: 0.0340830
- MR-STFT: 0.0918657
- EDC: 0.1883375
- M_NRMSE: 0.1158154
- M_SF: 0.1403269

Saved outputs observed locally:

- `hparams.json`
- `csv/metrics.csv`
- `csv/hparams.yaml`
- `tb/hparams.yaml`
- `eval_audio_metrics_validation.csv`
- `eval_gr_comparison.png` printed by the training notebook

## Quick Comparison

| Run | Model / target | Params | Crop | Batch | Loss function | What it predicts | Discretization / smoothing |
|---|---|---:|---:|---:|---|---|---|
| `tcn_gr_20260427_094122_with_recommendations_h100` | Causal TCN, normalized GR | 46913 | 3.0 s | 16 | `l1+diff_l1`, `diff_weight=0.1` | One normalized GR value per audio sample, output as a full `[1, T]` GR curve. | No discretization. No explicit output smoothing. |
| `gcn_gr_20260427_104716_gcn_split_fix` | Causal GCN, normalized GR | 104001 | 10.0 s | 8 | `warmup_masked_l1+diff_l1+second_diff_l1`, `diff_weight=1.0`, `smooth_weight=0.2` | One normalized GR value per audio sample, output as a full `[1, T]` GR curve. | No discretization. Warmup mask excludes first 118097 samples; second-difference term encourages smoother curves. |
| `lstm_gr_20260507_190047_lstm_v2` | Frame-rate LSTM, dB GR regression | 80065 | 10.0 s | 16 | `warmup_masked_l1_db+diff_l1_db`, `diff_weight=0.3` | One continuous GR dB value per 256-sample frame, 172.27 Hz frame rate, 1722 frames per 10 s crop. | No discretization. First-difference loss encourages smoother frame-to-frame GR. |
| `lstm_gr_20260520_121432_lstm_v3_discretized` | Frame-rate LSTM, discretized GR classification | 85015 | 10.0 s | 16 | `bce_gaussian_soft_targets`, `sigma_bins=2.0`, `warmup_frames=4` | A 151-bin logit distribution per 256-sample frame, then decoded to continuous GR dB per frame. | 151 bins over -30 to 0 dB, 0.2 dB/bin. Training uses Gaussian soft targets with 0.4 dB sigma. Inference uses local weighted average around argmax with ±5 bins. |

Run lengths:

- `tcn_gr_20260427_094122_with_recommendations_h100`: 428 CSV epochs, epochs 0-427; no wall-clock time printed because the notebook was interrupted.
- `gcn_gr_20260427_104716_gcn_split_fix`: 3 resumed CSV epochs, epochs 61-63; evaluation inventory shows the run reached epoch 63 with 39 checkpoints; no wall-clock time printed because the notebook was interrupted.
- `lstm_gr_20260507_190047_lstm_v2`: 76 CSV epochs, epochs 0-75; 10.2 min printed by notebook.
- `lstm_gr_20260520_121432_lstm_v3_discretized`: 136 CSV epochs, epochs 0-135; 17.3 min printed by notebook.

Best validation training loss from CSV:

- TCN: 0.0248658
- GCN: 0.0414249, resumed CSV only
- LSTM regression: 0.5930500
- LSTM discretized: 0.0756354

Validation audio metrics available locally:

- GCN: MAE 0.0016800, ESR 0.0301545, MR-STFT 0.0819903
- LSTM regression: MAE 0.0016921, ESR 0.0310604, MR-STFT 0.0828489
- LSTM discretized: MAE 0.0017904, ESR 0.0340830, MR-STFT 0.0918657
- TCN: no `eval_audio_metrics_validation.csv` found in the local run folder

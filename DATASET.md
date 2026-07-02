# Diff-SSL-G-Comp — Dataset Notes

Reference notes for the dataset used in this thesis (virtual analog modeling of an SSL G-series bus compressor). See `dataset_settings.md` for the full 220-setting inventory with download flags.

## Dataset

- **HuggingFace repo:** `amphion/SolidStateBusComp` (repo_type=dataset), pretty name *Diff-SSL-G-Comp*.
- **Paper:** Gu, Zhang, Juvela, Wu — *Diff-SSL-G-Comp: Towards a Large-Scale and Diverse Dataset for Virtual Analog Modeling*, arXiv:2504.04589 (2025).
- **Scale:** 175 real unmastered songs (Cambridge Multitrack Library) × 220 recorded compressor parameter combinations ≈ 2528 hours. Full repo ≈ 4.8 TB.
- **License / access:** CC-BY-NC-4.0, **gated**. Downloading files needs an HF token *and* acceptance of the dataset terms on the repo page. Tree/metadata via the public API works without auth; `resolve/` file URLs return 401 anonymously.

### Top-level folders

| Folder | Contents |
|---|---|
| `cambridge_unmastered_songs/` | Raw input songs |
| `processed_normalized/` | Normalized input audio fed to the units (175 files, `<song>_UnmasteredWAV.wav`) |
| `processed_ground_truth/` | Analog SSL hardware outputs = modeling target. 220 subfolders, one per setting |
| `processed_overloud/`, `processed_pspaudioware/`, `processed_softube/`, `processed_solid_state_logic/` | Commercial plugin renders (baselines) |

### Setting naming & parameter grid

Folder pattern: `threshold_<dB>_attack_<ms>_release_<s|auto>_ratio_<x>`. Output files inside: `<song>-exported.wav`.

- threshold: {-12, -8, -4, 0, 4, 8, 12} dB (annotated from the analog device, not the DAW)
- attack: {1, 3, 10, 30} ms
- release: {0.1, 0.4, 0.8, auto} s
- ratio: {2, 4, 10} :1

**Not a full factorial** — only 220 of the 336 combinations exist. Per-threshold counts: -12→12, -8→15, -4→48, 0→48, 4→48, 8→33, 12→16.

### Download pattern (single file)

```
https://huggingface.co/datasets/amphion/SolidStateBusComp/resolve/main/processed_ground_truth/<setting>/<song>-exported.wav
```
with header `Authorization: Bearer <hf_token>`. Files are Xet/LFS, ~50–85 MB each.

## Local layout (this machine)

- Data root: `data/Diff-SSL-G-Comp/` (on the external drive)
- `processed_normalized/` — all 175 song inputs present locally
- Other dirs under `data/`: `gr_curves/`, `ableton_compressed/`, and model-run outputs `diffssl_tvc_runs/`, `gr_pred_runs/`, `diffssl_gr_tfilm_runs/`

### Training set — `processed_ground_truth/` (10 settings × 10 songs)

Settings:

```
threshold_-12_attack_1_release_0.1_ratio_2
threshold_-12_attack_10_release_0.4_ratio_10
threshold_-8_attack_30_release_0.8_ratio_4
threshold_-4_attack_1_release_0.4_ratio_10
threshold_-4_attack_10_release_0.1_ratio_2
threshold_0_attack_3_release_0.8_ratio_4
threshold_4_attack_10_release_0.1_ratio_10
threshold_8_attack_1_release_0.1_ratio_10
threshold_8_attack_30_release_0.4_ratio_2
threshold_12_attack_3_release_0.8_ratio_2
```

Songs: Air, AncoraQui, BackroomInTulsa, Borderline, Ecstasy, Electrvm, LivingLie, NosPalpitants, OpenFire, SongForJohn.

### Test set — `test_ground_truth/` (kept separate from training)

Created 2026-07-01. Settings chosen by farthest-point selection in normalized parameter space (threshold linear; attack & ratio log; release linear), excluding `auto` releases and anything already downloaded; songs are 5 not in the training 10. One song per setting:

| Setting | Song file |
|---|---|
| `threshold_12_attack_1_release_0.1_ratio_2` | `54-exported.wav` |
| `threshold_4_attack_30_release_0.8_ratio_10` | `Convertible-exported.wav` |
| `threshold_-12_attack_1_release_0.8_ratio_2` | `IncidenteEnIntag-exported.wav` |
| `threshold_-8_attack_3_release_0.4_ratio_4` | `OralHygiene-exported.wav` |
| `threshold_4_attack_1_release_0.8_ratio_10` | `SuchFinePeople-exported.wav` |

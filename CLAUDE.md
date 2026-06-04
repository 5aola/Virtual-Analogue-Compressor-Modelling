# Project: Virtual Analogue Compressor Modelling

MSc thesis project — neural modelling of analog audio compressors (LA-2A, CL 1B, SSL G-Bus, etc.) using PyTorch and the NablaFX framework.

## Package Manager

Use **uv** exclusively. Never use `pip` or `pip install`.
- `uv add <pkg>` to add dependencies
- `uv run <script>` to run scripts
- `uv sync` to install/update from pyproject.toml

## Directory Layout

```
├── src/                  # Shared utility package (import as `from src.* import ...`)
│   ├── dsp.py            # NumPy RMS, gain-reduction, level helpers
│   ├── dsp_torch.py      # PyTorch RMS, gain-reduction helpers (training pipelines)
│   ├── audio_io.py       # Audio loading / stats (Essentia)
│   ├── losses.py         # Evaluation loss metrics (PyTorch)
│   ├── transfer.py       # FFT / transfer-function utilities
│   └── plotting.py       # Gain-curve estimation & comparison plots
├── 01_dset_filtering/    # Dataset filtering notebooks
├── 02_sota_analysis/     # SOTA evaluation & analysis scripts
│   └── eval_output/      # All SOTA eval outputs (plots, CSVs, WAVs)
├── 03_initial_GR_pred/   # GR-prediction model training & dataset
├── external/             # Cloned external projects (gitignored)
│   ├── nablafx-for-diffssl-compressor
│   ├── Optical-DRC-with-Selective-SSMs
│   ├── A-Comparative-Study-State-Based-main
│   ├── Conditioning-Methods-for-Neural-Audio-Effects-main
│   └── signaltrain
└── nablafx/              # Raw nablafx library (kept at repo root, gitignored)
```

## Import Rules

- `src/` is an editable package via pyproject.toml. Use `from src.dsp import ...` — no sys.path hacking needed.
- Numbered folders (`01_*`, `02_*`, `03_*`) are not valid Python packages. Scripts within them use bare sibling imports (e.g. `from dataset import ...`).
- `02_sota_analysis/utils.py` is a thin re-export layer. New code should import from `src.*` directly.
- External projects live under `external/`. Reference as `external/nablafx-for-diffssl-compressor/...`.
- Raw `nablafx` stays at repo root for sys.path compatibility. Scripts needing it add it via `sys.path.insert`.
- Do NOT add `sys.path.insert` for the project root or script's own directory — already handled.

## NablaFX Framework — Prefer Over Reimplementation

Local clone at `./nablafx/`. Always check before writing new code:
1. `nablafx.processors` before building new architectures (TCN, GCN, LSTM, S4, DDSP)
2. `nablafx.evaluation` before writing loss/metric functions
3. `nablafx.data` before writing dataset/dataloader code
4. `nablafx.core` models and systems for training pipelines
5. `./nablafx/cfg/` for config structure examples

Note: `nablafx.evaluation` is NOT re-exported from `nablafx.__init__`. Use `from nablafx.evaluation import ...`.

## Datasets

All datasets at `/Volumes/Saola's Drive/AllCode/thesis/data/`:
- **CL1B** — TubeTech CL 1B optical compressor (WAV stereo L=dry R=wet + pickle, 48kHz)
- **FET** — Softube FET plugin (single 24GB pickle, 48kHz)
- **LA2A** — Teletronix LA-2A hardware (paired mono WAVs, 44.1kHz)
- **PRESS** — u-he Presswerk plugin (single 24GB pickle, 48kHz)
- **PSP** — PSP MicroComp plugin (single 24GB pickle, 48kHz)
- **Diff-SSL-G-Comp** — SSL G-Bus hardware (WAV dirs, 44.1kHz, partially downloaded)

## External Projects — SOTA Reference Implementations

All in `external/`, all TensorFlow 2.15 except SignalTrain (PyTorch 1.0) and nablafx fork (PyTorch/Lightning). Use `/comparative-study-ref`, `/optical-drc-ref`, `/conditioning-methods-ref`, `/signaltrain-ref`, `/diffssl-ref` slash commands for detailed API references.

| Project | Paper | Year | Framework | Models | Conditioning | Weights? | Compressor datasets |
|---|---|---|---|---|---|---|---|
| **A-Comparative-Study-State-Based** | EURASIP JASM 2025 | 2025 | TF 2.15 / Keras 3.13 | LSTM, ED, **LRU**, S4D, S6 | FiLM+GLU (single layer) | 35 checkpoints (7 effects × 5 models) | CL1B (D=4), LA2A (D=2), + 5 other effects |
| **Optical-DRC-with-Selective-SSMs** | JAES Mar 2025 | 2025 | TF 2.15 / Keras 2.15 | LSTM, ED-CNN, S4D, Mamba, TCN + baselines | FiLM (static) + TemporalFiLM (dynamic) | 12 checkpoints (2 devices × 6 models) | CL1B (D=4), LA2A (D=2) |
| **Conditioning-Methods** | SMC 2024 | 2024 | TF 2.15 / Keras 2.15 | S4D only | ExtraInp, GAF, **FiLM-GLU**, FiLM-GCU | None | Synthetic compressor (D=2) |
| **nablafx-for-diffssl-compressor** | Diff-SSL 2024 | 2024 | PyTorch/Lightning | TCN, GCN, S4, LSTM, GreyBox | FiLM, TFiLM, TinyTFiLM, TVFiLM | 38 experiments | SSL G-Bus (D=4) |
| **signaltrain** | AES 147 (2019) | 2019 | PyTorch 1.0 | Trainable STFT autoencoder | Knob concatenation at bottleneck | 1 demo model | LA-2A (D=3) |

### Key conditioning limits across projects
- **Static params** (threshold, ratio): best with FiLM or input concatenation
- **Dynamic params** (attack, release): TemporalFiLM (GRU-based) or TVFiLM in Optical-DRC; single FiLM+GLU in Comparative Study works comparably
- **Polynomial FiLM** (order=3, cubic) outperforms linear FiLM (Conditioning Methods paper)
- **LRU** is the newest architecture (Comparative Study only) — linear recurrent unit, competitive with S4D/S6

## Conventions

- Audio tensor shape: `[batch, channels, time]` — always mono (channels=1)
- Control parameters (black-box): `[batch, num_controls]`
- Control parameters (grey-box static): `[batch, num_control_params, 1]`
- Control parameters (grey-box dynamic): `[batch, num_control_params, time]`

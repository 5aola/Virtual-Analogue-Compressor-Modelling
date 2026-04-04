"""
Shared utilities for Virtual Analogue Compressor Modelling.

Submodules
----------
dsp          – NumPy-based RMS, gain-reduction, level helpers, param parsing
dsp_torch    – PyTorch-based RMS, gain-reduction, param normalisation
audio_io     – Audio loading / stats (Essentia + torchaudio)
losses       – Evaluation loss metrics (PyTorch)
transfer     – FFT and transfer-function utilities
plotting     – Gain-curve estimation and comparison plots
"""

"""
Audio I/O and stats helpers (Essentia-based).
"""

import os

import essentia.standard as es
import numpy as np

SR = 44100


def collect_audio_files(root_dir):
    audio_files = []
    for file in os.listdir(root_dir):
        if file.endswith(".wav"):
            audio_files.append(os.path.join(root_dir, file))
    return audio_files


def load_audio(path, sr=SR, cut=(None, None)):
    """
    Load an audio file using Essentia and return its time vector and audio samples.

    Example:
        # Load the whole audio file
        t, audio = load_audio("path/to/file.wav")

        # Load audio between 2.0s and 5.5s
        t, audio = load_audio("path/to/file.wav", cut=(2.0, 5.5))

        # Load audio from 10.0s until the end of the file
        t, audio = load_audio("path/to/file.wav", cut=(10.0, None))
    """
    audio = es.MonoLoader(filename=path, sampleRate=sr)()
    t = np.arange(len(audio)) / sr
    if cut[0] is not None:
        audio = audio[int(cut[0] * sr) : min(int(cut[1] * sr), len(audio))]
        t = t[int(cut[0] * sr) : min(int(cut[1] * sr), len(t))]
    return t, audio


def get_audio_stats(audio, sr=SR):
    _, _, ebu_integrated, loudness_range = es.LoudnessEBUR128(
        hopSize=1024 / sr, startAtZero=True
    )(audio)
    return ebu_integrated, loudness_range

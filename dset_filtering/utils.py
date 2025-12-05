import numpy as np
import pandas as pd
import os
import essentia
import essentia.standard as es
from scipy.signal import hilbert


SSL_DSET_PATH = "/Volumes/Production Tools/coding_projs/THESIS/data_preprocesses/data/Diff-SSL-G-Comp"
SR = 44100


def collect_audio_files(root_dir):
    audio_files = []
    for file in os.listdir(root_dir):
        if file.endswith(".wav"):
            audio_files.append(os.path.join(root_dir, file))
    return audio_files


def load_audio(path, sr=SR, cut=(None, None)):
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


def window_rms(signal, window_size, in_dB=False):
    window_size = int(window_size)

    signal = np.square(signal)
    window = np.ones(window_size) / float(window_size)
    rms = np.sqrt(np.convolve(signal, window, "full"))
    rms = rms[: len(signal)]
    if in_dB:
        rms = to_dB(rms)

    return rms


def calc_gain_reduction(signal1, signal2, window_size):
    rms = window_rms(signal1, window_size, in_dB=True)
    rms_wet = window_rms(signal2, window_size, in_dB=True)
    gr_rms_dB = rms_wet - rms

    return gr_rms_dB


def to_dB(signal, ref=1.0):
    return 20 * np.log10(signal / ref)


def to_amplitude(signal, ref=1.0):
    return 10 ** (signal / 20) * ref


# if __name__ == "__main__":

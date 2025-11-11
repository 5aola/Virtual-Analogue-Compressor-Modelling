import numpy as np
import os
import essentia
import essentia.standard as es

import matplotlib.pyplot as plt
from pylab import plot, show, figure, imshow



SSL_DSET_PATH = '/Volumes/Production Tools/coding_projs/THESIS/data_preprocesses/data/Diff-SSL-G-Comp'

DRY_PATH = 'processed_normalized'
WET_PATH = 'processed_ground_truth'

def load_audio_files(root_dir):
    audio_files = []
    for file in os.listdir(root_dir):
        if file.endswith(".wav"):
            audio_files.append(os.path.join(root_dir, file))
    return audio_files

def load_audio(path):
    audio_st, sr, _, _, _, _ = es.AudioLoader(filename=path)()
    return audio_st, sr

def get_audio_stats(audio, sr):
    _, _, ebu_integrated, loudness_range = es.LoudnessEBUR128(hopSize=1024/sr, startAtZero=True)(audio)
    return ebu_integrated, loudness_range


files = load_audio_files(os.path.join(SSL_DSET_PATH, DRY_PATH))

stats = []
for file in files:
    audio, sr = load_audio(file)
    loudness, loudness_range = get_audio_stats(audio, sr)
    stats.append((loudness, loudness_range))

plot(stats)

plt.show()




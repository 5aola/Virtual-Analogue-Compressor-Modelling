import numpy as np

window_size = 1024
overlap = 0.5
fs = 44100


def calc_fft(signal, fs):
    fft_size = int(window_size / 2 + 1)
    fft = np.fft.fft(signal, fft_size)
    f = np.fft.fftfreq(fft_size, 1 / fs)
    return f, fft


def H1(signal1, signal2):
    auto_spectr = np.zeros(int(window_size / 2 + 1), dtype="complex128")
    cross_spectr = np.zeros(int(window_size / 2 + 1), dtype="complex128")
    for index in range(
        0,
        len(signal1) - window_size,
        int(window_size * (1 - overlap)),
    ):
        [f, signal1_f] = calc_fft(
            signal1[index : (index + window_size)],
            fs,
        )
        [f, signal2_f] = calc_fft(
            signal2[index : (index + window_size)],
            fs,
        )
        auto_spectr += np.abs(signal1_f) ** 2
        cross_spectr += signal2_f * np.conj(signal1_f)
    h1 = cross_spectr / auto_spectr
    return f, h1


def calc_FRAC(m1s1, m1s2, m2s1, m2s2):
    _, h1 = H1(m1s1, m1s2)
    _, h2 = H1(m2s1, m2s2)
    return np.abs((np.conj(h1) @ h2)) ** 2 / np.abs(
        (np.conj(h1) @ h1) * (np.conj(h2) @ h2)
    )

import numpy as np


def H1(self, ch1, ch2):
        auto_spectr = np.zeros(int(self.window_size / 2 + 1), dtype="complex128")
        cross_spectr = np.zeros(int(self.window_size / 2 + 1), dtype="complex128")
        for index in range(
            0,
            len(ch1) - self.window_size,
            int(self.window_size * (1 - self.overlap)),
        ):
            [f, ch1_f] = calc_fft(
                ch1[index : (index + self.window_size)],
                self.fs,
            )
            [f, ch2_f] = calc_fft(
                ch2[index : (index + self.window_size)],
                self.fs,
            )
            auto_spectr += np.abs(ch1_f) ** 2
            cross_spectr += ch2_f * np.conj(ch1_f)
        h1 = cross_spectr / auto_spectr
        return (f[(f>self.valid_freqs[0]) & (f<self.valid_freqs[1])], h1[(f>self.valid_freqs[0]) & (f<self.valid_freqs[1])])

def calc_FRAC(m1s1, m1s2, m2s1, m2s2):
    _, h1 = H1(m1s1, m1s2)
    _, h2 = H1(m2s1, m2s2)
    return np.abs((np.conj(h1)@h2))**2/np.abs((np.conj(h1)@h1)*(np.conj(h2)@h2))
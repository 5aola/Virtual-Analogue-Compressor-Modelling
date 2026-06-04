"""Data loading for raw WAV input/output pairs.

A *recording* is one (dry, wet) pair captured at a fixed set of device control
parameters.  Training is *stateful*: within a recording we walk contiguous
chunks in time and carry the model state across them (truncated BPTT), exactly
the regime the model is designed for.  A batch is a group of recordings advanced
through time together.

Two ways to specify data:

1. A manifest CSV with a header.  Required columns ``dry`` and ``wet`` (paths,
   absolute or relative to the manifest's directory).  Any further numeric
   columns are treated, in order, as the normalised control parameters.

       dry,wet,threshold,ratio,attack,release
       in.wav,out_a.wav,0.4,0.3,0.1,0.6
       in.wav,out_b.wav,0.4,0.3,0.9,0.2

2. A Python list of ``(dry_path, wet_path, params_list)`` tuples passed directly
   to :class:`WavPairsDataset`.

All audio is loaded mono.  Control parameters must already be normalised to
[0, 1] (see ``synth.make_dataset`` for an example of the convention).
"""

from __future__ import annotations

import csv
import os
import numpy as np
import torch

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None


def load_wav(path, expected_sr=None):
    if sf is None:
        raise RuntimeError("soundfile is required to read WAV files (pip install soundfile)")
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix to mono
    if expected_sr is not None and sr != expected_sr:
        raise ValueError(f"{path}: sample rate {sr} != expected {expected_sr}")
    return audio, sr


class WavPairsDataset:
    def __init__(self, source, n_params=None, sr=None, align=True):
        """
        Args:
            source: path to a manifest CSV, or a list of
                    (dry_path, wet_path, params_list) tuples.
            n_params: expected number of control parameters (inferred if None).
            sr: expected sample rate (validated if given).
            align: trim dry/wet of each pair to equal length.
        """
        rows = self._read_manifest(source) if isinstance(source, str) else list(source)
        self.sr = sr
        drys, wets, params = [], [], []
        for dry_p, wet_p, p in rows:
            d, sr_d = load_wav(dry_p, sr)
            w, sr_w = load_wav(wet_p, sr)
            self.sr = self.sr or sr_d
            if align:
                L = min(len(d), len(w))
                d, w = d[:L], w[:L]
            drys.append(d)
            wets.append(w)
            params.append(np.asarray(p, dtype=np.float32))
        self.dry = drys
        self.wet = wets
        self.params = params
        self.n_params = n_params if n_params is not None else (
            len(params[0]) if params and params[0].size else 0)

    @staticmethod
    def _read_manifest(path):
        base = os.path.dirname(os.path.abspath(path))
        rows = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            extra = [c for c in reader.fieldnames if c not in ("dry", "wet")]
            for r in reader:
                dry = r["dry"] if os.path.isabs(r["dry"]) else os.path.join(base, r["dry"])
                wet = r["wet"] if os.path.isabs(r["wet"]) else os.path.join(base, r["wet"])
                p = [float(r[c]) for c in extra if r.get(c) not in (None, "")]
                rows.append((dry, wet, p))
        return rows

    def __len__(self):
        return len(self.dry)

    def split(self, frac=0.9):
        """Split each recording along time into (train, val) datasets."""
        tr, va = [], []
        for d, w, p in zip(self.dry, self.wet, self.params):
            cut = int(len(d) * frac)
            tr.append((d[:cut], w[:cut], p))
            va.append((d[cut:], w[cut:], p))
        return _PreloadedDataset(tr, self.n_params, self.sr), \
               _PreloadedDataset(va, self.n_params, self.sr)

    def tbptt_loader(self, batch_size, seq_len, shuffle=True, device="cpu", drop_last=True):
        return TBPTTLoader(self, batch_size, seq_len, shuffle, device, drop_last)


class _PreloadedDataset(WavPairsDataset):
    def __init__(self, rows, n_params, sr):
        self.dry = [r[0] for r in rows]
        self.wet = [r[1] for r in rows]
        self.params = [np.asarray(r[2], dtype=np.float32) for r in rows]
        self.n_params = n_params
        self.sr = sr


class TBPTTLoader:
    """Iterates stateful chunks.  Yields dicts with keys:
        x (B,seq), y (B,seq), params (B,n_params), reset (bool)
    ``reset`` is True on the first chunk of a new recording group, telling the
    training loop to reinitialise the model state.
    """

    def __init__(self, ds, batch_size, seq_len, shuffle, device, drop_last):
        self.ds = ds
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.device = device
        self.drop_last = drop_last

    def __iter__(self):
        order = np.arange(len(self.ds))
        if self.shuffle:
            np.random.shuffle(order)
        for start in range(0, len(order), self.batch_size):
            grp = order[start : start + self.batch_size]
            if self.drop_last and len(grp) < self.batch_size:
                continue
            T = min(len(self.ds.dry[i]) for i in grp)
            T -= T % self.seq_len
            if T == 0:
                continue
            X = torch.tensor(np.stack([self.ds.dry[i][:T] for i in grp]), device=self.device)
            Y = torch.tensor(np.stack([self.ds.wet[i][:T] for i in grp]), device=self.device)
            P = torch.tensor(np.stack([self.ds.params[i] for i in grp]), device=self.device)
            for c, t in enumerate(range(0, T, self.seq_len)):
                yield {
                    "x": X[:, t : t + self.seq_len],
                    "y": Y[:, t : t + self.seq_len],
                    "params": P,
                    "reset": c == 0,
                }

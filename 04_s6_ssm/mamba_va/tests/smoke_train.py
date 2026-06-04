"""Fast end-to-end smoke test: a few training steps on a small synthetic
dataset must reduce the error-to-signal ratio.  Runs on CPU in seconds.

    PYTHONPATH=. python3 tests/smoke_train.py
"""

import numpy as np
import torch

from mamba_va import CompSSM
from mamba_va.losses import CombinedLoss, esr
from mamba_va.data import WavPairsDataset
from mamba_va.synth import make_dataset
from mamba_va.utils import detach_state


def build_small_ds(sr=8000, seconds=1.0, n=3):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        params = dict(threshold_db=float(rng.uniform(-36, -10)),
                      ratio=float(rng.uniform(2, 8)),
                      attack_ms=float(10 ** rng.uniform(0, 1.5)),
                      release_ms=float(10 ** rng.uniform(1.5, 2.6)))
        d, w, p = make_dataset(seconds=seconds, sr=sr, params=params, seed=i)
        rows.append((d, w, p))
    ds = WavPairsDataset.__new__(WavPairsDataset)
    ds.dry = [r[0] for r in rows]; ds.wet = [r[1] for r in rows]
    ds.params = [np.asarray(r[2], np.float32) for r in rows]
    ds.n_params = 4; ds.sr = sr
    return ds


def run():
    torch.manual_seed(0); np.random.seed(0)
    ds = build_small_ds()
    model = CompSSM(n_params=4, d_model=16, d_state=8, n_layers=2, n_bands=3)
    crit = CombinedLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def epoch_loss(train=True):
        model.train(train)
        state, tot, k = None, 0.0, 0
        for b in ds.tbptt_loader(3, 512, shuffle=False):
            if b["reset"]:
                state = None
            y_hat, state = model(b["x"], b["params"], state, parallel=True)
            l = crit(y_hat, b["y"])
            if train:
                opt.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            state = detach_state(state)
            tot += esr(y_hat, b["y"]).item(); k += 1
        return tot / k

    before = epoch_loss(train=False)
    for _ in range(6):
        epoch_loss(train=True)
    after = epoch_loss(train=False)
    print(f"ESR before {before:.4f} -> after {after:.4f}")
    assert after < before, "training did not reduce ESR"
    print("smoke_train passed")


if __name__ == "__main__":
    run()

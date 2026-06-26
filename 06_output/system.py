"""Lightning system for the audio→audio GR output-transformer LSTM.

Loss strategy is taken from the SOTA waveform-modelling references (not the GR
classification recipe):

  * **0.5·L1 + 0.5·MR-STFT** — the ``TimeAndFrequencyDomainLoss`` from
    ``external/nablafx-for-diffssl-compressor/nablafx/loss.py``, the native
    recipe for this SSL G-Bus dataset. MR-STFT (auraloss) is not
    energy-dominated like plain MSE, so quiet / low-coloration regions are
    actually fit.
  * optional **ESR** term — ``auraloss.time.ESRLoss``, the canonical RNN
    audio-effect loss used by Optical-DRC / the Comparative Study.

Metrics logged: ESR, RMSE (Optical-DRC ``Metrics.py`` ports), MAE, MSE.

Stateful TBPTT: the two LSTM states ``(s1, s2)`` are kept per phase
(train/val/test), restored before each forward and detached after, so
gradients never cross chunk boundaries and phases never share state — reset at
track boundary / epoch start. Mirrors ``02b_sota_training`` and the
``05_conditioning`` GR system.
"""

from __future__ import annotations

import auraloss
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule


def esr_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Error-to-signal ratio (Optical-DRC ``Metrics.ESR``)."""
    return torch.mean((y_pred - y_true) ** 2 / (y_true**2 + 1e-5))


def rmse_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Optical-DRC ``Metrics.RMSE``: mean(| |pred| − |true| |)."""
    return torch.mean(torch.abs(torch.abs(y_pred) - torch.abs(y_true)))


def _as_bool(value) -> bool:
    return bool(value.item()) if torch.is_tensor(value) else bool(value)


def _detach_state(state):
    if state is None:
        return None
    d = lambda s: None if s is None else (s[0].detach(), s[1].detach())
    # Generic over the number of carried LSTM states: 2 for the plain grey box,
    # 3 once the diffssl tvcond conditioning LSTM is threaded in (model_tfilm).
    return tuple(d(s) for s in state)


class OutputTransformerSystem(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        l1_weight: float = 0.5,
        mrstft_weight: float = 0.5,
        esr_weight: float = 0.0,
        warmup_samples: int = 0,
        scheduler: str = "cosine",
        max_epochs: int = 800,
        eta_min: float = 1e-6,
        lr_patience: int = 20,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.l1_weight = l1_weight
        self.mrstft_weight = mrstft_weight
        self.esr_weight = esr_weight
        self.warmup_samples = warmup_samples
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.lr_patience = lr_patience
        self.min_lr = min_lr

        self.l1 = nn.L1Loss()
        # nablafx-diffssl MultiResolutionSTFTLoss configuration
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 512],
            hop_sizes=[120, 240, 50],
            win_lengths=[600, 1200, 240],
            w_sc=1.0, w_log_mag=1.0, w_lin_mag=0.0,
        )
        self.esr = auraloss.time.ESRLoss() if esr_weight > 0 else None
        self._states: dict[str, tuple | None] = {"train": None, "val": None, "test": None}

    def forward(self, x):  # stateless convenience (used by eval)
        return self.model(x)

    def reset_phase_state(self, phase: str) -> None:
        self._states[phase] = None

    def _run_model(self, batch, phase: str):
        """Unpack the batch, run the (stateful) model, carry+detach state.

        Returns ``(pred, wet, mask, is_reset)``. Subclasses with extra inputs
        (e.g. a GR conditioning curve) override only this method."""
        x, wet, mask, is_reset = batch[:4]
        is_reset = _as_bool(is_reset)
        if is_reset:
            self.reset_phase_state(phase)
        pred, new_state = self.model(x, self._states[phase], return_state=True)  # [B,1,S]
        self._states[phase] = _detach_state(new_state)
        return pred, wet, mask, is_reset

    def _step(self, batch, phase: str):
        pred, wet, mask, is_reset = self._run_model(batch, phase)

        if is_reset and self.warmup_samples > 0:
            w = min(self.warmup_samples, max(pred.shape[-1] - 1, 0))
            pred, wet = pred[..., w:], wet[..., w:]

        valid = mask.bool()
        pv, tv = pred[valid], wet[valid]  # drop zero-padded (ended) rows

        td = self.l1(pv, tv)
        fd = self.mrstft(pv, tv)
        loss = self.l1_weight * td + self.mrstft_weight * fd
        if self.esr is not None:
            loss = loss + self.esr_weight * self.esr(pv, tv)

        with torch.no_grad():
            mae = F.l1_loss(pv, tv)
            mse = F.mse_loss(pv, tv)
            esr = esr_metric(tv, pv)
            rmse = rmse_metric(tv, pv)

        bs = int(mask.sum())
        self.log(f"loss/{phase}", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log(f"loss/{phase}_td", td, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"loss/{phase}_fd", fd, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"mae/{phase}", mae, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"mse/{phase}", mse, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"esr/{phase}", esr, on_step=False, on_epoch=True,
                 prog_bar=(phase != "train"), batch_size=bs)
        self.log(f"rmse/{phase}", rmse, on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def on_train_epoch_start(self):
        self.reset_phase_state("train")

    def on_validation_epoch_start(self):
        self.reset_phase_state("val")

    def on_test_epoch_start(self):
        self.reset_phase_state("test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        if self.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.max_epochs, eta_min=self.eta_min
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch", "frequency": 1},
            }
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=self.lr_patience, min_lr=self.min_lr
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "loss/val"},
        }

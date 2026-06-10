"""Lightning training system for TFiLM-conditioned GR LSTM."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

from gr_target import GR_DB_MAX, GR_DB_MIN, normalize_gr_01


class TFiLMGRSystem(LightningModule):
    """Stateful TBPTT across B parallel (song, setting) streams.

    Both recurrent states — the main LSTM's (h, c) and the TFiLM LSTM's
    hidden state — are kept per phase (train/val/test), restored before each
    forward and stashed detached afterwards, so gradients never cross chunk
    boundaries and phases never share state. nablafx's TFiLM stores its state
    on the module and has no detach method, hence the stash/restore here.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        warmup_frames: int = 4,
        scheduler: str = "cosine",
        max_epochs: int = 400,
        eta_min: float = 1e-6,
        lr_patience: int = 20,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.warmup_frames = warmup_frames
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.lr_patience = lr_patience
        self.min_lr = min_lr
        self._lstm_states: dict[str, tuple | None] = {"train": None, "val": None, "test": None}
        self._tfilm_states: dict[str, tuple | None] = {"train": None, "val": None, "test": None}

    def forward(self, dry, params):
        return self.model(dry, params)

    def reset_phase_states(self, phase: str) -> None:
        self._lstm_states[phase] = None
        self._tfilm_states[phase] = None

    def _step(self, batch, phase: str):
        dry, gr_db, params, mask, is_reset = batch
        if is_reset:
            self.reset_phase_states(phase)

        self.model.tfilm.hidden_state = self._tfilm_states[phase]
        pred, lstm_state = self.model(dry, params, self._lstm_states[phase], return_state=True)
        self._lstm_states[phase] = tuple(s.detach() for s in lstm_state)
        self._tfilm_states[phase] = tuple(s.detach() for s in self.model.tfilm.hidden_state)

        target = F.adaptive_avg_pool1d(normalize_gr_01(gr_db), pred.shape[-1])
        if is_reset and self.warmup_frames > 0:
            pred = pred[..., self.warmup_frames :]
            target = target[..., self.warmup_frames :]

        valid = mask.bool()
        loss = F.mse_loss(pred[valid], target[valid])
        mae_db = (GR_DB_MAX - GR_DB_MIN) * F.l1_loss(pred[valid].detach(), target[valid])
        bs = int(mask.sum())
        self.log(f"loss/{phase}", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log(f"mae_db/{phase}", mae_db, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def on_train_epoch_start(self):
        self.reset_phase_states("train")

    def on_validation_epoch_start(self):
        self.reset_phase_states("val")

    def on_test_epoch_start(self):
        self.reset_phase_states("test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
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

"""Lightning system for diffssl ``LSTM`` (tvcond) with stateful TBPTT.

Mirrors ``BlackBoxSystemWithTBPTT`` from
``external/nablafx-for-diffssl-compressor`` (manual optim, ``step_num_samples``
sub-steps, ``detach_states`` after each train step) while feeding the
multi-stream stateful dataloader from ``dataset.py`` (LSTM state carries across
consecutive ``segment_len`` chunks of each track; reset when ``is_reset``).
"""

from __future__ import annotations

import auraloss
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule


def esr_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_pred - y_true) ** 2 / (y_true**2 + 1e-5))


def rmse_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(torch.abs(y_pred) - torch.abs(y_true)))


def _as_bool(value) -> bool:
    return bool(value.item()) if torch.is_tensor(value) else bool(value)


class DiffSSLTVCLSTMSystem(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        step_num_samples: int = 4410,
        td_weight: float = 0.5,
        fd_weight: float = 0.5,
        lr_patience: int = 20,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.step_num_samples = step_num_samples
        self.td_weight = td_weight
        self.fd_weight = fd_weight
        self.lr_patience = lr_patience
        self.min_lr = min_lr
        self.automatic_optimization = False

        self.l1 = nn.L1Loss()
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 512],
            hop_sizes=[120, 240, 50],
            win_lengths=[600, 1200, 240],
            w_sc=1.0,
            w_log_mag=1.0,
            w_lin_mag=0.0,
        )

    def _loss_on_valid(self, pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor):
        pv, tv = pred[valid], target[valid]
        td = self.l1(pv, tv)
        fd = self.mrstft(pv, tv)
        return self.td_weight * td + self.fd_weight * fd, td, fd

    def _process_segment(self, dry, wet, params, mask, phase: str, is_reset: bool):
        if is_reset:
            self.model.reset_states()

        valid = mask.bool()
        seq_len = dry.shape[-1]
        pred_chunks: list[torch.Tensor] = []
        train_losses: list[torch.Tensor] = []

        if phase == "train":
            self.model.detach_states()
            opt = self.optimizers()

        for start in range(0, seq_len, self.step_num_samples):
            end = min(start + self.step_num_samples, seq_len)
            step_in = dry[..., start:end]
            step_tgt = wet[..., start:end]
            pred = self.model(step_in, params)
            pred_chunks.append(pred)

            if phase == "train":
                step_loss, _, _ = self._loss_on_valid(pred, step_tgt, valid)
                opt.zero_grad()
                self.manual_backward(step_loss)
                opt.step()
                self.model.detach_states()
                train_losses.append(step_loss.detach())

        pred_full = torch.cat(pred_chunks, dim=-1)
        if phase == "train" and train_losses:
            loss = torch.stack(train_losses).mean()
            _, td, fd = self._loss_on_valid(pred_full, wet, valid)
        else:
            loss, td, fd = self._loss_on_valid(pred_full, wet, valid)

        with torch.no_grad():
            pv, tv = pred_full[valid], wet[valid]
            mae = F.l1_loss(pv, tv)
            mse = F.mse_loss(pv, tv)
            esr = esr_metric(tv, pv)
            rmse = rmse_metric(tv, pv)

        return loss, td, fd, mae, mse, esr, rmse

    def _step(self, batch, phase: str):
        dry, wet, params, mask, is_reset = batch
        is_reset = _as_bool(is_reset)
        loss, td, fd, mae, mse, esr, rmse = self._process_segment(
            dry, wet, params, mask, phase, is_reset
        )
        bs = int(mask.sum())
        self.log(f"loss/{phase}", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log(f"loss/{phase}_td", td, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"loss/{phase}_fd", fd, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"mae/{phase}", mae, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"mse/{phase}", mse, on_step=False, on_epoch=True, batch_size=bs)
        self.log(
            f"esr/{phase}",
            esr,
            on_step=False,
            on_epoch=True,
            prog_bar=(phase != "train"),
            batch_size=bs,
        )
        self.log(f"rmse/{phase}", rmse, on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def on_train_epoch_start(self):
        self.model.reset_states()

    def on_validation_epoch_start(self):
        self.model.reset_states()

    def on_test_epoch_start(self):
        self.model.reset_states()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=self.lr_patience, min_lr=self.min_lr
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "loss/val"},
        }

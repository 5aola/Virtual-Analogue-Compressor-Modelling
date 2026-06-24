"""Lightning system mirroring diffssl ``BlackBoxSystemWithTBPTT``.

Per batch: ``reset_states()`` → TBPTT sub-steps of ``step_num_samples`` with
manual ``backward`` / ``step`` / ``detach_states()`` on train → log loss on the
full concatenated crop (diffssl ``loss/{phase}/tot`` recipe, logged as
``loss/{phase}`` here).
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

    def _loss(self, pred: torch.Tensor, target: torch.Tensor):
        td = self.l1(pred, target)
        fd = self.mrstft(pred, target)
        return self.td_weight * td + self.fd_weight * fd, td, fd

    def _common_step(self, batch, phase: str):
        dry, wet, params = batch
        train = phase == "train"

        self.model.reset_states()
        if train:
            self.model.detach_states()
            optimizer = self.optimizers()
            optimizer.zero_grad()

        seq_len = dry.shape[-1]
        pred_chunks: list[torch.Tensor] = []

        for start in range(0, seq_len, self.step_num_samples):
            end = min(start + self.step_num_samples, seq_len)
            step_pred = self.model(dry[..., start:end], params)
            pred_chunks.append(step_pred)

            if train:
                step_loss, _, _ = self._loss(step_pred, wet[..., start:end])
                self.manual_backward(step_loss)
                optimizer.step()
                self.model.detach_states()
                optimizer.zero_grad()

        pred = torch.cat(pred_chunks, dim=-1)
        loss, td, fd = self._loss(pred, wet)

        with torch.no_grad():
            mae = F.l1_loss(pred, wet)
            mse = F.mse_loss(pred, wet)
            esr = esr_metric(wet, pred)
            rmse = rmse_metric(wet, pred)

        bs = dry.shape[0]
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
            prog_bar=(not train),
            batch_size=bs,
        )
        self.log(f"rmse/{phase}", rmse, on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "val")

    def on_validation_epoch_end(self):
        # automatic_optimization=False → Lightning will not step the scheduler
        # for us, so drive ReduceLROnPlateau manually (matches diffssl
        # BlackBoxSystemWithTBPTT.on_validation_epoch_end).
        sch = self.lr_schedulers()
        if sch is not None and "loss/val" in self.trainer.logged_metrics:
            sch.step(self.trainer.logged_metrics["loss/val"])

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=self.lr_patience, min_lr=self.min_lr
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "monitor": "loss/val"},
        }

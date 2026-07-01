"""Lightning system for the GR-TFiLM diffssl LSTM.

Based on ``02b_sota_training/system.py`` (``DiffSSLTVCLSTMSystem`` → diffssl
``BlackBoxSystemWithTBPTT``): per batch, ``reset_states()`` → TBPTT sub-steps of
``step_num_samples`` with manual ``backward``/``step``/``detach_states()`` on
train → ``0.5·L1 + 0.5·MR-STFT`` loss, esr/rmse/mae/mse logging.

Two changes vs 02b — (1) each batch carries the sample-aligned **GR curve**,
sliced alongside dry/wet and passed to the model; (2) **training-speed** tweaks
for the fixed 100-epoch budget:

  * **bf16 autocast** around the LSTM forward (``use_amp``) — the recurrent
    compute is the bottleneck; the loss is still evaluated in fp32.
  * **cosine LR schedule** over ``max_epochs`` (``scheduler="cosine"``, stepped
    per epoch under manual optimization) instead of val-coupled ReduceLROnPlateau.
  * **no redundant full-crop MR-STFT recompute on train** — ``loss/train`` is the
    mean of the per-sub-step losses already computed for the gradient, so training
    no longer pays for an extra 3 s STFT every step (val/test still report the
    exact full-crop loss).
"""

from __future__ import annotations

import contextlib

import auraloss
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule


def esr_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_pred - y_true) ** 2 / (y_true**2 + 1e-5))


def rmse_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(torch.abs(y_pred) - torch.abs(y_true)))


class GRTFiLMSystem(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        step_num_samples: int = 4410,
        td_weight: float = 0.5,
        fd_weight: float = 0.5,
        # -- scheduler --
        scheduler: str = "cosine",     # "cosine" | "plateau" | "none"
        max_epochs: int = 100,         # cosine T_max
        eta_min: float = 1e-6,
        lr_patience: int = 20,         # plateau only
        min_lr: float = 1e-6,          # plateau only
        # -- speed --
        use_amp: bool = True,          # bf16 autocast around the forward (cuda only)
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.step_num_samples = step_num_samples
        self.td_weight = td_weight
        self.fd_weight = fd_weight
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.lr_patience = lr_patience
        self.min_lr = min_lr
        self.use_amp = use_amp
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

    def _amp(self):
        if self.use_amp and torch.cuda.is_available():
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _loss(self, pred: torch.Tensor, target: torch.Tensor):
        td = self.l1(pred, target)
        fd = self.mrstft(pred, target)
        return self.td_weight * td + self.fd_weight * fd, td, fd

    def _common_step(self, batch, phase: str):
        dry, gr, wet, params = batch
        train = phase == "train"

        self.model.reset_states()
        if train:
            self.model.detach_states()
            optimizer = self.optimizers()
            optimizer.zero_grad()

        seq_len = dry.shape[-1]
        pred_chunks: list[torch.Tensor] = []
        loss_sum = td_sum = fd_sum = 0.0
        n_sub = 0

        for start in range(0, seq_len, self.step_num_samples):
            end = min(start + self.step_num_samples, seq_len)
            with self._amp():
                step_pred = self.model(dry[..., start:end], gr[..., start:end], params)
            step_pred = step_pred.float()
            pred_chunks.append(step_pred)

            if train:
                step_loss, td_i, fd_i = self._loss(step_pred, wet[..., start:end])
                self.manual_backward(step_loss)
                optimizer.step()
                self.model.detach_states()
                optimizer.zero_grad()
                loss_sum += step_loss.detach()
                td_sum += td_i.detach()
                fd_sum += fd_i.detach()
                n_sub += 1

        pred = torch.cat(pred_chunks, dim=-1)

        if train:
            # mean of the per-sub-step losses already used for the gradient
            # (avoids an extra full-crop MR-STFT purely for logging).
            loss, td, fd = loss_sum / n_sub, td_sum / n_sub, fd_sum / n_sub
        else:
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

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, "test")

    def on_train_epoch_end(self):
        # manual optimization → Lightning won't step schedulers for us.
        if self.scheduler == "cosine":
            sch = self.lr_schedulers()
            if sch is not None:
                sch.step()

    def on_validation_epoch_end(self):
        if self.scheduler == "plateau":
            sch = self.lr_schedulers()
            if sch is not None and "loss/val" in self.trainer.logged_metrics:
                sch.step(self.trainer.logged_metrics["loss/val"])

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        if self.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.max_epochs, eta_min=self.eta_min
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
        if self.scheduler == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=self.lr_patience, min_lr=self.min_lr
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "monitor": "loss/val"},
            }
        return opt

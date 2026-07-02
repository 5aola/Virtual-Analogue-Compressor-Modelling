"""Lightning system for the gain-prior diffssl LSTM (4a model + 4c losses).

Training regime is identical to ``system_tfilm.GRTFiLMSystem`` (per batch:
``reset_states()`` → TBPTT sub-steps of ``step_num_samples`` with manual
backward/step/``detach_states()``; cosine LR; optional bf16 autocast).

Loss (the "4c" upgrades, each individually switchable back to the exact 02b
recipe for the comparability ablation):

    loss = td_weight · L1
         + fd_weight · MR-STFT            ("sota" 3-res, or "extended" with a
                                            4096-FFT resolution + linear-mag
                                            term to resolve release-scale
                                            energy errors)
         + env_weight · L_env             (L1 in dB between 1024-sample RMS
                                            envelopes of pred and wet — the
                                            differentiable form of the GR MAE
                                            eval metric, targeting the SOTA's
                                            weakest column)
         + pe_weight  · L_pe              (L1 after first-order pre-emphasis
                                            high-pass — transient/onset
                                            weighting, the known hard part of
                                            compressor modelling)

Setting ``env_weight=0, pe_weight=0, mrstft_variant="sota"`` reproduces the
02b loss exactly.

Extra logging: mean |Δg| (dB) and mean |color| per phase — the two heads'
share of the work, for interpretability (both start at 0 by construction).
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


# MR-STFT resolutions: "sota" == the 02b/auraloss set; "extended" adds a
# 4096-FFT resolution (win 2400 ≈ 54 ms) so release-scale (0.1-0.8 s) energy
# errors are resolved by more than frame-averaging, plus a linear-magnitude
# term (better conditioned than log-mag on quiet passages).
_MRSTFT_CFG = {
    "sota": dict(
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        w_sc=1.0, w_log_mag=1.0, w_lin_mag=0.0,
    ),
    "extended": dict(
        fft_sizes=[1024, 2048, 512, 4096],
        hop_sizes=[120, 240, 50, 480],
        win_lengths=[600, 1200, 240, 2400],
        w_sc=1.0, w_log_mag=1.0, w_lin_mag=1.0,
    ),
}


class GainPriorSystem(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        step_num_samples: int = 4410,
        # -- loss weights --
        td_weight: float = 0.5,
        fd_weight: float = 0.5,
        env_weight: float = 0.2,
        pe_weight: float = 0.1,
        mrstft_variant: str = "extended",   # "extended" | "sota"
        env_window: int = 1024,             # == the GR-export RMS window
        env_stride: int = 256,
        pe_coeff: float = 0.95,
        # -- scheduler --
        scheduler: str = "cosine",          # "cosine" | "plateau" | "none"
        max_epochs: int = 100,
        eta_min: float = 1e-6,
        lr_patience: int = 20,
        min_lr: float = 1e-6,
        # -- speed --
        use_amp: bool = True,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.step_num_samples = step_num_samples
        self.td_weight = td_weight
        self.fd_weight = fd_weight
        self.env_weight = env_weight
        self.pe_weight = pe_weight
        self.env_window = env_window
        self.env_stride = env_stride
        self.pe_coeff = pe_coeff
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.lr_patience = lr_patience
        self.min_lr = min_lr
        self.use_amp = use_amp
        self.automatic_optimization = False

        cfg = _MRSTFT_CFG[mrstft_variant]
        if step_num_samples < max(cfg["fft_sizes"]):
            raise ValueError(
                f"step_num_samples={step_num_samples} < largest FFT "
                f"{max(cfg['fft_sizes'])} ({mrstft_variant}); raise the TBPTT "
                f"sub-step or use mrstft_variant='sota'."
            )
        self.l1 = nn.L1Loss()
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(**cfg)

    def _amp(self):
        if self.use_amp and torch.cuda.is_available():
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # -- 4c loss components --------------------------------------------------
    def _env_db(self, x: torch.Tensor) -> torch.Tensor:
        # Short-time RMS energy in dB (window == GR export RMS window).
        # eps floors near-silence at -70 dB so silent regions contribute 0.
        p = F.avg_pool1d(x.pow(2), kernel_size=self.env_window, stride=self.env_stride)
        return 10.0 * torch.log10(p + 1e-7)

    def _preemph(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., 1:] - self.pe_coeff * x[..., :-1]

    def _loss(self, pred: torch.Tensor, target: torch.Tensor):
        td = self.l1(pred, target)
        fd = self.mrstft(pred, target)
        loss = self.td_weight * td + self.fd_weight * fd
        env = pe = pred.new_zeros(())
        if self.env_weight > 0:
            env = self.l1(self._env_db(pred), self._env_db(target))
            loss = loss + self.env_weight * env
        if self.pe_weight > 0:
            pe = self.l1(self._preemph(pred), self._preemph(target))
            loss = loss + self.pe_weight * pe
        return loss, {"td": td, "fd": fd, "env": env, "pe": pe}

    # -- TBPTT step (mirrors system_tfilm) ------------------------------------
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
        sums = {"loss": 0.0, "td": 0.0, "fd": 0.0, "env": 0.0, "pe": 0.0}
        delta_abs = color_abs = 0.0
        n_sub = 0

        for start in range(0, seq_len, self.step_num_samples):
            end = min(start + self.step_num_samples, seq_len)
            with self._amp():
                step_pred, step_delta, step_color = self.model(
                    dry[..., start:end], gr[..., start:end], params, return_parts=True
                )
            step_pred = step_pred.float()
            pred_chunks.append(step_pred)
            delta_abs += step_delta.detach().abs().mean().float()
            color_abs += step_color.detach().abs().mean().float()
            n_sub += 1

            if train:
                step_loss, comps = self._loss(step_pred, wet[..., start:end])
                self.manual_backward(step_loss)
                optimizer.step()
                self.model.detach_states()
                optimizer.zero_grad()
                sums["loss"] += step_loss.detach()
                for k in ("td", "fd", "env", "pe"):
                    sums[k] += comps[k].detach()

        pred = torch.cat(pred_chunks, dim=-1)

        if train:
            loss = sums["loss"] / n_sub
            comps = {k: sums[k] / n_sub for k in ("td", "fd", "env", "pe")}
        else:
            loss, comps = self._loss(pred, wet)

        with torch.no_grad():
            mae = F.l1_loss(pred, wet)
            mse = F.mse_loss(pred, wet)
            esr = esr_metric(wet, pred)
            rmse = rmse_metric(wet, pred)

        bs = dry.shape[0]
        self.log(f"loss/{phase}", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log(f"loss/{phase}_td", comps["td"], on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"loss/{phase}_fd", comps["fd"], on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"loss/{phase}_env", comps["env"], on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"loss/{phase}_pe", comps["pe"], on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"gain/{phase}_delta_db", delta_abs / n_sub, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"gain/{phase}_color", color_abs / n_sub, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"mae/{phase}", mae, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"mse/{phase}", mse, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"esr/{phase}", esr, on_step=False, on_epoch=True, prog_bar=(not train), batch_size=bs)
        self.log(f"rmse/{phase}", rmse, on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, "test")

    def on_train_epoch_end(self):
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

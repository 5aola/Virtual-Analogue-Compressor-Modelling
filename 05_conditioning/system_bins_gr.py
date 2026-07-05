"""Lightning system for the clean bins-head GR predictor (`model_bins_gr.BinsGRLSTM`).

The head-axis twin of `system_detector_gr.DetectorGRSystem`: identical
stateless 3 s crop regime, identical dry-energy-floor + warmup masking,
identical logging surface (loss/mae_db/delta_db). Only the loss changes with
the head — the predecessor bins recipe (`system.TFiLMGRBinsSystem`) stripped
of every training-dynamics patch:

    loss = BCE(logits, Gaussian soft targets)                 CREPE recipe
         + huber_weight · Huber(decoded dB, gr dB)            accuracy anchor
         + delta_weight · Huber(Δ decoded, Δ gr)              attack/release timing

REMOVED relative to `TFiLMGRBinsSystem` (the deconfounding point of this run):
LDS depth reweighting, conditioning dropout / null embedding, cold-start
mixing + pre-roll, TBPTT state stash. What remains is the bins head plus its
minimal loss, so any val→test difference vs the regression base is
attributable to the discretized output alone.

The Huber/Δ terms use the differentiable softmax-expectation decode
(`logits_to_expected_db`); the logged mae_db uses the robust local-average
decode (`logits_to_local_avg_db`) — the same decode `BinsGRLSTM.to_db` uses
at eval, so mae_db here matches the eval notebooks' metric.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

from gr_target import (
    gr_db_to_soft_target,
    logits_to_expected_db,
    logits_to_local_avg_db,
)


class BinsGRSystem(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        warmup_frames: int = 172,
        energy_floor_db: float = -60.0,
        sigma_bins: float = 2.0,
        huber_weight: float = 0.1,
        huber_beta_db: float = 1.0,
        delta_weight: float = 1.0,
        scheduler: str = "cosine",
        max_epochs: int = 150,
        eta_min: float = 1e-6,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.warmup_frames = warmup_frames
        self.energy_floor_db = energy_floor_db
        self.sigma_bins = sigma_bins
        self.huber_weight = huber_weight
        self.huber_beta_db = huber_beta_db
        self.delta_weight = delta_weight
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.eta_min = eta_min

    def forward(self, dry, params):
        return self.model(dry, params)

    def _step(self, batch, phase: str):
        dry, gr_db, params = batch
        logits = self.model(dry, params)                             # [B, bins, Tf]

        # exact hop alignment: trim to whole frames, then fixed-window pool
        Tf = logits.shape[-1]
        n = Tf * self.model.hop_size
        gr_frames = F.avg_pool1d(gr_db[..., :n], self.model.hop_size)
        frame_db = 10.0 * torch.log10(F.avg_pool1d(dry[..., :n] ** 2, self.model.hop_size) + 1e-12)

        valid = frame_db > self.energy_floor_db                      # [B, 1, Tf]
        if self.warmup_frames > 0:
            valid = valid.clone()
            valid[..., : self.warmup_frames] = False
        n_valid = valid.sum().clamp(min=1)

        def masked_huber(a, b, mask, m_sum):
            el = F.smooth_l1_loss(a, b, beta=self.huber_beta_db, reduction="none")
            return (el * mask).sum() / m_sum

        soft = gr_db_to_soft_target(gr_frames, self.sigma_bins)      # [B, bins, Tf]
        bce_el = F.binary_cross_entropy_with_logits(logits, soft, reduction="none")
        bce = (bce_el.mean(dim=1, keepdim=True) * valid).sum() / n_valid

        pred_db = logits_to_expected_db(logits)                      # differentiable decode
        huber = masked_huber(pred_db, gr_frames, valid, n_valid)

        vd = valid[..., 1:] & valid[..., :-1]
        dloss = masked_huber(
            pred_db[..., 1:] - pred_db[..., :-1],
            gr_frames[..., 1:] - gr_frames[..., :-1],
            vd,
            vd.sum().clamp(min=1),
        )

        loss = bce + self.huber_weight * huber + self.delta_weight * dloss

        with torch.no_grad():
            mae_db = ((logits_to_local_avg_db(logits) - gr_frames).abs() * valid).sum() / n_valid

        bs = dry.shape[0]
        self.log(f"loss/{phase}", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log(f"mae_db/{phase}", mae_db, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log(f"bce/{phase}", bce, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"huber_db/{phase}", huber, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"delta_db/{phase}", dloss, on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

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
        return opt

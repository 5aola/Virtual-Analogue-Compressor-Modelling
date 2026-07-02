"""Lightning system for the blackbox detector-LSTM GR predictor.

Replaces `TFiLMGRBinsSystem` (bins BCE + LDS reweighting + conditioning
dropout + per-phase state stash/restore) with a direct dB regression on
stateless 3 s crops:

    loss = Huber(pred_db, gr_db)                       accuracy
         + delta_weight · Huber(Δpred, Δgr)            attack/release timing

All terms are masked by the dry-energy floor (GR labels in silence are noise —
kept from the bins recipe) and by a crop-start warmup: the model starts every
crop from silence context + empty state while the label carries true
pre-history, so the first ~1 release time is irreducibly ambiguous and
excluded. Fresh state every batch == the deployment cold-start condition; no
TBPTT, no cold-start dataset mixing, no state bookkeeping.

`system_dsp_prior.DSPPriorGRSystem` is the greybox-ablation twin (adds a
prior-anchor loss term for the DSP path).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule


class DetectorGRSystem(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        warmup_frames: int = 172,
        energy_floor_db: float = -60.0,
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
        self.huber_beta_db = huber_beta_db
        self.delta_weight = delta_weight
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.eta_min = eta_min

    def forward(self, dry, params):
        return self.model(dry, params)

    def _step(self, batch, phase: str):
        dry, gr_db, params = batch
        pred = self.model(dry, params)

        # exact hop alignment: trim to whole frames, then fixed-window pool
        Tf = pred.shape[-1]
        n = Tf * self.model.hop_size
        gr_frames = F.avg_pool1d(gr_db[..., :n], self.model.hop_size)
        frame_db = 10.0 * torch.log10(F.avg_pool1d(dry[..., :n] ** 2, self.model.hop_size) + 1e-12)

        valid = frame_db > self.energy_floor_db
        if self.warmup_frames > 0:
            valid = valid.clone()
            valid[..., : self.warmup_frames] = False
        n_valid = valid.sum().clamp(min=1)

        def masked_huber(a, b, mask, m_sum):
            el = F.smooth_l1_loss(a, b, beta=self.huber_beta_db, reduction="none")
            return (el * mask).sum() / m_sum

        huber = masked_huber(pred, gr_frames, valid, n_valid)

        vd = valid[..., 1:] & valid[..., :-1]
        dloss = masked_huber(
            pred[..., 1:] - pred[..., :-1],
            gr_frames[..., 1:] - gr_frames[..., :-1],
            vd,
            vd.sum().clamp(min=1),
        )

        loss = huber + self.delta_weight * dloss

        with torch.no_grad():
            mae_db = ((pred - gr_frames).abs() * valid).sum() / n_valid

        bs = dry.shape[0]
        self.log(f"loss/{phase}", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log(f"mae_db/{phase}", mae_db, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
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

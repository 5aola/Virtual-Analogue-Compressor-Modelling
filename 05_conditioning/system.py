"""Lightning training system for TFiLM-conditioned GR LSTM."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

from gr_target import (
    GR_DB_MAX,
    GR_DB_MIN,
    gr_db_to_soft_target,
    logits_to_expected_db,
    logits_to_local_avg_db,
    normalize_gr_01,
)


def _as_bool(value) -> bool:
    if torch.is_tensor(value):
        return bool(value.item())
    return bool(value)


def _as_int(value) -> int:
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


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
        dry, gr_db, params, mask, is_reset = batch[:5]
        is_reset = _as_bool(is_reset)
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


class TFiLMGRBinsSystem(TFiLMGRSystem):
    """Discretized-head variant with a composite, dry-energy-masked loss:

        loss = BCE(bins, Gaussian soft targets)            CREPE recipe from 03
             + huber_weight * Huber(decoded dB, target dB)  direct regression
             + delta_weight * Huber(Δ decoded, Δ target)    attack/release timing

    All terms only count frames whose DRY input is above `energy_floor_db`
    (RMS dBFS) — GR is a ratio of tiny RMS values in silence/fades and those
    labels are noise. The Huber/delta terms use the differentiable softmax
    expectation decode; logged mae_db uses the robust local-average decode.

    Optional extras:
    - `depth_weights` [NUM_BINS]: LDS-style inverse-density weights (from
      dataset.compute_depth_weights); every loss term is weighted by the
      weight of its target's depth bin — counters regression-to-the-mean on
      the rare deep-GR tail.
    - `cond_dropout`: per-stream probability (train only, sampled at stream
      start, held for the whole epoch pass) of swapping the knob embedding
      for the model's learned null embedding — CFG-style conditioning dropout.
      Requires a model whose forward accepts `cond_drop` (StatefulCondLSTMGRBins).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        warmup_frames: int = 4,
        sigma_bins: float = 2.0,
        energy_floor_db: float = -60.0,
        huber_weight: float = 0.1,
        huber_beta_db: float = 1.0,
        delta_weight: float = 1.0,
        depth_weights: torch.Tensor | None = None,
        cond_dropout: float = 0.0,
        scheduler: str = "cosine",
        max_epochs: int = 800,
        eta_min: float = 1e-6,
        lr_patience: int = 20,
        min_lr: float = 1e-6,
    ):
        super().__init__(
            model=model,
            lr=lr,
            warmup_frames=warmup_frames,
            scheduler=scheduler,
            max_epochs=max_epochs,
            eta_min=eta_min,
            lr_patience=lr_patience,
            min_lr=min_lr,
        )
        self.sigma_bins = sigma_bins
        self.energy_floor_db = energy_floor_db
        self.huber_weight = huber_weight
        self.huber_beta_db = huber_beta_db
        self.delta_weight = delta_weight
        self.cond_dropout = cond_dropout
        self.register_buffer(
            "depth_weights",
            torch.ones(1) if depth_weights is None else depth_weights.float(),
        )
        self._cond_drop: dict[str, torch.Tensor | None] = {"train": None, "val": None, "test": None}

    def reset_phase_states(self, phase: str) -> None:
        super().reset_phase_states(phase)
        self._cond_drop[phase] = None

    def _depth_weight(self, gr_frames: torch.Tensor) -> torch.Tensor:
        n = self.depth_weights.numel()
        if n == 1:
            return torch.ones_like(gr_frames)
        idx = (
            ((gr_frames - GR_DB_MIN) / (GR_DB_MAX - GR_DB_MIN) * (n - 1))
            .round().clamp(0, n - 1).long()
        )
        return self.depth_weights[idx]

    def _step(self, batch, phase: str):
        dry, gr_db, params, mask, is_reset = batch[:5]
        is_reset = _as_bool(is_reset)
        batch_warmup_frames = None
        is_cold_start = False
        if len(batch) >= 7:
            batch_warmup_frames = _as_int(batch[5])
            is_cold_start = _as_bool(batch[6])

        if is_reset and not is_cold_start:
            self.reset_phase_states(phase)
            if phase == "train" and self.cond_dropout > 0:
                self._cond_drop[phase] = torch.rand(dry.shape[0], device=dry.device) < self.cond_dropout

        tfilm = getattr(self.model, "tfilm", None)
        if tfilm is not None and not is_cold_start:
            tfilm.hidden_state = self._tfilm_states[phase]
        elif tfilm is not None:
            tfilm.hidden_state = None

        cond_drop = self._cond_drop[phase]
        if is_cold_start:
            cond_drop = None
            if phase == "train" and self.cond_dropout > 0:
                cond_drop = torch.rand(dry.shape[0], device=dry.device) < self.cond_dropout
        kwargs = {"cond_drop": cond_drop} if cond_drop is not None else {}
        logits, lstm_state = self.model(
            dry,
            params,
            None if is_cold_start else self._lstm_states[phase],
            return_state=True,
            **kwargs,
        )
        if not is_cold_start:
            self._lstm_states[phase] = tuple(s.detach() for s in lstm_state)
        if tfilm is not None and not is_cold_start:
            self._tfilm_states[phase] = tuple(s.detach() for s in tfilm.hidden_state)
        elif tfilm is not None:
            tfilm.hidden_state = None

        Tf = logits.shape[-1]
        gr_frames = F.adaptive_avg_pool1d(gr_db, Tf)                    # [B,1,Tf] dB
        soft = gr_db_to_soft_target(gr_frames, self.sigma_bins)        # [B,bins,Tf]

        # dry-energy floor: frame RMS dBFS; silent frames have noise GR labels
        frame_db = 10.0 * torch.log10(F.adaptive_avg_pool1d(dry**2, Tf) + 1e-12)
        valid = (frame_db > self.energy_floor_db) & mask.view(-1, 1, 1).bool()

        warmup_frames = (
            batch_warmup_frames
            if batch_warmup_frames is not None
            else self.warmup_frames if is_reset else 0
        )
        if warmup_frames > 0:
            w = min(warmup_frames, max(Tf - 1, 0))
            logits, soft = logits[..., w:], soft[..., w:]
            gr_frames, valid = gr_frames[..., w:], valid[..., w:]

        n_valid = valid.sum().clamp(min=1)
        wv = self._depth_weight(gr_frames) * valid                     # [B,1,Tf]
        wv_sum = wv.sum().clamp(min=1.0)

        bce_el = F.binary_cross_entropy_with_logits(logits, soft, reduction="none")
        bce = (bce_el * wv).sum() / (wv_sum * logits.shape[1])

        pred_db = logits_to_expected_db(logits)                        # differentiable
        huber_el = F.smooth_l1_loss(pred_db, gr_frames, beta=self.huber_beta_db, reduction="none")
        huber = (huber_el * wv).sum() / wv_sum

        wd = wv[..., 1:] * valid[..., :-1]
        delta_el = F.smooth_l1_loss(
            pred_db[..., 1:] - pred_db[..., :-1],
            gr_frames[..., 1:] - gr_frames[..., :-1],
            beta=self.huber_beta_db,
            reduction="none",
        )
        delta = (delta_el * wd).sum() / wd.sum().clamp(min=1.0)

        loss = bce + self.huber_weight * huber + self.delta_weight * delta

        with torch.no_grad():
            mae_db = ((logits_to_local_avg_db(logits) - gr_frames).abs() * valid).sum() / n_valid

        bs = int(mask.sum())
        self.log(f"loss/{phase}", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log(f"mae_db/{phase}", mae_db, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log(f"bce/{phase}", bce, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"huber_db/{phase}", huber, on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"delta_db/{phase}", delta, on_step=False, on_epoch=True, batch_size=bs)
        return loss

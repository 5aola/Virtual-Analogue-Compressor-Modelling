"""
Train a simple (non-parametric) TCN to predict the gain-reduction envelope
of the SSL G-Bus compressor from dry audio input.

Model:  nablafx TCN  (no conditioning, num_controls=0)
Input:  dry audio          [B, 1, T]
Target: normalised GR (dB) [B, 1, T]   (from 1024-sample windowed RMS)
Loss:   Smooth L1 + temporal-difference (attack/release dynamics)

Usage:
    python C_initial_GR_pred/train_gr.py                          # defaults
    python C_initial_GR_pred/train_gr.py --max_epochs 200         # override
    python C_initial_GR_pred/train_gr.py --data_root /path/to/data --batch_size 8
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nablafx.processors import TCN

from C_initial_GR_pred.gr_dataset import (
    GainReductionDataModule,
    DEFAULT_DATA_ROOT,
    DEFAULT_SETTING,
    SAMPLE_LENGTH,
    SAMPLE_RATE,
)
from src.dsp_torch import RMS_WINDOW, denormalize_gr


# ---------------------------------------------------------------------------
# Lightning system
# ---------------------------------------------------------------------------


class GRPredictionSystem(pl.LightningModule):
    """Minimal Lightning module for GR-envelope prediction."""

    def __init__(
        self,
        processor: torch.nn.Module,
        lr: float = 1e-3,
        smooth_l1_beta: float = 0.5,
        diff_weight: float = 0.1,
        diff_beta: float = 0.1,
    ):
        super().__init__()
        self.processor = processor
        self.lr = lr
        self.smooth_l1_beta = smooth_l1_beta
        self.diff_weight = diff_weight
        self.diff_beta = diff_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.processor(x)

    def _step(self, batch: tuple, mode: str) -> torch.Tensor:
        dry, gr_target = batch
        if hasattr(self.processor, "reset_states"):
            self.processor.reset_states()
        gr_pred = self(dry)

        main_loss = F.smooth_l1_loss(gr_pred, gr_target, beta=self.smooth_l1_beta)

        dp = gr_pred[..., 1:] - gr_pred[..., :-1]
        dt = gr_target[..., 1:] - gr_target[..., :-1]
        diff_loss = F.smooth_l1_loss(dp, dt, beta=self.diff_beta)

        loss = main_loss + self.diff_weight * diff_loss

        self.log(f"loss/{mode}", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"loss/{mode}_main", main_loss, on_step=False, on_epoch=True)
        self.log(f"loss/{mode}_diff", diff_loss, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=10, factor=0.5
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "loss/val",
                "interval": "epoch",
            },
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GR-prediction TCN")

    # data
    p.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    p.add_argument("--settings_folder", type=str, default=DEFAULT_SETTING)
    p.add_argument("--sample_length", type=int, default=SAMPLE_LENGTH)
    p.add_argument("--sample_rate", type=int, default=SAMPLE_RATE)
    p.add_argument("--rms_window", type=int, default=RMS_WINDOW)
    p.add_argument("--train_split", type=float, default=0.8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)

    # model
    p.add_argument("--num_blocks", type=int, default=10)
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--channel_width", type=int, default=32)
    p.add_argument("--dilation_growth", type=int, default=2)
    p.add_argument("--causal", action="store_true", default=True)

    # loss
    p.add_argument("--smooth_l1_beta", type=float, default=0.5)
    p.add_argument("--diff_weight", type=float, default=0.1)
    p.add_argument("--diff_beta", type=float, default=0.1)

    # training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--accelerator", type=str, default="mps")
    p.add_argument("--precision", type=str, default="32-true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- data ----
    dm = GainReductionDataModule(
        data_root=args.data_root,
        settings_folder=args.settings_folder,
        sample_length=args.sample_length,
        sample_rate=args.sample_rate,
        rms_window=args.rms_window,
        train_split=args.train_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ---- model ----
    tcn = TCN(
        num_inputs=1,
        num_outputs=1,
        num_controls=0,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        dilation_growth=args.dilation_growth,
        channel_width=args.channel_width,
        causal=args.causal,
        cond_type=None,
        bias=False,
        batchnorm=True,
    )
    print(f"TCN receptive field: {tcn.rf} samples ({tcn.rf / args.sample_rate:.3f} s)")

    system = GRPredictionSystem(
        processor=tcn,
        lr=args.lr,
        smooth_l1_beta=args.smooth_l1_beta,
        diff_weight=args.diff_weight,
        diff_beta=args.diff_beta,
    )

    # ---- callbacks ----
    ckpt_cb = ModelCheckpoint(
        monitor="loss/val",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch}-{step}",
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    # ---- trainer ----
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        precision=args.precision,
        callbacks=[ckpt_cb, lr_cb],
        log_every_n_steps=10,
        default_root_dir="C_initial_GR_pred/lightning_logs",
    )

    trainer.fit(system, dm)


if __name__ == "__main__":
    main()

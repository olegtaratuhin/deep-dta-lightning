from __future__ import annotations

from typing import Any, Iterable, Protocol, TypedDict

import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch import LightningModule
from torchmetrics import MaxMetric, MeanMetric

from src.models.components import dta, metrics


class StepResult(TypedDict):
    loss: torch.Tensor
    preds: torch.Tensor
    targets: torch.Tensor


class OptimzerFactory(Protocol):
    def __call__(self, params: Iterable[torch.Tensor]) -> torch.optim.Optimizer:
        raise NotImplementedError()


class SchedulerFactory(Protocol):
    def __call__(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        raise NotImplementedError()


DTABatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _binarize(
    threshold: float,
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.sigmoid(preds - threshold), ((targets - threshold) > 0).int()


class DeepDTAModule(LightningModule):
    """DeepDTA model for affinity prediction.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: dta.DeepDTA,
        optimizer: OptimzerFactory,
        scheduler: SchedulerFactory | None = None,
        lr: float = 1e-3,
        classification_threshold: float | None = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt, note that model is already
        # saved in checkpoint, so we can ignore it
        self.save_hyperparameters(logger=False, ignore=["model"])

        self.model = model

        # we need to have this explicitly to be able to use auto-lr-finder callback
        self.lr = lr

        # optional threshold to calculate classification metrics
        self._threshold = classification_threshold

        # loss function
        self.criterion = nn.MSELoss()

        # concordance index
        self.train_cindex = metrics.CIndex()
        self.val_cindex = metrics.CIndex()
        self.test_cindex = metrics.CIndex()

        # R2
        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()
        self.test_r2 = torchmetrics.R2Score()

        # MAPE
        self.train_mape = torchmetrics.MeanAbsolutePercentageError()
        self.val_mape = torchmetrics.MeanAbsolutePercentageError()
        self.test_mape = torchmetrics.MeanAbsolutePercentageError()

        # AUPRC
        # although we do regression many papers have this metric
        # for this we need a dataset-specific threshold
        self.train_auprc = torchmetrics.AveragePrecision("binary")
        self.val_auprc = torchmetrics.AveragePrecision("binary")
        self.test_auprc = torchmetrics.AveragePrecision("binary")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation metric
        self.val_metric_best = MaxMetric()

    def forward(
        self,
        batch: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x_proteins, x_ligands, *_ = batch
        return self.model.forward(x_proteins=x_proteins, x_ligands=x_ligands)

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_cindex.reset()
        self.val_r2.reset()
        self.val_mape.reset()
        self.val_auprc.reset()
        self.val_metric_best.reset()

    def model_step(
        self,
        batch: DTABatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_protein, x_ligand, y_affinity = batch
        preds = self.forward(x_proteins=x_protein, x_ligands=x_ligand)
        loss = self.criterion(preds, y_affinity)
        return loss, preds, y_affinity

    def training_step(
        self,
        batch: DTABatch,
        batch_idx: int,
    ) -> StepResult:
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss.detach())
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        preds_view, targets_view = preds.detach(), targets.detach()

        self.train_cindex.update(preds_view, targets_view)
        self.log("train/cindex", self.train_cindex, on_step=True, on_epoch=True, prog_bar=True)

        self.train_r2.update(preds_view, targets_view)
        self.log("train/r2", self.train_r2, on_step=True, on_epoch=True, prog_bar=True)

        self.train_mape.update(preds_view, targets_view)
        self.log("train/mape", self.train_mape, on_step=True, on_epoch=True, prog_bar=True)

        if self._threshold is not None:
            self.train_auprc.update(*_binarize(self._threshold, preds_view, targets_view))
            self.log("train/auprc", self.train_auprc, on_step=True, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(
        self,
        batch: DTABatch,
        batch_idx: int,
    ) -> StepResult:
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)

        preds_view, targets_view = preds.detach(), targets.detach()

        self.val_cindex.update(preds_view, targets_view)
        self.log("val/cindex", self.val_cindex, on_step=True, on_epoch=True, prog_bar=True)

        self.val_r2.update(preds_view, targets_view)
        self.log("val/r2", self.val_r2, on_step=True, on_epoch=True, prog_bar=True)

        self.val_mape.update(preds_view, targets_view)
        self.log("val/mape", self.val_mape, on_step=True, on_epoch=True, prog_bar=True)

        if self._threshold is not None:
            self.val_auprc.update(*_binarize(self._threshold, preds_view, targets_view))
            self.log("val/auprc", self.val_auprc, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self) -> None:
        cindex = self.val_cindex.compute()  # get current val cindex
        self.val_metric_best(cindex)  # update best so far val cindex
        # log `val_metric_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/cindex_best", self.val_metric_best.compute(), prog_bar=True)

    def test_step(self, batch: DTABatch, batch_idx: int) -> StepResult:
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss.detach())
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        preds_view, targets_view = preds.detach(), targets.detach()

        self.test_cindex.update(preds_view, targets_view)
        self.log("test/cindex", self.test_cindex, on_step=False, on_epoch=True, prog_bar=True)

        self.test_r2.update(preds_view, targets_view)
        self.log("test/r2", self.test_r2, on_step=False, on_epoch=True, prog_bar=True)

        self.test_mape.update(preds_view, targets_view)
        self.log("test/mape", self.test_mape, on_step=False, on_epoch=True, prog_bar=True)

        if self._threshold is not None:
            self.test_auprc.update(*_binarize(self._threshold, preds_view, targets_view))
            self.log("test/auprc", self.test_auprc, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_test_epoch_end(self) -> None:
        pass

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters(), lr=self.lr)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

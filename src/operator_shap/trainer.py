"""Training loop with early stopping and best-checkpoint selection on the validation loss.

Single process. Losses per epoch go to ``train_log.csv`` in the run directory, the best model to ``best_model.pt``.
"""

import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    """Train ``model`` on dict batches; subclasses change the loss by overriding :meth:`_run_batch`.

    Parameters
    ----------
    model : nn.Module
        Model called on the batch dict.
    train_loader, val_loader : DataLoader
        Loaders of dict batches with the target at ``"y_true"``.
    criterion, val_criterion : nn.Module
        Training and validation loss.
    optimizer : torch.optim.Optimizer
    scheduler : torch.optim.lr_scheduler.LRScheduler
        Stepped once per epoch.
    config : DictConfig
        Full config; reads ``device``, ``out_dir``, and ``training``.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        val_criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.out_dir = Path(config.out_dir)
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.val_criterion = val_criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.amp_enabled = config.training.amp.enabled
        self.amp_dtype = getattr(torch, config.training.amp.dtype)
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def train(self) -> None:
        """Train until ``training.n_epochs`` or until the validation loss stalls for ``training.early_stopping_patience`` epochs."""
        log = []
        for epoch in range(self.config.training.n_epochs):
            metrics = {"epoch": epoch, **self.train_epoch(epoch), **self.evaluate(epoch), "lr": self.optimizer.param_groups[0]["lr"]}
            log.append(metrics)
            pd.DataFrame(log).to_csv(self.out_dir / "train_log.csv", index=False)
            logger.info(" ".join(f"{k}={v:.5g}" for k, v in metrics.items()))

            self.scheduler.step()

            if metrics["val_loss"] < self.best_val_loss - self.config.training.early_stopping_min_delta:
                self.best_val_loss = metrics["val_loss"]
                self.epochs_without_improvement = 0
                self.save_checkpoint(self.out_dir / "best_model.pt", epoch)
            else:
                self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.config.training.early_stopping_patience:
                logger.info("Early stopping triggered")
                break

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """One pass over the training loader; returns the mean batch loss."""
        self.model.train()
        total_loss = 0.0
        for data in tqdm(self.train_loader, desc=f"Training Epoch {epoch}"):
            batch_loss = self._run_batch(data)
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            total_loss += batch_loss.item()
        return {"train_loss": total_loss / len(self.train_loader)}

    @torch.no_grad()
    def evaluate(self, epoch: int) -> dict[str, float]:
        """Mean validation batch loss."""
        self.model.eval()
        total_loss = sum(self._run_batch(data, crit="val").item() for data in tqdm(self.val_loader, desc=f"Validation Epoch {epoch}"))
        return {"val_loss": total_loss / len(self.val_loader)}

    def _run_batch(self, data: dict[str, torch.Tensor], crit: str = "train") -> torch.Tensor:
        """Loss of one batch: the model output against ``data["y_true"]``."""
        criterion = self.criterion if crit == "train" else self.val_criterion
        data = {k: v.to(self.device) for k, v in data.items() if isinstance(v, torch.Tensor)}
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            return criterion(self.model(data), data["y_true"])

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        """Model weights and the full config, which :func:`operator_shap.backbone.load_model` rebuilds the model from."""
        torch.save({"model_state_dict": self.model.state_dict(), "config": self.config, "epoch": epoch, "best_val_loss": self.best_val_loss}, path)

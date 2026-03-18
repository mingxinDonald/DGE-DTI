"""Training loop for DGE-DTI models."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .metrics import compute_metrics, format_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """Manages the training and evaluation of a DTI model.

    Args:
        model: PyTorch model with a forward(drug_features, target_features) signature.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularisation coefficient.
        device: Torch device string (e.g., ``"cpu"``, ``"cuda"``).
        model_dir: Directory path for saving checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        model_dir: str = "outputs/models",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", patience=5, factor=0.5, verbose=True
        )
        self.criterion = nn.BCELoss()
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, np.ndarray, np.ndarray]:
        """Run one epoch over a DataLoader.

        Args:
            loader: DataLoader to iterate over.
            train: If True, updates model weights; otherwise runs in eval mode.

        Returns:
            Tuple of (mean_loss, all_labels, all_predictions).
        """
        self.model.train(train)
        total_loss = 0.0
        all_labels: list = []
        all_preds: list = []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                drug_feat = batch["drug_features"].to(self.device)
                target_feat = batch["target_features"].to(self.device)
                labels = batch["label"].to(self.device)

                preds = self.model(drug_feat, target_feat).squeeze(1)
                loss = self.criterion(preds, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * len(labels)
                all_labels.extend(labels.cpu().numpy().tolist())
                all_preds.extend(preds.detach().cpu().numpy().tolist())

        mean_loss = total_loss / max(len(all_labels), 1)
        return mean_loss, np.array(all_labels), np.array(all_preds)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 15,
    ) -> Dict[str, Any]:
        """Train the model with early stopping based on validation AUROC.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs: Maximum number of training epochs.
            early_stopping_patience: Number of epochs without improvement before stopping.

        Returns:
            Dictionary with training history (loss and metrics per epoch).
        """
        history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_auroc": [],
        }
        best_val_auroc = -1.0
        patience_counter = 0
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            train_loss, _, _ = self._run_epoch(train_loader, train=True)
            val_loss, val_labels, val_preds = self._run_epoch(val_loader, train=False)
            val_metrics = compute_metrics(val_labels, val_preds)
            val_auroc = val_metrics.get("auroc", float("nan"))

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auroc"].append(val_auroc)

            if not np.isnan(val_auroc):
                self.scheduler.step(val_auroc)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | %s",
                epoch,
                epochs,
                train_loss,
                val_loss,
                format_metrics(val_metrics),
            )

            if not np.isnan(val_auroc) and val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint("best_model.pt")
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping at epoch %d (best epoch: %d).", epoch, best_epoch)
                break

        logger.info("Training complete. Best val AUROC=%.4f at epoch %d.", best_val_auroc, best_epoch)
        return history

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on a DataLoader.

        Args:
            loader: DataLoader to evaluate on.

        Returns:
            Dictionary of evaluation metrics.
        """
        _, labels, preds = self._run_epoch(loader, train=False)
        return compute_metrics(labels, preds)

    def save_checkpoint(self, filename: str = "model.pt") -> str:
        """Save model state dict to disk.

        Args:
            filename: Filename (relative to ``model_dir``).

        Returns:
            Full path to the saved checkpoint.
        """
        path = os.path.join(self.model_dir, filename)
        torch.save(self.model.state_dict(), path)
        logger.info("Saved checkpoint to %s", path)
        return path

    def load_checkpoint(self, filename: str = "best_model.pt") -> None:
        """Load model state dict from disk.

        Args:
            filename: Filename (relative to ``model_dir``).
        """
        path = os.path.join(self.model_dir, filename)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info("Loaded checkpoint from %s", path)

"""Evaluation metrics for Drug-Target Interaction prediction."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute a comprehensive set of binary classification metrics.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_pred_proba: Predicted probabilities in [0, 1].
        threshold: Decision threshold to convert probabilities to labels.

    Returns:
        Dictionary mapping metric names to scalar values.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics: Dict[str, float] = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_pred_proba))
        metrics["auprc"] = float(average_precision_score(y_true, y_pred_proba))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")

    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format a metrics dictionary as a human-readable string.

    Args:
        metrics: Dictionary of metric name → value pairs.

    Returns:
        Formatted string suitable for logging.
    """
    parts = [f"{name}: {value:.4f}" for name, value in metrics.items()]
    return " | ".join(parts)

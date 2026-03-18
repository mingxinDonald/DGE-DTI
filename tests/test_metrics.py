"""Tests for training metrics."""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training.metrics import compute_metrics, format_metrics


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.1, 0.9, 0.9])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert abs(metrics["auroc"] - 1.0) < 1e-6

    def test_all_wrong_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.9, 0.9, 0.1, 0.1])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0
        assert metrics["auroc"] == 0.0

    def test_single_class_returns_nan_auroc(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.9, 0.8, 0.7, 0.6])
        metrics = compute_metrics(y_true, y_pred)
        assert np.isnan(metrics["auroc"])
        assert np.isnan(metrics["auprc"])

    def test_metric_keys_present(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0.3, 0.7])
        metrics = compute_metrics(y_true, y_pred)
        for key in ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]:
            assert key in metrics

    def test_threshold_effect(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0.4, 0.6])
        metrics_05 = compute_metrics(y_true, y_pred, threshold=0.5)
        metrics_07 = compute_metrics(y_true, y_pred, threshold=0.7)
        # With threshold 0.7 both predictions become 0, so recall should be 0
        assert metrics_07["recall"] == 0.0


class TestFormatMetrics:
    def test_returns_string(self):
        metrics = {"accuracy": 0.95, "auroc": 0.98}
        result = format_metrics(metrics)
        assert isinstance(result, str)

    def test_contains_metric_names(self):
        metrics = {"accuracy": 0.95, "auroc": 0.98}
        result = format_metrics(metrics)
        assert "accuracy" in result
        assert "auroc" in result

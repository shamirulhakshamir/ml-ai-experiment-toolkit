"""Tests for ModelEvaluator."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model_evaluator import ModelEvaluator


class TestClassificationEvaluator:
    def setup_method(self):
        self.evaluator = ModelEvaluator(task="classification")
        self.y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
        self.y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

    def test_evaluate_returns_all_metrics(self):
        results = self.evaluator.evaluate(self.y_true, self.y_pred)
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results

    def test_evaluate_accuracy(self):
        results = self.evaluator.evaluate(self.y_true, self.y_pred)
        assert results["accuracy"] == 0.8

    def test_evaluate_perfect_predictions(self):
        y = [1, 0, 1, 0]
        results = self.evaluator.evaluate(y, y)
        assert results["accuracy"] == 1.0
        assert results["f1"] == 1.0

    def test_evaluate_with_proba(self):
        y_true = [1, 0, 1, 1, 0]
        y_proba = [0.9, 0.1, 0.8, 0.7, 0.3]
        results = self.evaluator.evaluate_with_proba(y_true, y_proba, threshold=0.5)
        assert "auc" in results
        assert results["auc"] >= 0.0

    def test_threshold_analysis(self):
        y_true = [1, 0, 1, 1, 0, 0]
        y_proba = [0.9, 0.2, 0.7, 0.6, 0.4, 0.1]
        results = self.evaluator.threshold_analysis(y_true, y_proba)
        assert len(results) == 5  # default 5 thresholds
        assert results[0]["threshold"] == 0.3

    def test_compare_models(self):
        y_true = [1, 0, 1, 0, 1]
        preds = {
            "model_a": [1, 0, 1, 0, 1],
            "model_b": [1, 1, 0, 0, 1],
        }
        results = self.evaluator.compare_models(y_true, preds)
        assert "model_a" in results
        assert "model_b" in results
        # model_a is perfect so should be first
        assert list(results.keys())[0] == "model_a"

    def test_custom_metric(self):
        custom = {"always_one": lambda y, p: 1.0}
        ev = ModelEvaluator(task="classification", custom_metrics=custom)
        results = ev.evaluate(self.y_true, self.y_pred)
        assert results["always_one"] == 1.0


class TestRegressionEvaluator:
    def setup_method(self):
        self.evaluator = ModelEvaluator(task="regression")
        self.y_true = [3.0, -0.5, 2.0, 7.0]
        self.y_pred = [2.5, 0.0, 2.0, 8.0]

    def test_evaluate_returns_all_metrics(self):
        results = self.evaluator.evaluate(self.y_true, self.y_pred)
        assert "mse" in results
        assert "rmse" in results
        assert "mae" in results
        assert "r2" in results

    def test_evaluate_perfect_predictions(self):
        results = self.evaluator.evaluate(self.y_true, self.y_true)
        assert results["mse"] == 0.0
        assert results["r2"] == 1.0

    def test_r2_reasonable(self):
        results = self.evaluator.evaluate(self.y_true, self.y_pred)
        assert results["r2"] > 0.9

    def test_proba_raises_for_regression(self):
        with pytest.raises(ValueError):
            self.evaluator.evaluate_with_proba([1, 0], [0.5, 0.5])

    def test_threshold_raises_for_regression(self):
        with pytest.raises(ValueError):
            self.evaluator.threshold_analysis([1, 0], [0.5, 0.5])


class TestInvalidTask:
    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            ModelEvaluator(task="clustering")

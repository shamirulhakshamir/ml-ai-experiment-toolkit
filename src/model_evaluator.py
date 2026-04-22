"""
Model Evaluator — ML model evaluation framework.

Provides utilities to evaluate classification and regression models,
compare multiple models, and generate evaluation reports.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


class ModelEvaluator:
    """Evaluate ML models with standard and custom metrics."""

    CLASSIFICATION_METRICS = {
        "accuracy": accuracy_score,
        "precision": lambda y, p: precision_score(y, p, zero_division=0),
        "recall": lambda y, p: recall_score(y, p, zero_division=0),
        "f1": lambda y, p: f1_score(y, p, zero_division=0),
    }

    REGRESSION_METRICS = {
        "mse": mean_squared_error,
        "rmse": lambda y, p: float(np.sqrt(mean_squared_error(y, p))),
        "mae": mean_absolute_error,
        "r2": r2_score,
    }

    def __init__(self, task="classification", custom_metrics=None):
        """
        Initialize the evaluator.

        Parameters
        ----------
        task : str
            Either 'classification' or 'regression'.
        custom_metrics : dict, optional
            Mapping of metric_name -> callable(y_true, y_pred).
        """
        if task not in ("classification", "regression"):
            raise ValueError(f"Unknown task: {task}. Use 'classification' or 'regression'.")
        self.task = task
        self.metrics = dict(
            self.CLASSIFICATION_METRICS if task == "classification" else self.REGRESSION_METRICS
        )
        if custom_metrics:
            self.metrics.update(custom_metrics)

    def evaluate(self, y_true, y_pred):
        """
        Evaluate predictions against ground truth.

        Returns
        -------
        dict
            Metric name -> score.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        results = {}
        for name, fn in self.metrics.items():
            results[name] = round(fn(y_true, y_pred), 6)
        return results

    def evaluate_with_proba(self, y_true, y_proba, threshold=0.5):
        """
        Evaluate classification with probability scores.

        Parameters
        ----------
        y_true : array-like
            True binary labels.
        y_proba : array-like
            Predicted probabilities for the positive class.
        threshold : float
            Decision threshold for converting probabilities to labels.

        Returns
        -------
        dict
            Includes standard metrics plus AUC.
        """
        if self.task != "classification":
            raise ValueError("evaluate_with_proba is only for classification tasks.")
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        y_pred = (y_proba >= threshold).astype(int)
        results = self.evaluate(y_true, y_pred)
        results["auc"] = round(roc_auc_score(y_true, y_proba), 6)
        return results

    def compare_models(self, y_true, predictions_dict):
        """
        Compare multiple models' predictions.

        Parameters
        ----------
        y_true : array-like
            Ground truth labels/values.
        predictions_dict : dict
            Mapping of model_name -> y_pred array.

        Returns
        -------
        dict
            model_name -> metrics dict, sorted by primary metric (descending).
        """
        results = {}
        for model_name, y_pred in predictions_dict.items():
            results[model_name] = self.evaluate(y_true, y_pred)

        # Sort by primary metric
        primary = "f1" if self.task == "classification" else "r2"
        reverse = True  # Higher is better for both f1 and r2
        sorted_results = dict(
            sorted(results.items(), key=lambda x: x[1].get(primary, 0), reverse=reverse)
        )
        return sorted_results

    def threshold_analysis(self, y_true, y_proba, thresholds=None):
        """
        Analyze classification metrics across multiple thresholds.

        Parameters
        ----------
        y_true : array-like
            True binary labels.
        y_proba : array-like
            Predicted probabilities.
        thresholds : list of float, optional
            Thresholds to evaluate. Defaults to [0.3, 0.4, 0.5, 0.6, 0.7].

        Returns
        -------
        list of dict
            Each entry has 'threshold' and metric scores.
        """
        if self.task != "classification":
            raise ValueError("threshold_analysis is only for classification tasks.")
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        results = []
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            metrics = self.evaluate(y_true, y_pred)
            metrics["threshold"] = t
            results.append(metrics)
        return results

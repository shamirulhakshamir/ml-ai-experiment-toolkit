"""
Experiment Dashboard — Experiment tracking data layer.

Provides logging, comparison, and export of ML experiment runs
for downstream dashboard consumption.
"""

import json
import os
from datetime import datetime, timezone


class ExperimentRun:
    """A single experiment run with parameters, metrics, and metadata."""

    def __init__(self, run_id, experiment_name, parameters=None, tags=None):
        """
        Parameters
        ----------
        run_id : str
            Unique identifier for this run.
        experiment_name : str
            Name of the parent experiment.
        parameters : dict, optional
            Hyperparameters or configuration for this run.
        tags : dict, optional
            Arbitrary key-value tags.
        """
        self.run_id = run_id
        self.experiment_name = experiment_name
        self.parameters = parameters or {}
        self.tags = tags or {}
        self.metrics = {}
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.status = "running"

    def log_metric(self, name, value):
        """Log a single metric."""
        self.metrics[name] = value

    def log_metrics(self, metrics_dict):
        """Log multiple metrics at once."""
        self.metrics.update(metrics_dict)

    def complete(self):
        """Mark the run as completed."""
        self.status = "completed"
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def fail(self, error_message=""):
        """Mark the run as failed."""
        self.status = "failed"
        self.error_message = error_message
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self):
        """Serialize the run to a dictionary."""
        d = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "tags": self.tags,
            "status": self.status,
            "created_at": self.created_at,
        }
        if hasattr(self, "completed_at"):
            d["completed_at"] = self.completed_at
        if hasattr(self, "error_message"):
            d["error_message"] = self.error_message
        return d


class ExperimentTracker:
    """Track and compare multiple experiment runs."""

    def __init__(self):
        self.runs = {}

    def create_run(self, run_id, experiment_name, parameters=None, tags=None):
        """
        Create and register a new experiment run.

        Returns
        -------
        ExperimentRun
        """
        if run_id in self.runs:
            raise ValueError(f"Run ID '{run_id}' already exists.")
        run = ExperimentRun(run_id, experiment_name, parameters, tags)
        self.runs[run_id] = run
        return run

    def get_run(self, run_id):
        """Retrieve a run by ID."""
        if run_id not in self.runs:
            raise KeyError(f"Run ID '{run_id}' not found.")
        return self.runs[run_id]

    def list_runs(self, experiment_name=None, status=None):
        """
        List runs, optionally filtered by experiment name and/or status.

        Returns
        -------
        list of ExperimentRun
        """
        results = list(self.runs.values())
        if experiment_name is not None:
            results = [r for r in results if r.experiment_name == experiment_name]
        if status is not None:
            results = [r for r in results if r.status == status]
        return results

    def compare_runs(self, run_ids, metric_names=None):
        """
        Compare specified runs on given metrics.

        Parameters
        ----------
        run_ids : list of str
            Run IDs to compare.
        metric_names : list of str, optional
            Metrics to include. If None, includes all metrics.

        Returns
        -------
        list of dict
            One dict per run with run_id, parameters, and selected metrics.
        """
        comparison = []
        for rid in run_ids:
            run = self.get_run(rid)
            entry = {
                "run_id": run.run_id,
                "experiment_name": run.experiment_name,
                "parameters": run.parameters,
                "status": run.status,
            }
            if metric_names:
                entry["metrics"] = {k: run.metrics.get(k) for k in metric_names}
            else:
                entry["metrics"] = dict(run.metrics)
            comparison.append(entry)
        return comparison

    def best_run(self, experiment_name, metric_name, minimize=False):
        """
        Find the best run for an experiment by a given metric.

        Parameters
        ----------
        experiment_name : str
            Filter to this experiment.
        metric_name : str
            Metric to optimize.
        minimize : bool
            If True, lower is better. Default False (higher is better).

        Returns
        -------
        ExperimentRun or None
        """
        runs = self.list_runs(experiment_name=experiment_name, status="completed")
        valid = [r for r in runs if metric_name in r.metrics]
        if not valid:
            return None
        return (min if minimize else max)(valid, key=lambda r: r.metrics[metric_name])

    def summary_stats(self, experiment_name, metric_name):
        """
        Compute summary statistics for a metric across completed runs.

        Returns
        -------
        dict with keys: count, mean, std, min, max
        """
        runs = self.list_runs(experiment_name=experiment_name, status="completed")
        values = [r.metrics[metric_name] for r in runs if metric_name in r.metrics]
        if not values:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        import numpy as np
        arr = np.array(values)
        return {
            "count": len(values),
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std()), 6),
            "min": round(float(arr.min()), 6),
            "max": round(float(arr.max()), 6),
        }

    def export_to_json(self, filepath):
        """Export all runs to a JSON file."""
        data = [run.to_dict() for run in self.runs.values()]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        return filepath

    def export_to_dict(self):
        """Export all runs as a list of dicts."""
        return [run.to_dict() for run in self.runs.values()]

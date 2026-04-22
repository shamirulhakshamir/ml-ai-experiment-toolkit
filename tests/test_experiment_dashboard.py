"""Tests for ExperimentTracker and ExperimentRun."""

import pytest
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.experiment_dashboard import ExperimentRun, ExperimentTracker


class TestExperimentRun:
    def test_create_run(self):
        run = ExperimentRun("run_1", "exp_a", parameters={"lr": 0.01})
        assert run.run_id == "run_1"
        assert run.experiment_name == "exp_a"
        assert run.parameters == {"lr": 0.01}
        assert run.status == "running"

    def test_log_metric(self):
        run = ExperimentRun("run_1", "exp_a")
        run.log_metric("accuracy", 0.95)
        assert run.metrics["accuracy"] == 0.95

    def test_log_metrics(self):
        run = ExperimentRun("run_1", "exp_a")
        run.log_metrics({"accuracy": 0.95, "f1": 0.90})
        assert run.metrics["accuracy"] == 0.95
        assert run.metrics["f1"] == 0.90

    def test_complete(self):
        run = ExperimentRun("run_1", "exp_a")
        run.complete()
        assert run.status == "completed"
        assert hasattr(run, "completed_at")

    def test_fail(self):
        run = ExperimentRun("run_1", "exp_a")
        run.fail("OOM error")
        assert run.status == "failed"
        assert run.error_message == "OOM error"

    def test_to_dict(self):
        run = ExperimentRun("run_1", "exp_a", parameters={"lr": 0.01}, tags={"version": "v1"})
        run.log_metric("accuracy", 0.95)
        run.complete()
        d = run.to_dict()
        assert d["run_id"] == "run_1"
        assert d["metrics"]["accuracy"] == 0.95
        assert d["status"] == "completed"
        assert "completed_at" in d


class TestExperimentTracker:
    def setup_method(self):
        self.tracker = ExperimentTracker()

    def test_create_run(self):
        run = self.tracker.create_run("r1", "exp_a", parameters={"lr": 0.01})
        assert run.run_id == "r1"
        assert "r1" in self.tracker.runs

    def test_duplicate_run_raises(self):
        self.tracker.create_run("r1", "exp_a")
        with pytest.raises(ValueError):
            self.tracker.create_run("r1", "exp_a")

    def test_get_run(self):
        self.tracker.create_run("r1", "exp_a")
        run = self.tracker.get_run("r1")
        assert run.run_id == "r1"

    def test_get_missing_run_raises(self):
        with pytest.raises(KeyError):
            self.tracker.get_run("nonexistent")

    def test_list_runs_all(self):
        self.tracker.create_run("r1", "exp_a")
        self.tracker.create_run("r2", "exp_b")
        assert len(self.tracker.list_runs()) == 2

    def test_list_runs_by_experiment(self):
        self.tracker.create_run("r1", "exp_a")
        self.tracker.create_run("r2", "exp_b")
        self.tracker.create_run("r3", "exp_a")
        runs = self.tracker.list_runs(experiment_name="exp_a")
        assert len(runs) == 2

    def test_list_runs_by_status(self):
        r1 = self.tracker.create_run("r1", "exp_a")
        r2 = self.tracker.create_run("r2", "exp_a")
        r1.complete()
        runs = self.tracker.list_runs(status="completed")
        assert len(runs) == 1
        assert runs[0].run_id == "r1"

    def test_compare_runs(self):
        r1 = self.tracker.create_run("r1", "exp_a")
        r2 = self.tracker.create_run("r2", "exp_a")
        r1.log_metric("accuracy", 0.9)
        r2.log_metric("accuracy", 0.85)
        comparison = self.tracker.compare_runs(["r1", "r2"], metric_names=["accuracy"])
        assert len(comparison) == 2
        assert comparison[0]["metrics"]["accuracy"] == 0.9

    def test_compare_runs_all_metrics(self):
        r1 = self.tracker.create_run("r1", "exp_a")
        r1.log_metrics({"accuracy": 0.9, "f1": 0.88})
        comparison = self.tracker.compare_runs(["r1"])
        assert "accuracy" in comparison[0]["metrics"]
        assert "f1" in comparison[0]["metrics"]

    def test_best_run(self):
        r1 = self.tracker.create_run("r1", "exp_a")
        r2 = self.tracker.create_run("r2", "exp_a")
        r3 = self.tracker.create_run("r3", "exp_a")
        r1.log_metric("accuracy", 0.85)
        r1.complete()
        r2.log_metric("accuracy", 0.92)
        r2.complete()
        r3.log_metric("accuracy", 0.88)
        r3.complete()
        best = self.tracker.best_run("exp_a", "accuracy")
        assert best.run_id == "r2"

    def test_best_run_minimize(self):
        r1 = self.tracker.create_run("r1", "exp_a")
        r2 = self.tracker.create_run("r2", "exp_a")
        r1.log_metric("loss", 0.3)
        r1.complete()
        r2.log_metric("loss", 0.1)
        r2.complete()
        best = self.tracker.best_run("exp_a", "loss", minimize=True)
        assert best.run_id == "r2"

    def test_best_run_no_completed(self):
        self.tracker.create_run("r1", "exp_a")
        best = self.tracker.best_run("exp_a", "accuracy")
        assert best is None

    def test_summary_stats(self):
        for i, acc in enumerate([0.85, 0.90, 0.88, 0.92]):
            r = self.tracker.create_run(f"r{i}", "exp_a")
            r.log_metric("accuracy", acc)
            r.complete()
        stats = self.tracker.summary_stats("exp_a", "accuracy")
        assert stats["count"] == 4
        assert stats["min"] == 0.85
        assert stats["max"] == 0.92
        assert stats["mean"] is not None

    def test_summary_stats_empty(self):
        stats = self.tracker.summary_stats("exp_a", "accuracy")
        assert stats["count"] == 0

    def test_export_to_json(self):
        r1 = self.tracker.create_run("r1", "exp_a")
        r1.log_metric("accuracy", 0.9)
        r1.complete()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            self.tracker.export_to_json(filepath)
            with open(filepath) as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["run_id"] == "r1"
        finally:
            os.unlink(filepath)

    def test_export_to_dict(self):
        r1 = self.tracker.create_run("r1", "exp_a")
        r1.log_metric("accuracy", 0.9)
        data = self.tracker.export_to_dict()
        assert len(data) == 1
        assert data[0]["metrics"]["accuracy"] == 0.9

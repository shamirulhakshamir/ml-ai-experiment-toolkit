"""Tests for FeatureSelector."""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feature_selector import FeatureSelector


@pytest.fixture
def classification_data():
    """Generate a simple classification dataset."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "informative_1": np.random.randn(n),
        "informative_2": np.random.randn(n),
        "noisy": np.random.randn(n) * 0.01,
        "constant": np.ones(n),
        "redundant": np.random.randn(n),
    })
    # Make informative_1 correlated with target
    y = (X["informative_1"] + X["informative_2"] > 0).astype(int)
    # Make redundant almost identical to informative_1
    X["redundant"] = X["informative_1"] + np.random.randn(n) * 0.001
    return X, y


@pytest.fixture
def regression_data():
    """Generate a simple regression dataset."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "feat_a": np.random.randn(n),
        "feat_b": np.random.randn(n),
        "feat_c": np.random.randn(n) * 0.001,  # low variance
    })
    y = 3 * X["feat_a"] + 2 * X["feat_b"] + np.random.randn(n) * 0.1
    return X, y


class TestVarianceFilter:
    def test_removes_constant_feature(self, classification_data):
        X, _ = classification_data
        selector = FeatureSelector(task="classification")
        passed = selector.variance_filter(X, threshold=0.01)
        assert "constant" not in passed

    def test_keeps_informative_features(self, classification_data):
        X, _ = classification_data
        selector = FeatureSelector(task="classification")
        passed = selector.variance_filter(X, threshold=0.01)
        assert "informative_1" in passed
        assert "informative_2" in passed


class TestMutualInformation:
    def test_returns_series(self, classification_data):
        X, y = classification_data
        selector = FeatureSelector(task="classification")
        scores = selector.mutual_information_scores(X, y)
        assert isinstance(scores, pd.Series)
        assert len(scores) == X.shape[1]

    def test_informative_scored_higher(self, classification_data):
        X, y = classification_data
        selector = FeatureSelector(task="classification")
        scores = selector.mutual_information_scores(X, y)
        # informative_1 should score higher than noisy
        assert scores["informative_1"] > scores["noisy"]

    def test_regression_mi(self, regression_data):
        X, y = regression_data
        selector = FeatureSelector(task="regression")
        scores = selector.mutual_information_scores(X, y)
        assert len(scores) == 3


class TestCorrelationFilter:
    def test_removes_redundant_feature(self, classification_data):
        X, _ = classification_data
        selector = FeatureSelector(task="classification")
        passed = selector.correlation_filter(X, threshold=0.95)
        # Either informative_1 or redundant should be removed (they're nearly identical)
        has_informative = "informative_1" in passed
        has_redundant = "redundant" in passed
        # At least one should be removed
        assert not (has_informative and has_redundant)

    def test_keeps_uncorrelated_features(self, classification_data):
        X, _ = classification_data
        selector = FeatureSelector(task="classification")
        passed = selector.correlation_filter(X, threshold=0.95)
        assert "informative_2" in passed


class TestRFERanking:
    def test_returns_ranking_series(self, classification_data):
        X, y = classification_data
        selector = FeatureSelector(task="classification")
        ranking = selector.rfe_ranking(X, y, n_features_to_select=2)
        assert isinstance(ranking, pd.Series)
        assert (ranking == 1).sum() == 2

    def test_regression_rfe(self, regression_data):
        X, y = regression_data
        selector = FeatureSelector(task="regression")
        ranking = selector.rfe_ranking(X, y, n_features_to_select=2)
        assert (ranking == 1).sum() == 2


class TestConsensusSelect:
    def test_returns_list(self, classification_data):
        X, y = classification_data
        selector = FeatureSelector(task="classification")
        selected = selector.consensus_select(X, y, top_k=2)
        assert isinstance(selected, list)
        assert len(selected) == 2

    def test_selects_informative_features(self, classification_data):
        X, y = classification_data
        selector = FeatureSelector(task="classification")
        selected = selector.consensus_select(X, y, top_k=2)
        # At least one informative feature should be selected
        informative = {"informative_1", "informative_2", "redundant"}
        assert len(set(selected) & informative) >= 1


class TestInvalidTask:
    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            FeatureSelector(task="unsupervised")

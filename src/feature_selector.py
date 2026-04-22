"""
Feature Selector — Automated feature selection pipeline.

Combines multiple feature selection strategies:
- Variance threshold filtering
- Mutual information scoring
- Correlation-based redundancy removal
- Recursive feature elimination (RFE)
- Consensus ranking across methods
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class FeatureSelector:
    """Automated feature selection with multiple strategies."""

    def __init__(self, task="classification", random_state=42):
        """
        Parameters
        ----------
        task : str
            'classification' or 'regression'.
        random_state : int
            Random seed for reproducibility.
        """
        if task not in ("classification", "regression"):
            raise ValueError(f"Unknown task: {task}")
        self.task = task
        self.random_state = random_state

    def variance_filter(self, X, threshold=0.01):
        """
        Remove features with variance below threshold.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        threshold : float
            Minimum variance to keep a feature.

        Returns
        -------
        list of str
            Feature names that pass the variance filter.
        """
        X = pd.DataFrame(X)
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        mask = selector.get_support()
        return list(X.columns[mask])

    def mutual_information_scores(self, X, y):
        """
        Compute mutual information between each feature and the target.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : array-like
            Target variable.

        Returns
        -------
        pd.Series
            MI scores indexed by feature name, sorted descending.
        """
        X = pd.DataFrame(X)
        mi_func = mutual_info_classif if self.task == "classification" else mutual_info_regression
        scores = mi_func(X, y, random_state=self.random_state)
        mi_series = pd.Series(scores, index=X.columns, name="mi_score")
        return mi_series.sort_values(ascending=False)

    def correlation_filter(self, X, threshold=0.90):
        """
        Remove highly correlated features (keeps the first of each correlated pair).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        threshold : float
            Correlation threshold above which one feature is dropped.

        Returns
        -------
        list of str
            Feature names after removing redundant correlated features.
        """
        X = pd.DataFrame(X)
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
        return [col for col in X.columns if col not in to_drop]

    def rfe_ranking(self, X, y, n_features_to_select=None):
        """
        Rank features using recursive feature elimination.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : array-like
            Target variable.
        n_features_to_select : int, optional
            Number of features to select. Defaults to half.

        Returns
        -------
        pd.Series
            RFE ranking (1 = selected) indexed by feature name, sorted ascending.
        """
        from sklearn.feature_selection import RFE

        X = pd.DataFrame(X)
        if n_features_to_select is None:
            n_features_to_select = max(1, len(X.columns) // 2)

        estimator = (
            RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            if self.task == "classification"
            else RandomForestRegressor(n_estimators=50, random_state=self.random_state)
        )
        rfe = RFE(estimator, n_features_to_select=n_features_to_select)
        rfe.fit(X, y)
        ranking = pd.Series(rfe.ranking_, index=X.columns, name="rfe_rank")
        return ranking.sort_values()

    def consensus_select(self, X, y, top_k=None, variance_threshold=0.01, corr_threshold=0.90):
        """
        Combine multiple selection methods into a consensus ranking.

        Each method assigns a rank to every feature. The consensus score is
        the average rank across methods. Lower is better.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : array-like
            Target variable.
        top_k : int, optional
            Number of top features to return. Defaults to half.
        variance_threshold : float
            Threshold for variance filter.
        corr_threshold : float
            Threshold for correlation filter.

        Returns
        -------
        list of str
            Top-k feature names by consensus ranking.
        """
        X = pd.DataFrame(X)
        all_features = list(X.columns)
        n = len(all_features)
        if top_k is None:
            top_k = max(1, n // 2)

        # 1. Variance filter rank: passing features get rank 1, others get n
        var_passed = set(self.variance_filter(X, threshold=variance_threshold))
        var_rank = pd.Series(
            [1 if f in var_passed else n for f in all_features],
            index=all_features,
        )

        # 2. MI rank
        mi_scores = self.mutual_information_scores(X, y)
        mi_rank = mi_scores.rank(ascending=False)

        # 3. Correlation filter rank
        corr_passed = set(self.correlation_filter(X, threshold=corr_threshold))
        corr_rank = pd.Series(
            [1 if f in corr_passed else n for f in all_features],
            index=all_features,
        )

        # 4. RFE rank
        rfe_rank = self.rfe_ranking(X, y, n_features_to_select=top_k)

        # Average ranks
        consensus = (var_rank + mi_rank + corr_rank + rfe_rank) / 4.0
        consensus = consensus.sort_values()
        return list(consensus.head(top_k).index)

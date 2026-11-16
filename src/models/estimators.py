"""Small reusable estimators shared by several models."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone


class ColumnBaseline(BaseEstimator, RegressorMixin):
    """'Model' that simply returns one existing column, e.g. a lag = 'same as last period'."""

    def __init__(self, column="sales_lag_6"):
        self.column = column

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X[self.column].to_numpy(dtype=float)


class NonNegativeRegressor(BaseEstimator, RegressorMixin):
    """Wraps any regressor and clips predictions at 0 (demand cannot be negative)."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.estimator_.predict(X), 0, None)


class RatioToLevelRegressor(BaseEstimator, RegressorMixin):
    """Predict y / (level + offset) and multiply back.

    For panels of series on very different scales (products selling 1 vs 5,000 units),
    this puts every series on a common scale so the model learns shared patterns,
    instead of spending its splits on telling big sellers from small ones.
    `level_col` must be a past-only statistic (e.g. a shifted rolling mean) -> no leakage.
    """

    def __init__(self, estimator=None, level_col="units_roll_mean_4", offset=1.0):
        self.estimator = estimator
        self.level_col = level_col
        self.offset = offset

    def _level(self, X):
        return X[self.level_col].to_numpy(dtype=float) + self.offset

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, np.asarray(y, dtype=float) / self._level(X))
        return self

    def predict(self, X):
        return np.clip(self.estimator_.predict(X) * self._level(X), 0, None)

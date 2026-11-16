"""Shared evaluation tools for every model in the project (Module 8 builds on these)."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


def regression_metrics(y_true, y_pred):
    """MAE, RMSE, R2, MAPE, WAPE and Bias.

    MAE  : average size of the error, in the target's units (£) - easiest to explain.
    RMSE : like MAE but punishes large misses more.
    R2   : share of the variance explained (1 = perfect, 0 = no better than the mean).
    MAPE : average % error. Only defined when every actual value is > 0, and it
           over-weights small actuals (e.g. short Sundays) -> read it with care.
    WAPE : total absolute error / total actual. A % error that is robust to small days.
    Bias : mean(predicted - actual). > 0 = over-forecasting on average.
    Bias_pct : total predicted / total actual - 1, in %. -30 = forecasts add up to 30% too little.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs(err / y_true)) * 100) if np.all(y_true > 0) else float("nan"),
        "WAPE": float(np.abs(err).sum() / np.abs(y_true).sum() * 100),
        "Bias": float(err.mean()),
        "Bias_pct": float(err.sum() / y_true.sum() * 100) if y_true.sum() else float("nan"),
    }


def chronological_split(frame, test_size):
    """Last `test_size` rows = test. No shuffling: the test period is strictly after training."""
    if not 0 < test_size < len(frame):
        raise ValueError(f"test_size must be between 1 and {len(frame) - 1}")
    return frame.iloc[:-test_size], frame.iloc[-test_size:]


def time_series_cv(model, X, y, n_splits=5, test_size=None):
    """Expanding-window cross-validation: every validation fold lies AFTER its training data.

    fold 1: train [====]          validate [--]
    fold 2: train [======]        validate [--]
    fold 3: train [========]      validate [--]
    """
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    rows = []
    for fold, (tr, va) in enumerate(splitter.split(X), start=1):
        fitted = clone(model).fit(X.iloc[tr], y.iloc[tr])
        rows.append({"fold": fold, "train_end": X.index[tr[-1]], **regression_metrics(y.iloc[va], fitted.predict(X.iloc[va]))})
    return pd.DataFrame(rows)


def series_bias_pct(y_true, y_pred, series_ids):
    """Sum over series of |that series' total error| / total actual, in %.

    Unlike total bias, errors in different series cannot cancel out: over-forecasting
    small products no longer hides under-forecasting the best sellers.
    Always >= |total Bias_pct|.
    """
    y_true = np.asarray(y_true, dtype=float)
    err = pd.Series(np.asarray(y_pred, dtype=float) - y_true)
    return float(err.groupby(np.asarray(series_ids)).sum().abs().sum() / y_true.sum() * 100)


def grouped_time_series_cv(model, X, y, periods, n_splits=4, periods_per_fold=4, series_ids=None):
    """Expanding-window CV for PANEL data (many series, e.g. products x weeks).

    Folds are cut on whole time periods, never on rows, so all products of a given
    week are in the same fold and training always ends before validation starts.
    periods: array-like with the time period (e.g. week start) of every row.
    series_ids: optional series label per row -> adds SeriesBias_pct to each fold.
    """
    periods = np.asarray(periods)
    unique = np.sort(pd.unique(periods))
    needed = n_splits * periods_per_fold
    if needed >= len(unique):
        raise ValueError(f"Need more than {needed} distinct periods, got {len(unique)}")
    rows = []
    for fold in range(n_splits):
        start = len(unique) - (n_splits - fold) * periods_per_fold
        val_periods = unique[start:start + periods_per_fold]
        tr = periods < val_periods[0]
        va = np.isin(periods, val_periods)
        fitted = clone(model).fit(X[tr], y[tr])
        pred = fitted.predict(X[va])
        row = {"fold": fold + 1, "train_end": unique[start - 1], "val_start": val_periods[0],
               **regression_metrics(y[va], pred)}
        if series_ids is not None:
            row["SeriesBias_pct"] = series_bias_pct(y[va], pred, np.asarray(series_ids)[va])
        rows.append(row)
    return pd.DataFrame(rows)

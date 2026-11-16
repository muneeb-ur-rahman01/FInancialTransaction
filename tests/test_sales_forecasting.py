"""Tests for src/models/evaluation.py and src/models/sales_forecasting.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from src.models import sales_forecasting as SF
from src.models.evaluation import chronological_split, regression_metrics, time_series_cv


def test_regression_metrics_known_values():
    m = regression_metrics([100, 200], [110, 180])
    assert m["MAE"] == pytest.approx(15)
    assert m["RMSE"] == pytest.approx(np.sqrt((100 + 400) / 2))
    assert m["MAPE"] == pytest.approx((10 + 10) / 2)          # 10% and 10%
    assert m["WAPE"] == pytest.approx(30 / 300 * 100)
    assert m["Bias"] == pytest.approx(-5)


def test_mape_undefined_when_actual_is_zero():
    assert np.isnan(regression_metrics([0, 10], [1, 10])["MAPE"])


def test_chronological_split_keeps_order():
    frame = pd.DataFrame({"y": range(10)}, index=pd.date_range("2011-01-01", periods=10))
    train, test = chronological_split(frame, 3)
    assert len(train) == 7 and len(test) == 3
    assert train.index.max() < test.index.min()
    with pytest.raises(ValueError):
        chronological_split(frame, 10)


def test_time_series_cv_validates_only_on_the_future():
    idx = pd.date_range("2011-01-01", periods=60)
    X = pd.DataFrame({"x": np.arange(60.0)}, index=idx)
    y = pd.Series(2 * np.arange(60.0), index=idx)
    folds = time_series_cv(LinearRegression(), X, y, n_splits=4, test_size=5)
    assert len(folds) == 4
    assert folds["train_end"].is_monotonic_increasing
    assert folds["MAE"].max() < 1e-8                          # perfect linear relation


def test_column_baseline_returns_the_column():
    X = pd.DataFrame({"sales_lag_6": [1.0, 2.0, 3.0]})
    assert SF.ColumnBaseline("sales_lag_6").fit(X).predict(X).tolist() == [1.0, 2.0, 3.0]


def test_level_scaled_regressor_round_trip():
    rng = np.random.default_rng(0)
    level = rng.uniform(10, 100, 200)
    X = pd.DataFrame({SF.LEVEL_COL: level, "x": rng.normal(size=200)})
    y = level * 1.5                                           # y is exactly 1.5 x level
    model = SF.LevelScaledRegressor(LinearRegression()).fit(X, y)
    assert np.allclose(model.predict(X), y)
    clone(model)                                              # must be clonable for CV


def test_pipeline_one_hot_encodes_weekday():
    X = pd.DataFrame({"weekday": [0, 1, 2, 6] * 10, "sales_lag_6": np.arange(40.0)})
    pipe = SF.make_pipeline(LinearRegression()).fit(X, np.arange(40.0))
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    assert {"weekday_0", "weekday_6", "sales_lag_6"} <= set(names)


def test_feature_columns_exclude_target_and_non_recursive_lags():
    table = pd.DataFrame(columns=["net_sales", "sales_lag_1", "orders_lag_1", "customers_lag_6",
                                  "avg_order_value_lag_1", "weekday"])
    assert SF.feature_columns(table) == ["sales_lag_1", "weekday"]


def test_candidates_include_baselines_and_all_families():
    families = {family for family, _, _ in SF.candidate_models()}
    assert families == {"Baseline", "Ridge", "Random Forest", "XGBoost"}


def test_experiment_on_synthetic_series_beats_naive():
    """End-to-end on a synthetic weekly-seasonal series: the pipeline runs and ML beats 'yesterday'."""
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2011-01-03", periods=200)
    weekly = np.array([1.0, 1.2, 1.1, 1.3, 0.9])[idx.day_of_week.to_numpy() % 5]
    y = pd.Series(1000 * weekly + rng.normal(0, 30, len(idx)), index=idx)
    t = pd.DataFrame({"net_sales": y, "weekday": idx.day_of_week, "is_sunday": 0, "month": idx.month,
                      "day_of_month": idx.day, "week_of_year": idx.isocalendar().week.to_numpy(),
                      "day_of_year": idx.day_of_year, "trend": np.arange(len(idx)),
                      "days_since_prev_trading_day": 1}, index=idx)
    for k in (1, 2, 3, 6, 12, 18, 24):
        t[f"sales_lag_{k}"] = y.shift(k)
    past = y.shift(1)
    for w in (6, 24):
        t[f"sales_roll_mean_{w}"] = past.rolling(w).mean()
        t[f"sales_roll_std_{w}"] = past.rolling(w).std()
    t["sales_same_weekday_mean_4"] = pd.concat([y.shift(k) for k in (6, 12, 18, 24)], axis=1).mean(axis=1)
    t = t.dropna()
    result = SF.run_experiment(t, SF.feature_columns(t))
    scores = result["test_table"].set_index("model")["MAE"]
    assert scores[result["champion_name"]] < scores["Naive: yesterday"]

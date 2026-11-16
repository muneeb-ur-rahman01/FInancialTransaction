"""Tests for src/models/demand_forecasting.py, the panel CV and the shared estimators."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from src.models import demand_forecasting as D
from src.models.estimators import NonNegativeRegressor, RatioToLevelRegressor
from src.models.evaluation import grouped_time_series_cv, regression_metrics, series_bias_pct


def test_bias_pct():
    assert regression_metrics([10, 30], [5, 15])["Bias_pct"] == pytest.approx(-50)
    assert regression_metrics([10, 30], [10, 30])["Bias_pct"] == pytest.approx(0)


def test_planning_score_penalises_bias_both_ways():
    assert D.planning_score(60, -30) == pytest.approx(90)
    assert D.planning_score(60, 30) == pytest.approx(90)
    assert D.planning_score(65, 0) < D.planning_score(60, -30)


def test_series_bias_catches_errors_that_cancel_in_total():
    y = [100, 100, 10, 10]
    pred = [50, 50, 60, 60]                      # big seller -100, small seller +100
    ids = ["big", "big", "small", "small"]
    assert regression_metrics(y, pred)["Bias_pct"] == pytest.approx(0)          # total hides it
    assert series_bias_pct(y, pred, ids) == pytest.approx(200 / 220 * 100)     # per product does not


def test_ratio_to_level_regressor():
    rng = np.random.default_rng(0)
    level = rng.uniform(1, 1000, 300)
    X = pd.DataFrame({"units_roll_mean_4": level, "noise": rng.normal(size=300)})
    y = 2 * (level + 1)                                                         # exactly 2x the level
    model = RatioToLevelRegressor(LinearRegression()).fit(X, y)
    assert np.allclose(model.predict(X), y)
    clone(model)


def _cv_row(family, model, score, std):
    return {"family": family, "model": model, "cv_score": score, "cv_score_std": std}


def test_champion_rule_keeps_baseline_when_gain_is_within_noise():
    table = pd.DataFrame([_cv_row("Baseline", "B", 100, 5), _cv_row("XGBoost", "X", 97, 2)])
    champion, base, ml, text = D.choose_champion(table)
    assert (champion, base, ml) == ("B", "B", "X")
    assert text.startswith("Baseline kept")


def test_champion_rule_picks_ml_when_gain_is_convincing():
    table = pd.DataFrame([_cv_row("Baseline", "B", 100, 5), _cv_row("XGBoost", "X", 90, 2)])
    champion, _, _, text = D.choose_champion(table)
    assert champion == "X" and text.startswith("ML wins")


def test_non_negative_regressor_clips():
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    model = NonNegativeRegressor(LinearRegression()).fit(X, [-3.0, -1.0, 1.0, 3.0])
    assert model.predict(X).min() == 0
    clone(model)


def test_grouped_cv_never_mixes_a_period_across_train_and_validation():
    weeks = np.repeat(pd.date_range("2011-01-03", periods=12, freq="W-MON"), 3)   # 3 products per week
    X = pd.DataFrame({"x": np.arange(36.0)})
    y = pd.Series(np.arange(36.0))
    folds = grouped_time_series_cv(LinearRegression(), X, y, weeks, n_splits=3, periods_per_fold=2)
    assert len(folds) == 3
    assert (folds["train_end"] < folds["val_start"]).all()
    assert folds["val_start"].tolist() == list(pd.date_range("2011-01-03", periods=12, freq="W-MON")[6::2])
    with pytest.raises(ValueError):
        grouped_time_series_cv(LinearRegression(), X, y, weeks, n_splits=6, periods_per_fold=2)


def test_split_by_weeks_keeps_whole_weeks():
    data = pd.DataFrame({"week": np.repeat(pd.date_range("2011-01-03", periods=10, freq="W-MON"), 2),
                         "units_sold": 1})
    train, test = D.split_by_weeks(data, test_weeks=3)
    assert test["week"].nunique() == 3 and len(test) == 6
    assert train["week"].max() < test["week"].min()


def test_log_count_ridge_is_stable_on_huge_lags():
    """The prototype's failure: log-target Ridge on raw lags exploded. Logged features must keep it sane."""
    rng = np.random.default_rng(0)
    n = 500
    lag = rng.lognormal(3, 1.5, n)
    X = pd.DataFrame({"units_lag_1": lag, "week_of_year": rng.integers(1, 53, n)})
    y = np.round(lag * rng.uniform(0.5, 1.5, n))
    model = NonNegativeRegressor(D.log_target(D.log_count_ridge(list(X.columns), alpha=1))).fit(X, y)
    X_big = X.assign(units_lag_1=X["units_lag_1"] * 50)            # far outside the training range
    assert np.isfinite(model.predict(X_big)).all()
    assert model.predict(X_big).max() < 1e7


def test_candidates_cover_all_families_and_objectives():
    names = [(f, n) for f, n, _ in D.candidate_models(["units_lag_1", "units_roll_mean_4"])]
    assert {f for f, _ in names} == {"Baseline", "Ridge", "Random Forest", "XGBoost"}
    assert any("Poisson" in n for _, n in names)
    assert any("absolute error" in n for _, n in names)


def test_experiment_end_to_end_on_synthetic_panel():
    rng = np.random.default_rng(3)
    weeks = pd.date_range("2011-01-03", periods=40, freq="W-MON")
    rows = []
    for code, scale in [("A", 5), ("B", 50), ("C", 500)]:
        units = rng.poisson(scale * (1 + 0.3 * np.sin(np.arange(40) / 4)))
        s = pd.Series(units, dtype=float)
        frame = pd.DataFrame({"StockCode": code, "week": weeks, "product": code, "units_sold": units,
                              "week_of_year": weeks.isocalendar().week.to_numpy(), "month": weeks.month,
                              "trading_days": 6, "product_age_weeks": np.arange(40)})
        for k in (1, 2, 3, 4):
            frame[f"units_lag_{k}"] = s.shift(k)
        frame["units_roll_mean_4"] = s.shift(1).rolling(4).mean()
        frame["units_roll_std_4"] = s.shift(1).rolling(4).std()
        frame["units_expanding_mean"] = s.shift(1).expanding().mean()
        frame["orders_lag_1"] = frame["units_lag_1"] / 3
        frame["customers_lag_1"] = frame["units_lag_1"] / 4
        frame["price_lag_1"] = 2.0
        rows.append(frame)
    data = pd.concat(rows).dropna().sort_values(["week", "StockCode"]).reset_index(drop=True)
    result = D.run_experiment(data, D.feature_columns(data), verbose=False)
    assert result["champion_name"] in set(result["cv_table"]["model"])
    assert len(result["test"]["week"].unique()) == D.TEST_WEEKS
    assert (result["test_preds"][result["champion_name"]] >= 0).all()


def test_business_rule_prefers_tied_baseline():
    table = pd.DataFrame([_cv_row("Baseline", "Last week", 100, 4),
                          _cv_row("Baseline", "Mean of last 4 weeks", 103, 3),
                          _cv_row("XGBoost", "X", 98, 2)])
    champion, best_base, _, text = D.choose_champion(table)
    assert champion == "Mean of last 4 weeks" and best_base == "Last week"
    assert "Business rule" in text


def test_business_rule_does_not_override_a_clearly_better_baseline():
    table = pd.DataFrame([_cv_row("Baseline", "Last week", 100, 2),
                          _cv_row("Baseline", "Mean of last 4 weeks", 110, 3),
                          _cv_row("XGBoost", "X", 99, 2)])
    assert D.choose_champion(table)[0] == "Last week"

"""Unit tests for src/analysis/statistical_tests.py.

Statistical functions are tested on synthetic data where we KNOW the truth:
a clearly shifted group must be detected, identical groups must not,
white noise is stationary, a random walk is not.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import statistical_tests as st

RNG = np.random.default_rng(0)


def test_describe_distribution_basic_values():
    d = st.describe_distribution([1, 2, 2, 3, 10])
    assert d["n"] == 5
    assert d["mean"] == pytest.approx(3.6)
    assert d["median"] == 2
    assert d["mode"] == 2
    assert d["variance"] == pytest.approx(np.var([1, 2, 2, 3, 10], ddof=1))
    assert d["iqr"] == pytest.approx(1.0)
    assert d["skewness"] > 0                       # long right tail


def test_mode_is_nan_when_no_value_repeats():
    assert np.isnan(st.describe_distribution([1.1, 2.2, 3.3])["mode"])


def test_normality_detects_lognormal_and_log_fixes_it():
    data = RNG.lognormal(mean=3, sigma=1, size=5000)
    rep = st.normality_report(data)
    assert not rep.loc["raw", "normal_at_5pct"]
    assert abs(rep.loc["log", "skewness"]) < 0.2   # log of lognormal is normal


def test_compare_two_groups_detects_shift():
    a = RNG.normal(12, 2, 400)
    b = RNG.normal(10, 2, 400)
    r = st.compare_two_groups(a, b, alternative="greater")
    assert r["p_value"] < 0.001
    assert r["rank_biserial"] > 0.3
    lo, hi = r["median_diff_ci95"]
    assert lo < 2 < hi + 0.5 and lo > 0            # CI excludes 0, near the true shift of 2


def test_compare_two_groups_no_effect():
    a = RNG.normal(10, 2, 400)
    b = RNG.normal(10, 2, 400)
    r = st.compare_two_groups(a, b, bootstrap=False)
    assert r["p_value"] > 0.01
    assert abs(r["rank_biserial"]) < 0.15


def test_rank_biserial_extremes():
    r = st.compare_two_groups([10, 11, 12], [1, 2, 3], bootstrap=False)
    assert r["rank_biserial"] == pytest.approx(1.0)     # every a beats every b


def test_compare_many_groups_with_holm():
    groups = {"x": RNG.normal(0, 1, 200), "y": RNG.normal(0, 1, 200), "z": RNG.normal(3, 1, 200)}
    overall, pairwise = st.compare_many_groups(groups)
    assert overall["p_value"] < 0.001
    assert overall["k"] == 3
    sig_pairs = set(map(tuple, pairwise.loc[pairwise["significant"], ["group_a", "group_b"]].values))
    assert sig_pairs == {("x", "z"), ("y", "z")}
    assert (pairwise["p_holm"] >= pairwise["p_raw"]).all()   # correction never lowers p


def test_chi_square():
    independent = st.chi_square_independence([[100, 100], [200, 200]])
    assert independent["p_value"] == pytest.approx(1.0)
    assert independent["cramers_v"] == pytest.approx(0.0)
    dependent = st.chi_square_independence([[90, 10], [10, 90]])
    assert dependent["p_value"] < 0.001
    assert dependent["cramers_v"] == pytest.approx(0.8)


def test_stationarity():
    noise = RNG.normal(0, 1, 400)
    walk = np.cumsum(RNG.normal(0, 1, 400))
    assert st.stationarity_report(noise)["conclusion"] == "stationary"
    assert st.stationarity_report(walk)["conclusion"] == "non-stationary"


def test_ljung_box():
    noise = st.ljung_box(RNG.normal(0, 1, 500), lags=[5])
    ar = np.zeros(500)
    for t in range(1, 500):
        ar[t] = 0.8 * ar[t - 1] + RNG.normal()
    dependent = st.ljung_box(ar, lags=[5])
    assert noise["lb_pvalue"].iloc[0] > 0.01
    assert dependent["lb_pvalue"].iloc[0] < 0.001


def test_correlation_test():
    x = np.arange(100, dtype=float)
    r = st.correlation_test(x, x ** 3)
    assert r["r"] == pytest.approx(1.0)                          # monotonic -> Spearman 1
    assert st.correlation_test(x, x ** 3, method="pearson")["r"] < 1


def test_daily_features_totals():
    from src.data_processing.cleaning import RAW_COLUMNS, clean_transactions

    rows = [
        ("1", "11111", "MUG", 2, "2011-01-03 10:00", 5.0, 1.0, "United Kingdom"),
        ("2", "11111", "MUG", 4, "2011-01-03 11:00", 5.0, 2.0, "France"),
        ("C3", "11111", "MUG", -1, "2011-01-04 10:00", 5.0, 1.0, "United Kingdom"),
        ("4", "22222", "PLATE", 1, "2011-01-04 12:00", 3.0, 1.0, "United Kingdom"),
    ]
    raw = pd.DataFrame(rows, columns=RAW_COLUMNS)
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
    daily = st.daily_features(clean_transactions(raw))
    assert daily["net_sales"].tolist() == [30.0, -2.0]
    assert daily["orders"].tolist() == [2, 1]
    assert daily.loc["2011-01-03", "avg_order_value"] == pytest.approx(15.0)
    assert daily["weekday"].tolist() == ["Monday", "Tuesday"]

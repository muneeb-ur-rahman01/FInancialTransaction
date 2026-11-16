"""Unit tests for src/data_processing/features.py.

The most important property of a feature table is NO LEAKAGE: a feature at
time t must not change when we change any value at time >= t. Several tests
below check exactly that by editing the "future" and comparing.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_processing import features as F
from src.data_processing.cleaning import RAW_COLUMNS, clean_transactions


def make_raw(rows):
    raw = pd.DataFrame(rows, columns=RAW_COLUMNS)
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
    return raw


@pytest.fixture
def daily_raw():
    """30 consecutive Mon-Fri + Sun trading days, one sale of known value per day."""
    days = pd.bdate_range("2011-01-03", periods=40).union(pd.date_range("2011-01-02", periods=8, freq="W-SUN"))
    rows = [(str(1000 + i), "11111", "MUG", i + 1, f"{d.date()} 10:00", 1.0, 1.0, "United Kingdom")
            for i, d in enumerate(days)]
    return make_raw(rows)


def test_time_features():
    df = pd.DataFrame({"InvoiceDate": pd.to_datetime(["2011-12-04 15:30"])})   # a Sunday
    out = F.add_time_features(df)
    row = out.iloc[0]
    assert (row["Year"], row["Month"], row["Day"], row["Hour"]) == (2011, 12, 4, 15)
    assert row["Weekday"] == 6 and row["WeekdayName"] == "Sunday"
    assert row["Quarter"] == 4


def test_daily_lags_and_rolling_use_only_the_past(daily_raw):
    t = F.build_daily_features(clean_transactions(daily_raw))
    y = t["net_sales"]
    assert t["sales_lag_1"].iloc[5] == y.iloc[4]
    assert t["sales_lag_6"].iloc[10] == y.iloc[4]
    assert t["sales_roll_mean_6"].iloc[10] == pytest.approx(y.iloc[4:10].mean())   # rows 4..9, not 10
    assert np.isnan(t["sales_lag_6"].iloc[5])                                    # no history yet


def test_daily_features_do_not_leak(daily_raw):
    base = F.build_daily_features(clean_transactions(daily_raw))
    k = 30
    changed = daily_raw.copy()
    changed.loc[k, "Quantity"] = 99_999                     # change ONE future day
    after = F.build_daily_features(clean_transactions(changed))
    feature_cols = [c for c in base.columns if c != "net_sales"]
    pd.testing.assert_frame_equal(base.iloc[: k + 1][feature_cols], after.iloc[: k + 1][feature_cols])
    assert after["sales_lag_1"].iloc[k + 1] != base["sales_lag_1"].iloc[k + 1]   # ...but it feeds later rows


def test_daily_calendar_flags(daily_raw):
    t = F.build_daily_features(clean_transactions(daily_raw))
    assert (t.loc[t.index.day_of_week == 6, "is_sunday"] == 1).all()
    assert t["trend"].tolist() == list(range(len(t)))


@pytest.fixture
def product_raw():
    rows = [
        # product A sells in weeks 1, 2 and 4 (week 3 has no sales -> must be a 0 row)
        ("1", "11111", "MUG", 10, "2011-01-03 10:00", 2.0, 1.0, "United Kingdom"),
        ("2", "11111", "MUG", 20, "2011-01-10 10:00", 2.0, 2.0, "United Kingdom"),
        ("3", "11111", "MUG", 40, "2011-01-24 10:00", 2.5, 1.0, "United Kingdom"),
        # product B starts in week 2
        ("4", "22222", "PLATE", 5, "2011-01-11 10:00", 4.0, 1.0, "United Kingdom"),
        ("5", "22222", "PLATE", 7, "2011-01-18 10:00", 4.0, 1.0, "United Kingdom"),
    ]
    return make_raw(rows)


def test_product_panel_zero_fills_and_starts_at_first_sale(product_raw):
    p = F.build_product_weekly_features(clean_transactions(product_raw), min_active_weeks=1)
    a = p[p["StockCode"] == "11111"].set_index("week")
    b = p[p["StockCode"] == "22222"].set_index("week")
    assert a["units_sold"].tolist() == [10, 20, 0, 40]
    assert b.index.min() == pd.Timestamp("2011-01-10")                            # no rows before launch
    assert b["units_sold"].tolist() == [5, 7, 0]                                  # zero-filled to the end
    assert a["units_lag_1"].tolist()[1:] == [10, 20, 0]
    assert a.loc["2011-01-24", "price_lag_1"] == 2.0                              # last price BEFORE the week
    assert a.loc["2011-01-24", "units_expanding_mean"] == pytest.approx(10)       # mean(10, 20, 0)


def test_product_min_active_weeks_filter(product_raw):
    p = F.build_product_weekly_features(clean_transactions(product_raw), min_active_weeks=3)
    assert set(p["StockCode"]) == {"11111"}


@pytest.fixture
def customer_raw():
    rows = [
        ("1", "11111", "MUG", 10, "2011-01-03 10:00", 2.0, 1.0, "United Kingdom"),   # 20
        ("2", "11111", "MUG", 10, "2011-01-13 10:00", 3.0, 1.0, "United Kingdom"),   # 30
        ("C3", "11111", "MUG", -2, "2011-01-20 10:00", 3.0, 1.0, "United Kingdom"),  # -6 return
        ("4", "22222", "PLATE", 1, "2011-01-05 10:00", 50.0, 2.0, "France"),         # 50
        ("5", "22222", "PLATE", 1, "2011-03-01 10:00", 50.0, 2.0, "France"),         # after snapshot
    ]
    return make_raw(rows)


def test_rfm_values(customer_raw):
    c = F.build_customer_features(clean_transactions(customer_raw), snapshot="2011-02-01")
    one = c.loc[1]
    assert one["frequency"] == 2
    assert one["monetary"] == pytest.approx(44)                 # 20 + 30 - 6
    assert one["recency_days"] == (pd.Timestamp("2011-02-01") - pd.Timestamp("2011-01-13 10:00")).days
    assert one["avg_order_value"] == pytest.approx(25)
    assert one["return_rate"] == pytest.approx(6 / 50)
    assert c.loc[2, "is_uk"] == 0


def test_rfm_ignores_transactions_after_snapshot(customer_raw):
    c = F.build_customer_features(clean_transactions(customer_raw), snapshot="2011-02-01")
    assert c.loc[2, "frequency"] == 1                           # the March order is invisible
    assert c.loc[2, "monetary"] == pytest.approx(50)


def test_rfm_scores_range(customer_raw):
    c = F.build_customer_features(clean_transactions(customer_raw))
    for col in ("R_score", "F_score", "M_score"):
        assert c[col].between(1, 5).all()


def test_invoice_history_uses_only_previous_orders(customer_raw):
    inv = F.build_invoice_features(clean_transactions(customer_raw))
    assert inv.loc["1", "prior_orders"] == 0
    assert np.isnan(inv.loc["1", "prior_median_order_value"])
    assert inv.loc["2", "prior_orders"] == 1
    assert inv.loc["2", "prior_median_order_value"] == pytest.approx(20)   # not including itself (30)
    assert inv.loc["2", "value_vs_customer_median"] == pytest.approx(1.5)
    assert inv.loc["2", "days_since_prev_order"] == pytest.approx(10)


def test_invoice_label_marks_entry_corrections():
    raw = make_raw([
        ("1", "11111", "MUG", 5000, "2011-01-03 10:00", 2.0, 1.0, "United Kingdom"),
        ("C2", "11111", "MUG", -5000, "2011-01-03 10:15", 2.0, 1.0, "United Kingdom"),
        ("3", "11111", "MUG", 5, "2011-01-04 10:00", 2.0, 1.0, "United Kingdom"),
    ])
    inv = F.build_invoice_features(clean_transactions(raw))
    assert inv.loc["1", "has_entry_correction"]
    assert not inv.loc["3", "has_entry_correction"]

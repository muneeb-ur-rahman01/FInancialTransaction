"""Unit tests for src/analysis/eda.py on a tiny dataset with known answers."""

import numpy as np
import pandas as pd
import pytest

from src.analysis import eda
from src.data_processing.cleaning import RAW_COLUMNS, clean_transactions


@pytest.fixture
def clean():
    rows = [
        # Jan: customer 1 buys twice, customer 2 once (one line is a keying error, reversed)
        ("100", "11111", "MUG", 10, "2011-01-03 10:00", 2.0, 1.0, "United Kingdom"),       # 20
        ("101", "22222", "PLATE", 5, "2011-01-04 11:00", 4.0, 1.0, "United Kingdom"),      # 20
        ("102", "11111", "MUG", 1000, "2011-01-05 12:00", 2.0, 2.0, "France"),             # keying error
        ("C103", "11111", "MUG", -1000, "2011-01-05 12:10", 2.0, 2.0, "France"),           # its reversal
        ("104", "22222", "PLATE", 10, "2011-01-05 13:00", 4.0, 2.0, "France"),             # 40
        # Feb: a genuine return and an anonymous sale; postage is not product revenue
        ("105", "11111", "MUG", 30, "2011-02-01 09:00", 2.0, np.nan, "United Kingdom"),    # 60
        ("C106", "22222", "PLATE", -2, "2011-02-02 09:00", 4.0, 1.0, "United Kingdom"),    # -8
        ("107", "POST", "POSTAGE", 1, "2011-02-02 10:00", 15.0, 2.0, "France"),            # excluded
    ]
    raw = pd.DataFrame(rows, columns=RAW_COLUMNS)
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
    return clean_transactions(raw)


def test_monthly_net_and_rates(clean):
    m = eda.monthly_summary(clean)
    jan, feb = m.loc["2011-01"], m.loc["2011-02"]
    assert jan["net_sales"] == pytest.approx(80)          # 20 + 20 + 40, error pair nets to 0
    assert feb["net_sales"] == pytest.approx(52)          # 60 - 8
    assert jan["cancellation_rate"] == pytest.approx(2000 / 2080)
    assert jan["genuine_cancellation_rate"] == pytest.approx(0)   # error pair excluded
    assert feb["genuine_cancellation_rate"] == pytest.approx(8 / 60)
    assert jan["trading_days"] == 3
    assert bool(feb["is_partial"])                        # data stops on 2 Feb


def test_net_total_matches_revenue_lines(clean):
    total = clean.loc[clean["IsRevenueLine"], "TotalAmount"].sum()
    assert eda.monthly_summary(clean)["net_sales"].sum() == pytest.approx(total)
    assert eda.daily_summary(clean)["net_sales"].sum() == pytest.approx(total)
    assert eda.country_summary(clean)["net_sales"].sum() == pytest.approx(total)
    assert eda.product_summary(clean)["net_sales"].sum() == pytest.approx(total)


def test_invoice_summary_excludes_reversed_and_non_products(clean):
    inv = eda.invoice_summary(clean)
    assert "102" not in inv.index                          # reversed keying error
    assert "107" not in inv.index                          # postage only
    assert inv.loc["104", "order_value"] == pytest.approx(40)


def test_customer_summary(clean):
    c = eda.customer_summary(clean)
    assert c.loc[1, "orders"] == 2
    assert c.loc[1, "net_sales"] == pytest.approx(32)      # 20 + 20 - 8
    assert c.loc[2, "orders"] == 1                         # reversed order is not an order
    assert c["net_sales"].sum() == pytest.approx(72)       # anonymous sale excluded


def test_product_summary_nets_units(clean):
    p = eda.product_summary(clean)
    assert p.loc["11111", "net_quantity"] == 40            # 10 + 1000 - 1000 + 30
    assert p.loc["22222", "net_quantity"] == 13            # 5 + 10 - 2
    assert "POST" not in p.index


def test_weekday_hour_is_average_per_trading_day(clean):
    mat = eda.weekday_hour_matrix(clean)
    # Wednesday 2011-01-05: 1 trading Wednesday; Wednesday 2011-02-02: second one.
    wed_total = clean[clean["IsRevenueLine"] & (clean["InvoiceDate"].dt.day_name() == "Wednesday")]["TotalAmount"].sum()
    assert mat.loc["Wednesday"].sum() == pytest.approx(wed_total / 2)


def test_pareto():
    curve = eda.pareto_curve([50, 30, 20, -5])
    assert curve["value_share"].iloc[-1] == pytest.approx(1.0)
    assert curve["value_share"].iloc[0] == pytest.approx(0.5)
    assert eda.top_share([80, 10, 5, 5, 0], fraction=0.2) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        eda.pareto_curve([0, -1])

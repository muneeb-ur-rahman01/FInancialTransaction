"""Tests for src/models/predict.py (Module 9).

The key test proves the forecasting feature row built at prediction time is IDENTICAL to the
row Module 6 built for training - otherwise the model would see different inputs in production.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import predict as PR

ROOT = Path(__file__).resolve().parents[1]
ARTEFACTS = [ROOT / "artifacts/models/sales_forecaster.joblib", ROOT / "artifacts/models/demand_forecaster.joblib",
             ROOT / "artifacts/models/customer_segmentation.joblib", ROOT / "artifacts/models/anomaly_detector.joblib",
             ROOT / "data/processed/features_daily.parquet"]
needs_artefacts = pytest.mark.skipif(not all(p.exists() for p in ARTEFACTS), reason="run the pipeline first")


def test_next_trading_days_skips_saturdays():
    days = PR.next_trading_days("2011-12-09", 4)          # Friday
    assert [d.day_name() for d in days] == ["Sunday", "Monday", "Tuesday", "Wednesday"]


@needs_artefacts
def test_feature_row_matches_training_features():
    daily = pd.read_parquet(PR.DAILY_FILE)
    i = 200
    date, prev = daily.index[i], daily.index[i - 1]
    row = PR.daily_feature_row(daily["net_sales"].iloc[:i].to_numpy(), date, int(daily["trend"].iloc[i]), prev)
    expected = daily.iloc[i]
    for col, value in row.items():
        assert value == pytest.approx(expected[col]), col


@needs_artefacts
def test_forecast_sales_shape_and_range():
    f = PR.forecast_sales(days=6)
    assert len(f) == 6 and (f["weekday"] != "Saturday").all()
    assert f["date"].is_monotonic_increasing
    assert (f["low_80"] <= f["forecast"]).all() and (f["forecast"] <= f["high_80"]).all()
    assert f["forecast"].between(1_000, 200_000).all()
    with pytest.raises(ValueError):
        PR.forecast_sales(days=0)


@needs_artefacts
def test_forecast_demand_is_mean_of_last_four_weeks():
    f = PR.forecast_demand(top=20)
    assert len(f) == 20 and f["forecast"].is_monotonic_decreasing
    assert (f["forecast"] - f["mean_last_4_weeks"]).abs().max() <= 0.5      # champion = 4-week mean
    one = PR.forecast_demand(codes=["85123a"])
    assert one["StockCode"].iloc[0] == "85123A"
    with pytest.raises(KeyError):
        PR.forecast_demand(codes=["NOT-A-CODE"])


@needs_artefacts
def test_segments():
    best = PR.segment_for(recency_days=3, frequency=40, monetary=20_000)
    worst = PR.segment_for(recency_days=300, frequency=1, monetary=80)
    assert best["segment"] == "Champions"
    assert worst["segment"] == "Hibernating"
    assert best["action"]
    table = pd.read_parquet(PR.SEGMENTS_FILE)
    cid = int(table["CustomerID"].iloc[0])
    assert PR.customer_segment(cid)["segment"] == table["segment"].iloc[0]
    with pytest.raises(KeyError):
        PR.customer_segment(1)


@needs_artefacts
def test_score_lines_flags_a_keying_error_and_explains_it():
    lines = pd.DataFrame({"StockCode": ["85123A", "85123A"], "Quantity": [6, 60_000], "UnitPrice": [2.55, 2.55]})
    out = PR.score_lines(lines)
    assert not out.loc[0, "flagged"] and out.loc[1, "flagged"]
    assert "usual" in out.loc[1, "reason"]
    assert out.loc[0, "reason"] == "looks normal"
    with pytest.raises(ValueError):
        PR.score_lines(pd.DataFrame({"StockCode": ["85123A"], "Quantity": [-1], "UnitPrice": [1.0]}))

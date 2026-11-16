"""
Module 9 - Prediction System
============================
One place to get predictions from every saved model. Used by main.py (command line) and by
the Streamlit dashboard, so both always give the same answers.

  forecast_sales(days)          next N trading days of net sales (recursive, with an 80% range)
  forecast_demand(top, codes)   next week's units per product
  customer_segment(customer_id) segment of an existing customer
  segment_for(recency, frequency, monetary)   segment for any RFM profile ("what-if")
  score_lines(lines)            anomaly score + alert flag + plain-language reason per order line

Every function loads the artefacts written by Module 7 (artifacts/models/*.joblib) and the
processed data; nothing is retrained here.
"""

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.data_processing.features import DAILY_FILE, build_product_weekly_features
from src.models.anomaly_detection import MODEL_FILE as ANOMALY_FILE
from src.models.customer_segmentation import ACTIONS, MODEL_FILE as SEGMENT_FILE, SEGMENTS_FILE
from src.models.demand_forecasting import MODEL_FILE as DEMAND_FILE
from src.models.sales_forecasting import MODEL_FILE as SALES_FILE
from src.utils.paths import CLEAN_DATA_FILE, MODEL_REPORTS_DIR

MAX_SALES_HORIZON = 24          # 4 trading weeks; recursive errors compound beyond this
CLOSED_WEEKDAY = 5              # Saturday: the business never trades (Module 4)


def _require(path, command):
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. Build it first with:  {command}")


@lru_cache(maxsize=None)
def load_bundle(name):
    paths = {"sales": (SALES_FILE, "python -m src.models.sales_forecasting"),
             "demand": (DEMAND_FILE, "python -m src.models.demand_forecasting"),
             "segments": (SEGMENT_FILE, "python -m src.models.customer_segmentation"),
             "anomaly": (ANOMALY_FILE, "python -m src.models.anomaly_detection")}
    path, command = paths[name]
    _require(path, command)
    return joblib.load(path)


# --------------------------------------------------------------------------
# Sales: recursive multi-day forecast
# --------------------------------------------------------------------------
def next_trading_days(after, n):
    """The next n trading days after `after` (every day except Saturday).

    Public holidays are not known to the model: last year the business closed 24 Dec - 3 Jan.
    """
    days, d = [], pd.Timestamp(after).normalize()
    while len(days) < n:
        d += pd.Timedelta(days=1)
        if d.day_of_week != CLOSED_WEEKDAY:
            days.append(d)
    return pd.DatetimeIndex(days)


def daily_feature_row(history, date, trend, prev_date):
    """Features for one future trading day, built exactly like Module 6's daily table.

    history: net sales of all previous trading days (oldest first), actual or forecast.
    """
    y = np.asarray(history, dtype=float)
    lag = lambda k: y[-k] if len(y) >= k else np.nan  # noqa: E731
    row = {
        "weekday": date.day_of_week, "is_sunday": int(date.day_of_week == 6), "month": date.month,
        "day_of_month": date.day, "week_of_year": int(date.isocalendar().week), "day_of_year": date.day_of_year,
        "trend": trend, "days_since_prev_trading_day": (date - prev_date).days,
    }
    for k in (1, 2, 3, 6, 12, 18, 24):
        row[f"sales_lag_{k}"] = lag(k)
    for w in (6, 24):
        window = y[-w:]
        row[f"sales_roll_mean_{w}"] = window.mean() if len(window) == w else np.nan
        row[f"sales_roll_std_{w}"] = window.std(ddof=1) if len(window) == w else np.nan
    same = [lag(k) for k in (6, 12, 18, 24)]
    row["sales_same_weekday_mean_4"] = np.mean(same) if not np.isnan(same).any() else np.nan
    return row


def _residual_quantiles(lower=0.10, upper=0.90):
    """Empirical one-day-ahead errors of the champion on the test period -> an 80% range."""
    path = MODEL_REPORTS_DIR / "sales_forecasting_test_predictions.csv"
    bundle = load_bundle("sales")
    if not path.exists():
        return 0.0, 0.0
    preds = pd.read_csv(path, index_col=0)
    col = f"{bundle['model_family']} ({bundle['model_name']})"
    if col not in preds:
        return 0.0, 0.0
    resid = preds["actual"] - preds[col]
    return float(resid.quantile(lower)), float(resid.quantile(upper))


def forecast_sales(days=12):
    """Forecast the next `days` trading days; each forecast is fed back as history for the next.

    Returns columns: date, weekday, forecast, low_80, high_80.
    The 80% range comes from one-day-ahead test errors, so it is optimistic for later days.
    """
    if not 1 <= days <= MAX_SALES_HORIZON:
        raise ValueError(f"days must be between 1 and {MAX_SALES_HORIZON}")
    bundle = load_bundle("sales")
    _require(DAILY_FILE, "python -m src.data_processing.features")
    daily = pd.read_parquet(DAILY_FILE)
    history = list(daily["net_sales"].to_numpy())
    trend = int(daily["trend"].iloc[-1])
    prev = daily.index[-1]
    lo, hi = _residual_quantiles()

    rows = []
    for date in next_trading_days(prev, days):
        trend += 1
        feats = pd.DataFrame([daily_feature_row(history, date, trend, prev)])[bundle["features"]]
        value = float(bundle["model"].predict(feats)[0])
        rows.append({"date": date, "weekday": date.day_name(), "forecast": value,
                     "low_80": max(0.0, value + lo), "high_80": value + hi})
        history.append(value)
        prev = date
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Demand: next week per product
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _next_week_features():
    _require(CLEAN_DATA_FILE, "python -m src.data_processing.cleaning")
    df = pd.read_parquet(CLEAN_DATA_FILE)
    panel = build_product_weekly_features(df, future_weeks=1)
    next_week = panel["week"].max()
    return panel[panel["week"] == next_week].set_index("StockCode"), next_week


def forecast_demand(top=None, codes=None):
    """Next week's units per product (products with >= 4 weeks of history).

    Returns columns: StockCode, product, week, forecast, last_week, mean_last_4_weeks.
    """
    bundle = load_bundle("demand")
    rows, week = _next_week_features()
    rows = rows.dropna(subset=bundle["features"])
    if codes:
        codes = [str(c).strip().upper() for c in codes]
        missing = [c for c in codes if c not in rows.index]
        if missing:
            raise KeyError(f"No forecast for {missing}: unknown code or under 4 weeks of sales history")
        rows = rows.loc[codes]
    out = pd.DataFrame({
        "product": rows["product"], "week": week,
        "forecast": np.round(bundle["model"].predict(rows[bundle["features"]])).astype(int),
        "last_week": rows["units_lag_1"].astype(int),
        "mean_last_4_weeks": rows["units_roll_mean_4"].round(1),
    }).reset_index()
    out = out.sort_values("forecast", ascending=False).reset_index(drop=True)
    return out.head(top) if top else out


# --------------------------------------------------------------------------
# Customer segments
# --------------------------------------------------------------------------
def segment_for(recency_days, frequency, monetary):
    """Segment for any RFM profile, using the saved preprocessing + K-Means pipeline."""
    if frequency < 1 or recency_days < 0:
        raise ValueError("frequency must be >= 1 and recency_days >= 0")
    bundle = load_bundle("segments")
    X = pd.DataFrame({"recency_days": [recency_days], "frequency": [frequency], "monetary": [monetary]})
    cluster = int(bundle["model"].predict(X[bundle["features"]])[0])
    name = bundle["cluster_names"][cluster]
    return {"cluster": cluster, "segment": name, "action": ACTIONS.get(name.split(" (")[0], "")}


@lru_cache(maxsize=1)
def _segments_table():
    _require(SEGMENTS_FILE, "python -m src.models.customer_segmentation")
    return pd.read_parquet(SEGMENTS_FILE).set_index("CustomerID")


def customer_segment(customer_id):
    table = _segments_table()
    cid = int(customer_id)
    if cid not in table.index:
        raise KeyError(f"Customer {cid} not found (unknown ID, or no real order before the snapshot)")
    row = table.loc[cid]
    return {"customer_id": cid, "segment": row["segment"], "lapsing": bool(row["lapsing"]),
            "recency_days": int(row["recency_days"]), "orders": int(row["frequency"]),
            "net_spend": float(row["monetary"]), "country": row["country"],
            "action": ACTIONS.get(row["segment"].split(" (")[0], "")}


# --------------------------------------------------------------------------
# Anomaly scoring of new order lines
# --------------------------------------------------------------------------
def score_lines(lines):
    """Score order lines with the saved detector.

    lines: DataFrame with StockCode, Quantity, UnitPrice and optionally InvoiceNo, InvoiceDate,
           Country, CustomerID. Missing optional columns get neutral defaults.
    Returns the input plus anomaly_score, flagged, usual_qty, usual_price, reason.
    """
    bundle = load_bundle("anomaly")
    scorer = bundle["scorer"]
    lines = lines.copy().reset_index(drop=True)
    lines["StockCode"] = lines["StockCode"].astype(str).str.strip().str.upper()
    defaults = {"InvoiceNo": "NEW", "InvoiceDate": pd.Timestamp.now().floor("min"),
                "Country": "United Kingdom", "CustomerID": np.nan}
    for col, value in defaults.items():
        if col not in lines:
            lines[col] = value
    lines["InvoiceDate"] = pd.to_datetime(lines["InvoiceDate"])
    if (lines["Quantity"] <= 0).any() or (lines["UnitPrice"] <= 0).any():
        raise ValueError("Quantity and UnitPrice must be positive for sale lines")

    scores = scorer.score(lines)
    norms = scorer.norms_["table"].reindex(lines["StockCode"])
    usual_qty = norms["usual_qty"].to_numpy()
    usual_price = norms["usual_price"].to_numpy()
    out = lines.assign(anomaly_score=scores, flagged=scores >= bundle["threshold"],
                       usual_qty=usual_qty, usual_price=usual_price)
    out["reason"] = [_reason(r) for r in out.itertuples()]
    return out


def _reason(r):
    parts = []
    if np.isnan(r.usual_qty):
        parts.append("new product (no sales history)")
    else:
        if r.Quantity >= 5 * r.usual_qty:
            parts.append(f"quantity {r.Quantity:,} is {r.Quantity / r.usual_qty:,.0f}x the usual {r.usual_qty:,.0f}")
        if r.UnitPrice >= 3 * r.usual_price:
            parts.append(f"price £{r.UnitPrice:,.2f} is {r.UnitPrice / r.usual_price:,.0f}x the usual "
                         f"£{r.usual_price:,.2f}")
    if r.flagged and not parts:
        parts.append(f"line value £{r.Quantity * r.UnitPrice:,.0f} is in the top 1% of all lines")
    return "; ".join(parts) if parts else "looks normal"

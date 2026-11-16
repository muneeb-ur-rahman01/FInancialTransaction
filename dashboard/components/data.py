"""Data access for the dashboard: cached loaders, global filters, cached aggregations.

The dashboard never recomputes models. It reads what the pipeline produced:
  data/processed/*.parquet          (Modules 3 and 6)
  reports/model_reports/*           (Module 7)
and reuses the tested aggregation functions in src/analysis/eda.py, so every
number on screen matches the reports.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import eda  # noqa: E402
from src.data_processing.features import CUSTOMER_FILE, DAILY_FILE, INVOICE_FILE, PRODUCT_WEEKLY_FILE  # noqa: E402
from src.utils.paths import CLEAN_DATA_FILE, MODEL_REPORTS_DIR  # noqa: E402

HOW_TO_BUILD = {
    "clean": "python -m src.data_processing.cleaning",
    "features": "python -m src.data_processing.features",
    "sales": "python -m src.models.sales_forecasting",
    "demand": "python -m src.models.demand_forecasting",
    "segments": "python -m src.models.customer_segmentation",
    "anomaly": "python -m src.models.anomaly_detection",
    "report": "python -m src.models.evaluation_report",
}


def require(path, step):
    """Stop the page with a clear instruction if a pipeline output is missing."""
    if not Path(path).exists():
        st.error(f"Missing `{Path(path).relative_to(PROJECT_ROOT)}`. Run this from the project root first:")
        st.code(HOW_TO_BUILD[step], language="powershell")
        st.stop()


# --------------------------------------------------------------------------
# Loaders (cache_resource = loaded once, shared, never copied; treat as read-only)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading transactions...")
def load_clean():
    require(CLEAN_DATA_FILE, "clean")
    return pd.read_parquet(CLEAN_DATA_FILE)


@st.cache_resource(show_spinner=False)
def load_parquet(path, step):
    require(path, step)
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_report(name, step):
    path = MODEL_REPORTS_DIR / name
    require(path, step)
    return json.loads(path.read_text(encoding="utf-8"))


def load_product_weekly():
    return load_parquet(PRODUCT_WEEKLY_FILE, "features")


def load_customers():
    return load_parquet(CUSTOMER_FILE, "features")


def load_invoices():
    return load_parquet(INVOICE_FILE, "features")


def load_flagged_lines():
    from src.models.anomaly_detection import FLAGGED_FILE

    return load_parquet(FLAGGED_FILE, "anomaly")


def load_invoice_scores():
    from src.models.anomaly_detection import INVOICE_SCORES_FILE

    return load_parquet(INVOICE_SCORES_FILE, "anomaly")


# --------------------------------------------------------------------------
# Module 9 predictions (same functions as `python main.py ...`)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Forecasting the next trading days...")
def sales_forecast(days):
    from src.models.predict import forecast_sales

    return forecast_sales(days)


@st.cache_data(show_spinner="Building next week's features for every product...")
def demand_forecast():
    from src.models.predict import forecast_demand

    return forecast_demand()


def load_daily_history():
    return load_parquet(DAILY_FILE, "features")


def load_segments():
    from src.models.customer_segmentation import SEGMENTS_FILE

    return load_parquet(SEGMENTS_FILE, "segments")


@st.cache_data(show_spinner=False)
def load_sales_predictions():
    path = MODEL_REPORTS_DIR / "sales_forecasting_test_predictions.csv"
    require(path, "sales")
    return pd.read_csv(path, index_col=0, parse_dates=True)


@st.cache_resource(show_spinner=False)
def load_demand_predictions():
    path = MODEL_REPORTS_DIR / "demand_forecasting_test_predictions.parquet"
    require(path, "demand")
    return pd.read_parquet(path)


# --------------------------------------------------------------------------
# Global filters (sidebar). Values live in session_state so they survive page switches.
# --------------------------------------------------------------------------
def sidebar_filters():
    df = load_clean()
    lo, hi = df["InvoiceDate"].min().date(), df["InvoiceDate"].max().date()
    ss = st.session_state
    ss.setdefault("f_dates", (lo, hi))
    ss.setdefault("f_countries", [])

    st.sidebar.header("Filters")
    dates = st.sidebar.date_input("Date range", value=ss["f_dates"], min_value=lo, max_value=hi)
    if isinstance(dates, tuple) and len(dates) == 2:
        ss["f_dates"] = dates
    countries = sorted(df["Country"].astype(str).unique())
    ss["f_countries"] = st.sidebar.multiselect("Countries", countries, default=ss["f_countries"],
                                               placeholder="All countries")
    st.sidebar.caption("Filters apply to the analysis pages. Model pages always show the full evaluation.")
    start, end = ss["f_dates"]
    return str(start), str(end), tuple(ss["f_countries"])


def filtered(key):
    start, end, countries = key
    df = load_clean()
    mask = (df["InvoiceDate"] >= start) & (df["InvoiceDate"] < pd.Timestamp(end) + pd.Timedelta(days=1))
    if countries:
        mask &= df["Country"].isin(countries)
    return df[mask]


# --------------------------------------------------------------------------
# Cached aggregations, keyed by the filter values (not by the DataFrame)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def kpis(key):
    df = filtered(key)
    rev = eda.revenue_lines(df)
    sales = eda.sale_lines(df)
    gross = rev.loc[rev["TotalAmount"] > 0, "TotalAmount"].sum()
    returned = -rev.loc[rev["TotalAmount"] < 0, "TotalAmount"].sum()
    orders = sales["InvoiceNo"].nunique()
    order_values = sales.groupby("InvoiceNo")["TotalAmount"].sum()
    return {
        "net_sales": rev["TotalAmount"].sum(),
        "orders": orders,
        "customers": sales["CustomerID"].nunique(),
        "median_order": order_values.median() if orders else 0.0,
        "return_rate": returned / gross if gross else 0.0,
        "rows": len(df),
    }


@st.cache_data(show_spinner=False)
def monthly(key):
    return eda.monthly_summary(filtered(key))


@st.cache_data(show_spinner=False)
def daily(key):
    return eda.daily_summary(filtered(key))


@st.cache_data(show_spinner=False)
def countries(key):
    return eda.country_summary(filtered(key))


@st.cache_data(show_spinner=False)
def products(key):
    return eda.product_summary(filtered(key))


@st.cache_data(show_spinner=False)
def invoices(key):
    return eda.invoice_summary(filtered(key))


@st.cache_data(show_spinner=False)
def customers(key):
    return eda.customer_summary(filtered(key))


@st.cache_data(show_spinner=False)
def weekday_hour(key):
    return eda.weekday_hour_matrix(filtered(key))


@st.cache_data(show_spinner=False)
def country_month(key):
    return eda.country_month_share(filtered(key), top_n=8)


@st.cache_data(show_spinner=False)
def transaction_types(key):
    df = filtered(key)
    out = df.groupby("TransactionType", observed=True).agg(rows=("RowID", "size"), amount=("TotalAmount", "sum"))
    quality = {
        "duplicates": int(df["IsDuplicate"].sum()),
        "entry_corrections": int((df["IsEntryCorrection"] & df["IsReversedSale"]).sum()),
        "entry_correction_value": float(df.loc[df["IsEntryCorrection"] & df["IsReversedSale"], "TotalAmount"].sum()),
        "missing_customer_share": float(df["CustomerID"].isna().mean()) if len(df) else 0.0,
        "non_product_rows": int((~df["IsProduct"]).sum()),
    }
    return out.reset_index(), quality


@st.cache_data(show_spinner=False)
def price_bands(key):
    s = eda.sale_lines(filtered(key))
    bands = pd.cut(s["UnitPrice"], [0, 1, 2, 5, 10, float("inf")], labels=["< £1", "£1–2", "£2–5", "£5–10", "> £10"])
    return s.groupby(bands, observed=True)["Quantity"].median().rename("median_quantity").reset_index()

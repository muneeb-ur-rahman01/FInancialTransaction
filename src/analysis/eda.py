"""
Module 4 - Exploratory Data Analysis: reusable aggregations
===========================================================
Every function takes the CLEAN table (data/processed/transactions_clean.parquet)
and returns a small DataFrame. No plotting here, so the same numbers can feed
the EDA script, the statistics module and later the Streamlit dashboard.

Definitions used everywhere (agreed in Module 3):
  * Revenue = NET product sales = sum of TotalAmount over IsRevenueLine rows
    (product sales are positive, product cancellations negative).
  * Order counts use SALE invoices only (a cancellation is not an order).
  * Order-size statistics exclude ORDER-ENTRY CORRECTIONS (IsEntryCorrection:
    a sale cancelled with identical quantity/price within 1 hour, e.g. the
    80,995-unit keying error), because that order never really happened.
    Slower exact reversals are genuine returns and are kept.
"""

import numpy as np
import pandas as pd

from src.utils.paths import CLEAN_DATA_FILE

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def load_clean(path=CLEAN_DATA_FILE):
    return pd.read_parquet(path)


def revenue_lines(df):
    """Product sales + product cancellations: the rows that make up net revenue."""
    return df[df["IsRevenueLine"]]


def sale_lines(df, exclude_corrections=True):
    """Product sale lines only (orders that really happened, if exclude_corrections)."""
    mask = (df["TransactionType"] == "sale") & df["IsProduct"]
    if exclude_corrections:
        mask &= ~df["IsEntryCorrection"]
    return df[mask]


# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------
def monthly_summary(df):
    """Gross, cancelled and net sales per month, plus activity counts.

    net_per_trading_day makes a partial month (Dec 2011 has 8 trading days)
    comparable with full months.
    """
    rev = revenue_lines(df).assign(
        month=lambda d: d["InvoiceDate"].dt.to_period("M"),
        gross=lambda d: d["TotalAmount"].clip(lower=0),
        cancelled=lambda d: d["TotalAmount"].clip(upper=0),
    )
    sales = rev[rev["TransactionType"] == "sale"].assign(day=lambda d: d["InvoiceDate"].dt.normalize())

    out = rev.groupby("month").agg(
        gross_sales=("gross", "sum"),
        cancellations=("cancelled", "sum"),
        net_sales=("TotalAmount", "sum"),
    )
    by_month = sales.groupby("month")
    out["invoices"] = by_month["InvoiceNo"].nunique()
    out["customers"] = by_month["CustomerID"].nunique()
    out["trading_days"] = by_month["day"].nunique()
    out["net_per_trading_day"] = out["net_sales"] / out["trading_days"]
    out["cancellation_rate"] = -out["cancellations"] / out["gross_sales"]

    # Same rate without order-entry corrections (e.g. the 80,995-unit order cancelled
    # 12 minutes later); this is the rate of genuine customer returns.
    genuine = rev[~rev["IsEntryCorrection"]]
    g = genuine.groupby("month")
    out["genuine_cancellation_rate"] = -g["cancelled"].sum() / g["gross"].sum()

    last_date = df["InvoiceDate"].max()
    out["is_partial"] = (out.index == last_date.to_period("M")) & (
        last_date.normalize() < out.index.max().to_timestamp(how="end").normalize()
    )
    return out


def daily_summary(df, window=7):
    """Net sales per trading day with a rolling mean over `window` trading days."""
    rev = revenue_lines(df)
    daily = rev.groupby(rev["InvoiceDate"].dt.normalize())["TotalAmount"].sum().rename("net_sales").to_frame()
    daily.index.name = "date"
    daily["rolling_mean"] = daily["net_sales"].rolling(window, min_periods=window).mean()
    return daily


def weekday_hour_matrix(df):
    """Average net sales per trading day for every weekday x hour cell.

    Dividing by the number of trading days of each weekday removes the bias
    of some weekdays occurring more often in the year (47 Mondays vs 53 Wednesdays).
    """
    rev = revenue_lines(df).assign(
        day=lambda d: d["InvoiceDate"].dt.normalize(),
        weekday=lambda d: d["InvoiceDate"].dt.day_name(),
        hour=lambda d: d["InvoiceDate"].dt.hour,
    )
    totals = rev.pivot_table(index="weekday", columns="hour", values="TotalAmount", aggfunc="sum", fill_value=0)
    days_per_weekday = rev.groupby("weekday")["day"].nunique()
    matrix = totals.div(days_per_weekday, axis=0)
    order = [d for d in WEEKDAY_ORDER if d in matrix.index]
    return matrix.loc[order]


# --------------------------------------------------------------------------
# Geography, products, customers
# --------------------------------------------------------------------------
def country_summary(df):
    rev = revenue_lines(df)
    sales = sale_lines(df)
    out = rev.groupby("Country", observed=True).agg(net_sales=("TotalAmount", "sum"))
    out["customers"] = sales.groupby("Country", observed=True)["CustomerID"].nunique()
    out["invoices"] = sales.groupby("Country", observed=True)["InvoiceNo"].nunique()
    out["avg_order_value"] = sales.groupby("Country", observed=True)["TotalAmount"].sum() / out["invoices"]
    out["share"] = out["net_sales"] / out["net_sales"].sum()
    return out.sort_values("net_sales", ascending=False)


def product_summary(df):
    """Net units and net revenue per product (cancellations already subtracted)."""
    rev = revenue_lines(df)
    sales = sale_lines(df)
    out = rev.groupby("StockCode").agg(
        product=("ProductName", "first"),
        net_quantity=("Quantity", "sum"),
        net_sales=("TotalAmount", "sum"),
    )
    out["invoices"] = sales.groupby("StockCode")["InvoiceNo"].nunique()
    out["customers"] = sales.groupby("StockCode")["CustomerID"].nunique()
    out["median_price"] = sales.groupby("StockCode")["UnitPrice"].median()
    return out.fillna({"invoices": 0, "customers": 0}).astype({"invoices": int, "customers": int})


def invoice_summary(df):
    """One row per real order: value, number of lines and units."""
    sales = sale_lines(df)
    return sales.groupby("InvoiceNo").agg(
        date=("InvoiceDate", "min"),
        customer=("CustomerID", "first"),
        country=("Country", "first"),
        lines=("StockCode", "size"),
        units=("Quantity", "sum"),
        order_value=("TotalAmount", "sum"),
    )


def customer_summary(df):
    """Per known customer: orders, net spend, average order value, activity window."""
    rev = revenue_lines(df)
    rev = rev[rev["CustomerID"].notna()]
    sales = sale_lines(df)
    sales = sales[sales["CustomerID"].notna()]

    out = rev.groupby("CustomerID").agg(net_sales=("TotalAmount", "sum"))
    by_cust = sales.groupby("CustomerID")
    # Customers whose only activity was a return/reversed order have 0 real orders.
    out["orders"] = by_cust["InvoiceNo"].nunique().reindex(out.index).fillna(0).astype(int)
    out["avg_order_value"] = by_cust["TotalAmount"].sum() / out["orders"].replace(0, np.nan)
    out["first_purchase"] = by_cust["InvoiceDate"].min()
    out["last_purchase"] = by_cust["InvoiceDate"].max()
    out["country"] = by_cust["Country"].first()
    return out


def pareto_curve(values):
    """Cumulative share of the total held by the largest entities.

    Returns columns: entity_share (0..1) and value_share (0..1).
    Negative values (net refunders) are treated as 0 for the curve.
    """
    v = np.sort(np.clip(np.asarray(values, dtype=float), 0, None))[::-1]
    total = v.sum()
    if total == 0:
        raise ValueError("Pareto curve needs a positive total")
    n = len(v)
    return pd.DataFrame({
        "entity_share": np.arange(1, n + 1) / n,
        "value_share": np.cumsum(v) / total,
    })


def top_share(values, fraction=0.2):
    """Share of the total held by the top `fraction` of entities (e.g. top 20%)."""
    curve = pareto_curve(values)
    k = max(1, int(np.ceil(fraction * len(curve))))
    return float(curve["value_share"].iloc[k - 1])


def country_month_share(df, top_n=8):
    """For the top countries: each month's share of that country's yearly net sales.

    Normalising per country lets a small country's seasonality be compared
    with the UK's on one colour scale.
    """
    rev = revenue_lines(df).assign(month=lambda d: d["InvoiceDate"].dt.to_period("M"))
    top = country_summary(df).head(top_n).index
    table = (
        rev[rev["Country"].isin(top)]
        .pivot_table(index="Country", columns="month", values="TotalAmount", aggfunc="sum", fill_value=0, observed=True)
        .reindex(top)
    )
    return table.div(table.sum(axis=1), axis=0)

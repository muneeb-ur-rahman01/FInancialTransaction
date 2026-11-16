"""
Module 6 - Feature Engineering
==============================
Builds one feature table per machine-learning objective:

  Table                         Row = one ...          Used for (Module 7)
  ----------------------------  ---------------------  --------------------------
  features_daily.parquet        trading day            sales forecasting
  features_product_weekly       product x week         product demand prediction
  features_customers (RFM)      customer               customer segmentation
  features_invoices             order (invoice)        anomaly detection

THE ANTI-LEAKAGE RULE
  A feature for time t may only use information known BEFORE t.
  * Lags use .shift(k) with k >= 1  -> value from k periods ago.
  * Rolling statistics are computed on the SHIFTED series, so the current
    value is never part of its own feature.
  * "History" features (customer's previous orders, product's past demand)
    use expanding statistics shifted by one.
  * RFM is computed from transactions strictly before a snapshot date.
  The tests in tests/test_features.py prove this by changing a future value
  and checking that no past feature changes.

Run from the project root:
    python -m src.data_processing.features
"""

import numpy as np
import pandas as pd

from src.analysis import eda
from src.analysis.statistical_tests import daily_features
from src.utils.paths import PROCESSED_DIR

DAILY_FILE = PROCESSED_DIR / "features_daily.parquet"
PRODUCT_WEEKLY_FILE = PROCESSED_DIR / "features_product_weekly.parquet"
CUSTOMER_FILE = PROCESSED_DIR / "features_customers.parquet"
INVOICE_FILE = PROCESSED_DIR / "features_invoices.parquet"


# --------------------------------------------------------------------------
# Transaction level: calendar features
# --------------------------------------------------------------------------
def add_time_features(df, date_col="InvoiceDate"):
    """Year, Month, Day, Weekday, Hour ... taken from the timestamp itself (no leakage possible)."""
    out = df.copy()
    d = out[date_col].dt
    out["Year"] = d.year.astype("int16")
    out["Quarter"] = d.quarter.astype("int8")
    out["Month"] = d.month.astype("int8")
    out["WeekOfYear"] = d.isocalendar().week.astype("int8").to_numpy()
    out["Day"] = d.day.astype("int8")
    out["DayOfYear"] = d.day_of_year.astype("int16")
    out["Weekday"] = d.day_of_week.astype("int8")        # 0 = Monday ... 6 = Sunday
    out["WeekdayName"] = d.day_name()
    out["Hour"] = d.hour.astype("int8")
    return out


def week_start(dates):
    """Monday of the week each timestamp falls in (weeks run Monday-Sunday)."""
    return dates.dt.to_period("W-SUN").dt.start_time


# --------------------------------------------------------------------------
# 1. Daily table - sales forecasting
# --------------------------------------------------------------------------
def build_daily_features(df, lags=(1, 2, 3, 6, 12, 18, 24), windows=(6, 24)):
    """One row per trading day. Target: net_sales.

    Lags are counted in TRADING days. Module 5 showed a 6-trading-day cycle,
    so lag 6 = same weekday last week. Calendar features describe the target
    day itself; they are known in advance, so they are allowed.
    """
    base = daily_features(df)
    y = base["net_sales"]

    out = pd.DataFrame(index=base.index)
    out["net_sales"] = y                                               # TARGET

    # Calendar (known in advance)
    idx = out.index
    out["weekday"] = idx.day_of_week.astype("int8")
    out["is_sunday"] = (idx.day_of_week == 6).astype("int8")
    out["month"] = idx.month.astype("int8")
    out["day_of_month"] = idx.day.astype("int8")
    out["week_of_year"] = idx.isocalendar().week.astype("int8").to_numpy()
    out["day_of_year"] = idx.day_of_year.astype("int16")
    out["trend"] = np.arange(len(out), dtype="int32")                  # trading-day counter
    gap = idx.to_series().diff().dt.days
    out["days_since_prev_trading_day"] = gap.fillna(1).astype("int16").to_numpy()   # >2 after a closure

    # Lags of the target
    for k in lags:
        out[f"sales_lag_{k}"] = y.shift(k)

    # Rolling statistics on the PAST only (shift first, then roll)
    past = y.shift(1)
    for w in windows:
        out[f"sales_roll_mean_{w}"] = past.rolling(w, min_periods=w).mean()
        out[f"sales_roll_std_{w}"] = past.rolling(w, min_periods=w).std()

    # Average of the same weekday over the last 4 weeks
    out["sales_same_weekday_mean_4"] = pd.concat([y.shift(k) for k in (6, 12, 18, 24)], axis=1).mean(axis=1, skipna=False)

    # Lagged activity drivers (Module 5: number of orders drives revenue)
    for col in ("orders", "customers", "avg_order_value"):
        out[f"{col}_lag_1"] = base[col].shift(1)
        out[f"{col}_lag_6"] = base[col].shift(6)

    return out


# --------------------------------------------------------------------------
# 2. Product x week table - demand prediction
# --------------------------------------------------------------------------
def build_product_weekly_features(df, min_active_weeks=20, lags=(1, 2, 3, 4), window=4, future_weeks=0):
    """One row per product per week, from the product's first sale week onwards.

    future_weeks > 0 appends that many weeks AFTER the data (Module 9 forecasting). Their
    units_sold is left missing (unknown), while lag/rolling features are computed from the real
    history exactly as for past weeks. They are assumed to have 6 trading days (Mon-Fri + Sun).

    Target: units_sold = units on real sale lines (order-entry corrections excluded).
    Demand is measured when customers order; a later return does not mean they
    wanted less at the time, so returns are not subtracted here (unlike revenue).

    Weeks with no sales get units_sold = 0: "no demand" is information, and
    leaving those weeks out would make demand look higher than it is.
    Only products selling in >= min_active_weeks weeks are kept, so every
    product has enough history to learn from.
    """
    sales = eda.sale_lines(df).assign(week=lambda d: week_start(d["InvoiceDate"]))
    agg = sales.groupby(["StockCode", "week"]).agg(
        units_sold=("Quantity", "sum"),
        orders=("InvoiceNo", "nunique"),
        customers=("CustomerID", "nunique"),
        median_price=("UnitPrice", "median"),
    )

    active = agg.groupby(level="StockCode").size()
    products = active[active >= min_active_weeks].index
    agg = agg.loc[agg.index.get_level_values("StockCode").isin(products)]

    # Calendar of all weeks + number of trading days the business was open in each
    all_days = df["InvoiceDate"].dt.normalize().drop_duplicates()
    trading_days = all_days.groupby(week_start(all_days)).size()
    weeks = trading_days.index.sort_values()
    last_actual_week = weeks.max()
    if future_weeks:
        future = pd.date_range(last_actual_week + pd.Timedelta(weeks=1), periods=future_weeks, freq="7D")
        trading_days = pd.concat([trading_days, pd.Series(6, index=future)])
        weeks = trading_days.index.sort_values()

    grid = pd.MultiIndex.from_product([products, weeks], names=["StockCode", "week"])
    panel = agg.reindex(grid).reset_index()
    first_week = agg.reset_index().groupby("StockCode")["week"].min()
    panel = panel[panel["week"] >= panel["StockCode"].map(first_week)].copy()
    panel = panel.sort_values(["StockCode", "week"]).reset_index(drop=True)

    known = panel["week"] <= last_actual_week          # only real weeks get "no sales = 0"
    cols = ["units_sold", "orders", "customers"]
    panel.loc[known, cols] = panel.loc[known, cols].fillna(0)
    names = sales.drop_duplicates("StockCode").set_index("StockCode")["ProductName"]
    panel["product"] = panel["StockCode"].map(names)

    g = panel.groupby("StockCode", sort=False)
    y = g["units_sold"]

    # Calendar (known in advance)
    panel["week_of_year"] = panel["week"].dt.isocalendar().week.astype("int8").to_numpy()
    panel["month"] = panel["week"].dt.month.astype("int8")
    panel["trading_days"] = panel["week"].map(trading_days).astype("int8")
    panel["product_age_weeks"] = g.cumcount().astype("int16")

    # Lags and rolling statistics of demand (past weeks only)
    for k in lags:
        panel[f"units_lag_{k}"] = y.shift(k)
    past = y.shift(1)
    panel[f"units_roll_mean_{window}"] = past.groupby(panel["StockCode"]).transform(
        lambda s: s.rolling(window, min_periods=window).mean())
    panel[f"units_roll_std_{window}"] = past.groupby(panel["StockCode"]).transform(
        lambda s: s.rolling(window, min_periods=window).std())
    panel["units_expanding_mean"] = past.groupby(panel["StockCode"]).transform(lambda s: s.expanding().mean())
    panel["orders_lag_1"] = g["orders"].shift(1)
    panel["customers_lag_1"] = g["customers"].shift(1)

    # Price: last price seen BEFORE this week (this week's price is only known after selling)
    panel["price_lag_1"] = g["median_price"].transform(lambda s: s.ffill().shift(1))
    panel = panel.drop(columns=["median_price", "orders", "customers"])
    return panel


# --------------------------------------------------------------------------
# 3. Customer table - RFM segmentation
# --------------------------------------------------------------------------
def build_customer_features(df, snapshot=None):
    """RFM and behaviour features per known customer, using only data BEFORE `snapshot`.

    Recency   : days since last real order (smaller = better)
    Frequency : number of real orders (distinct sale invoices)
    Monetary  : net spend (sales minus returns)
    Customers who never placed a real order (only returns) are excluded:
    they have no Recency and cannot be segmented as buyers.
    """
    if snapshot is None:
        snapshot = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    snapshot = pd.Timestamp(snapshot)
    hist = df[df["InvoiceDate"] < snapshot]

    sales = eda.sale_lines(hist)
    sales = sales[sales["CustomerID"].notna()]
    rev = eda.revenue_lines(hist)
    rev = rev[rev["CustomerID"].notna()]

    g = sales.groupby("CustomerID")
    out = pd.DataFrame({
        "last_purchase": g["InvoiceDate"].max(),
        "first_purchase": g["InvoiceDate"].min(),
        "frequency": g["InvoiceNo"].nunique(),
        "gross_spend": g["TotalAmount"].sum(),
        "total_units": g["Quantity"].sum(),
        "distinct_products": g["StockCode"].nunique(),
        "country": g["Country"].first().astype(str),
    })
    net = rev.groupby("CustomerID")["TotalAmount"].sum()
    returned = -rev.loc[rev["TotalAmount"] < 0].groupby("CustomerID")["TotalAmount"].sum()

    out["monetary"] = net.reindex(out.index)
    out["recency_days"] = (snapshot - out["last_purchase"]).dt.days
    out["tenure_days"] = (snapshot - out["first_purchase"]).dt.days
    out["avg_order_value"] = out["gross_spend"] / out["frequency"]
    active_span = (out["last_purchase"] - out["first_purchase"]).dt.days
    out["avg_days_between_orders"] = (active_span / (out["frequency"] - 1)).where(out["frequency"] > 1)
    out["orders_per_month"] = out["frequency"] / (out["tenure_days"] / 30.44).clip(lower=1)
    out["return_rate"] = (returned.reindex(out.index).fillna(0) / out["gross_spend"]).clip(0, 1)
    out["is_uk"] = (out["country"] == "United Kingdom").astype("int8")

    # Classic RFM scores 1-5 (5 = best). rank(method="first") breaks the many ties
    # in frequency, otherwise qcut cannot form 5 equal groups.
    out["R_score"] = pd.qcut(out["recency_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    out["F_score"] = pd.qcut(out["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["M_score"] = pd.qcut(out["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["RFM_score"] = out["R_score"] + out["F_score"] + out["M_score"]

    out["snapshot"] = snapshot
    out.index = out.index.astype("int64")
    return out


# --------------------------------------------------------------------------
# 4. Invoice table - anomaly detection
# --------------------------------------------------------------------------
def build_invoice_features(df):
    """One row per sale invoice, INCLUDING order-entry corrections (we want to detect them).

    has_entry_correction is a LABEL for evaluating the detector later - it is
    never to be used as an input feature (it is only known after the cancellation).
    """
    lines = df[(df["TransactionType"] == "sale") & df["IsProduct"]]
    inv = lines.groupby("InvoiceNo").agg(
        date=("InvoiceDate", "min"),
        customer=("CustomerID", "first"),
        country=("Country", "first"),
        n_lines=("StockCode", "size"),
        n_products=("StockCode", "nunique"),
        units=("Quantity", "sum"),
        order_value=("TotalAmount", "sum"),
        mean_unit_price=("UnitPrice", "mean"),
        max_unit_price=("UnitPrice", "max"),
        max_line_quantity=("Quantity", "max"),
        max_line_value=("TotalAmount", "max"),
        duplicate_lines=("IsDuplicate", "sum"),
        has_entry_correction=("IsEntryCorrection", "any"),     # LABEL, not a feature
    )
    inv["country"] = inv["country"].astype(str)
    inv["hour"] = inv["date"].dt.hour.astype("int8")
    inv["weekday"] = inv["date"].dt.day_of_week.astype("int8")
    inv["is_uk"] = (inv["country"] == "United Kingdom").astype("int8")
    inv["has_customer"] = inv["customer"].notna().astype("int8")
    inv["max_line_share"] = inv["max_line_value"] / inv["order_value"]   # one line dominating the order?

    # Customer history BEFORE this invoice ("invoice frequency" and usual order size)
    inv = inv.sort_values("date")
    known = inv["customer"].notna()
    g = inv[known].groupby("customer")
    inv["prior_orders"] = g.cumcount().reindex(inv.index)
    inv["prior_median_order_value"] = g["order_value"].transform(lambda s: s.shift(1).expanding().median())
    inv["days_since_prev_order"] = g["date"].diff().dt.total_seconds().div(86400).reindex(inv.index)
    inv["value_vs_customer_median"] = inv["order_value"] / inv["prior_median_order_value"]
    return inv


# --------------------------------------------------------------------------
def main():
    df = eda.load_clean()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", 40)

    tables = {
        DAILY_FILE: build_daily_features(df),
        PRODUCT_WEEKLY_FILE: build_product_weekly_features(df),
        CUSTOMER_FILE: build_customer_features(df),
        INVOICE_FILE: build_invoice_features(df),
    }
    for path, table in tables.items():
        table.to_parquet(path)
        print("=" * 78)
        print(f"{path.name}: {table.shape[0]:,} rows x {table.shape[1]} columns")
        print("=" * 78)
        print("Columns:", list(table.columns))
        missing = table.isna().sum()
        missing = missing[missing > 0]
        print("Missing values (expected only in lag/history columns at the start of each series):")
        print(missing.to_string() if len(missing) else "  none")
        print()

    daily = tables[DAILY_FILE]
    print("Daily table, first trading days of the week before last (lags filled):")
    print(daily.iloc[-12:-6, :12].round(1).to_string())

    products = tables[PRODUCT_WEEKLY_FILE]
    print(f"\nProduct-week table: {products['StockCode'].nunique():,} products, "
          f"{products['week'].nunique()} weeks, zero-demand weeks: {(products['units_sold'] == 0).mean():.1%}")

    rfm = tables[CUSTOMER_FILE]
    print(f"\nCustomer table: {len(rfm):,} customers, snapshot {rfm['snapshot'].iloc[0].date()}")
    print(rfm[["recency_days", "frequency", "monetary", "avg_order_value", "orders_per_month", "return_rate"]]
          .describe().round(2).to_string())

    inv = tables[INVOICE_FILE]
    print(f"\nInvoice table: {len(inv):,} invoices; invoices containing an order-entry correction "
          f"(evaluation label): {int(inv['has_entry_correction'].sum()):,}")


if __name__ == "__main__":
    main()

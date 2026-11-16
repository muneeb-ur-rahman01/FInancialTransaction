"""
Module 2 - Data Quality Analysis
================================
File    : notebooks/02_data_quality_check.py
Run     : python notebooks/02_data_quality_check.py     (from the project root)
Output  : printed to the terminal AND saved to reports/data_quality_report.txt

Golden rule of this script: it only MEASURES. It never drops, fills or edits a
single row. Cleaning decisions come afterwards (Module 3), based on what we see.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# "../data/raw/..." only works when the terminal is inside notebooks/.
# Building the path from this file's own location works from ANY folder.
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:  # running cell-by-cell in an interactive window
    PROJECT_ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "Online_Retail.xlsx"
REPORT_FILE = PROJECT_ROOT / "reports" / "data_quality_report.txt"

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")


class Tee:
    """Write everything we print to the terminal and to a report file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def pct(part, whole):
    return f"{part:,} ({part / whole * 100:.2f}%)" if whole else f"{part:,}"


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def load_raw():
    df = pd.read_excel(RAW_FILE)

    # InvoiceNo and StockCode look numeric but are really LABELS:
    # '536365' vs 'C536379' (cancellation), '22423' vs '85123A' vs 'POST'.
    # Excel gives us a mix of int and str, so we force text.
    # The "string" dtype keeps real missing values as <NA> instead of 'nan'.
    for col in ["InvoiceNo", "StockCode", "Description", "Country"]:
        df[col] = df[col].astype("string")

    # Working column for analysis only (kept in memory, raw file is untouched).
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
    return df


# --------------------------------------------------------------------------
# 1. Overview
# --------------------------------------------------------------------------
def check_overview(df):
    section("1. OVERVIEW")
    print(f"Rows x Columns : {df.shape[0]:,} x {df.shape[1] - 1} (+ TotalAmount helper)")
    print(f"Memory         : {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"Date range     : {df['InvoiceDate'].min()}  ->  {df['InvoiceDate'].max()}")
    print(f"Invoices       : {df['InvoiceNo'].nunique():,}")
    print(f"Stock codes    : {df['StockCode'].nunique():,}")
    print(f"Customers      : {df['CustomerID'].nunique():,} (known IDs)")
    print(f"Countries      : {df['Country'].nunique():,}")
    print("\nData types:")
    print(df.dtypes.to_string())


# --------------------------------------------------------------------------
# 2. Missing values - how many, and more importantly WHICH rows
# --------------------------------------------------------------------------
def check_missing(df):
    section("2. MISSING VALUES")
    n = len(df)
    miss = pd.DataFrame({"missing": df.isna().sum(), "percent": df.isna().mean() * 100})
    print(miss.to_string())

    # --- Description -------------------------------------------------------
    no_desc = df["Description"].isna()
    print("\n-- Missing Description: what kind of rows are they? --")
    print(f"Rows                          : {no_desc.sum():,}")
    print(f"...with UnitPrice == 0        : {(no_desc & (df['UnitPrice'] == 0)).sum():,}")
    print(f"...with CustomerID missing    : {(no_desc & df['CustomerID'].isna()).sum():,}")
    print(f"...with negative Quantity     : {(no_desc & (df['Quantity'] < 0)).sum():,}")
    print(f"Revenue on these rows         : {df.loc[no_desc, 'TotalAmount'].sum():,.2f}")

    # --- CustomerID --------------------------------------------------------
    no_cust = df["CustomerID"].isna()
    sales = (df["Quantity"] > 0) & (df["UnitPrice"] > 0) & df["InvoiceNo"].str.match(r"^\d+$")
    rev_all = df.loc[sales, "TotalAmount"].sum()
    rev_anon = df.loc[sales & no_cust, "TotalAmount"].sum()
    print("\n-- Missing CustomerID: is it random noise or a real customer group? --")
    print(f"Rows without CustomerID       : {pct(no_cust.sum(), n)}")
    print(f"Invoices without CustomerID   : {pct(df.loc[no_cust, 'InvoiceNo'].nunique(), df['InvoiceNo'].nunique())}")
    print(f"Sales revenue without ID      : {rev_anon:,.2f} of {rev_all:,.2f} ({rev_anon / rev_all * 100:.2f}%)")

    # Is the ID missing for a whole invoice, or only for some of its lines?
    per_invoice = df.groupby("InvoiceNo")["CustomerID"].agg(lambda s: s.isna().mean())
    partial = ((per_invoice > 0) & (per_invoice < 1)).sum()
    print(f"Invoices with PARTLY missing ID: {partial:,}  (0 means: missing for the whole invoice)")

    by_country = (
        df.assign(no_cust=no_cust)
        .groupby("Country")["no_cust"]
        .agg(rows="size", missing="sum")
        .query("missing > 0")
        .assign(percent=lambda t: t["missing"] / t["rows"] * 100)
        .sort_values("missing", ascending=False)
    )
    print("\nMissing CustomerID by country:")
    print(by_country.to_string())

    by_month = (
        df.assign(no_cust=no_cust, month=df["InvoiceDate"].dt.to_period("M"))
        .groupby("month")["no_cust"]
        .mean()
        .mul(100)
    )
    print("\n% of rows without CustomerID per month:")
    print(by_month.round(1).to_string())


# --------------------------------------------------------------------------
# 3. Duplicates - are they errors or legitimate repeated lines?
# --------------------------------------------------------------------------
def check_duplicates(df):
    section("3. DUPLICATE RECORDS")
    cols = [c for c in df.columns if c != "TotalAmount"]
    extra = df.duplicated(subset=cols, keep="first")
    involved = df.duplicated(subset=cols, keep=False)
    print(f"Extra copies (keep='first')   : {pct(extra.sum(), len(df))}")
    print(f"Rows involved (keep=False)    : {involved.sum():,}")
    print(f"Invoices affected             : {df.loc[involved, 'InvoiceNo'].nunique():,}")
    print(f"Copies with known CustomerID  : {df.loc[extra, 'CustomerID'].notna().sum():,}")
    print(f"Copies on cancellations       : {df.loc[extra, 'InvoiceNo'].str.startswith('C').sum():,}")
    print(f"Revenue on extra copies       : {df.loc[extra, 'TotalAmount'].sum():,.2f}")

    # A full-row duplicate always shares InvoiceNo, so it is the same product,
    # same quantity, same price repeated INSIDE one invoice.
    sizes = df[involved].groupby(cols, dropna=False).size()
    print("\nHow many times is an identical line repeated?")
    print(sizes.value_counts().sort_index().rename_axis("copies").rename("groups").to_string())

    print("\nQuantity on duplicated lines (a scanner double-read is usually quantity 1):")
    print(df.loc[extra, "Quantity"].describe().to_string())

    # The other kind of repetition: same product twice in an invoice but with
    # a DIFFERENT quantity/price. That is clearly not a duplicate record.
    same_item = df.duplicated(subset=["InvoiceNo", "StockCode"], keep=False)
    print(f"\nRows where the product appears >1 time in its invoice (any quantity): {same_item.sum():,}")

    print("\nExample of one duplicated group:")
    first_key = df.loc[extra, cols].iloc[0]
    example = df[(df["InvoiceNo"] == first_key["InvoiceNo"]) & (df["StockCode"] == first_key["StockCode"])]
    print(example[cols].to_string())


# --------------------------------------------------------------------------
# 4. Transaction semantics: invoice type x quantity sign x price sign
# --------------------------------------------------------------------------
def check_semantics(df):
    section("4. TRANSACTION SEMANTICS (InvoiceNo prefix, Quantity sign, UnitPrice sign)")
    prefix = df["InvoiceNo"].str.extract(r"^([A-Za-z]*)")[0]
    print("InvoiceNo prefixes ('' = normal numeric invoice):")
    print(prefix.value_counts(dropna=False).to_string())

    inv_type = np.select(
        [prefix == "", prefix == "C"], ["Normal", "Cancellation"], default="Other"
    )
    sign = lambda s: np.select([s > 0, s < 0], ["positive", "negative"], default="zero")
    table = (
        df.assign(invoice_type=inv_type, quantity=sign(df["Quantity"]), price=sign(df["UnitPrice"]))
        .groupby(["invoice_type", "quantity", "price"])
        .agg(
            rows=("InvoiceNo", "size"),
            invoices=("InvoiceNo", "nunique"),
            has_customer=("CustomerID", lambda s: s.notna().sum()),
            total_amount=("TotalAmount", "sum"),
        )
    )
    print("\nEvery row of the dataset falls into exactly one of these groups:")
    print(table.to_string())

    cancelled = prefix == "C"
    neg_qty = df["Quantity"] < 0
    print(f"\nCancellation rows              : {pct(cancelled.sum(), len(df))}")
    print(f"Cancellation invoices          : {pct(df.loc[cancelled, 'InvoiceNo'].nunique(), df['InvoiceNo'].nunique())}")
    print(f"Negative-quantity rows         : {neg_qty.sum():,}")
    print(f"  negative AND cancellation    : {(neg_qty & cancelled).sum():,}")
    print(f"  negative but NOT cancellation: {(neg_qty & ~cancelled).sum():,}")

    odd = neg_qty & ~cancelled
    print("\nNegative quantity WITHOUT a 'C' invoice - most common descriptions:")
    print(df.loc[odd, "Description"].value_counts(dropna=False).head(15).to_string())

    print("\nNegative UnitPrice rows:")
    print(df[df["UnitPrice"] < 0].drop(columns="TotalAmount").to_string())

    zero = df["UnitPrice"] == 0
    print(f"\nZero-price rows                : {zero.sum():,}")
    print(f"  with positive quantity       : {(zero & (df['Quantity'] > 0)).sum():,}")
    print(f"  with negative quantity       : {(zero & neg_qty).sum():,}")
    print(f"  with a known CustomerID      : {(zero & df['CustomerID'].notna()).sum():,}")
    print("  most common descriptions:")
    print(df.loc[zero, "Description"].value_counts(dropna=False).head(15).to_string())

    # Does each cancellation refer to a sale we can actually see?
    canc = df[cancelled & df["CustomerID"].notna()].copy()
    canc["abs_qty"] = -canc["Quantity"]
    first_sale = (
        df[(prefix == "") & (df["Quantity"] > 0)]
        .groupby(["CustomerID", "StockCode", "Quantity"])["InvoiceDate"]
        .min()
        .rename("first_sale_date")
        .reset_index()
        .rename(columns={"Quantity": "abs_qty"})
    )
    canc = canc.merge(first_sale, on=["CustomerID", "StockCode", "abs_qty"], how="left")
    exact = (canc["first_sale_date"] <= canc["InvoiceDate"]).sum()
    bought = df[(prefix == "") & (df["Quantity"] > 0)].groupby(["CustomerID", "StockCode"])["InvoiceDate"].min()
    canc = canc.join(bought.rename("first_buy_date"), on=["CustomerID", "StockCode"])
    any_buy = (canc["first_buy_date"] <= canc["InvoiceDate"]).sum()
    print(f"\nCancellation rows with a CustomerID            : {len(canc):,}")
    print(f"  earlier sale, same customer+product+quantity : {pct(exact, len(canc))}  (full cancellation)")
    print(f"  earlier sale, same customer+product          : {pct(any_buy, len(canc))}  (incl. partial returns)")
    print(f"  no earlier sale visible in the data          : {pct(len(canc) - any_buy, len(canc))}  (sold before Dec 2010, or manual credit)")


# --------------------------------------------------------------------------
# 5. StockCode - products vs. fees, postage, manual adjustments
# --------------------------------------------------------------------------
def check_stock_codes(df):
    section("5. STOCK CODES THAT ARE NOT PRODUCTS")
    # Real products start with 5 digits (22423, 85123A). Anything else is
    # a service line: postage, bank charges, discounts, manual corrections...
    is_product = df["StockCode"].str.match(r"^\d{5}")
    special = (
        df[~is_product]
        .groupby("StockCode")
        .agg(
            rows=("InvoiceNo", "size"),
            description=("Description", lambda s: s.dropna().mode().iloc[0] if s.notna().any() else "<none>"),
            min_price=("UnitPrice", "min"),
            max_price=("UnitPrice", "max"),
            total_amount=("TotalAmount", "sum"),
        )
        .sort_values("rows", ascending=False)
    )
    print(f"Rows with a non-product code: {pct((~is_product).sum(), len(df))}")
    print(special.to_string())


# --------------------------------------------------------------------------
# 6. Outliers - measured on genuine sales lines only
# --------------------------------------------------------------------------
def check_outliers(df):
    section("6. OUTLIER INVESTIGATION (rows with Quantity > 0 and UnitPrice > 0)")
    sales = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    qs = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999, 1.0]
    print("Percentiles:")
    table = sales[["Quantity", "UnitPrice", "TotalAmount"]].quantile(qs)
    table.index = [f"p{q * 100:g}" for q in qs]
    print(table.to_string())

    print("\nShape of the distributions (skew 0 = symmetric; retail data is far from it):")
    print(sales[["Quantity", "UnitPrice", "TotalAmount"]].agg(["mean", "median", "std", "skew"]).to_string())

    print("\nIQR rule (value > Q3 + 1.5*IQR):")
    for col in ["Quantity", "UnitPrice", "TotalAmount"]:
        q1, q3 = sales[col].quantile([0.25, 0.75])
        upper = q3 + 1.5 * (q3 - q1)
        flagged = sales[col] > upper
        share = sales.loc[flagged, "TotalAmount"].sum() / sales["TotalAmount"].sum() * 100
        print(f"  {col:<12} upper fence {upper:>8.2f} -> {pct(flagged.sum(), len(sales))} rows, carrying {share:.1f}% of revenue")

    show = ["InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice", "CustomerID", "TotalAmount"]
    print("\nTop 8 quantities (all rows, both signs - look for sale/cancel pairs):")
    print(df.loc[df["Quantity"].abs().nlargest(8).index, show].to_string())
    print("\nTop 8 unit prices (all rows):")
    print(df.loc[df["UnitPrice"].abs().nlargest(8).index, show].to_string())


# --------------------------------------------------------------------------
# 7. Consistency
# --------------------------------------------------------------------------
def check_consistency(df):
    section("7. DATA CONSISTENCY")
    desc = df["Description"]
    print(f"Descriptions with leading/trailing spaces : {(desc != desc.str.strip()).sum():,}")
    print(f"Descriptions fully lowercase (staff notes) : {(desc == desc.str.lower()).sum():,}")
    lower = df[(desc == desc.str.lower()).fillna(False)]
    print(lower["Description"].value_counts().head(10).to_string())

    per_code = df.groupby("StockCode")["Description"].nunique()
    print(f"\nStockCodes with more than one description  : {(per_code > 1).sum():,} of {len(per_code):,}")
    worst = per_code.idxmax()
    print(f"Example - code {worst}:")
    print(df.loc[df["StockCode"] == worst, "Description"].value_counts(dropna=False).to_string())

    codes = df["StockCode"].dropna()
    print(f"\nStockCodes differing only by letter case   : {codes.nunique() - codes.str.upper().nunique():,}")

    cust_per_inv = df.groupby("InvoiceNo")["CustomerID"].nunique()
    country_per_inv = df.groupby("InvoiceNo")["Country"].nunique()
    country_per_cust = df.groupby("CustomerID")["Country"].nunique()
    time_per_inv = df.groupby("InvoiceNo")["InvoiceDate"].nunique()
    print(f"Invoices linked to >1 CustomerID           : {(cust_per_inv > 1).sum():,}")
    print(f"Invoices linked to >1 Country              : {(country_per_inv > 1).sum():,}")
    print(f"Invoices with >1 timestamp                 : {(time_per_inv > 1).sum():,}")
    print(f"Customers linked to >1 Country             : {(country_per_cust > 1).sum():,}")

    ids = df["CustomerID"].dropna()
    print(f"CustomerIDs that are not whole numbers     : {(ids != ids.round()).sum():,}  (float only because of NaN)")
    print(f"Rows with Country == 'Unspecified'         : {(df['Country'] == 'Unspecified').sum():,}")
    print(f"UnitPrice with more than 2 decimals        : {(df['UnitPrice'].mul(100).round(6) % 1 != 0).sum():,}")

    monthly = df.groupby(df["InvoiceDate"].dt.to_period("M")).agg(
        rows=("InvoiceNo", "size"),
        invoices=("InvoiceNo", "nunique"),
        first_day=("InvoiceDate", lambda s: s.min().date()),
        last_day=("InvoiceDate", lambda s: s.max().date()),
        net_amount=("TotalAmount", "sum"),
    )
    print("\nCoverage per month (a partial month must not be compared with full months):")
    print(monthly.to_string())

    days = pd.Series(df["InvoiceDate"].dt.normalize().unique())
    print("\nTrading days per weekday (is any weekday missing?):")
    print(days.dt.day_name().value_counts().to_string())
    print(f"Hours of activity: {df['InvoiceDate'].dt.hour.min()}:00 to {df['InvoiceDate'].dt.hour.max()}:00")


# --------------------------------------------------------------------------
# 8. What the money looks like under different definitions
# --------------------------------------------------------------------------
def check_revenue(df):
    section("8. REVENUE RECONCILIATION")
    is_c = df["InvoiceNo"].str.startswith("C")
    is_product = df["StockCode"].str.match(r"^\d{5}")
    is_numeric_invoice = df["InvoiceNo"].str.match(r"^\d+$")
    sale = (df["Quantity"] > 0) & (df["UnitPrice"] > 0) & is_numeric_invoice
    rows = {
        "Naive sum of Quantity*UnitPrice (everything)": df["TotalAmount"].sum(),
        "Gross sales (numeric invoice, qty>0, price>0)": df.loc[sale, "TotalAmount"].sum(),
        "  sales: real products": df.loc[sale & is_product, "TotalAmount"].sum(),
        "  sales: non-product codes (postage, fees...)": df.loc[sale & ~is_product, "TotalAmount"].sum(),
        "Cancellations / credits ('C' invoices)": df.loc[is_c, "TotalAmount"].sum(),
        "  credits: real products": df.loc[is_c & is_product, "TotalAmount"].sum(),
        "  credits: non-product codes": df.loc[is_c & ~is_product, "TotalAmount"].sum(),
        "Bad-debt adjustments ('A' invoices)": df.loc[df["InvoiceNo"].str.startswith("A"), "TotalAmount"].sum(),
        "Product sales with known CustomerID": df.loc[sale & is_product & df["CustomerID"].notna(), "TotalAmount"].sum(),
        "Product sales minus product credits": df.loc[(sale | is_c) & is_product, "TotalAmount"].sum(),
    }
    for label, value in rows.items():
        print(f"{label:<50} {value:>16,.2f}")


def main():
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as fh:
        original = sys.stdout
        sys.stdout = Tee(original, fh)
        try:
            print("DATA QUALITY REPORT - Online Retail")
            print(f"Source: {RAW_FILE.relative_to(PROJECT_ROOT)}")
            df = load_raw()
            check_overview(df)
            check_missing(df)
            check_duplicates(df)
            check_semantics(df)
            check_stock_codes(df)
            check_outliers(df)
            check_consistency(df)
            check_revenue(df)
        finally:
            sys.stdout = original
    print(f"\nReport saved to {REPORT_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

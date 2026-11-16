"""
Module 3 - Data Cleaning
========================
Turns the raw Online Retail file into one clean, labelled table.

Design decisions (from the Module 2 data quality report):
  * NO row is deleted. Every raw row survives, gets a RowID that points back
    to the raw file, and receives labels. Each later analysis/model filters
    the rows it needs. This keeps the work reproducible and reversible.
  * Duplicates are FLAGGED (IsDuplicate), not dropped.
  * Revenue is NET: product sales minus product cancellations
    (rows where IsRevenueLine is True; their TotalAmount is already signed).
  * Rows without a CustomerID are kept (they are 16% of sales). Only
    customer-level work (RFM, segmentation) will exclude them.

Run from the project root:
    python -m src.data_processing.cleaning
"""

import numpy as np
import pandas as pd

from src.utils.paths import CLEAN_DATA_FILE, RAW_DATA_FILE, REPORTS_DIR

RAW_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]

# Real merchandise codes start with five digits (22423, 85123A).
# Everything else is postage, fees, discounts, manual corrections, vouchers...
PRODUCT_CODE_PATTERN = r"^\d{5}"

TRANSACTION_TYPES = ["sale", "cancellation", "stock_adjustment", "accounting_adjustment", "other"]

# An exact sale<->cancellation pair this fast is an order-entry correction (the order
# never really happened). Slower exact reversals are genuine customer returns.
# Measured on the data: 452 pairs < 1 h hold £292K; 2,097 pairs > 1 day hold £94K.
ENTRY_CORRECTION_MAX_HOURS = 1.0


def _mask(series):
    """Convert a nullable boolean Series into a plain NumPy bool array (NA -> False)."""
    return series.fillna(False).to_numpy(dtype=bool)


# --------------------------------------------------------------------------
# Step 0 - load and validate
# --------------------------------------------------------------------------
def load_raw(path=RAW_DATA_FILE):
    """Read the original Excel file. The file itself is never modified."""
    return pd.read_excel(path, dtype={"InvoiceNo": str, "StockCode": str})


def validate_schema(df):
    """Fail early and clearly if the input does not look like the Online Retail data."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Input DataFrame is empty")


# --------------------------------------------------------------------------
# Step 1 - consistent types and text
# --------------------------------------------------------------------------
def standardise(df):
    """Fix types and text formatting. Values are tidied, never invented."""
    out = df.copy()

    # Traceability: RowID = position in the raw file.
    if "RowID" not in out.columns:
        out.insert(0, "RowID", np.arange(len(out), dtype="int64"))

    # Codes are labels, not numbers. Upper-casing merges the 112 codes that
    # only differed by letter case (e.g. '85123a' vs '85123A').
    for col in ["InvoiceNo", "StockCode"]:
        out[col] = out[col].astype("string").str.strip().str.upper()

    # Collapse repeated spaces and trim; an empty text becomes a real missing value.
    out["Description"] = (
        out["Description"].astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace("", pd.NA)
    )
    out["Country"] = out["Country"].astype("string").str.strip()

    # CustomerID was float only because NaN forced it; Int64 holds integers AND <NA>.
    out["CustomerID"] = out["CustomerID"].astype("Int64")
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"])
    out["Quantity"] = out["Quantity"].astype("int32")   # max |80,995| fits easily
    out["UnitPrice"] = out["UnitPrice"].astype("float64")
    return out


# --------------------------------------------------------------------------
# Step 2 - what kind of transaction is each row?
# --------------------------------------------------------------------------
def classify_transactions(df):
    """Label every row by its business meaning and compute TotalAmount.

    TransactionType
      sale                  : normal invoice, quantity > 0, price > 0
      cancellation          : invoice starts with 'C' (customer credit / return)
      stock_adjustment      : normal invoice with price 0 (damages, 'check', found stock)
      accounting_adjustment : invoice starts with 'A' (bad-debt write-offs)
      other                 : anything not covered above (should be 0 rows; checked)
    """
    out = df.copy()
    invoice = out["InvoiceNo"]
    qty, price = out["Quantity"], out["UnitPrice"]

    is_cancel = _mask(invoice.str.startswith("C"))
    is_accounting = _mask(invoice.str.startswith("A"))
    is_numeric = _mask(invoice.str.fullmatch(r"\d+"))

    conditions = [
        is_accounting,
        is_cancel,
        is_numeric & (price.to_numpy() == 0),
        is_numeric & (qty.to_numpy() > 0) & (price.to_numpy() > 0),
    ]
    choices = ["accounting_adjustment", "cancellation", "stock_adjustment", "sale"]
    out["TransactionType"] = pd.Categorical(
        np.select(conditions, choices, default="other"), categories=TRANSACTION_TYPES
    )

    out["IsProduct"] = _mask(out["StockCode"].str.match(PRODUCT_CODE_PATTERN))

    # Signed amount: positive for sales, negative for cancellations, 0 for stock adjustments.
    out["TotalAmount"] = qty * price

    # The rows that make up NET product revenue.
    out["IsRevenueLine"] = out["TransactionType"].isin(["sale", "cancellation"]) & out["IsProduct"]
    return out


# --------------------------------------------------------------------------
# Step 3 - duplicates are flagged, not removed
# --------------------------------------------------------------------------
def flag_duplicates(df):
    """IsDuplicate is True for the 2nd, 3rd... copy of an identical line (first copy stays False)."""
    out = df.copy()
    out["IsDuplicate"] = out.duplicated(subset=RAW_COLUMNS, keep="first")
    return out


# --------------------------------------------------------------------------
# Step 4 - one clean name per product
# --------------------------------------------------------------------------
def add_product_names(df):
    """ProductName = the most frequent description used on real SALES of that StockCode.

    650 codes carry several descriptions, many of them staff notes
    ('wrongly marked', 'found'). Sales lines hold the real catalogue name,
    so we prefer them and fall back to any description, then 'UNKNOWN'.
    The original Description column is left in place.
    """
    out = df.copy()

    def most_common(rows):
        counts = rows.dropna(subset=["Description"]).groupby(["StockCode", "Description"]).size()
        counts = counts.reset_index(name="n").sort_values(
            ["StockCode", "n", "Description"], ascending=[True, False, True]
        )
        return counts.drop_duplicates("StockCode").set_index("StockCode")["Description"]

    from_sales = most_common(out[out["TransactionType"] == "sale"])
    from_any = most_common(out)
    names = from_sales.combine_first(from_any)

    out["ProductName"] = out["StockCode"].map(names).fillna("UNKNOWN").astype("string")
    return out


# --------------------------------------------------------------------------
# Step 5 - pair cancellations with the sale they reverse
# --------------------------------------------------------------------------
def flag_reversals(df):
    """Link each cancellation to the most recent earlier sale it exactly reverses.

    Exact match = same CustomerID, StockCode, UnitPrice and the same absolute
    quantity. This catches order-entry mistakes such as the 80,995-unit sale
    cancelled 12 minutes later. Each sale can be reversed at most once.

    Adds:
      IsReversedSale    : sale that was later fully cancelled
      IsReversal        : cancellation that fully reverses an earlier sale
      MatchedRowID      : RowID of the partner row (<NA> if unmatched)
      ReversalLagHours  : time between the sale and its cancellation (both rows)
      IsEntryCorrection : pair cancelled within ENTRY_CORRECTION_MAX_HOURS -> an
                          order-entry mistake, not a real order + return
    """
    out = df.copy()
    out["IsReversedSale"] = False
    out["IsReversal"] = False
    out["MatchedRowID"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out["ReversalLagHours"] = np.nan
    out["IsEntryCorrection"] = False

    keys = ["CustomerID", "StockCode", "UnitPrice", "AbsQuantity"]
    has_customer = out["CustomerID"].notna()

    def side(type_name):
        rows = out.loc[has_customer & (out["TransactionType"] == type_name),
                       ["RowID", "InvoiceDate", "CustomerID", "StockCode", "UnitPrice", "Quantity"]]
        rows = rows.assign(
            AbsQuantity=rows["Quantity"].abs(),
            CustomerID=rows["CustomerID"].astype("int64"),
            StockCode=rows["StockCode"].astype(str),
        ).drop(columns="Quantity")
        if type_name == "sale":
            rows["SaleDate"] = rows["InvoiceDate"]      # kept: merge_asof keeps only the left date
        return rows.sort_values("InvoiceDate")

    sales, cancels = side("sale"), side("cancellation")
    if sales.empty or cancels.empty:
        return out

    matched = pd.merge_asof(
        cancels, sales, on="InvoiceDate", by=keys,
        direction="backward", suffixes=("_cancel", "_sale"),
    )
    matched = (
        matched.dropna(subset=["RowID_sale"])
        .drop_duplicates("RowID_sale", keep="first")   # one sale <-> one cancellation
    )
    cancel_ids = matched["RowID_cancel"].astype("int64").to_numpy()
    sale_ids = matched["RowID_sale"].astype("int64").to_numpy()

    position = pd.Series(np.arange(len(out)), index=out["RowID"].to_numpy())
    c_pos, s_pos = position[cancel_ids].to_numpy(), position[sale_ids].to_numpy()

    col = out.columns.get_loc
    out.iloc[c_pos, col("IsReversal")] = True
    out.iloc[s_pos, col("IsReversedSale")] = True
    out.iloc[c_pos, col("MatchedRowID")] = sale_ids
    out.iloc[s_pos, col("MatchedRowID")] = cancel_ids

    lag_hours = ((matched["InvoiceDate"] - matched["SaleDate"]) / pd.Timedelta(hours=1)).to_numpy()
    out.iloc[c_pos, col("ReversalLagHours")] = lag_hours
    out.iloc[s_pos, col("ReversalLagHours")] = lag_hours
    out["IsEntryCorrection"] = (out["IsReversal"] | out["IsReversedSale"]) & (
        out["ReversalLagHours"] <= ENTRY_CORRECTION_MAX_HOURS
    )
    return out


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def clean_transactions(raw):
    """Full cleaning pipeline. Returns a NEW DataFrame with the same number of rows."""
    validate_schema(raw)
    df = standardise(raw)
    df = classify_transactions(df)
    df = flag_duplicates(df)
    df = add_product_names(df)
    df = flag_reversals(df)

    df["Country"] = df["Country"].astype("category")  # 38 repeated values -> saves memory

    if len(df) != len(raw):
        raise AssertionError("Cleaning must not add or remove rows")
    return df


def build_summary(df):
    """A small, human-readable audit of the cleaned table."""
    by_type = (
        df.groupby("TransactionType", observed=False)
        .agg(rows=("RowID", "size"),
             product_rows=("IsProduct", "sum"),
             with_customer=("CustomerID", "count"),
             total_amount=("TotalAmount", "sum"))
    )
    rev = df[df["IsRevenueLine"]]
    figures = pd.Series({
        "rows": len(df),
        "duplicate_rows_flagged": int(df["IsDuplicate"].sum()),
        "reversed_sales": int(df["IsReversedSale"].sum()),
        "reversal_cancellations": int(df["IsReversal"].sum()),
        "entry_correction_pairs": int((df["IsEntryCorrection"] & df["IsReversal"]).sum()),
        "entry_correction_value": round(df.loc[df["IsEntryCorrection"] & df["IsReversedSale"], "TotalAmount"].sum(), 2),
        "unclassified_rows": int((df["TransactionType"] == "other").sum()),
        "gross_product_sales": round(rev.loc[rev["TotalAmount"] > 0, "TotalAmount"].sum(), 2),
        "product_cancellations": round(rev.loc[rev["TotalAmount"] < 0, "TotalAmount"].sum(), 2),
        "net_product_sales": round(rev["TotalAmount"].sum(), 2),
    })
    return by_type, figures


def main():
    print("Reading raw Excel file (about a minute)...")
    raw = load_raw()
    clean = clean_transactions(raw)

    CLEAN_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(CLEAN_DATA_FILE, index=False)

    by_type, figures = build_summary(clean)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    text = (
        "CLEANING SUMMARY\n\nRows by transaction type:\n" + by_type.to_string()
        + "\n\nKey figures:\n" + figures.to_string()
        + f"\n\nMemory: {clean.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
    )
    print(text)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "cleaning_summary.txt").write_text(text, encoding="utf-8")

    print(f"\nColumns: {list(clean.columns)}")
    print(f"Saved clean data -> {CLEAN_DATA_FILE}")


if __name__ == "__main__":
    main()

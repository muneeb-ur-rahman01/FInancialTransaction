"""Unit tests for src/data_processing/cleaning.py.

Each test uses a tiny hand-made DataFrame where we KNOW the right answer,
one row for every situation found in the data quality report.
Run from the project root:  python -m pytest -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data_processing.cleaning import RAW_COLUMNS, clean_transactions, validate_schema


@pytest.fixture
def raw():
    rows = [
        # InvoiceNo, StockCode, Description, Qty, InvoiceDate, UnitPrice, CustomerID, Country
        ("536365", "85123A", "WHITE HANGING HEART ", 6, "2010-12-01 08:26", 2.55, 17850.0, "United Kingdom"),  # 0 sale
        ("536365", "85123A", "WHITE HANGING HEART ", 6, "2010-12-01 08:26", 2.55, 17850.0, "United Kingdom"),  # 1 duplicate of 0
        ("536370", "22728", "ALARM CLOCK", 24, "2010-12-01 09:00", 3.75, 12583.0, "France"),                  # 2 sale, later reversed
        ("C536379", "22728", "ALARM CLOCK", -24, "2010-12-01 09:30", 3.75, 12583.0, "France"),               # 3 reverses row 2
        ("C536380", "22728", "ALARM CLOCK", -2, "2010-12-02 10:00", 3.75, 12583.0, "France"),                # 4 partial return
        ("536381", "21777", None, -10, "2010-12-02 11:00", 0.0, np.nan, "United Kingdom"),                   # 5 stock write-off
        ("A563186", "B", "Adjust bad debt", 1, "2011-08-12 14:51", -11062.06, np.nan, "United Kingdom"),    # 6 accounting
        ("536382", "POST", "POSTAGE", 1, "2010-12-03 12:00", 18.0, 12583.0, "France"),                       # 7 non-product sale
        ("536383", "85123a", "  white  hanging heart", 2, "2010-12-03 13:00", 2.55, np.nan, "United Kingdom"),  # 8 anonymous, lower-case code
    ]
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


@pytest.fixture
def clean(raw):
    return clean_transactions(raw)


def test_no_rows_removed_and_input_untouched(raw, clean):
    before = raw.copy()
    clean_transactions(raw)
    pd.testing.assert_frame_equal(raw, before)          # input not mutated
    assert len(clean) == len(raw)
    assert clean["RowID"].tolist() == list(range(len(raw)))


def test_transaction_types(clean):
    expected = ["sale", "sale", "sale", "cancellation", "cancellation",
                "stock_adjustment", "accounting_adjustment", "sale", "sale"]
    assert clean["TransactionType"].astype(str).tolist() == expected
    assert (clean["TransactionType"] == "other").sum() == 0


def test_product_flag_and_revenue_lines(clean):
    assert not clean.loc[7, "IsProduct"]                # POST is not merchandise
    assert not clean.loc[6, "IsProduct"]                # B (bad debt) is not merchandise
    assert clean.loc[[0, 2, 3, 8], "IsProduct"].all()
    assert clean["IsRevenueLine"].tolist() == [True, True, True, True, True, False, False, False, True]


def test_net_revenue_is_signed_sum(clean):
    net = clean.loc[clean["IsRevenueLine"], "TotalAmount"].sum()
    expected = 6 * 2.55 * 2 + 24 * 3.75 - 24 * 3.75 - 2 * 3.75 + 2 * 2.55
    assert net == pytest.approx(expected)


def test_duplicates_flagged_not_dropped(clean):
    assert clean["IsDuplicate"].tolist() == [False, True] + [False] * 7


def test_text_and_codes_standardised(clean):
    assert clean.loc[8, "StockCode"] == "85123A"
    assert clean.loc[0, "Description"] == "WHITE HANGING HEART"
    assert clean.loc[8, "Description"] == "white hanging heart"   # original wording kept...
    assert clean.loc[8, "ProductName"] == "WHITE HANGING HEART"   # ...but one canonical name
    assert clean.loc[5, "ProductName"] == "UNKNOWN"
    assert pd.isna(clean.loc[8, "CustomerID"])
    assert clean.loc[0, "CustomerID"] == 17850


def test_reversal_pairs_are_exact_and_one_to_one(clean):
    assert clean.loc[2, "IsReversedSale"] and clean.loc[3, "IsReversal"]
    assert clean.loc[2, "MatchedRowID"] == 3 and clean.loc[3, "MatchedRowID"] == 2
    assert not clean.loc[4, "IsReversal"]               # partial return: no exact sale
    assert clean["IsReversedSale"].sum() == 1
    assert clean["IsReversal"].sum() == 1


def test_fast_reversal_is_entry_correction(clean):
    assert clean.loc[2, "ReversalLagHours"] == pytest.approx(0.5)
    assert clean.loc[2, "IsEntryCorrection"] and clean.loc[3, "IsEntryCorrection"]
    assert not clean.loc[4, "IsEntryCorrection"]


def test_slow_reversal_is_a_genuine_return(raw):
    raw.loc[3, "InvoiceDate"] = pd.Timestamp("2010-12-05 09:00")   # cancelled 4 days later
    out = clean_transactions(raw)
    assert out.loc[3, "IsReversal"]                                 # still an exact pair...
    assert not out.loc[3, "IsEntryCorrection"]                      # ...but a real return
    assert not out.loc[2, "IsEntryCorrection"]


def test_cancellation_before_sale_is_not_matched(raw):
    raw.loc[3, "InvoiceDate"] = pd.Timestamp("2010-11-30 09:00")   # cancel BEFORE the sale
    out = clean_transactions(raw)
    assert out["IsReversal"].sum() == 0


def test_schema_validation():
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(pd.DataFrame({"InvoiceNo": ["1"]}))

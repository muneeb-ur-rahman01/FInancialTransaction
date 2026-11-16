"""Smoke tests: every dashboard page runs without an exception on the real pipeline outputs.

Uses Streamlit's headless AppTest runner. Skipped automatically if the pipeline outputs
(clean data, feature tables, model reports) have not been generated yet.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "dashboard"
REQUIRED = [
    ROOT / "data/processed/transactions_clean.parquet",
    ROOT / "data/processed/features_customers.parquet",
    ROOT / "reports/model_reports/sales_forecasting_test_predictions.csv",
    ROOT / "reports/model_reports/demand_forecasting_test_predictions.parquet",
    ROOT / "data/processed/customer_segments.parquet",
    ROOT / "data/processed/flagged_lines.parquet",
    ROOT / "reports/model_reports/model_evaluation_summary.json",
]
PAGES = [DASHBOARD / "app.py"] + sorted((DASHBOARD / "pages").glob("*.py"))

pytestmark = pytest.mark.skipif(not all(p.exists() for p in REQUIRED),
                                reason="pipeline outputs missing - run the pipeline first")


@pytest.fixture(autouse=True)
def dashboard_on_path(monkeypatch):
    # `streamlit run dashboard/app.py` puts dashboard/ on sys.path; AppTest on a page file does not.
    monkeypatch.syspath_prepend(str(DASHBOARD))


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_runs_without_errors(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(page), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.title) == 1


def test_country_filter_changes_the_overview():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=120).run()
    all_orders = at.metric[1].value
    at.sidebar.multiselect[0].set_value(["France"]).run()
    assert not at.exception
    assert at.metric[1].value != all_orders

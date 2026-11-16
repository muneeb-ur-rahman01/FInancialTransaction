"""Tests for src/models/anomaly_detection.py."""

import numpy as np
import pandas as pd
import pytest

from src.models import anomaly_detection as A


def make_lines(n=400, seed=0, errors=()):
    rng = np.random.default_rng(seed)
    codes = rng.choice(["11111", "22222", "33333"], n)
    usual = {"11111": 6, "22222": 12, "33333": 24}
    qty = np.array([max(1, int(rng.normal(usual[c], 2))) for c in codes])
    lines = pd.DataFrame({
        "InvoiceNo": [str(1000 + i // 4) for i in range(n)],
        "InvoiceDate": pd.date_range("2011-01-03 09:00", periods=n, freq="6h"),
        "StockCode": codes, "Quantity": qty, "UnitPrice": 2.0, "Country": "United Kingdom",
        "CustomerID": 12345.0, A.LABEL: False,
    })
    for i in errors:
        lines.loc[i, "Quantity"] = 5000
        lines.loc[i, A.LABEL] = True
    lines["TotalAmount"] = lines["Quantity"] * lines["UnitPrice"]
    return lines


def test_product_norms_and_relative_quantity():
    lines = make_lines()
    norms = A.product_norms(lines)
    feats = A.line_features(lines, norms)
    assert set(feats.columns) == set(A.FEATURES)
    assert abs(feats["qty_vs_product"].median()) < 0.3                      # typical line ~ its product's norm
    new = lines.head(1).assign(StockCode="99999", Quantity=int(norms["usual_qty"]))
    assert A.line_features(new, norms)["qty_vs_product"].iloc[0] == pytest.approx(0)   # unknown product -> overall norm


def test_norms_come_only_from_the_fitting_data():
    fit = make_lines(seed=1)
    later = make_lines(seed=2).assign(Quantity=1000)                      # future lines must not change the norm
    scorer = A.LineAnomalyScorer("rule", column="qty_vs_product").fit(fit)
    before = scorer.norms_["table"].copy()
    scorer.score(later)
    pd.testing.assert_frame_equal(before, scorer.norms_["table"])


def test_detection_metrics_known_values():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([9, 1, 8, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    m = A.detection_metrics(y, scores, threshold=5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 7)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["lift"] == pytest.approx(0.5 / 0.2)


def test_alert_threshold_flags_requested_share():
    scores = np.arange(1000, dtype=float)
    t = A.alert_threshold(scores, flag_rate=0.01)
    assert (scores >= t).sum() == pytest.approx(10, abs=1)


def test_split_periods_is_chronological():
    lines = make_lines(n=2000)
    fit, val, test = A.split_periods(lines, pd.Timestamp("2011-03-01"), pd.Timestamp("2011-04-01"))
    assert fit["InvoiceDate"].max() < val["InvoiceDate"].min()
    assert val["InvoiceDate"].max() < test["InvoiceDate"].min()
    assert len(fit) + len(val) + len(test) == len(lines)


@pytest.mark.parametrize("family,name", [("Rule", "Largest quantity"), ("Rule", "Quantity vs product norm"),
                                         ("Unsupervised", "Isolation Forest")])
def test_detectors_rank_planted_errors_on_top(family, name):
    lines = make_lines(n=600, errors=(50, 200, 400))
    scorer = {n: s for f, n, s in A.candidate_scorers()}[name]
    scores = scorer.clone().fit(lines, lines[A.LABEL]).score(lines)
    top3 = set(np.argsort(-scores)[:3])
    assert top3 == {50, 200, 400}


def test_score_everything_outputs():
    lines = make_lines(n=800, errors=(10, 500))
    scorer = A.LineAnomalyScorer("rule", column="log_qty")
    final, threshold, scored, invoices = A.score_everything(lines, scorer)
    assert scored.loc[[10, 500], "flagged"].all()
    assert scored["flagged"].mean() == pytest.approx(A.FLAG_RATE, abs=0.01)
    assert invoices["known_error"].sum() == 2
    assert invoices.loc[invoices["known_error"], "flagged"].all()


def test_saved_report_metrics_are_numbers(tmp_path, monkeypatch):
    """Regression test: numpy scalars must not end up as strings in the JSON report."""
    import json

    for name in ("MODELS_DIR", "MODEL_REPORTS_DIR", "PROCESSED_DIR"):
        monkeypatch.setattr(A, name, tmp_path)
    monkeypatch.setattr(A, "MODEL_FILE", tmp_path / "m.joblib")
    monkeypatch.setattr(A, "FLAGGED_FILE", tmp_path / "f.parquet")
    monkeypatch.setattr(A, "INVOICE_SCORES_FILE", tmp_path / "i.parquet")
    # 3-hourly lines over 2011 -> errors land in the fit (Jan), validation (Aug) and test (Nov/Dec) periods
    lines = make_lines(n=3000, errors=(100, 1800, 2600, 2900))
    lines["ProductName"] = lines["StockCode"]
    lines["InvoiceDate"] = pd.date_range("2011-01-01", periods=3000, freq="3h")
    result = A.run_experiment(lines, verbose=False)
    final, threshold, scored, invoices = A.score_everything(lines, result["scorers"][result["champion"]])
    A.save(result, final, threshold, scored, invoices, A.pr_curves(result))
    report = json.loads((tmp_path / "anomaly_detection.json").read_text())
    assert all(isinstance(report["test_metrics"][k], (int, float)) for k in ("tp", "fp", "fn", "tn", "precision"))

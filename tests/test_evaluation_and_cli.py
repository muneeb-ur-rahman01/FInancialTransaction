"""Tests for Module 8 (consolidated evaluation report) and the main.py command line."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "model_reports"
needs_reports = pytest.mark.skipif(
    not all((REPORTS / n).exists() for n in ["sales_forecasting.json", "demand_forecasting.json",
                                               "customer_segmentation.json", "anomaly_detection.json"]),
    reason="run the pipeline first")


@needs_reports
def test_summary_has_one_comparable_row_per_task(tmp_path, monkeypatch):
    from src.models import evaluation_report as E

    summary = E.build_summary(E.load_reports())
    tasks = summary["tasks"]
    assert [t["task"][:2] for t in tasks] == ["7A", "7B", "7C", "7D"]
    for t in tasks:
        assert isinstance(t["champion_value"], float) and isinstance(t["reference_value"], float)
        assert t["verdict"] and t["limitation"] and t["evaluation"]
    sales, demand, seg, anom = tasks
    assert sales["lower_is_better"] and sales["champion_value"] < sales["reference_value"]
    assert seg["champion_value"] > seg["reference_value"]          # K-Means separates better than rules
    assert 0 <= anom["champion_value"] <= 1

    monkeypatch.setattr(E, "SUMMARY_MD", tmp_path / "summary.md")
    E.write_markdown(summary)
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "## 7D Anomaly detection" in text and "| Task | Champion |" in text


def run_cli(*args):
    return subprocess.run([sys.executable, "main.py", *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})


def test_cli_help_lists_every_command():
    out = run_cli("--help").stdout
    for cmd in ["pipeline", "forecast-sales", "forecast-demand", "segment", "check-line", "evaluate", "dashboard", "test"]:
        assert cmd in out


def test_cli_rejects_bad_input_cleanly():
    r = run_cli("segment")
    assert r.returncode != 0 and "Give --customer" in r.stderr
    r = run_cli("pipeline", "--from", "nonsense")
    assert r.returncode != 0 and "invalid choice" in r.stderr


@needs_reports
def test_cli_predictions_run():
    r = run_cli("check-line", "--code", "85123A", "--quantity", "5000", "--price", "2.55")
    assert r.returncode == 0 and "SUSPICIOUS" in r.stdout
    r = run_cli("forecast-sales", "--days", "6")
    assert r.returncode == 0 and "Total" in r.stdout
    r = run_cli("segment", "--customer", "1")
    assert r.returncode != 0 and "not found" in r.stderr


def test_pipeline_steps_are_ordered():
    sys.path.insert(0, str(ROOT))
    import main

    assert main.STEP_ORDER == ["quality", "clean", "features", "eda", "stats", "models", "report"]
    for _, args, _ in main.PIPELINE:
        target = args[-1]
        assert (ROOT / target).exists() if target.endswith(".py") else \
            (ROOT / (target.replace(".", "/") + ".py")).exists()

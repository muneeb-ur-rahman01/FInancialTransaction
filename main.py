"""
Financial Transaction Analysis & Prediction System - single entry point
=======================================================================
Run from the project root (with the virtual environment active):

  python main.py pipeline                     rebuild everything: clean -> features -> analysis -> models -> report
  python main.py pipeline --from models       start at a later step (steps: quality, clean, features, eda,
                                              stats, models, report)

  python main.py forecast-sales --days 12     next 12 trading days of net sales (with an 80% range)
  python main.py forecast-demand --top 20     next week's units for the 20 biggest products
  python main.py forecast-demand --code 85123A 22423
  python main.py segment --customer 12347     segment + recommended action for a customer
  python main.py segment --recency 10 --frequency 8 --monetary 2500     what-if profile
  python main.py check-line --code 85123A --quantity 5000 --price 2.55  is this order line suspicious?

  python main.py evaluate                     regenerate the consolidated evaluation report (Module 8)
  python main.py dashboard                    open the Streamlit dashboard
  python main.py test                         run the test suite

Add --save to any forecast command to write a CSV to reports/predictions/.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREDICTIONS_DIR = ROOT / "reports" / "predictions"

# (name, command, what it produces) - run in this order
PIPELINE = [
    ("quality", ["notebooks/02_data_quality_check.py"], "data quality report"),
    ("clean", ["-m", "src.data_processing.cleaning"], "clean transactions"),
    ("features", ["-m", "src.data_processing.features"], "feature tables"),
    ("eda", ["notebooks/04_eda.py"], "EDA findings + figures"),
    ("stats", ["notebooks/05_statistical_analysis.py"], "statistical findings + figures"),
    ("models", ["-m", "src.models.sales_forecasting"], "7A sales forecaster"),
    ("models", ["-m", "src.models.demand_forecasting"], "7B demand forecaster"),
    ("models", ["-m", "src.models.customer_segmentation"], "7C customer segments"),
    ("models", ["-m", "src.models.anomaly_detection"], "7D anomaly detector"),
    ("report", ["-m", "src.models.evaluation_report"], "Module 8 evaluation summary"),
]
STEP_ORDER = list(dict.fromkeys(step for step, _, _ in PIPELINE))


def run_pipeline(start):
    steps = [s for s in PIPELINE if STEP_ORDER.index(s[0]) >= STEP_ORDER.index(start)]
    total = time.time()
    for i, (step, args, what) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {step:<8} -> {what}", flush=True)
        t0 = time.time()
        result = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", env={**__import__("os").environ,
                                                                         "PYTHONIOENCODING": "utf-8"})
        if result.returncode != 0:
            print(result.stdout[-3000:])
            print(result.stderr[-3000:])
            sys.exit(f"Step '{step}' ({what}) failed - see the error above. Fix it, then rerun with "
                     f"`python main.py pipeline --from {step}`.")
        print(f"         done in {time.time() - t0:,.0f}s")
    print(f"\nPipeline finished in {(time.time() - total) / 60:,.1f} min. Open the dashboard: python main.py dashboard")


def save(frame, name):
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / name
    frame.to_csv(path, index=False)
    print(f"\nSaved -> {path.relative_to(ROOT)}")


def cmd_forecast_sales(a):
    from src.models.predict import forecast_sales, load_bundle

    f = forecast_sales(a.days)
    b = load_bundle("sales")
    print(f"Net product sales forecast - {b['model_family']} ({b['model_name']}), trained to {b['trained_on'][1]}")
    print(f"{'Date':<12}{'Day':<11}{'Forecast':>11}{'80% range':>24}")
    for r in f.itertuples():
        print(f"{r.date:%Y-%m-%d}  {r.weekday:<10}£{r.forecast:>9,.0f}   £{r.low_80:>8,.0f} - £{r.high_80:>8,.0f}")
    print(f"{'Total':<23}£{f['forecast'].sum():>9,.0f}")
    print("\nNote: holidays are unknown to the model (last year the business closed 24 Dec - 3 Jan).")
    if a.save:
        save(f, "sales_forecast.csv")


def cmd_forecast_demand(a):
    import pandas as pd

    from src.models.predict import forecast_demand

    f = forecast_demand(top=None if a.code else a.top, codes=a.code)
    print(f"Units forecast for the week of {pd.Timestamp(f['week'].iloc[0]):%d %b %Y} (mean of the last 4 weeks)\n")
    print(f[["StockCode", "product", "forecast", "last_week", "mean_last_4_weeks"]].to_string(index=False))
    if a.save:
        save(f, "demand_forecast.csv")


def cmd_segment(a):
    from src.models.predict import customer_segment, segment_for

    if a.customer is not None:
        s = customer_segment(a.customer)
        print(f"Customer {s['customer_id']} ({s['country']}): {s['segment']}"
              f"{'  - LAPSING: on the win-back list' if s['lapsing'] else ''}")
        print(f"  last order {s['recency_days']} days ago, {s['orders']} orders, net spend £{s['net_spend']:,.0f}")
    else:
        if None in (a.recency, a.frequency, a.monetary):
            sys.exit("Give --customer ID, or all of --recency --frequency --monetary")
        s = segment_for(a.recency, a.frequency, a.monetary)
        print(f"Profile (recency {a.recency} days, {a.frequency} orders, £{a.monetary:,.0f}): {s['segment']}")
    print(f"  action: {s['action']}")


def cmd_check_line(a):
    import pandas as pd

    from src.models.predict import score_lines

    r = score_lines(pd.DataFrame({"StockCode": [a.code], "Quantity": [a.quantity], "UnitPrice": [a.price],
                                  "Country": [a.country]})).iloc[0]
    print(f"{r['StockCode']}: {a.quantity:,} x £{a.price:,.2f} = £{a.quantity * a.price:,.2f}")
    print(f"  {'SUSPICIOUS - please check before confirming' if r['flagged'] else 'OK - no alert'}")
    print(f"  reason: {r['reason']}")


def main():
    parser = argparse.ArgumentParser(description="Financial Transaction Analysis & Prediction System",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pipeline", help="rebuild data, analysis, models and reports")
    p.add_argument("--from", dest="start", choices=STEP_ORDER, default="quality", help="first step to run")

    p = sub.add_parser("forecast-sales", help="forecast the next trading days of net sales")
    p.add_argument("--days", type=int, default=12, help="number of trading days (1-24)")
    p.add_argument("--save", action="store_true")

    p = sub.add_parser("forecast-demand", help="forecast next week's units per product")
    p.add_argument("--top", type=int, default=20, help="show the N largest products")
    p.add_argument("--code", nargs="+", help="one or more stock codes")
    p.add_argument("--save", action="store_true")

    p = sub.add_parser("segment", help="customer segment and recommended action")
    p.add_argument("--customer", type=int)
    p.add_argument("--recency", type=int, help="days since last order")
    p.add_argument("--frequency", type=int, help="number of orders")
    p.add_argument("--monetary", type=float, help="net spend in £")

    p = sub.add_parser("check-line", help="anomaly check for one order line")
    p.add_argument("--code", required=True)
    p.add_argument("--quantity", type=int, required=True)
    p.add_argument("--price", type=float, required=True)
    p.add_argument("--country", default="United Kingdom")

    sub.add_parser("evaluate", help="regenerate the consolidated evaluation report")
    sub.add_parser("dashboard", help="open the Streamlit dashboard")
    sub.add_parser("test", help="run the test suite")

    a = parser.parse_args()
    try:
        if a.command == "pipeline":
            run_pipeline(a.start)
        elif a.command == "forecast-sales":
            cmd_forecast_sales(a)
        elif a.command == "forecast-demand":
            cmd_forecast_demand(a)
        elif a.command == "segment":
            cmd_segment(a)
        elif a.command == "check-line":
            cmd_check_line(a)
        elif a.command == "evaluate":
            from src.models.evaluation_report import main as report_main

            report_main()
        elif a.command == "dashboard":
            subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"], cwd=ROOT)
        elif a.command == "test":
            sys.exit(subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT).returncode)
    except (FileNotFoundError, KeyError, ValueError) as e:
        sys.exit(f"Error: {e.args[0] if e.args else e}")


if __name__ == "__main__":
    main()

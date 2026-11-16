"""
Module 8 - Consolidated Model Evaluation
========================================
Reads the evaluation each Module 7 task saved and writes ONE comparable summary:
  reports/model_reports/model_evaluation_summary.md    (human-readable report)
  reports/model_reports/model_evaluation_summary.json  (same content, for the dashboard)
  reports/figures/08_01_champion_vs_baseline.png

Nothing is re-trained or re-scored here, so the numbers are exactly those of the saved models.

Run from the project root:
    python -m src.models.evaluation_report
"""

import json
from datetime import datetime

import pandas as pd

from src.utils.paths import MODEL_REPORTS_DIR

REPORTS = {
    "sales": ("sales_forecasting.json", "python -m src.models.sales_forecasting"),
    "demand": ("demand_forecasting.json", "python -m src.models.demand_forecasting"),
    "segments": ("customer_segmentation.json", "python -m src.models.customer_segmentation"),
    "anomaly": ("anomaly_detection.json", "python -m src.models.anomaly_detection"),
}
SUMMARY_MD = MODEL_REPORTS_DIR / "model_evaluation_summary.md"
SUMMARY_JSON = MODEL_REPORTS_DIR / "model_evaluation_summary.json"


def load_reports():
    out = {}
    for key, (name, command) in REPORTS.items():
        path = MODEL_REPORTS_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"{name} missing - run:  {command}")
        out[key] = json.loads(path.read_text(encoding="utf-8"))
    return out


def label(row):
    return row["model"] if row["family"] == "Baseline" else f"{row['family']} ({row['model']})"


# --------------------------------------------------------------------------
# One summary row per task: champion vs the strongest alternative, on the SAME metric
# --------------------------------------------------------------------------
def summarise_sales(rep):
    comp = pd.DataFrame(rep["test_comparison"])
    base = comp[comp["family"] == "Baseline"].sort_values("MAE").iloc[0]
    champ = rep["test_metrics"]
    return {
        "task": "7A Sales forecasting", "type": "Regression (time series)",
        "question": "Net product sales of the next trading day",
        "evaluation": f"Time-series CV ({rep['cv_setup'].split(' on')[0]}) for selection; "
                      f"test {rep['test_period'][0]} to {rep['test_period'][1]} scored once",
        "champion": f"{rep['model_family']} ({rep['model_name']})",
        "metric": "MAE (£ per day, lower is better)", "lower_is_better": True,
        "champion_value": champ["MAE"], "reference": f"Best baseline: {base['model']}", "reference_value": base["MAE"],
        "other_metrics": {"WAPE %": champ["WAPE"], "RMSE £": champ["RMSE"], "R2": champ["R2"],
                          "MAPE %": champ["MAPE"], "Bias £/day": champ["Bias"]},
        "verdict": f"ML beats the best simple rule by {(1 - champ['MAE'] / base['MAE']) * 100:.1f}% on MAE: a real "
                   "but modest gain. Daily sales are dominated by unpredictable large wholesale orders.",
        "limitation": "One year of history: no model has seen an autumn ramp-up before, so every model "
                      f"under-forecasts Q4 (on average £{abs(champ['Bias']):,.0f} per day too low).",
    }


def summarise_demand(rep):
    comp = pd.DataFrame(rep["test_comparison"])
    ml = comp[comp["family"] != "Baseline"].sort_values("score").iloc[0]
    champ = rep["test_metrics"]
    return {
        "task": "7B Product demand", "type": "Regression (panel of 2,237 products)",
        "question": "Units each product sells next week",
        "evaluation": f"Week-based expanding CV ({CV_TEXT(rep)}) for selection; test weeks {rep['test_weeks'][0]} "
                      f"to {rep['test_weeks'][1]} scored once",
        "champion": rep["model_name"] + " (simple rule)",
        "metric": "Score = WAPE + product-level bias % (lower is better)", "lower_is_better": True,
        "champion_value": champ["score"], "reference": f"Best ML model: {label(ml)}", "reference_value": ml["score"],
        "other_metrics": {"WAPE %": champ["WAPE"], "Product bias %": champ["ProductBias_pct"],
                          "Total bias %": champ["Bias_pct"], "MAE units": champ["MAE"], "R2": champ["R2"]},
        "verdict": "No ML model convincingly beat the 4-week average: ML had lower weekly error but systematically "
                   "under-forecast the best sellers. Simpler model kept (plus a business rule preferring it over "
                   "'last week' when tied).",
        "limitation": "Sales history alone cannot anticipate promotions or new launches; products with under "
                      "4 weeks of history are not forecast (cold start).",
    }


def CV_TEXT(rep):  # noqa: N802 - short helper
    return rep["cv_setup"].replace("week-based expanding CV: ", "").replace(" on training weeks", "")


def summarise_segments(rep):
    algos = pd.DataFrame(rep["algorithm_comparison"]).set_index("method")
    base = algos.loc[[m for m in algos.index if "baseline" in m][0]]
    return {
        "task": "7C Customer segmentation", "type": "Unsupervised clustering",
        "question": "Which natural groups of customers exist?",
        "evaluation": f"k = 2..8 judged on silhouette and bootstrap stability (ARI); {rep['customers']:,} customers",
        "champion": f"K-Means, k = {rep['k']}",
        "metric": "Silhouette (separation, higher is better)", "lower_is_better": False,
        "champion_value": rep["silhouette"], "reference": "Rule-based RFM segments", "reference_value": base["silhouette"],
        "other_metrics": {"Stability ARI": rep["stability_ari"],
                          "Davies-Bouldin": algos.loc["K-Means", "davies_bouldin"],
                          "Lapsing valuable customers": rep["lapsing_customers"]},
        "verdict": f"K-Means separates customers {rep['silhouette'] / base['silhouette']:.1f}x better than fixed RFM "
                   "rules and is stable under resampling. Champions are 16% of customers and 64% of revenue.",
        "limitation": "Clusters follow overall distance, not business meaning: some 'regulars' are long silent, "
                      "handled by an explicit 90-day lapsing flag rather than more (unstable) clusters.",
    }


def summarise_anomaly(rep):
    m = rep["test_metrics"]
    comp = pd.DataFrame(rep["comparison"]).set_index("method")
    iforest = comp.loc["Isolation Forest"]
    return {
        "task": "7D Anomaly detection", "type": "Rare-event detection (line level)",
        "question": "Which order lines are likely keying errors?",
        "evaluation": f"Fit {rep['periods']['fit']}, validation {rep['periods']['validation']} (selection), "
                      f"test {rep['periods']['test']} (scored once); alert = top {rep['flag_rate']:.0%} of lines",
        "champion": f"{rep['family']}: {rep['method']}",
        "metric": "Recall of known errors at the alert threshold (higher is better)", "lower_is_better": False,
        "champion_value": m["recall"], "reference": "Isolation Forest (unsupervised ML)",
        "reference_value": iforest["test_recall"],
        "other_metrics": {"Precision": m["precision"], "F1": m["f1"], "Lift vs random": m["lift"],
                          "ROC-AUC": m["roc_auc"], "Average precision": m["average_precision"],
                          "TP": m["tp"], "FP": m["fp"], "FN": m["fn"], "TN": m["tn"]},
        "verdict": f"A transparent rule catches {m['recall']:.0%} of known errors in a queue {m['lift']:.0f}x more "
                   "concentrated than random checks, including every catastrophic keying error; ML detectors did worse.",
        "limitation": "The answer key mixes typos with quick customer cancellations of normal lines, which no "
                      "detector can see; measured precision is a lower bound (the queue finds unlabelled errors too).",
    }


def build_summary(reports):
    rows = [summarise_sales(reports["sales"]), summarise_demand(reports["demand"]),
            summarise_segments(reports["segments"]), summarise_anomaly(reports["anomaly"])]
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "principles": [
            "Split by time: models learn from the past and are tested on the most recent period.",
            "Baselines first: every model is compared with a simple rule; complexity must earn its place.",
            "Select on cross-validation / validation only; the test period is scored once.",
            "Preprocessing and product/customer norms are fitted on training data only (no leakage).",
            "Metrics match the problem: MAE/WAPE for sales, WAPE + product bias for demand, silhouette + "
            "stability for clusters, precision/recall/F1 for rare anomalies (accuracy is meaningless there).",
        ],
        "tasks": rows,
    }


def fmt(value, metric):
    if "Recall" in metric or "Precision" in metric:
        return f"{value:.1%}"
    if "£" in metric:
        return f"£{value:,.0f}"
    if "Silhouette" in metric:
        return f"{value:.3f}"
    return f"{value:.1f}"


def write_markdown(summary):
    lines = ["# Model Evaluation Summary", "", f"_Generated {summary['created']} by `python -m src.models.evaluation_report`._",
             "", "## Evaluation principles", ""]
    lines += [f"{i}. {p}" for i, p in enumerate(summary["principles"], 1)]
    lines += ["", "## Overview", "",
              "| Task | Champion | Metric | Champion | Reference | Reference value |",
              "|---|---|---|---|---|---|"]
    for t in summary["tasks"]:
        lines.append(f"| {t['task']} | {t['champion']} | {t['metric']} | {fmt(t['champion_value'], t['metric'])} | "
                     f"{t['reference']} | {fmt(t['reference_value'], t['metric'])} |")
    for t in summary["tasks"]:
        lines += ["", f"## {t['task']}", "", f"- **Question:** {t['question']}", f"- **Type:** {t['type']}",
                  f"- **Evaluation:** {t['evaluation']}", f"- **Champion:** {t['champion']}",
                  f"- **Verdict:** {t['verdict']}", f"- **Limitation:** {t['limitation']}", "",
                  "| Metric | Value |", "|---|---|"]
        for k, v in t["other_metrics"].items():
            lines.append(f"| {k} | {v:,.4g} |" if isinstance(v, float) else f"| {k} | {v:,} |")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(summary):
    import matplotlib.pyplot as plt

    from src.utils import plotting as P

    P.apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, t in zip(axes, summary["tasks"]):
        names = ["Champion", "Reference"]
        values = [t["champion_value"], t["reference_value"]]
        ax.bar(names, values, color=[P.SERIES[0], P.MUTED], width=0.6)
        for x, v in enumerate(values):
            ax.text(x, v, fmt(v, t["metric"]), ha="center", va="bottom", fontsize=10, color=P.INK)
        ax.set_title(t["task"], fontsize=11)
        ax.set_ylabel(t["metric"].split(" (")[0], fontsize=9)
        ax.set_xticks([0, 1], [t["champion"][:22], t["reference"].split(": ")[-1][:22]], fontsize=8)
        ax.set_ylim(0, max(values) * 1.2)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Champion vs reference, each on its own metric  (7A, 7B: lower is better;  7C, 7D: higher is better)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    P.save(fig, "08_01_champion_vs_baseline.png")


def main():
    summary = build_summary(load_reports())
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    write_markdown(summary)
    plot_summary(summary)
    for t in summary["tasks"]:
        print(f"{t['task']:<26} {t['champion']:<45} {t['metric'].split(' (')[0]}: "
              f"{fmt(t['champion_value'], t['metric'])} vs {fmt(t['reference_value'], t['metric'])} ({t['reference']})")
    print(f"\nSaved -> {SUMMARY_MD}\nSaved -> {SUMMARY_JSON}\nSaved -> reports/figures/08_01_champion_vs_baseline.png")


if __name__ == "__main__":
    main()

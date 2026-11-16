"""
Module 7D - Anomaly Detection
=============================
Goal   : flag sale lines that are likely order-entry errors (or otherwise unusual) so a person
         can check them BEFORE the order is confirmed.
Input  : data/processed/transactions_clean.parquet          (Module 3)
Output : artifacts/models/anomaly_detector.joblib            (scorer + alert threshold)
         data/processed/flagged_lines.parquet                 (every line above the threshold)
         data/processed/invoice_anomaly_scores.parquet        (one row per order, worst line)
         reports/model_reports/anomaly_detection.json
         reports/figures/07d_*.png

What we learned while designing it (prototypes)
  * Order level does not work: a corrected order is usually a normal order with ONE mistyped
    line, so order totals look normal (test AUC ~0.6). Detection must happen per LINE, comparing
    each line with what is normal for that product.
  * Ground truth is partial: the 424 known errors (IsEntryCorrection, cancelled within 1 hour) are
    one kind of anomaly. They are used only to EVALUATE and SELECT, never as input features.

Candidates (all score lines; higher = more suspicious)
  Rules         largest quantity / largest line value / quantity relative to the product's norm
  Unsupervised  Isolation Forest - learns "normal" without labels
  Supervised    Logistic Regression and XGBoost trained on past known errors
Protocol (time-based, no leakage)
  fit period        -> 2011-07-31   candidates fitted here (incl. product norms)
  validation period 2011-08-01 -> 2011-10-16   champion = highest average precision here
  test period       2011-10-17 -> 2011-12-09   scored once
  Alert threshold = top FLAG_RATE of TRAINING-period scores, so the test period never sets it.
Metrics: rare positives (0.08% of lines) make accuracy meaningless (flag nothing = 99.9%), so we
use average precision (area under the precision-recall curve), ROC-AUC, and precision, recall,
F1 and the confusion matrix at the alert threshold. Lift = precision / base rate.

Run from the project root:
    python -m src.models.anomaly_detection
"""

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from src.analysis.eda import load_clean
from src.utils.paths import MODEL_REPORTS_DIR, MODELS_DIR, PROCESSED_DIR

VAL_START = pd.Timestamp("2011-08-01")
TEST_START = pd.Timestamp("2011-10-17")
FLAG_RATE = 0.01                     # review queue = top 1% of lines (~100 lines per week)
SEED = 42
LABEL = "IsEntryCorrection"

MODEL_FILE = MODELS_DIR / "anomaly_detector.joblib"
FLAGGED_FILE = PROCESSED_DIR / "flagged_lines.parquet"
INVOICE_SCORES_FILE = PROCESSED_DIR / "invoice_anomaly_scores.parquet"

FEATURES = ["log_qty", "log_value", "qty_vs_product", "price_vs_product", "log_product_history",
            "log_invoice_lines", "hour", "is_uk", "has_customer"]


# --------------------------------------------------------------------------
# Line-level data and features
# --------------------------------------------------------------------------
def load_lines(df):
    """All product sale lines (including the ones later corrected - those are what we want to find)."""
    lines = df[(df["TransactionType"] == "sale") & df["IsProduct"]].sort_values("InvoiceDate")
    return lines.reset_index(drop=True)


def product_norms(lines):
    """What is normal per product, learned from the lines given (= the fitting period only)."""
    table = lines.groupby("StockCode").agg(usual_qty=("Quantity", "median"), usual_price=("UnitPrice", "median"),
                                           history=("Quantity", "size"))
    return {"table": table, "usual_qty": float(lines["Quantity"].median()),
            "usual_price": float(lines["UnitPrice"].median())}


def line_features(lines, norms):
    """Features known when the line is keyed in. Unknown products fall back to overall norms."""
    t = lines[["StockCode"]].join(norms["table"], on="StockCode")
    usual_qty = t["usual_qty"].fillna(norms["usual_qty"]).to_numpy()
    usual_price = t["usual_price"].fillna(norms["usual_price"]).to_numpy()
    qty = lines["Quantity"].to_numpy(dtype=float)
    price = lines["UnitPrice"].to_numpy(dtype=float)
    return pd.DataFrame({
        "log_qty": np.log1p(qty),
        "log_value": np.log1p(qty * price),
        "qty_vs_product": np.log(qty / usual_qty),                 # 0 = usual quantity, +2.3 = 10x usual
        "price_vs_product": np.log(price / usual_price),
        "log_product_history": np.log1p(t["history"].fillna(0).to_numpy()),
        "log_invoice_lines": np.log1p(lines.groupby("InvoiceNo")["Quantity"].transform("size").to_numpy()),
        "hour": lines["InvoiceDate"].dt.hour.to_numpy(),
        "is_uk": (lines["Country"] == "United Kingdom").to_numpy().astype(int),
        "has_customer": lines["CustomerID"].notna().to_numpy().astype(int),
    }, index=lines.index)


# --------------------------------------------------------------------------
# Scorers: one interface for rules, unsupervised and supervised methods
# --------------------------------------------------------------------------
class LineAnomalyScorer:
    """fit(lines, y) learns product norms (+ a model if any); score(lines) -> higher = more suspicious."""

    def __init__(self, kind, column=None, estimator=None):
        self.kind = kind              # "rule" | "unsupervised" | "supervised"
        self.column = column
        self.estimator = estimator

    def fit(self, lines, y=None):
        self.norms_ = product_norms(lines)
        X = line_features(lines, self.norms_)[FEATURES]
        if self.kind == "unsupervised":
            self.model_ = clone(self.estimator).fit(X)
        elif self.kind == "supervised":
            self.model_ = clone(self.estimator).fit(X, np.asarray(y, dtype=int))
        return self

    def score(self, lines):
        X = line_features(lines, self.norms_)[FEATURES]
        if self.kind == "rule":
            return X[self.column].to_numpy(dtype=float)
        if self.kind == "unsupervised":
            return -self.model_.score_samples(X)
        return self.model_.predict_proba(X)[:, 1]

    def clone(self):
        return LineAnomalyScorer(self.kind, self.column, self.estimator)


def candidate_scorers(pos_weight=100.0):
    return [
        ("Rule", "Largest quantity", LineAnomalyScorer("rule", column="log_qty")),
        ("Rule", "Largest line value", LineAnomalyScorer("rule", column="log_value")),
        ("Rule", "Quantity vs product norm", LineAnomalyScorer("rule", column="qty_vs_product")),
        ("Unsupervised", "Isolation Forest", LineAnomalyScorer("unsupervised", estimator=Pipeline([
            ("scale", RobustScaler()),
            ("model", IsolationForest(n_estimators=300, random_state=SEED, n_jobs=1))]))),
        ("Supervised", "Logistic Regression", LineAnomalyScorer("supervised", estimator=Pipeline([
            ("scale", RobustScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000))]))),
        ("Supervised", "XGBoost", LineAnomalyScorer("supervised", estimator=XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight, eval_metric="aucpr", random_state=SEED, n_jobs=4))),
    ]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def detection_metrics(y, scores, threshold):
    y = np.asarray(y, dtype=int)
    flagged = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, flagged, labels=[0, 1]).ravel()
    precision = precision_score(y, flagged, zero_division=0)
    return {
        "average_precision": float(average_precision_score(y, scores)) if y.any() else float("nan"),
        "roc_auc": float(roc_auc_score(y, scores)) if 0 < y.sum() < len(y) else float("nan"),
        "precision": float(precision),
        "recall": float(recall_score(y, flagged, zero_division=0)),
        "f1": float(f1_score(y, flagged, zero_division=0)),
        "flagged": int(flagged.sum()), "flag_rate": float(flagged.mean()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "base_rate": float(y.mean()),
        "lift": float(precision / y.mean()) if y.mean() else float("nan"),
    }


def alert_threshold(scores, flag_rate=FLAG_RATE):
    """Score above which the top `flag_rate` of (training) lines lie."""
    return float(np.quantile(scores, 1 - flag_rate))


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------
def split_periods(lines, val_start=VAL_START, test_start=TEST_START):
    d = lines["InvoiceDate"]
    return lines[d < val_start], lines[(d >= val_start) & (d < test_start)], lines[d >= test_start]


def run_experiment(lines, verbose=True):
    fit, val, test = split_periods(lines)
    train = pd.concat([fit, val])
    pos_weight = (1 - fit[LABEL].mean()) / max(fit[LABEL].mean(), 1e-9)

    rows, scorers, test_scores = [], {}, {}
    for family, name, scorer in candidate_scorers(pos_weight):
        # 1) selection: fit on the fit period, judge on the validation period
        s_val = scorer.clone().fit(fit, fit[LABEL]).score(val)
        # 2) final evaluation: refit on the whole training period, threshold from training scores, score test once
        final = scorer.clone().fit(train, train[LABEL])
        threshold = alert_threshold(final.score(train))
        s_test = final.score(test)
        m = detection_metrics(test[LABEL], s_test, threshold)
        rows.append({"family": family, "method": name,
                     "val_average_precision": average_precision_score(val[LABEL], s_val),
                     "val_roc_auc": roc_auc_score(val[LABEL], s_val), **{f"test_{k}": v for k, v in m.items()}})
        scorers[name], test_scores[name] = scorer, s_test
        if verbose:
            print(f"  {family:<12} {name:<26} val AP {rows[-1]['val_average_precision']:.4f} | test AP "
                  f"{m['average_precision']:.4f}  AUC {m['roc_auc']:.3f}  recall {m['recall']:.1%}  "
                  f"precision {m['precision']:.1%}  lift x{m['lift']:.0f}")
    table = pd.DataFrame(rows).sort_values("val_average_precision", ascending=False).reset_index(drop=True)
    champion = table.iloc[0]["method"]
    return {"fit": fit, "val": val, "test": test, "table": table, "champion": champion,
            "champion_family": table.iloc[0]["family"], "scorers": scorers, "test_scores": test_scores}


def pr_curves(result, max_points=150):
    out = {}
    y = result["test"][LABEL].to_numpy()
    for name, scores in result["test_scores"].items():
        p, r, _ = precision_recall_curve(y, scores)
        idx = np.unique(np.linspace(0, len(p) - 1, min(max_points, len(p))).astype(int))
        out[name] = {"precision": p[idx].round(5).tolist(), "recall": r[idx].round(5).tolist()}
    return out


def score_everything(lines, scorer):
    """Production: refit the champion on all lines, set the threshold, score every line and order."""
    final = scorer.clone().fit(lines, lines[LABEL])
    scores = final.score(lines)
    threshold = alert_threshold(scores)
    feats = line_features(lines, final.norms_)
    history = np.expm1(feats["log_product_history"]).round().astype(int)
    # With fewer than 3 sales of a product its "usual" quantity is mostly the line itself -> not shown
    usual = np.exp(np.log(lines["Quantity"]) - feats["qty_vs_product"]).where(history >= 3)
    scored = lines.assign(anomaly_score=scores, flagged=scores >= threshold, product_lines=history,
                          usual_qty=usual, qty_vs_usual=lines["Quantity"] / usual)
    invoices = scored.groupby("InvoiceNo").agg(
        date=("InvoiceDate", "min"), customer=("CustomerID", "first"), country=("Country", "first"),
        order_value=("TotalAmount", "sum"), lines=("StockCode", "size"),
        max_score=("anomaly_score", "max"), flagged_lines=("flagged", "sum"), known_error=(LABEL, "any"))
    invoices["country"] = invoices["country"].astype(str)
    invoices["flagged"] = invoices["flagged_lines"] > 0
    return final, threshold, scored, invoices


def save(result, final, threshold, scored, invoices, curves):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    champ = result["table"].set_index("method").loc[result["champion"]]
    bundle = {
        "scorer": final, "threshold": threshold, "flag_rate": FLAG_RATE,
        "method": result["champion"], "family": result["champion_family"], "features": FEATURES,
        # .item() turns numpy scalars into plain Python numbers (otherwise JSON would store them as text)
        "test_metrics": {k.replace("test_", ""): (v.item() if hasattr(v, "item") else v)
                         for k, v in champ.items() if k.startswith("test_")},
        "periods": {"fit": f"-> {VAL_START.date() - pd.Timedelta(days=1)}",
                    "validation": f"{VAL_START.date()} -> {TEST_START.date() - pd.Timedelta(days=1)}",
                    "test": f"{TEST_START.date()} ->"},
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, MODEL_FILE)

    cols = ["InvoiceNo", "InvoiceDate", "StockCode", "ProductName", "Quantity", "usual_qty", "qty_vs_usual",
            "product_lines",
            "UnitPrice", "TotalAmount", "CustomerID", "Country", "anomaly_score", LABEL]
    flagged = scored.loc[scored["flagged"], cols].sort_values("anomaly_score", ascending=False)
    flagged["Country"] = flagged["Country"].astype(str)
    flagged.to_parquet(FLAGGED_FILE, index=False)
    invoices.reset_index().to_parquet(INVOICE_SCORES_FILE, index=False)

    report = {k: v for k, v in bundle.items() if k != "scorer"}
    report["selection"] = "champion = highest average precision on the validation period"
    report["comparison"] = result["table"].to_dict(orient="records")
    report["pr_curves_test"] = curves
    report["production"] = {"lines_scored": int(len(scored)), "lines_flagged": int(scored["flagged"].sum()),
                            "known_errors_flagged": int((scored["flagged"] & scored[LABEL]).sum()),
                            "known_errors_total": int(scored[LABEL].sum()),
                            "orders_flagged": int(invoices["flagged"].sum())}
    (MODEL_REPORTS_DIR / "anomaly_detection.json").write_text(json.dumps(report, indent=2, default=str),
                                                                encoding="utf-8")
    return bundle, report


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_results(result, curves, scored):
    import matplotlib.pyplot as plt

    from src.utils import plotting as P

    P.apply_style()
    champion = result["champion"]
    base_rate = result["test"][LABEL].mean()

    fig, ax = plt.subplots(figsize=(9, 5.2))
    shown = [champion, "Isolation Forest", "Logistic Regression"]
    for name, color in zip(dict.fromkeys(shown), P.SERIES):
        c = curves[name]
        ax.plot(c["recall"], c["precision"], color=color, linewidth=2.5 if name == champion else 1.8,
                label=f"{name}{' (champion)' if name == champion else ''}")
    ax.axhline(base_rate, color=P.MUTED, linestyle="--", linewidth=1, label=f"Random ({base_rate:.2%})")
    ax.set_yscale("log")
    ax.set_xlabel("Recall: share of known errors found")
    ax.set_ylabel("Precision: share of alerts that are known errors (log)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}" if v < 0.1 else f"{v:.0%}"))
    ax.set_title("Precision-recall on the test period")
    ax.legend(loc="upper right")
    fig.tight_layout()
    P.save(fig, "07d_01_precision_recall.png")

    m = result["table"].set_index("method").loc[champion]
    cm = np.array([[m["test_tn"], m["test_fp"]], [m["test_fn"], m["test_tp"]]])
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    ax.imshow(np.log1p(cm), cmap=P.SEQUENTIAL)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=13, fontweight="bold",
                color="white" if np.log1p(v) > np.log1p(cm).max() * 0.6 else P.INK)
    ax.set_xticks([0, 1], ["Not flagged", "Flagged"])
    ax.set_yticks([0, 1], ["Normal line", "Known error"])
    ax.set_title(f"Confusion matrix - {champion} (test)", fontsize=11)
    ax.grid(False)
    fig.tight_layout()
    P.save(fig, "07d_02_confusion_matrix.png")

    sample = scored.sample(min(60000, len(scored)), random_state=SEED)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sample = sample[sample["usual_qty"].notna()]
    normal = sample[~sample["flagged"] & ~sample[LABEL]]
    ax.scatter(normal["usual_qty"], normal["Quantity"], s=4, color=P.BLUE_LIGHT, alpha=0.3, linewidths=0,
               label="Normal line (sample)")
    flagged = scored[scored["flagged"] & ~scored[LABEL] & scored["usual_qty"].notna()]
    ax.scatter(flagged["usual_qty"], flagged["Quantity"], s=10, color=P.SERIES[0], alpha=0.6, linewidths=0,
               label="Flagged")
    known = scored[scored[LABEL] & scored["usual_qty"].notna()]
    ax.scatter(known["usual_qty"], known["Quantity"], s=40, color=P.SERIES[1], edgecolor=P.SURFACE, linewidth=1,
               label="Known entry error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Product's usual quantity per line (log)")
    ax.set_ylabel("Quantity on this line (log)")
    ax.set_title("Lines vs their product's norm")
    ax.legend(loc="upper left", markerscale=1.5)
    fig.tight_layout()
    P.save(fig, "07d_03_lines_vs_norm.png")


# --------------------------------------------------------------------------
def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}" if abs(v) < 1 else f"{v:,.1f}")

    lines = load_lines(load_clean())
    fit, val, test = split_periods(lines)
    print(f"Product sale lines: {len(lines):,}; known entry errors: {int(lines[LABEL].sum())} "
          f"({lines[LABEL].mean():.3%})")
    for name, part in [("fit", fit), ("validation", val), ("test", test)]:
        print(f"  {name:<10} {part['InvoiceDate'].min().date()} -> {part['InvoiceDate'].max().date()}: "
              f"{len(part):,} lines, {int(part[LABEL].sum())} known errors")
    print(f"Alert threshold: top {FLAG_RATE:.0%} of training-period scores\n\nComparing detectors (1-3 minutes)...")

    result = run_experiment(lines)
    print("\nRanked by validation average precision (selection), with the test period scored once:")
    cols = ["family", "method", "val_average_precision", "test_average_precision", "test_roc_auc", "test_precision",
            "test_recall", "test_f1", "test_lift", "test_tp", "test_fp", "test_fn"]
    print(result["table"][cols].to_string(index=False))

    m = result["table"].set_index("method").loc[result["champion"]]
    print(f"\nCHAMPION: {result['champion_family']} - {result['champion']}")
    print(f"Test period, alert = top {FLAG_RATE:.0%}: {int(m['test_flagged']):,} lines flagged, "
          f"{int(m['test_tp'])} of {int(m['test_tp'] + m['test_fn'])} known errors caught "
          f"(recall {m['test_recall']:.1%}), precision {m['test_precision']:.2%} = {m['test_lift']:.0f}x the base rate")
    print(f"Confusion matrix [[TN, FP], [FN, TP]] = [[{int(m['test_tn']):,}, {int(m['test_fp']):,}], "
          f"[{int(m['test_fn'])}, {int(m['test_tp'])}]]")

    curves = pr_curves(result)
    final, threshold, scored, invoices = score_everything(lines, result["scorers"][result["champion"]])
    bundle, report = save(result, final, threshold, scored, invoices, curves)
    plot_results(result, curves, scored)
    prod = report["production"]
    print(f"\nProduction scoring (all data): {prod['lines_flagged']:,} lines flagged in {prod['orders_flagged']:,} orders; "
          f"{prod['known_errors_flagged']} of {prod['known_errors_total']} known errors among them")
    print("\nTop 10 flagged lines:")
    top = scored[scored["flagged"]].nlargest(10, "anomaly_score")
    print(top[["InvoiceNo", "InvoiceDate", "ProductName", "Quantity", "usual_qty", "product_lines", "TotalAmount", LABEL]]
          .to_string(index=False))
    print(f"\nSaved model    -> {MODEL_FILE}")
    print(f"Saved flags    -> {FLAGGED_FILE}")
    print(f"Saved invoices -> {INVOICE_SCORES_FILE}")
    print(f"Saved report   -> {MODEL_REPORTS_DIR / 'anomaly_detection.json'}")
    print("Saved charts   -> reports/figures/07d_01..03_*.png")


if __name__ == "__main__":
    # Run through the importable module, so saved models reference
    # src.models.anomaly_detection.<class> instead of __main__.<class> (otherwise only this script could load them).
    from src.models.anomaly_detection import main as _main

    _main()

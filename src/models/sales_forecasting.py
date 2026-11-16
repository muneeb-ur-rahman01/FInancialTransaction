"""
Module 7A - Sales Forecasting
=============================
Goal   : predict tomorrow's (next trading day's) net product sales.
Input  : data/processed/features_daily.parquet          (Module 6)
Output : artifacts/models/sales_forecaster.joblib
         reports/model_reports/sales_forecasting.json + model comparison CSV
         reports/figures/07a_*.png

Scientific procedure
  1. Chronological split: the final 48 trading days (8 trading weeks, mid Oct - 9 Dec 2011)
     are the TEST set and are touched exactly once, at the end.
  2. Baselines first (yesterday, same weekday last week, ...). A model that cannot beat
     them is useless however sophisticated it is.
  3. Candidates: Ridge (linear), Random Forest and XGBoost (tree ensembles), each with a
     small hyper-parameter grid and two target options:
       raw      : predict £ sales directly
       relative : predict sales / recent level (24-day mean), multiply back.
                  Trees cannot predict above the highest value seen in training; the
                  Q4 peak IS higher, so predicting a ratio can help them extrapolate.
  4. Model selection by expanding-window time-series CV on the TRAINING period only.
  5. The CV winner is scored once on the test set, then refit on all data and saved.

Features: calendar + lags/rolling stats of SALES only. Lags of orders/customers are left
out on purpose: to forecast several days ahead (Module 9) we feed predictions back in as
lags, and we only predict sales - not orders.

Run from the project root:
    python -m src.models.sales_forecasting
"""

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.data_processing.features import DAILY_FILE
from src.models.estimators import ColumnBaseline  # noqa: F401  (re-exported for tests/pickles)
from src.models.evaluation import chronological_split, regression_metrics, time_series_cv
from src.utils.paths import MODEL_REPORTS_DIR, MODELS_DIR

TARGET = "net_sales"
LEVEL_COL = "sales_roll_mean_24"
TEST_DAYS = 48          # 8 trading weeks
CV_SPLITS = 5
CV_DAYS = 24            # each validation fold = 4 trading weeks
EXCLUDED_PREFIXES = ("orders_", "customers_", "avg_order_value_")
MODEL_FILE = MODELS_DIR / "sales_forecaster.joblib"
SEED = 42


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------
class LevelScaledRegressor(BaseEstimator, RegressorMixin):
    """Fit `estimator` on y / level and multiply predictions back by the level.

    level = X[level_col], a PAST-only rolling mean, so no leakage is introduced.
    """

    def __init__(self, estimator=None, level_col=LEVEL_COL):
        self.estimator = estimator
        self.level_col = level_col

    def fit(self, X, y):
        level = X[self.level_col].to_numpy(dtype=float)
        self.estimator_ = clone(self.estimator).fit(X, np.asarray(y, dtype=float) / level)
        return self

    def predict(self, X):
        return self.estimator_.predict(X) * X[self.level_col].to_numpy(dtype=float)


def make_pipeline(model):
    """Weekday -> one-hot (it is a category, not a number); everything else -> standardised.

    The preprocessing lives INSIDE the pipeline, so it is fitted on training folds only
    and is saved together with the model.
    """
    prep = ColumnTransformer(
        [("weekday", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["weekday"])],
        remainder=StandardScaler(),
        verbose_feature_names_out=False,
    )
    return Pipeline([("prep", prep), ("model", model)])


def candidate_models():
    """Every model we compare, as (family, name, estimator)."""
    cands = [
        ("Baseline", "Naive: yesterday", ColumnBaseline("sales_lag_1")),
        ("Baseline", "Seasonal naive: same weekday last week", ColumnBaseline("sales_lag_6")),
        ("Baseline", "Mean of same weekday, last 4 weeks", ColumnBaseline("sales_same_weekday_mean_4")),
        ("Baseline", "Mean of last 6 trading days", ColumnBaseline("sales_roll_mean_6")),
    ]
    grids = {
        "Ridge": [(f"alpha={a}", Ridge(alpha=a)) for a in (1, 10, 100)],
        "Random Forest": [
            (f"leaf={leaf}, max_features={mf}",
             RandomForestRegressor(n_estimators=300, min_samples_leaf=leaf, max_features=mf,
                                   random_state=SEED, n_jobs=1))   # tiny data: parallelism adds only overhead
            for leaf in (3, 10) for mf in (0.5, 1.0)
        ],
        "XGBoost": [
            (f"depth={d}, trees={n}",
             XGBRegressor(n_estimators=n, max_depth=d, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, random_state=SEED, n_jobs=-1))
            for d in (2, 3) for n in (200, 400)
        ],
    }
    for family, options in grids.items():
        for label, model in options:
            cands.append((family, f"{label}, raw", make_pipeline(model)))
            cands.append((family, f"{label}, relative", LevelScaledRegressor(make_pipeline(model))))
    return cands


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def feature_columns(table):
    return [c for c in table.columns if c != TARGET and not c.startswith(EXCLUDED_PREFIXES)]


def load_dataset(path=DAILY_FILE):
    """Daily table restricted to rows where every feature exists (first 24 days have no full history)."""
    table = pd.read_parquet(path)
    cols = feature_columns(table)
    return table[cols + [TARGET]].dropna(), cols


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------
def run_experiment(data, features):
    train, test = chronological_split(data, TEST_DAYS)
    X_tr, y_tr = train[features], train[TARGET]
    X_te, y_te = test[features], test[TARGET]

    # 1) Cross-validate every candidate on the TRAINING period
    rows, estimators = [], {}
    for family, name, est in candidate_models():
        cv = time_series_cv(est, X_tr, y_tr, n_splits=CV_SPLITS, test_size=CV_DAYS)
        rows.append({"family": family, "model": name, "cv_MAE": cv["MAE"].mean(), "cv_MAE_std": cv["MAE"].std(),
                     "cv_RMSE": cv["RMSE"].mean(), "cv_WAPE": cv["WAPE"].mean()})
        estimators[name] = est
    cv_table = pd.DataFrame(rows).sort_values("cv_MAE").reset_index(drop=True)

    # 2) Best configuration of each family (by CV only)
    finalists = cv_table.loc[cv_table.groupby("family")["cv_MAE"].idxmin()].sort_values("cv_MAE")

    # 3) Champion = lowest CV MAE overall; if no ML model beats the best baseline, the baseline wins
    champion_name = finalists.iloc[0]["model"]

    # 4) Score finalists + all baselines ONCE on the test period
    to_test = list(dict.fromkeys(list(finalists["model"]) +
                                 list(cv_table.loc[cv_table["family"] == "Baseline", "model"])))
    test_rows, test_preds = [], {}
    for name in to_test:
        fitted = clone(estimators[name]).fit(X_tr, y_tr)
        pred = fitted.predict(X_te)
        test_preds[name] = pred
        family = cv_table.loc[cv_table["model"] == name, "family"].iloc[0]
        test_rows.append({"family": family, "model": name, **regression_metrics(y_te, pred)})
    test_table = pd.DataFrame(test_rows).sort_values("MAE").reset_index(drop=True)

    return {
        "train": train, "test": test, "cv_table": cv_table, "finalists": finalists,
        "test_table": test_table, "test_preds": test_preds, "champion_name": champion_name,
        "champion_estimator": estimators[champion_name],
    }


def champion_importance(result, features, n_repeats=20):
    """Permutation importance on the TEST set: how much does MAE worsen if a feature is shuffled?"""
    fitted = clone(result["champion_estimator"]).fit(result["train"][features], result["train"][TARGET])
    imp = permutation_importance(
        fitted, result["test"][features], result["test"][TARGET],
        scoring="neg_mean_absolute_error", n_repeats=n_repeats, random_state=SEED,
    )
    return pd.DataFrame({"feature": features, "mae_increase": imp.importances_mean, "std": imp.importances_std}) \
        .sort_values("mae_increase", ascending=False).reset_index(drop=True)


def save_champion(result, data, features, importance):
    """Refit the champion on ALL data (train + test) for production and save it with its metadata."""
    final = clone(result["champion_estimator"]).fit(data[features], data[TARGET])
    test_metrics = result["test_table"].set_index("model").loc[result["champion_name"]].drop("family").to_dict()
    bundle = {
        "model": final,
        "model_family": result["finalists"].iloc[0]["family"],
        "model_name": result["champion_name"],
        "features": features,
        "target": TARGET,
        "trained_on": [str(data.index.min().date()), str(data.index.max().date())],
        "test_period": [str(result["test"].index.min().date()), str(result["test"].index.max().date())],
        "test_metrics": test_metrics,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_FILE)

    MODEL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {k: v for k, v in bundle.items() if k != "model"}
    report["cv_setup"] = f"TimeSeriesSplit(n_splits={CV_SPLITS}, test_size={CV_DAYS}) on the training period"
    report["finalists_cv"] = result["finalists"].to_dict(orient="records")
    report["test_comparison"] = result["test_table"].to_dict(orient="records")
    report["permutation_importance_top10"] = importance.head(10).to_dict(orient="records")
    (MODEL_REPORTS_DIR / "sales_forecasting.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    result["cv_table"].to_csv(MODEL_REPORTS_DIR / "sales_forecasting_cv_all_candidates.csv", index=False)

    families = result["test_table"].set_index("model")["family"]
    preds = pd.DataFrame({"actual": result["test"][TARGET]})
    for name, values in result["test_preds"].items():
        preds[name if families[name] == "Baseline" else f"{families[name]} ({name})"] = values
    preds.to_csv(MODEL_REPORTS_DIR / "sales_forecasting_test_predictions.csv")
    return bundle


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_results(result, importance):
    import matplotlib.pyplot as plt

    from src.utils import plotting as P

    P.apply_style()
    test = result["test"]
    champ = result["champion_name"]
    baseline = "Mean of same weekday, last 4 weeks"

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(test.index, test[TARGET], color=P.INK, linewidth=2, label="Actual")
    family = result["test_table"].set_index("model").loc[champ, "family"]
    ax.plot(test.index, result["test_preds"][champ], color=P.SERIES[0], linewidth=2,
            label=f"Champion: {family} ({champ})")
    ax.plot(test.index, result["test_preds"][baseline], color=P.SERIES[1], linewidth=1.5, linestyle="--",
            label=f"Baseline: {baseline}")
    ax.yaxis.set_major_formatter(P.money())
    ax.set_title("Test period: actual vs forecast daily net sales")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    P.save(fig, "07a_01_test_forecast.png")

    t = result["test_table"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = [P.MUTED if f == "Baseline" else P.SERIES[0] for f in t["family"]]
    ax.barh([f"{f}: {m}" if f != "Baseline" else m for f, m in zip(t["family"], t["model"])], t["MAE"],
            height=0.6, color=colors)
    for y, v in enumerate(t["MAE"]):
        ax.text(v, y, f"  £{v:,.0f}", va="center", color=P.INK_SECONDARY, fontsize=9)
    ax.xaxis.set_major_formatter(P.money())
    ax.set_title("Test MAE by model (lower is better; grey = baselines)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    P.save(fig, "07a_02_model_comparison.png")

    top = importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(top["feature"], top["mae_increase"], xerr=top["std"], height=0.6, color=P.SERIES[0],
            error_kw={"ecolor": P.MUTED, "elinewidth": 1})
    ax.xaxis.set_major_formatter(P.money())
    ax.set_title("Which features matter? MAE increase when shuffled (test set)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    P.save(fig, "07a_03_feature_importance.png")


# --------------------------------------------------------------------------
def main():
    pd.set_option("display.width", 170)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

    data, features = load_dataset()
    print(f"Rows with full history: {len(data)} trading days ({data.index.min().date()} -> {data.index.max().date()})")
    print(f"Features ({len(features)}): {features}")
    print(f"Test period: last {TEST_DAYS} trading days. Candidates are chosen by CV on the training period only.")
    print("Running time-series cross-validation for all candidates (about a minute)...\n")

    result = run_experiment(data, features)
    tr, te = result["train"], result["test"]
    print(f"Train: {tr.index.min().date()} -> {tr.index.max().date()} ({len(tr)} days)")
    print(f"Test : {te.index.min().date()} -> {te.index.max().date()} ({len(te)} days)\n")

    print("Cross-validation, top 10 of all candidates (training period only):")
    print(result["cv_table"].head(10).to_string(index=False))
    print("\nBest configuration per family (CV):")
    print(result["finalists"].to_string(index=False))
    print(f"\nChampion chosen by CV: {result['champion_name']}")

    print("\nTEST SET - scored once:")
    print(result["test_table"].to_string(index=False))

    champ = result["test_table"].set_index("model").loc[result["champion_name"]]
    best_base = result["test_table"][result["test_table"]["family"] == "Baseline"].iloc[0]
    gain = (1 - champ["MAE"] / best_base["MAE"]) * 100
    print(f"\nChampion test MAE £{champ['MAE']:,.0f} vs best baseline ({best_base['model']}) £{best_base['MAE']:,.0f}"
          f" -> {gain:+.1f}% improvement")
    print(f"Champion WAPE {champ['WAPE']:.1f}% | MAPE {champ['MAPE']:.1f}% (MAPE is inflated by short Sundays)")

    importance = champion_importance(result, features)
    print("\nPermutation importance (champion, test set) - MAE increase in £ when a feature is shuffled:")
    print(importance.head(10).to_string(index=False))

    bundle = save_champion(result, data, features, importance)
    plot_results(result, importance)
    print(f"\nSaved model  -> {MODEL_FILE}  (refit on {bundle['trained_on'][0]} .. {bundle['trained_on'][1]})")
    print(f"Saved report -> {MODEL_REPORTS_DIR / 'sales_forecasting.json'}")
    print("Saved charts -> reports/figures/07a_01..03_*.png")


if __name__ == "__main__":
    # Run through the importable module, so saved models reference
    # src.models.sales_forecasting.<class> instead of __main__.<class> (otherwise only this script could load them).
    from src.models.sales_forecasting import main as _main

    _main()

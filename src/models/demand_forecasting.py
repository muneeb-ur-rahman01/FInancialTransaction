"""
Module 7B - Product Demand Prediction
=====================================
Goal   : predict next week's units sold for every product.
Input  : data/processed/features_product_weekly.parquet   (Module 6)
Output : artifacts/models/demand_forecaster.joblib
         reports/model_reports/demand_forecasting.json + CV table CSV
         reports/figures/07b_*.png

Scientific procedure
  1. Chronological split on WEEKS: the final 8 weeks (17 Oct - 9 Dec 2011) are the test set.
  2. Baselines: last week, mean of last 4 weeks, the product's long-run mean.
  3. Candidates suited to COUNT data on wildly different scales (1 to 5,000 units/week):
       Ridge          - with log-transformed count features (a log-target Ridge on raw
                        lags extrapolates to absurd values)
       Random Forest
       XGBoost        - squared error, log target, Poisson (counts), absolute error (MAE)
       Ratio variants - predict units / (4-week mean + 1): puts all products on one scale
  4. Week-based expanding CV on the training weeks (4 folds x 4 weeks).
  5. Champion rule: an ML model replaces the best baseline only if its CV score is better
     by MORE than the baseline's own fold-to-fold standard deviation. Otherwise the gain is
     indistinguishable from noise and the simpler model ships.

Metric: score = WAPE + ProductBias%  (lower is better)
  WAPE          total absolute error / total units (MAPE is undefined: 23% of weeks are 0).
  ProductBias%  sum over products of |that product's total error| / total units.
                WAPE alone rewards predicting the median (the WAPE-best model under-forecast
                total units by 33%). Total bias alone lets errors cancel out: a model can
                look unbiased overall while capping best sellers at a fraction of their sales.
                Product-level bias catches both, and is always >= |total bias|.

Cold start: rows need 4 weeks of product history. Brand-new products are not forecast here.

Run from the project root:
    python -m src.models.demand_forecasting
"""

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBRegressor

from src.data_processing.features import PRODUCT_WEEKLY_FILE
from src.models.estimators import ColumnBaseline, NonNegativeRegressor, RatioToLevelRegressor
from src.models.evaluation import grouped_time_series_cv, regression_metrics, series_bias_pct
from src.utils.paths import MODEL_REPORTS_DIR, MODELS_DIR

TARGET = "units_sold"
ID_COLS = ["StockCode", "week", "product"]
TEST_WEEKS = 8
CV_SPLITS = 4
CV_WEEKS = 4
MODEL_FILE = MODELS_DIR / "demand_forecaster.joblib"
SEED = 42
COUNT_PREFIXES = ("units_", "orders_", "customers_", "price_")   # skewed, >= 0 -> log1p for Ridge

# Business decision (project owner, 7B review): when baselines are statistically tied, prefer the
# 4-week mean. "Last week" repeats every one-off spike a week later, which means costly re-orders;
# the 4-week mean smooths them out and has better week-to-week accuracy (lower WAPE).
PREFERRED_BASELINE = "Mean of last 4 weeks"


def planning_score(wape, product_bias_pct):
    """Accuracy + product-level bias, both in %. Lower is better."""
    return wape + np.abs(product_bias_pct)


def feature_columns(table):
    return [c for c in table.columns if c not in ID_COLS + [TARGET]]


def load_dataset(path=PRODUCT_WEEKLY_FILE):
    table = pd.read_parquet(path)
    cols = feature_columns(table)
    data = table.dropna(subset=cols).sort_values(["week", "StockCode"]).reset_index(drop=True)
    return data, cols


def split_by_weeks(data, test_weeks=TEST_WEEKS):
    weeks = np.sort(data["week"].unique())
    is_test = data["week"].isin(weeks[-test_weeks:])
    return data[~is_test], data[is_test]


def demand_metrics(frame, pred):
    m = regression_metrics(frame[TARGET], pred)
    m.pop("MAPE")
    m["ProductBias_pct"] = series_bias_pct(frame[TARGET], pred, frame["StockCode"])
    m["score"] = float(planning_score(m["WAPE"], m["ProductBias_pct"]))
    return m


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------
def log_count_ridge(features, alpha):
    count_cols = [c for c in features if c.startswith(COUNT_PREFIXES)]
    prep = ColumnTransformer(
        [("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), count_cols)],
        remainder="passthrough", verbose_feature_names_out=False,
    )
    return Pipeline([("log", prep), ("scale", StandardScaler()), ("model", Ridge(alpha=alpha))])


def log_target(model):
    """Train on log1p(y), predict with expm1. Approximates the MEDIAN -> runs low on totals."""
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)


def xgb(objective="reg:squarederror", depth=6):
    return XGBRegressor(objective=objective, n_estimators=400, max_depth=depth, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=5, random_state=SEED, n_jobs=4)


def rf(leaf):
    # n_jobs=1: the installed joblib/scikit-learn combination floods warnings when parallel
    return RandomForestRegressor(n_estimators=100, min_samples_leaf=leaf, max_features=0.5, max_samples=0.3,
                                 random_state=SEED, n_jobs=1)


def candidate_models(features):
    cands = [
        ("Baseline", "Last week", ColumnBaseline("units_lag_1")),
        ("Baseline", "Mean of last 4 weeks", ColumnBaseline("units_roll_mean_4")),
        ("Baseline", "Product long-run mean", ColumnBaseline("units_expanding_mean")),
        ("Ridge", "raw target", NonNegativeRegressor(log_count_ridge(features, 1))),
        ("Ridge", "log target", NonNegativeRegressor(log_target(log_count_ridge(features, 1)))),
    ]
    for leaf in (20, 50):
        cands.append(("Random Forest", f"leaf={leaf}", NonNegativeRegressor(rf(leaf))))
    cands.append(("Random Forest", "leaf=20, ratio target", RatioToLevelRegressor(rf(20))))
    for depth in (4, 6):
        cands += [
            ("XGBoost", f"depth={depth}, squared error", NonNegativeRegressor(xgb(depth=depth))),
            ("XGBoost", f"depth={depth}, log target", NonNegativeRegressor(log_target(xgb(depth=depth)))),
            ("XGBoost", f"depth={depth}, Poisson", NonNegativeRegressor(xgb("count:poisson", depth))),
            ("XGBoost", f"depth={depth}, absolute error", NonNegativeRegressor(xgb("reg:absoluteerror", depth))),
        ]
    for obj, name in (("reg:squarederror", "squared error"), ("count:poisson", "Poisson"),
                      ("reg:absoluteerror", "absolute error")):
        cands.append(("XGBoost", f"depth=4, {name}, ratio target", RatioToLevelRegressor(xgb(obj, 4))))
    return cands


def choose_champion(cv_table, preferred_baseline=PREFERRED_BASELINE):
    """Pick the champion from the CV results.

    1. The best ML model wins only if it beats the best baseline by more than the
       baseline's fold-to-fold standard deviation (otherwise the gain is noise).
    2. If a baseline is kept and `preferred_baseline` is statistically tied with the
       CV-best baseline (within that std), the preferred one is used (business rule).
    Returns (champion, best_baseline, best_ml, decision_text).
    """
    baselines = cv_table[cv_table["family"] == "Baseline"].sort_values("cv_score")
    base = baselines.iloc[0]
    ml = cv_table[cv_table["family"] != "Baseline"].sort_values("cv_score").iloc[0]
    margin = base["cv_score"] - ml["cv_score"]
    if margin > base["cv_score_std"]:
        decision = (f"ML wins: {ml['family']} ({ml['model']}) beats the best baseline by {margin:.1f} points, "
                    f"more than the baseline's fold-to-fold std of {base['cv_score_std']:.1f}.")
        return ml["model"], base["model"], ml["model"], decision

    decision = (f"Baseline kept: the best ML model, {ml['family']} ({ml['model']}), differs by only "
                f"{margin:+.1f} points, within the baseline's fold-to-fold std of {base['cv_score_std']:.1f} "
                f"- not a convincing improvement.")
    champion = base["model"]
    if preferred_baseline and preferred_baseline in set(baselines["model"]) and preferred_baseline != champion:
        pref = baselines.set_index("model").loc[preferred_baseline]
        gap = pref["cv_score"] - base["cv_score"]
        if gap <= base["cv_score_std"]:
            champion = preferred_baseline
            decision += (f" Business rule: '{preferred_baseline}' is statistically tied with '{base['model']}' "
                         f"(gap {gap:.1f} <= std {base['cv_score_std']:.1f}) and is preferred because it smooths "
                         f"one-off spikes instead of repeating them.")
    return champion, base["model"], ml["model"], decision


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------
def run_experiment(data, features, verbose=True):
    train, test = split_by_weeks(data)
    X_tr, y_tr = train[features], train[TARGET]

    rows, estimators, families = [], {}, {}
    cands = candidate_models(features)
    for i, (family, name, est) in enumerate(cands, start=1):
        cv = grouped_time_series_cv(est, X_tr, y_tr, train["week"].to_numpy(), n_splits=CV_SPLITS,
                                    periods_per_fold=CV_WEEKS, series_ids=train["StockCode"].to_numpy())
        fold_scores = planning_score(cv["WAPE"], cv["SeriesBias_pct"])
        rows.append({"family": family, "model": name, "cv_score": fold_scores.mean(),
                     "cv_score_std": fold_scores.std(), "cv_WAPE": cv["WAPE"].mean(),
                     "cv_ProductBias_pct": cv["SeriesBias_pct"].mean(), "cv_Bias_pct": cv["Bias_pct"].mean()})
        estimators[name], families[name] = est, family
        if verbose:
            r = rows[-1]
            print(f"  [{i:2d}/{len(cands)}] {family:<13} {name:<36} WAPE {r['cv_WAPE']:5.1f}%  "
                  f"product bias {r['cv_ProductBias_pct']:5.1f}%  total bias {r['cv_Bias_pct']:+6.1f}%  "
                  f"-> score {r['cv_score']:5.1f}")
    cv_table = pd.DataFrame(rows).sort_values("cv_score").reset_index(drop=True)
    finalists = cv_table.loc[cv_table.groupby("family")["cv_score"].idxmin()].sort_values("cv_score")
    champion, best_baseline, best_ml, decision = choose_champion(cv_table)

    to_test = list(dict.fromkeys(list(finalists["model"]) +
                                 list(cv_table.loc[cv_table["family"] == "Baseline", "model"])))
    test_rows, preds = [], {}
    for name in to_test:
        preds[name] = clone(estimators[name]).fit(X_tr, y_tr).predict(test[features])
        test_rows.append({"family": families[name], "model": name, **demand_metrics(test, preds[name])})
    test_table = pd.DataFrame(test_rows).sort_values("score").reset_index(drop=True)

    return {"train": train, "test": test, "cv_table": cv_table, "finalists": finalists,
            "test_table": test_table, "test_preds": preds, "champion_name": champion,
            "best_baseline": best_baseline, "best_ml": best_ml, "decision": decision,
            "families": families, "estimators": estimators, "champion_estimator": estimators[champion]}


def label(result, name):
    family = result["families"][name]
    return name if family == "Baseline" else f"{family} ({name})"


def per_product_view(result):
    """Planner's view: champion vs the strongest other option, per product and per week."""
    champ = result["champion_name"]
    challenger = result["best_ml"] if champ == result["best_baseline"] else result["best_baseline"]
    test = result["test"].assign(champion=result["test_preds"][champ],
                                 challenger=result["test_preds"][challenger])
    test["err_champion"] = (test["champion"] - test[TARGET]).abs()
    test["err_challenger"] = (test["challenger"] - test[TARGET]).abs()
    by_product = test.groupby("StockCode").agg(
        product=("product", "first"), units=(TARGET, "sum"), forecast=("champion", "sum"),
        err_champion=("err_champion", "sum"), err_challenger=("err_challenger", "sum"))
    by_product["bias_pct"] = (by_product["forecast"] / by_product["units"].replace(0, np.nan) - 1) * 100
    by_product["champion_better"] = by_product["err_champion"] < by_product["err_challenger"]
    weekly = test.groupby("week")[[TARGET, "champion", "challenger"]].sum()
    return test, by_product.sort_values("units", ascending=False), weekly, challenger


def model_importance(result, name, features, n_repeats=5):
    fitted = clone(result["estimators"][name]).fit(result["train"][features], result["train"][TARGET])
    imp = permutation_importance(fitted, result["test"][features], result["test"][TARGET],
                                 scoring="neg_mean_absolute_error", n_repeats=n_repeats, random_state=SEED)
    return pd.DataFrame({"feature": features, "mae_increase": imp.importances_mean, "std": imp.importances_std}) \
        .sort_values("mae_increase", ascending=False).reset_index(drop=True)


def save_champion(result, data, features, importance):
    final = clone(result["champion_estimator"]).fit(data[features], data[TARGET])
    champ_row = result["test_table"].set_index("model").loc[result["champion_name"]]
    bundle = {
        "model": final,
        "model_family": champ_row["family"],
        "model_name": result["champion_name"],
        "selection_decision": result["decision"],
        "features": features,
        "target": TARGET,
        "min_history_weeks": 4,
        "trained_on_weeks": [str(data["week"].min().date()), str(data["week"].max().date())],
        "test_weeks": [str(result["test"]["week"].min().date()), str(result["test"]["week"].max().date())],
        "test_metrics": champ_row.drop("family").to_dict(),
        "products": int(data["StockCode"].nunique()),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_FILE)

    MODEL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {k: v for k, v in bundle.items() if k != "model"}
    report["metric"] = "score = WAPE + ProductBias% (lower is better)"
    report["champion_label"] = label(result, result["champion_name"])
    report["cv_setup"] = f"week-based expanding CV: {CV_SPLITS} folds x {CV_WEEKS} weeks on training weeks"
    report["finalists_cv"] = result["finalists"].to_dict(orient="records")
    report["test_comparison"] = result["test_table"].to_dict(orient="records")
    report["best_ml_model"] = label(result, result["best_ml"])
    report["best_ml_permutation_importance_top10"] = importance.head(10).to_dict(orient="records")
    (MODEL_REPORTS_DIR / "demand_forecasting.json").write_text(json.dumps(report, indent=2, default=str),
                                                                 encoding="utf-8")
    result["cv_table"].to_csv(MODEL_REPORTS_DIR / "demand_forecasting_cv_all_candidates.csv", index=False)

    preds = result["test"][["StockCode", "product", "week", TARGET]].copy()
    for name, values in result["test_preds"].items():
        preds[label(result, name)] = values
    preds.to_parquet(MODEL_REPORTS_DIR / "demand_forecasting_test_predictions.parquet", index=False)
    return bundle


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_results(result, test, by_product, weekly, challenger, importance):
    import matplotlib.pyplot as plt

    from src.utils import plotting as P

    P.apply_style()
    champ_label = f"Champion: {label(result, result['champion_name'])}"
    chall_label = f"Challenger: {label(result, challenger)}"

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(weekly.index, weekly[TARGET], color=P.INK, marker="o", markersize=5, label="Actual")
    ax.plot(weekly.index, weekly["champion"], color=P.SERIES[0], marker="o", markersize=5, label=champ_label)
    ax.plot(weekly.index, weekly["challenger"], color=P.SERIES[1], linestyle="--", label=chall_label)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e3:g}K"))
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Units, all modelled products")
    ax.set_title("Test weeks: total units sold, actual vs forecast")
    ax.legend(loc="lower left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    P.save(fig, "07b_01_weekly_total_demand.png")

    t = result["test_table"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    colors = [P.MUTED if f == "Baseline" else P.SERIES[0] for f in t["family"]]
    ax.barh([label(result, m) for m in t["model"]], t["score"], height=0.6, color=colors)
    for y, (sc, w, b) in enumerate(zip(t["score"], t["WAPE"], t["ProductBias_pct"])):
        ax.text(sc, y, f"  {sc:.1f}  (WAPE {w:.1f}% + product bias {b:.1f}%)", va="center",
                color=P.INK_SECONDARY, fontsize=9)
    ax.set_xlim(right=t["score"].max() * 1.6)
    ax.set_title("Test score = WAPE + product bias %  (lower is better; grey = baselines)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    P.save(fig, "07b_02_model_comparison.png")

    top = by_product.head(4).index
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True)
    for ax, code in zip(axes.flat, top):
        s = test[test["StockCode"] == code]
        ax.plot(s["week"], s[TARGET], color=P.INK, marker="o", markersize=4, label="Actual")
        ax.plot(s["week"], s["champion"], color=P.SERIES[0], marker="o", markersize=4, label="Champion")
        ax.plot(s["week"], s["challenger"], color=P.SERIES[1], linestyle="--", label="Challenger")
        ax.set_title(f"{by_product.loc[code, 'product']} ({code})", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelrotation=30)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Best sellers in the test weeks.  {champ_label}  |  {chall_label}", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    P.save(fig, "07b_03_top_products.png")

    imp = importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(imp["feature"], imp["mae_increase"], xerr=imp["std"], height=0.6, color=P.SERIES[0],
            error_kw={"ecolor": P.MUTED, "elinewidth": 1})
    ax.set_xlabel("MAE increase (units) when the feature is shuffled")
    ax.set_title(f"Feature importance - best ML model: {label(result, result['best_ml'])}")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    P.save(fig, "07b_04_feature_importance.png")


# --------------------------------------------------------------------------
def main():
    pd.set_option("display.width", 190)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

    data, features = load_dataset()
    train, test = split_by_weeks(data)
    print(f"Product-weeks with 4+ weeks of history: {len(data):,} ({data['StockCode'].nunique():,} products)")
    print(f"Train weeks: {train['week'].min().date()} -> {train['week'].max().date()} ({len(train):,} rows)")
    print(f"Test weeks : {test['week'].min().date()} -> {test['week'].max().date()} ({len(test):,} rows)")
    print(f"Features ({len(features)}): {features}")
    print(f"Zero-demand rows: {(data[TARGET] == 0).mean():.1%}  |  test mean {test[TARGET].mean():.1f}, "
          f"median {test[TARGET].median():.0f}, max {test[TARGET].max():,.0f} units")
    print("\nWeek-based cross-validation of all candidates (3-5 minutes)...")

    result = run_experiment(data, features)
    print("\nBest configuration per family (CV):")
    print(result["finalists"].to_string(index=False))
    print(f"\nSELECTION: {result['decision']}")

    print("\nTEST WEEKS - scored once:")
    print(result["test_table"].to_string(index=False))

    test_view, by_product, weekly, challenger = per_product_view(result)
    print(f"\nTotal units over the test weeks: actual {weekly[TARGET].sum():,.0f} | champion "
          f"{weekly['champion'].sum():,.0f} | challenger, {label(result, challenger)}: {weekly['challenger'].sum():,.0f}")
    print(f"Products where the champion has lower error than the challenger: {by_product['champion_better'].mean():.1%}")
    print("\nTop 10 products by test volume (champion forecast):")
    print(by_product.head(10)[["product", "units", "forecast", "bias_pct", "champion_better"]].to_string())

    importance = model_importance(result, result["best_ml"], features)
    print(f"\nPermutation importance of the best ML model, {label(result, result['best_ml'])}:")
    print(importance.head(10).to_string(index=False))

    bundle = save_champion(result, data, features, importance)
    plot_results(result, test_view, by_product, weekly, challenger, importance)
    print(f"\nSaved model  -> {MODEL_FILE}  ({bundle['model_family']}: {bundle['model_name']})")
    print(f"Saved report -> {MODEL_REPORTS_DIR / 'demand_forecasting.json'}")
    print("Saved charts -> reports/figures/07b_01..04_*.png")


if __name__ == "__main__":
    # Run through the importable module, so saved models reference
    # src.models.demand_forecasting.<class> instead of __main__.<class> (otherwise only this script could load them).
    from src.models.demand_forecasting import main as _main

    _main()

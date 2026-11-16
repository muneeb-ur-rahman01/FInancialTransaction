"""
Module 7C - Customer Segmentation (RFM + K-Means)
=================================================
Goal   : group customers by behaviour so each group can get a different business action.
Input  : data/processed/features_customers.parquet         (Module 6, snapshot 10 Dec 2011)
Output : artifacts/models/customer_segmentation.joblib      (preprocessing + K-Means, one pipeline)
         artifacts/preprocessors/rfm_preprocessor.joblib     (the fitted transform on its own)
         data/processed/customer_segments.parquet            (one row per customer + segment)
         reports/model_reports/customer_segmentation.json + k-selection CSV
         reports/figures/07c_*.png

Why these steps
  * Features: Recency, Frequency, Monetary. All three are heavily right-skewed (Module 5), and
    K-Means uses squared Euclidean distance, so a few huge customers would dominate. We take
    log1p (monetary clipped at 0 first: 30-odd customers returned more than they bought) and then
    standardise, so each dimension contributes equally.
  * Choosing k (no labels exist, so we judge structure):
      silhouette   - how well separated the clusters are (-1..1, higher = better)
      Davies-Bouldin / Calinski-Harabasz - two more separation indices, for confirmation
      stability    - refit on bootstrap resamples, compare with the Adjusted Rand Index (ARI);
                     segments that change when the data is resampled are not real
    Rule: the MOST DETAILED k >= 3 that is stable (mean ARI >= 0.90) and whose silhouette is
    within 0.02 of the best k >= 3. k = 2 is excluded: "active vs inactive" is too coarse to act on.
  * Baseline first: the rule-based RFM segments (fixed score thresholds) are scored in the same
    space; K-Means must separate customers better than hand-written rules.
  * Algorithm comparison at the chosen k: K-Means vs Gaussian Mixture vs Ward hierarchical.

Run from the project root:
    python -m src.models.customer_segmentation
"""

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.data_processing.features import CUSTOMER_FILE
from src.utils.paths import MODEL_REPORTS_DIR, MODELS_DIR, PREPROCESSORS_DIR, PROCESSED_DIR

RFM_COLS = ["recency_days", "frequency", "monetary"]
K_RANGE = range(2, 9)
MIN_K = 3
MIN_STABILITY = 0.90
SILHOUETTE_TOLERANCE = 0.02
N_BOOTSTRAP = 20
SEED = 42

# Business overlay on top of the clusters: K-Means assigns by overall distance, so a customer who
# bought often but long ago can still sit nearest the "regulars" centre. Valuable customers with
# no order for this many days are flagged as lapsing -> a ready win-back list.
LAPSING_DAYS = 90
VALUABLE_SEGMENTS = ("Champions", "Loyal regulars")

MODEL_FILE = MODELS_DIR / "customer_segmentation.joblib"
PREPROCESSOR_FILE = PREPROCESSORS_DIR / "rfm_preprocessor.joblib"
SEGMENTS_FILE = PROCESSED_DIR / "customer_segments.parquet"

# Fixed display order (and colour order) of segment names everywhere in the project
SEGMENT_ORDER = ["Champions", "Loyal regulars", "New customers", "At risk", "Hibernating"]
ACTIONS = {
    "Champions": "Reward and retain: early access, referral programme, dedicated account contact.",
    "Loyal regulars": "Grow: cross-sell and upsell, loyalty pricing, nudge towards Champion behaviour.",
    "New customers": "Onboard: welcome series and a second-order incentive while interest is fresh.",
    "At risk": "Win back now: they used to buy often but have gone quiet - personal outreach.",
    "Hibernating": "Low-cost reactivation only (seasonal campaign); do not spend heavily.",
}


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def log_rfm(X):
    """log1p of each RFM column; monetary is clipped at 0 (net refunders count as 0 spend)."""
    X = pd.DataFrame(X, columns=RFM_COLS) if not isinstance(X, pd.DataFrame) else X[RFM_COLS]
    return np.log1p(X.clip(lower=0).to_numpy(dtype=float))


def make_preprocessor():
    return Pipeline([("log", FunctionTransformer(log_rfm)), ("scale", StandardScaler())])


def make_model(k):
    return Pipeline([("prep", make_preprocessor()),
                     ("kmeans", KMeans(n_clusters=k, n_init=50, random_state=SEED))])


# --------------------------------------------------------------------------
# Baseline: rule-based RFM segments (also used by the dashboard)
# --------------------------------------------------------------------------
RULE_SEGMENTS = ["Champions", "Loyal", "New / promising", "Need attention", "At risk", "Hibernating"]


def rule_based_segments(frame):
    """Classic RFM rules on 1-5 quintile scores (5 = best). The first matching rule wins."""
    r, f = frame["R_score"], frame["F_score"]
    rules = [(r >= 4) & (f >= 4), (r >= 3) & (f >= 3), (r >= 4) & (f <= 2), (r == 3), (r <= 2) & (f >= 3)]
    return pd.Series(np.select(rules, RULE_SEGMENTS[:5], default="Hibernating"), index=frame.index)


# --------------------------------------------------------------------------
# Choosing k
# --------------------------------------------------------------------------
def bootstrap_stability(Z, k, n_boot=N_BOOTSTRAP, seed=SEED):
    """Mean/std ARI between the full-data clustering and clusterings fitted on bootstrap samples."""
    rng = np.random.default_rng(seed)
    reference = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
    scores = []
    for b in range(n_boot):
        idx = rng.choice(len(Z), len(Z), replace=True)
        boot = KMeans(n_clusters=k, n_init=10, random_state=seed + b + 1).fit(Z[idx])
        scores.append(adjusted_rand_score(reference, boot.predict(Z)))
    return float(np.mean(scores)), float(np.std(scores))


def evaluate_k(Z, k_range=K_RANGE, n_boot=N_BOOTSTRAP):
    rows = []
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Z)
        stab, stab_sd = bootstrap_stability(Z, k, n_boot)
        rows.append({"k": k, "inertia": labels.inertia_,
                     "silhouette": silhouette_score(Z, labels.labels_),
                     "davies_bouldin": davies_bouldin_score(Z, labels.labels_),
                     "calinski_harabasz": calinski_harabasz_score(Z, labels.labels_),
                     "stability_ari": stab, "stability_ari_std": stab_sd,
                     "smallest_cluster_share": np.bincount(labels.labels_).min() / len(Z)})
    return pd.DataFrame(rows)


def choose_k(table, min_k=MIN_K, min_stability=MIN_STABILITY, tolerance=SILHOUETTE_TOLERANCE):
    eligible = table[table["k"] >= min_k]
    best_sil = eligible["silhouette"].max()
    ok = eligible[(eligible["stability_ari"] >= min_stability) & (eligible["silhouette"] >= best_sil - tolerance)]
    if ok.empty:
        k = int(eligible.sort_values("stability_ari", ascending=False).iloc[0]["k"])
        return k, f"No k met every criterion; chose the most stable k = {k}."
    k = int(ok["k"].max())
    row = ok.set_index("k").loc[k]
    return k, (f"k = {k}: the most detailed segmentation that is stable (ARI {row['stability_ari']:.2f} >= "
               f"{min_stability}) and as well separated as the best option (silhouette {row['silhouette']:.3f}, "
               f"best for k >= {min_k}: {best_sil:.3f}, tolerance {tolerance}).")


def compare_algorithms(Z, k, baseline_labels):
    """Silhouette / Davies-Bouldin of the rule-based baseline and three clustering algorithms."""
    candidates = {
        "Rule-based RFM segments (baseline)": baseline_labels,
        "K-Means": KMeans(n_clusters=k, n_init=50, random_state=SEED).fit_predict(Z),
        "Gaussian Mixture": GaussianMixture(n_components=k, n_init=5, random_state=SEED).fit_predict(Z),
        "Ward hierarchical": AgglomerativeClustering(n_clusters=k).fit_predict(Z),
    }
    rows = []
    for name, labels in candidates.items():
        labels = pd.factorize(np.asarray(labels))[0]
        rows.append({"method": name, "clusters": int(len(np.unique(labels))),
                     "silhouette": silhouette_score(Z, labels), "davies_bouldin": davies_bouldin_score(Z, labels)})
    return pd.DataFrame(rows).sort_values("silhouette", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Naming the clusters
# --------------------------------------------------------------------------
def name_clusters(centroids):
    """Turn standardised centroids into business names.

    centroids: DataFrame indexed by cluster with columns R, F, M in standardised log space,
               where R = -recency (so higher always means better).
    Highest overall value -> Champions; lowest -> Hibernating; the rest by their shape:
      clearly lapsed (R < -0.5) but used to buy often (F > 0) -> At risk
      recent (R > 0) but few orders (F < 0)                    -> New customers
      otherwise                                                -> Loyal regulars
    Duplicate names get a spend qualifier.
    """
    value = centroids[["R", "F", "M"]].sum(axis=1)
    names = {}
    names[value.idxmax()] = "Champions"
    names[value.idxmin()] = "Hibernating"
    for c in centroids.index:
        if c in names:
            continue
        r, f = centroids.loc[c, "R"], centroids.loc[c, "F"]
        if r < -0.5 and f > 0:
            names[c] = "At risk"
        elif r > 0 and f < 0:
            names[c] = "New customers"
        else:
            names[c] = "Loyal regulars"
    counts = pd.Series(names).value_counts()
    for dup in counts[counts > 1].index:
        members = [c for c, n in names.items() if n == dup]
        for c in sorted(members, key=lambda c: centroids.loc[c, "M"], reverse=True):
            rank = sorted(members, key=lambda m: centroids.loc[m, "M"], reverse=True).index(c)
            names[c] = f"{dup} ({'higher' if rank == 0 else 'lower'} spend)" if len(members) == 2 else f"{dup} {rank + 1}"
    return names


def add_lapsing_flag(customers, days=LAPSING_DAYS):
    """True for valuable customers (Champions / Loyal regulars) with no order for `days` or more."""
    base = customers["segment"].str.split(" (", regex=False).str[0]
    return base.isin(VALUABLE_SEGMENTS) & (customers["recency_days"] >= days)


def segment_profiles(customers):
    total_rev = customers["monetary"].clip(lower=0).sum()
    prof = customers.groupby("segment").agg(
        customers=("frequency", "size"),
        median_recency_days=("recency_days", "median"),
        median_orders=("frequency", "median"),
        median_spend=("monetary", "median"),
        total_spend=("monetary", "sum"),
        median_order_value=("avg_order_value", "median"),
        return_rate=("return_rate", "mean"),
        uk_share=("is_uk", "mean"),
    )
    prof["lapsing"] = customers.groupby("segment")["lapsing"].sum()
    prof["customer_share"] = prof["customers"] / prof["customers"].sum()
    prof["revenue_share"] = prof["total_spend"].clip(lower=0) / total_rev
    order = [s for s in SEGMENT_ORDER if s in prof.index] + [s for s in prof.index if s not in SEGMENT_ORDER]
    prof = prof.loc[order]
    prof["action"] = [ACTIONS.get(s.split(" (")[0], "") for s in prof.index]
    return prof


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------
def run_experiment(customers, k_range=K_RANGE, n_boot=N_BOOTSTRAP):
    prep = make_preprocessor().fit(customers[RFM_COLS])
    Z = prep.transform(customers[RFM_COLS])

    k_table = evaluate_k(Z, k_range, n_boot)
    k, k_reason = choose_k(k_table)
    algorithms = compare_algorithms(Z, k, rule_based_segments(customers))

    model = make_model(k).fit(customers[RFM_COLS])
    labels = model.predict(customers[RFM_COLS])
    centers = pd.DataFrame(model.named_steps["kmeans"].cluster_centers_, columns=["R", "F", "M"])
    centers["R"] = -centers["R"]                       # high recency (days) is bad -> flip so higher = better
    names = name_clusters(centers)

    out = customers.copy()
    out["cluster"] = labels
    out["segment"] = out["cluster"].map(names)
    out["lapsing"] = add_lapsing_flag(out)
    return {"Z": Z, "k_table": k_table, "k": k, "k_reason": k_reason, "algorithms": algorithms,
            "model": model, "preprocessor": prep, "names": names, "centers": centers,
            "customers": out, "profiles": segment_profiles(out)}


def save(result):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSORS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cust = result["customers"]
    km_row = result["algorithms"].set_index("method").loc["K-Means"]
    k_row = result["k_table"].set_index("k").loc[result["k"]]
    bundle = {
        "model": result["model"],
        "k": result["k"],
        "cluster_names": {int(c): n for c, n in result["names"].items()},
        "features": RFM_COLS,
        "snapshot": str(pd.Timestamp(cust["snapshot"].iloc[0]).date()),
        "customers": int(len(cust)),
        "silhouette": float(km_row["silhouette"]),
        "stability_ari": float(k_row["stability_ari"]),
        "k_reason": result["k_reason"],
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, MODEL_FILE)
    joblib.dump(result["preprocessor"], PREPROCESSOR_FILE)

    keep = ["cluster", "segment", "lapsing", "recency_days", "frequency", "monetary", "avg_order_value", "return_rate",
            "country", "is_uk", "R_score", "F_score", "M_score", "first_purchase", "last_purchase"]
    cust[keep].rename_axis("CustomerID").reset_index().to_parquet(SEGMENTS_FILE, index=False)

    report = {k: v for k, v in bundle.items() if k != "model"}
    report["lapsing_rule"] = f"segment in {list(VALUABLE_SEGMENTS)} and no order for >= {LAPSING_DAYS} days"
    report["lapsing_customers"] = int(cust["lapsing"].sum())
    report["lapsing_value"] = float(cust.loc[cust["lapsing"], "monetary"].sum())
    report["selection_rule"] = (f"most detailed k >= {MIN_K} with bootstrap ARI >= {MIN_STABILITY} and silhouette "
                                f"within {SILHOUETTE_TOLERANCE} of the best k >= {MIN_K}")
    report["k_selection"] = result["k_table"].to_dict(orient="records")
    report["algorithm_comparison"] = result["algorithms"].to_dict(orient="records")
    report["profiles"] = result["profiles"].reset_index().to_dict(orient="records")
    report["standardised_centroids"] = result["centers"].assign(
        segment=result["centers"].index.map(result["names"])).to_dict(orient="records")
    (MODEL_REPORTS_DIR / "customer_segmentation.json").write_text(json.dumps(report, indent=2, default=str),
                                                                    encoding="utf-8")
    result["k_table"].to_csv(MODEL_REPORTS_DIR / "customer_segmentation_k_selection.csv", index=False)
    return bundle


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def segment_colors(names):
    from src.utils import plotting as P

    palette = P.SERIES + ["#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    ordered = [s for s in SEGMENT_ORDER if s in names] + sorted(s for s in names if s not in SEGMENT_ORDER)
    return {s: palette[i % len(palette)] for i, s in enumerate(ordered)}, ordered


def plot_results(result):
    import matplotlib.pyplot as plt

    from src.utils import plotting as P

    P.apply_style()
    t, k = result["k_table"], result["k"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, col, title in [(axes[0], "inertia", "Elbow: within-cluster spread"),
                           (axes[1], "silhouette", "Silhouette (higher = better separated)"),
                           (axes[2], "stability_ari", "Stability: bootstrap ARI")]:
        ax.plot(t["k"], t[col], color=P.SERIES[0], marker="o", markersize=7)
        if col == "stability_ari":
            ax.fill_between(t["k"], t[col] - t["stability_ari_std"], t[col] + t["stability_ari_std"],
                            color=P.BLUE_LIGHT, alpha=0.35, linewidth=0)
            ax.axhline(MIN_STABILITY, color=P.MUTED, linestyle="--", linewidth=1)
        ax.axvline(k, color=P.SERIES[1], linewidth=1.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Number of segments (k)")
        ax.set_xticks(list(t["k"]))
        ax.grid(axis="x", visible=False)
    axes[0].annotate(f"chosen k = {k}", (k, t.set_index("k").loc[k, "inertia"]), xytext=(8, 8),
                     textcoords="offset points", color=P.INK, fontsize=9)
    fig.tight_layout()
    P.save(fig, "07c_01_choosing_k.png")

    cust = result["customers"]
    colors, order = segment_colors(cust["segment"].unique())
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in order:
        part = cust[cust["segment"] == s]
        ax.scatter(part["recency_days"], part["monetary"].clip(lower=1), s=14, color=colors[s], alpha=0.6,
                   edgecolor=P.SURFACE, linewidth=0.4, label=f"{s} ({len(part):,})")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(P.money())
    ax.set_xlabel("Days since last order (lower = more recent)")
    ax.set_ylabel("Net spend (log scale)")
    ax.set_title("Customer segments: recency vs spend")
    ax.legend(loc="lower left", markerscale=2)
    fig.tight_layout()
    P.save(fig, "07c_02_segments_scatter.png")

    centers = result["centers"].assign(segment=result["centers"].index.map(result["names"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s in order:
        row = centers[centers["segment"] == s].iloc[0]
        ax.plot(["Recency\n(recent = high)", "Frequency", "Monetary"], [row["R"], row["F"], row["M"]],
                color=colors[s], marker="o", markersize=8, label=s)
    ax.axhline(0, color=P.MUTED, linewidth=1, linestyle="--")
    ax.set_ylabel("Standardised value (0 = average customer)")
    ax.set_title("Segment profiles (snake plot)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    P.save(fig, "07c_03_segment_profiles.png")

    prof = result["profiles"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    y = np.arange(len(prof))
    ax.barh(y + 0.18, prof["customer_share"], height=0.34, color=P.MUTED, label="Share of customers")
    ax.barh(y - 0.18, prof["revenue_share"], height=0.34, color=P.SERIES[0], label="Share of revenue")
    for i, (cs, rs) in enumerate(zip(prof["customer_share"], prof["revenue_share"])):
        ax.text(cs, i + 0.18, f"  {cs:.0%}", va="center", fontsize=9, color=P.INK_SECONDARY)
        ax.text(rs, i - 0.18, f"  {rs:.0%}", va="center", fontsize=9, color=P.INK_SECONDARY)
    ax.set_yticks(y, prof.index)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Customers vs revenue by segment")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    P.save(fig, "07c_04_customers_vs_revenue.png")


# --------------------------------------------------------------------------
def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}" if abs(v) < 10 else f"{v:,.0f}")

    customers = pd.read_parquet(CUSTOMER_FILE)
    print(f"Customers: {len(customers):,} (snapshot {pd.Timestamp(customers['snapshot'].iloc[0]).date()})")
    print(f"Features: {RFM_COLS} -> log1p -> standardised")
    print(f"Evaluating k = {K_RANGE.start}..{K_RANGE.stop - 1} with {N_BOOTSTRAP} bootstrap refits each (1-2 minutes)...\n")

    result = run_experiment(customers)
    print("Choosing k:")
    print(result["k_table"].to_string(index=False))
    print(f"\nDECISION: {result['k_reason']}")

    print("\nBaseline and algorithm comparison at the chosen k (same standardised RFM space):")
    print(result["algorithms"].to_string(index=False))

    print("\nSegment profiles:")
    prof = result["profiles"]
    print(prof.drop(columns="action").to_string())
    lap = result["customers"][result["customers"]["lapsing"]]
    print(f"\nLapsing valuable customers (Champions / Loyal regulars, no order for {LAPSING_DAYS}+ days): "
          f"{len(lap):,} customers, lifetime net spend £{lap['monetary'].sum():,.0f} -> win-back list")
    print("\nRecommended actions:")
    for seg, action in prof["action"].items():
        print(f"  {seg:<16} {action}")

    bundle = save(result)
    plot_results(result)
    print(f"\nSaved model        -> {MODEL_FILE}")
    print(f"Saved preprocessor -> {PREPROCESSOR_FILE}")
    print(f"Saved segments     -> {SEGMENTS_FILE} ({bundle['customers']:,} customers)")
    print(f"Saved report       -> {MODEL_REPORTS_DIR / 'customer_segmentation.json'}")
    print("Saved charts       -> reports/figures/07c_01..04_*.png")


if __name__ == "__main__":
    # Run through the importable module, so saved models reference
    # src.models.customer_segmentation.<class> instead of __main__.<class> (otherwise only this script could load them).
    from src.models.customer_segmentation import main as _main

    _main()

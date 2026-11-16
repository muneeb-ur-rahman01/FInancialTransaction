"""
Module 5 - Statistical Analysis: reusable statistical tools
===========================================================
Principles applied throughout:
  * Our data is heavily skewed (skew 17 for order value), so we default to
    NON-PARAMETRIC methods (medians, ranks, Mann-Whitney, Kruskal-Wallis,
    Spearman) that do not assume normality.
  * A p-value only says "is there an effect?". With 20,000 orders almost any
    difference is "significant", so every test also reports an EFFECT SIZE
    ("how big is it?") and, where useful, a bootstrap confidence interval.
  * Running many tests inflates false positives -> Holm correction for
    pairwise comparisons.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

from src.analysis import eda

ALPHA = 0.05


def verdict(p, alpha=ALPHA):
    return "significant" if p < alpha else "not significant"


# --------------------------------------------------------------------------
# Data at the right level for each question
# --------------------------------------------------------------------------
def daily_features(df):
    """One row per trading day: net sales, orders, customers, units, avg order value."""
    rev = eda.revenue_lines(df)
    sales = eda.sale_lines(df)
    day_rev = rev["InvoiceDate"].dt.normalize()
    day_sales = sales["InvoiceDate"].dt.normalize()

    out = pd.DataFrame({
        "net_sales": rev.groupby(day_rev)["TotalAmount"].sum(),
        "net_units": rev.groupby(day_rev)["Quantity"].sum(),
        "orders": sales.groupby(day_sales)["InvoiceNo"].nunique(),
        "customers": sales.groupby(day_sales)["CustomerID"].nunique(),
        "order_value_sum": sales.groupby(day_sales)["TotalAmount"].sum(),
    })
    out.index.name = "date"
    out = out.fillna({"orders": 0, "customers": 0, "order_value_sum": 0})
    out["avg_order_value"] = out["order_value_sum"] / out["orders"].replace(0, np.nan)
    out["weekday"] = out.index.day_name()
    out["month"] = out.index.to_period("M")
    return out.drop(columns="order_value_sum")


# --------------------------------------------------------------------------
# Descriptive statistics
# --------------------------------------------------------------------------
def describe_distribution(values):
    """Central tendency, spread and shape of one numeric variable."""
    s = pd.Series(values, dtype=float).dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    # A mode only exists if some value repeats; for continuous data every value is
    # unique and pandas would just return the smallest one, which is meaningless.
    counts = s.value_counts()
    mode = counts.index[0] if len(counts) and counts.iloc[0] > 1 else np.nan
    return pd.Series({
        "n": len(s),
        "mean": s.mean(),
        "median": s.median(),
        "mode": mode,
        "variance": s.var(ddof=1),
        "std": s.std(ddof=1),
        "cv": s.std(ddof=1) / s.mean() if s.mean() else np.nan,   # spread relative to the mean
        "min": s.min(),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "max": s.max(),
        "skewness": s.skew(),
        "excess_kurtosis": s.kurt(),        # 0 for a normal distribution
    })


def normality_report(values):
    """D'Agostino-Pearson K^2 test of normality, on the raw and the log scale.

    Shapiro-Wilk is limited to ~5,000 points, K^2 works for any n >= 20.
    Caveat: with very large n even tiny deviations are 'significant', so read
    the skewness/kurtosis (effect) next to the p-value.
    """
    s = pd.Series(values, dtype=float).dropna()
    rows = {}
    for label, data in [("raw", s), ("log", np.log(s[s > 0]))]:
        k2, p = stats.normaltest(data)
        rows[label] = {"n": len(data), "skewness": data.skew(), "excess_kurtosis": data.kurt(),
                       "K2": k2, "p_value": p, "normal_at_5pct": p >= ALPHA}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------------
def correlation_matrices(frame):
    """Pearson (linear, outlier-sensitive) and Spearman (rank, monotonic)."""
    num = frame.select_dtypes("number")
    return num.corr(method="pearson"), num.corr(method="spearman")


def correlation_test(x, y, method="spearman"):
    x, y = pd.Series(x, dtype=float), pd.Series(y, dtype=float)
    ok = x.notna() & y.notna()
    fn = stats.spearmanr if method == "spearman" else stats.pearsonr
    r, p = fn(x[ok], y[ok])
    return {"method": method, "n": int(ok.sum()), "r": float(r), "p_value": float(p)}


# --------------------------------------------------------------------------
# Hypothesis tests
# --------------------------------------------------------------------------
def bootstrap_median_diff(a, b, n_boot=2000, seed=42, ci=0.95):
    """Percentile bootstrap CI for median(a) - median(b)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = np.median(rng.choice(a, a.size)) - np.median(rng.choice(b, b.size))
    lo, hi = np.quantile(diffs, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def compare_two_groups(a, b, alternative="two-sided", bootstrap=True):
    """Mann-Whitney U test + rank-biserial effect size + bootstrap CI.

    rank_biserial in [-1, 1]: P(a > b) - P(a < b). |r| ~0.1 small, ~0.3 medium, ~0.5 large.
    """
    a = pd.Series(a, dtype=float).dropna().to_numpy()
    b = pd.Series(b, dtype=float).dropna().to_numpy()
    u, p = stats.mannwhitneyu(a, b, alternative=alternative)
    result = {
        "n_a": a.size, "n_b": b.size,
        "median_a": float(np.median(a)), "median_b": float(np.median(b)),
        "U": float(u), "p_value": float(p),
        "rank_biserial": float(2 * u / (a.size * b.size) - 1),
    }
    if bootstrap:
        result["median_diff_ci95"] = bootstrap_median_diff(a, b)
    return result


def compare_many_groups(groups):
    """Kruskal-Wallis across k groups + epsilon^2 effect size + Holm-corrected pairwise tests.

    groups: dict {name: values}. epsilon^2 ~0.01 small, ~0.08 medium, ~0.26 large.
    """
    clean = {k: pd.Series(v, dtype=float).dropna().to_numpy() for k, v in groups.items()}
    h, p = stats.kruskal(*clean.values())
    n = sum(v.size for v in clean.values())
    overall = {"H": float(h), "p_value": float(p), "epsilon_squared": float(h / (n - 1)), "n": n, "k": len(clean)}

    names = list(clean)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = clean[names[i]], clean[names[j]]
            u, pp = stats.mannwhitneyu(a, b, alternative="two-sided")
            pairs.append({"group_a": names[i], "group_b": names[j],
                          "median_a": np.median(a), "median_b": np.median(b),
                          "p_raw": pp, "rank_biserial": 2 * u / (a.size * b.size) - 1})
    pairwise = pd.DataFrame(pairs)
    if not pairwise.empty:
        reject, p_holm, _, _ = multipletests(pairwise["p_raw"], alpha=ALPHA, method="holm")
        pairwise["p_holm"] = p_holm
        pairwise["significant"] = reject
    return overall, pairwise


def chi_square_independence(table):
    """Chi-square test of independence on a contingency table + Cramer's V."""
    table = np.asarray(table, dtype=float)
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    n = table.sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof),
            "cramers_v": float(v), "min_expected": float(expected.min())}


# --------------------------------------------------------------------------
# Time-series diagnostics (needed before forecasting)
# --------------------------------------------------------------------------
def stationarity_report(series):
    """ADF and KPSS test the SAME question from opposite sides.

    ADF : H0 = series has a unit root (NON-stationary). Small p -> stationary.
    KPSS: H0 = series IS stationary.                     Small p -> non-stationary.
    Agreement between both gives a confident answer.
    """
    s = pd.Series(series, dtype=float).dropna()
    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC", result_object=False)
    with warnings.catch_warnings():          # KPSS warns when p is beyond its lookup table
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
    adf_says = "stationary" if adf_p < ALPHA else "non-stationary"
    kpss_says = "non-stationary" if kpss_p < ALPHA else "stationary"
    conclusion = adf_says if adf_says == kpss_says else "inconclusive (tests disagree)"
    return {"adf_stat": float(adf_stat), "adf_p": float(adf_p), "adf_says": adf_says,
            "kpss_stat": float(kpss_stat), "kpss_p": float(kpss_p), "kpss_says": kpss_says,
            "conclusion": conclusion}


def ljung_box(series, lags=(1, 6, 12)):
    """Ljung-Box: H0 = no autocorrelation up to lag k. Small p -> today depends on past days."""
    s = pd.Series(series, dtype=float).dropna()
    return acorr_ljungbox(s, lags=list(lags), return_df=True)

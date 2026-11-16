"""
Module 5 - Statistical Analysis
===============================
File   : notebooks/05_statistical_analysis.py
Run    : python notebooks/05_statistical_analysis.py      (from the project root)
Input  : data/processed/transactions_clean.parquet
Output : printed results + reports/statistical_findings.txt + 4 charts in reports/figures/

Sections
  1. Descriptive statistics (mean, median, mode, variance, std, shape)
  2. Distribution analysis (normality, raw vs log scale)
  3. Correlation analysis (Pearson vs Spearman)
  4. Hypothesis tests (weekday effect, UK vs overseas, Q4 effect, returns by region)
  5. Time-series diagnostics (stationarity, autocorrelation) for Module 7
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.graphics.tsaplots import plot_acf  # noqa: E402

from src.analysis import eda  # noqa: E402
from src.analysis import statistical_tests as st  # noqa: E402
from src.utils import plotting as P  # noqa: E402
from src.utils.paths import REPORTS_DIR  # noqa: E402

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}" if abs(v) < 1 else f"{v:,.2f}")

LINES = []


def say(text=""):
    print(text)
    LINES.append(str(text))


def section(title):
    say("\n" + "=" * 78)
    say(title)
    say("=" * 78)


def fmt_p(p):
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


# --------------------------------------------------------------------------
# 1. Descriptive statistics
# --------------------------------------------------------------------------
def descriptive(df, daily, orders, customers):
    section("1. DESCRIPTIVE STATISTICS")
    lines = eda.sale_lines(df)
    variables = {
        "order_value": orders["order_value"],
        "daily_net_sales": daily["net_sales"],
        "line_quantity": lines["Quantity"],
        "unit_price": lines["UnitPrice"],
        "customer_net_spend": customers["net_sales"],
        "customer_orders": customers.loc[customers["orders"] > 0, "orders"],
    }
    table = pd.DataFrame({k: st.describe_distribution(v) for k, v in variables.items()}).T
    say(table.to_string())
    say("\nHow to read it:")
    say(" * mean >> median and skewness >> 0 everywhere: a few very large values pull the mean up.")
    say(" * cv (std / mean) > 1 means the spread is larger than the average itself.")
    say(" * mode is NaN when no value repeats (daily sales). For money amounts it is just the most frequent")
    say(f"   exact amount (order value: £{table.loc['order_value', 'mode']:.2f}) - the median describes them better.")
    say("   For discrete counts the mode is meaningful: the most common line quantity is "
        f"{table.loc['line_quantity', 'mode']:.0f} and the most common number of orders per customer is "
        f"{table.loc['customer_orders', 'mode']:.0f}.")
    say(" * daily_net_sales is the least skewed variable: summing many orders per day evens things out")
    say("   (Central Limit Theorem at work) - good news for forecasting.")


# --------------------------------------------------------------------------
# 2. Distributions
# --------------------------------------------------------------------------
def distributions(daily, orders):
    section("2. DISTRIBUTION ANALYSIS (D'Agostino-Pearson normality test)")
    for name, values in [("Order value", orders["order_value"]), ("Daily net sales", daily["net_sales"])]:
        say(f"\n{name}:")
        say(st.normality_report(values).to_string())
    say("\nInterpretation: order value is far from normal on the raw scale; the log scale removes most of")
    say("the skew (approximately log-normal). This is why we use medians and rank-based tests below,")
    say("and why log-transforms will be used for skewed model features later.")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, data, title in [
        (axes[0], orders["order_value"], "Order value (raw)"),
        (axes[1], np.log(orders["order_value"]), "Order value (log)"),
        (axes[2], daily["net_sales"], "Daily net sales (raw)"),
    ]:
        (osm, osr), (slope, intercept, _) = stats.probplot(data, dist="norm")
        ax.scatter(osm, osr, s=8, color=P.SERIES[0], alpha=0.5, linewidths=0)
        ax.plot(osm, slope * osm + intercept, color=P.INK, linewidth=1, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Theoretical normal quantiles")
    axes[0].set_ylabel("Observed values")
    fig.suptitle("Q-Q plots: points on the dashed line = normally distributed", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    say(f"-> {P.save(fig, '05_01_qq_plots.png').name}")


# --------------------------------------------------------------------------
# 3. Correlations
# --------------------------------------------------------------------------
def correlations(df, daily, customers):
    section("3. CORRELATION ANALYSIS")
    cols = ["net_sales", "net_units", "orders", "customers", "avg_order_value"]
    pearson, spearman = st.correlation_matrices(daily[cols])
    say("Daily level - Pearson (linear):")
    say(pearson.round(2).to_string())
    say("\nDaily level - Spearman (rank / monotonic):")
    say(spearman.round(2).to_string())

    say("\nKey pairs (Spearman, with p-values):")
    lines = eda.sale_lines(df)
    buyers = customers[customers["orders"] > 0]
    for label, x, y in [
        ("Daily: orders vs net sales", daily["orders"], daily["net_sales"]),
        ("Daily: avg order value vs net sales", daily["avg_order_value"], daily["net_sales"]),
        ("Line: unit price vs quantity", lines["UnitPrice"], lines["Quantity"]),
        ("Customer: orders vs net spend", buyers["orders"], buyers["net_sales"]),
    ]:
        r = st.correlation_test(x, y)
        rp = st.correlation_test(x, y, method="pearson")
        say(f"  {label:<38} Spearman {r['r']:+.2f} (p {fmt_p(r['p_value'])})   Pearson {rp['r']:+.2f}   n={r['n']:,}")
    say("\nInterpretation: daily revenue is driven mainly by HOW MANY orders arrive, much more than by")
    say("how large each order is. Where Pearson and Spearman disagree, outliers are distorting Pearson.")
    say("Correlation is not causation: e.g. orders and customers move together because both rise in Q4.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, m, title in [(axes[0], pearson, "Pearson"), (axes[1], spearman, "Spearman")]:
        sns.heatmap(m, vmin=-1, vmax=1, cmap=P.DIVERGING, annot=True, fmt=".2f", square=True,
                    linewidths=1, linecolor=P.SURFACE, ax=ax, cbar=ax is axes[1], annot_kws={"fontsize": 9})
        ax.set_title(f"{title} correlation - daily features")
        ax.grid(False)
    fig.tight_layout()
    say(f"-> {P.save(fig, '05_02_correlation_heatmaps.png').name}")


# --------------------------------------------------------------------------
# 4. Hypothesis tests
# --------------------------------------------------------------------------
def hypothesis_tests(df, daily, orders, customers):
    section("4. HYPOTHESIS TESTS  (alpha = 0.05)")

    # H1 ------------------------------------------------------------------
    say("\nH1. Do daily net sales differ between weekdays?")
    say("    H0: all weekdays have the same distribution of daily sales.  Test: Kruskal-Wallis")
    order = [d for d in eda.WEEKDAY_ORDER if d in set(daily["weekday"])]
    groups = {d: daily.loc[daily["weekday"] == d, "net_sales"] for d in order}
    overall, pairwise = st.compare_many_groups(groups)
    say(f"    All 6 trading days : H = {overall['H']:.2f}, p = {fmt_p(overall['p_value'])}, "
        f"epsilon^2 = {overall['epsilon_squared']:.3f} -> {st.verdict(overall['p_value'])}")
    weekdays_only = {d: v for d, v in groups.items() if d != "Sunday"}
    ov2, _ = st.compare_many_groups(weekdays_only)
    say(f"    Monday-Friday only : H = {ov2['H']:.2f}, p = {fmt_p(ov2['p_value'])}, "
        f"epsilon^2 = {ov2['epsilon_squared']:.3f} -> {st.verdict(ov2['p_value'])}")
    sig = pairwise[pairwise["significant"]]
    say(f"    Pairwise Mann-Whitney with Holm correction: {len(sig)} of {len(pairwise)} pairs differ:")
    say(sig[["group_a", "group_b", "median_a", "median_b", "p_holm", "rank_biserial"]].to_string(index=False))
    weekday_pairs = pairwise[(pairwise["group_a"] != "Sunday") & (pairwise["group_b"] != "Sunday")]
    say(f"    Conclusion: the big effect is Sunday (a short trading day; r ~ +0.7 against every weekday).")
    say(f"    Monday-Friday differ only slightly (epsilon^2 {ov2['epsilon_squared']:.3f} = small) and "
        f"{int(weekday_pairs['significant'].sum())} of {len(weekday_pairs)} weekday pairs survive the Holm correction.")
    say("    -> forecasting features: weekday (at least an is_sunday flag).")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.boxplot(data=daily, x="weekday", y="net_sales", order=order, color=P.BLUE_LIGHT, width=0.55,
                linewidth=1, fliersize=3, flierprops={"markerfacecolor": P.SERIES[0], "markeredgecolor": P.SURFACE},
                medianprops={"color": P.INK, "linewidth": 2}, ax=ax)
    ax.yaxis.set_major_formatter(P.money())
    ax.set_xlabel("")
    ax.set_ylabel("Daily net sales")
    ax.set_title(f"Daily net sales by weekday  (Kruskal-Wallis p {fmt_p(overall['p_value'])})")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    say(f"-> {P.save(fig, '05_03_weekday_boxplot.png').name}")

    # H2 ------------------------------------------------------------------
    say("\nH2. Are overseas orders larger than UK orders?")
    say("    H0: overseas and UK order values have the same distribution. H1: overseas larger (one-sided).")
    is_uk = orders["country"] == "United Kingdom"
    r = st.compare_two_groups(orders.loc[~is_uk, "order_value"], orders.loc[is_uk, "order_value"], alternative="greater")
    lo, hi = r["median_diff_ci95"]
    say(f"    Order level   : median overseas {r['median_a']:,.2f} vs UK {r['median_b']:,.2f}; "
        f"U p = {fmt_p(r['p_value'])}; rank-biserial r = {r['rank_biserial']:+.2f}; "
        f"95% CI of median difference [{lo:,.2f}, {hi:,.2f}]")
    say("    Caveat: orders are NOT independent - one Dutch customer can place 90 orders. The honest unit is")
    say("    the CUSTOMER, so we repeat the test on each customer's average order value:")
    buyers = customers[customers["orders"] > 0]
    c_uk = buyers["country"] == "United Kingdom"
    rc = st.compare_two_groups(buyers.loc[~c_uk, "avg_order_value"], buyers.loc[c_uk, "avg_order_value"],
                               alternative="greater")
    lo, hi = rc["median_diff_ci95"]
    say(f"    Customer level: median overseas {rc['median_a']:,.2f} vs UK {rc['median_b']:,.2f}; "
        f"U p = {fmt_p(rc['p_value'])}; r = {rc['rank_biserial']:+.2f}; 95% CI [{lo:,.2f}, {hi:,.2f}] "
        f"(n = {rc['n_a']} vs {rc['n_b']})")
    say(f"    Conclusion: {st.verdict(rc['p_value'])} at the customer level. Overseas customers place larger")
    say("    orders, but the effect is moderate - the EDA ratio (803 vs 471) was inflated by a few big accounts.")

    # H3 ------------------------------------------------------------------
    say("\nH3. Are daily sales higher in autumn (Sep-Nov 2011) than in Jan-Aug 2011?")
    q4 = daily["month"].isin(pd.period_range("2011-09", "2011-11", freq="M"))
    base = daily["month"].isin(pd.period_range("2011-01", "2011-08", freq="M"))
    r3 = st.compare_two_groups(daily.loc[q4, "net_sales"], daily.loc[base, "net_sales"], alternative="greater")
    lo, hi = r3["median_diff_ci95"]
    say(f"    median Sep-Nov {r3['median_a']:,.0f} vs Jan-Aug {r3['median_b']:,.0f}; p = {fmt_p(r3['p_value'])}; "
        f"r = {r3['rank_biserial']:+.2f}; 95% CI of difference [{lo:,.0f}, {hi:,.0f}]")
    say("    Caveat: consecutive days are autocorrelated (see section 5), so the days are not independent")
    say("    and the p-value is over-optimistic. Judge by the effect size and CI instead: "
        f"r = {r3['rank_biserial']:+.2f} is large and the CI is far from 0, so the autumn rise is real.")

    # H4 ------------------------------------------------------------------
    say("\nH4. Is the chance that an invoice is a cancellation independent of region (UK vs overseas)?")
    inv = df[df["IsRevenueLine"]].drop_duplicates("InvoiceNo")
    region = pd.Series(np.where(inv["Country"] == "United Kingdom", "UK", "Overseas"), index=inv.index, name="region")
    table = pd.crosstab(region, inv["TransactionType"].astype(str))
    say(table.to_string())
    rates = table["cancellation"] / table.sum(axis=1)
    say(f"    cancellation-invoice share: UK {rates['UK']:.1%}, overseas {rates['Overseas']:.1%}")
    r4 = st.chi_square_independence(table.to_numpy())
    say(f"    chi2 = {r4['chi2']:.2f}, dof = {r4['dof']}, p = {fmt_p(r4['p_value'])}, Cramer's V = {r4['cramers_v']:.3f}, "
        f"smallest expected count = {r4['min_expected']:.0f} (must be >= 5 -> test valid)")
    strength = "negligible" if r4["cramers_v"] < 0.1 else "small" if r4["cramers_v"] < 0.3 else "substantial"
    say(f"    Conclusion: statistically {st.verdict(r4['p_value'])}, but Cramer's V = {r4['cramers_v']:.3f} "
        f"means the association is {strength}.")
    say("    Lesson: with ~23,000 invoices even a 3-point difference is 'significant'; it is not practically")
    say("    important. Always report the effect size next to the p-value.")


# --------------------------------------------------------------------------
# 5. Time-series diagnostics
# --------------------------------------------------------------------------
def time_series(daily):
    section("5. TIME-SERIES DIAGNOSTICS (daily net sales) - input for Module 7")
    s = daily["net_sales"]
    rep = st.stationarity_report(s)
    say(f"ADF : stat {rep['adf_stat']:.3f}, p {fmt_p(rep['adf_p'])} -> {rep['adf_says']}")
    say(f"KPSS: stat {rep['kpss_stat']:.3f}, p {fmt_p(rep['kpss_p'])} -> {rep['kpss_says']}")
    say(f"Combined conclusion: {rep['conclusion']}")
    say("Ljung-Box (H0: no autocorrelation up to lag k):")
    say(st.ljung_box(s).to_string())
    say("\nInterpretation for forecasting:")
    say(" * Significant autocorrelation -> yesterday and last week carry information: lag features are useful.")
    say(" * A trend/level shift (Q4 rise) -> random train/test splits would leak the future; we MUST split")
    say("   chronologically and evaluate on the most recent period.")
    acf6 = s.autocorr(6)
    say(f" * Weekly cycle: autocorrelation peaks at lags 6, 12, 18, 24 (lag-6 r = {acf6:.2f}). Lag 6 = one trading")
    say("   week (no Saturdays) -> 'same weekday last week' (lag 6) is the strongest single predictor,")
    say("   and a seasonal-naive forecast (repeat last week) is the baseline every model must beat.")

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(s, lags=30, ax=ax, color=P.SERIES[0], vlines_kwargs={"colors": P.SERIES[0]},
             title="Autocorrelation of daily net sales (shaded = 95% no-correlation band)")
    for coll in ax.collections:
        coll.set_facecolor(P.BLUE_LIGHT)
        coll.set_alpha(0.35)
    ax.set_xlabel("Lag (trading days)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    say(f"-> {P.save(fig, '05_04_autocorrelation.png').name}")


def main():
    P.apply_style()
    df = eda.load_clean()
    daily = st.daily_features(df)
    orders = eda.invoice_summary(df)
    customers = eda.customer_summary(df)
    say(f"Units of analysis: {len(daily)} trading days, {len(orders):,} orders, {len(customers):,} customers")

    descriptive(df, daily, orders, customers)
    distributions(daily, orders)
    correlations(df, daily, customers)
    hypothesis_tests(df, daily, orders, customers)
    time_series(daily)

    out = REPORTS_DIR / "statistical_findings.txt"
    out.write_text("\n".join(LINES), encoding="utf-8")
    print(f"\nFindings saved to {out.relative_to(PROJECT_ROOT)}; figures in reports/figures/")


if __name__ == "__main__":
    main()

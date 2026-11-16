"""
Module 4 - Exploratory Data Analysis
====================================
File   : notebooks/04_eda.py
Run    : python notebooks/04_eda.py        (from the project root)
Input  : data/processed/transactions_clean.parquet   (made by Module 3)
Output : printed findings + reports/eda_findings.txt + 9 charts in reports/figures/

Questions answered
  Univariate   : How big is a typical order? How skewed is spending?
  Bivariate    : How do price and quantity relate? Which countries/products drive sales?
  Time         : Is there a trend? Seasonality? Which days and hours are busy?
  Multivariate : Do all countries share the same seasonality? Weekday x hour pattern?
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.analysis import eda  # noqa: E402
from src.utils import plotting as P  # noqa: E402
from src.utils.paths import REPORTS_DIR  # noqa: E402

pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

LINES = []  # collected findings, also written to a text file


def say(text=""):
    print(text)
    LINES.append(str(text))


def section(title):
    say("\n" + "=" * 78)
    say(title)
    say("=" * 78)


def short(name, n=34):
    name = str(name)
    return name if len(name) <= n else name[: n - 1] + "…"


# --------------------------------------------------------------------------
# 1. Monthly trend
# --------------------------------------------------------------------------
def monthly(df):
    section("1. MONTHLY TREND")
    m = eda.monthly_summary(df)
    say(m.to_string())
    full = m[~m["is_partial"]]
    say(f"\nBest month : {full['net_sales'].idxmax()}  ({full['net_sales'].max():,.0f})")
    say(f"Worst month: {full['net_sales'].idxmin()}  ({full['net_sales'].min():,.0f})")
    say(f"Nov 2011 vs Feb 2011 net sales per trading day: "
        f"x{m.loc['2011-11', 'net_per_trading_day'] / m.loc['2011-02', 'net_per_trading_day']:.1f}")
    if m["is_partial"].any():
        p = m[m["is_partial"]].iloc[0]
        say(f"Partial month {m[m['is_partial']].index[0]}: only {p['trading_days']} trading days "
            f"- compare it per trading day, never by total.")

    labels = [p.strftime("%b %y") for p in m.index]
    x = np.arange(len(m))
    colors = [P.BLUE_LIGHT if partial else P.SERIES[0] for partial in m["is_partial"]]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9.5), sharex=True)
    panels = [
        ("net_sales", "Net product sales per month", P.money()),
        ("net_per_trading_day", "Net sales per trading day (fair for the partial month)", P.money()),
        ("genuine_cancellation_rate", "Cancellation rate: cancelled ÷ gross, excluding order-entry corrections",
         plt.FuncFormatter(lambda v, _: f"{v:.1%}")),
    ]
    for ax, (col, title, fmt) in zip(axes, panels):
        ax.bar(x, m[col], width=0.65, color=colors, edgecolor=P.SURFACE, linewidth=2)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(fmt)
        ax.grid(axis="x", visible=False)
    axes[-1].set_xticks(x, labels)
    if m["is_partial"].any():
        i = int(np.flatnonzero(m["is_partial"])[0])
        axes[0].annotate(f"{int(m['trading_days'].iloc[i])} days only", (i, m["net_sales"].iloc[i]),
                         xytext=(0, 6), textcoords="offset points", ha="center", color=P.INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    say(f"-> {P.save(fig, '04_01_monthly_trend.png').name}")
    return m


# --------------------------------------------------------------------------
# 2. Daily trend
# --------------------------------------------------------------------------
def daily(df):
    section("2. DAILY TREND")
    d = eda.daily_summary(df)
    say(f"Trading days: {len(d)}   mean {d['net_sales'].mean():,.0f}   median {d['net_sales'].median():,.0f}")
    say("Top 5 days:")
    say(d["net_sales"].nlargest(5).to_string())
    say("Lowest 5 days:")
    say(d["net_sales"].nsmallest(5).to_string())

    # Break the line across closures (e.g. Christmas) instead of drawing a fake straight segment.
    gaps = d.index.to_series().diff().dt.days > 3
    breaks = pd.DataFrame(index=d.index[gaps] - pd.Timedelta(days=1), columns=d.columns, dtype=float)
    d = pd.concat([d, breaks]).sort_index()

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(d.index, d["net_sales"], color=P.BLUE_LIGHT, linewidth=1, label="Daily net sales")
    ax.plot(d.index, d["rolling_mean"], color=P.SERIES[0], linewidth=2, label="7-trading-day average")
    ax.set_title("Daily net product sales")
    ax.yaxis.set_major_formatter(P.money())
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    say(f"-> {P.save(fig, '04_02_daily_trend.png').name}")


# --------------------------------------------------------------------------
# 3. Weekday x hour
# --------------------------------------------------------------------------
def weekday_hour(df):
    section("3. WEEKDAY x HOUR (average net sales per trading day)")
    mat = eda.weekday_hour_matrix(df)
    say("Average per trading day, by weekday:")
    say(mat.sum(axis=1).round(0).to_string())
    say("Average per trading day, by hour (summed over weekdays / 6):")
    say((mat.sum(axis=0) / len(mat)).round(0).to_string())
    say("Note: there are no Saturday transactions at all - the business does not trade on Saturdays.")

    mat = mat.loc[:, (mat != 0).any(axis=0)]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    sns.heatmap(mat, cmap=P.SEQUENTIAL, linewidths=1, linecolor=P.SURFACE, ax=ax,
                cbar_kws={"format": P.money(), "label": "Avg net sales per trading day"})
    ax.set_title("When does revenue happen? Weekday × hour")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("")
    ax.grid(False)
    fig.tight_layout()
    say(f"-> {P.save(fig, '04_03_weekday_hour_heatmap.png').name}")


# --------------------------------------------------------------------------
# 4. Countries
# --------------------------------------------------------------------------
def countries(df):
    section("4. COUNTRIES")
    c = eda.country_summary(df)
    say(c.head(15).to_string())
    uk = c.loc["United Kingdom"]
    say(f"\nUnited Kingdom: {uk['share']:.1%} of net sales, {int(uk['customers']):,} known customers.")
    say(f"Average order value: UK {uk['avg_order_value']:,.0f} vs rest of world "
        f"{c.drop('United Kingdom')['net_sales'].sum() / c.drop('United Kingdom')['invoices'].sum():,.0f}")

    top = c.drop("United Kingdom").head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(top.index.astype(str), top["net_sales"], height=0.65, color=P.SERIES[0])
    for y, v in enumerate(top["net_sales"]):
        ax.text(v, y, f"  {v / 1e3:,.0f}K", va="center", color=P.INK_SECONDARY, fontsize=9)
    ax.set_title(f"Top 12 countries outside the UK  (the UK alone is {uk['share']:.0%} of net sales)")
    ax.xaxis.set_major_formatter(P.money())
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    say(f"-> {P.save(fig, '04_04_countries.png').name}")
    return c


# --------------------------------------------------------------------------
# 5. Products
# --------------------------------------------------------------------------
def products(df):
    section("5. PRODUCTS")
    p = eda.product_summary(df)
    say(f"Products sold: {len(p):,}")
    say(f"Top 20% of products hold {eda.top_share(p['net_sales']):.1%} of net sales")
    say("\nTop 10 by net revenue:")
    say(p.nlargest(10, "net_sales")[["product", "net_sales", "net_quantity", "invoices", "median_price"]].to_string())
    say("\nTop 10 by net units:")
    say(p.nlargest(10, "net_quantity")[["product", "net_quantity", "net_sales", "invoices", "median_price"]].to_string())
    say(f"\nProducts with negative net sales (more returned than sold in this window): {(p['net_sales'] < 0).sum():,}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, col, title, fmt in [
        (axes[0], "net_sales", "Top 10 products by net revenue", P.money()),
        (axes[1], "net_quantity", "Top 10 products by net units", plt.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}K")),
    ]:
        top = p.nlargest(10, col).iloc[::-1]
        ax.barh([short(n) for n in top["product"]], top[col], height=0.65, color=P.SERIES[0])
        ax.set_title(title)
        ax.xaxis.set_major_formatter(fmt)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    say(f"-> {P.save(fig, '04_05_top_products.png').name}")
    return p


# --------------------------------------------------------------------------
# 6. Order value distribution (univariate)
# --------------------------------------------------------------------------
def order_values(df):
    section("6. ORDER VALUE DISTRIBUTION (real orders, order-entry corrections excluded)")
    inv = eda.invoice_summary(df)
    v = inv["order_value"]
    say(v.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())
    say(f"Skewness: {v.skew():.1f}  (mean is {v.mean() / v.median():.1f}x the median -> use the MEDIAN as 'typical')")
    say(f"Lines per order: median {inv['lines'].median():.0f}, mean {inv['lines'].mean():.1f}")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bins = np.logspace(np.log10(max(v.min(), 0.5)), np.log10(v.max()), 60)
    ax.hist(v, bins=bins, color=P.SERIES[0], edgecolor=P.SURFACE, linewidth=0.8)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(P.money())
    med = v.median()
    ax.axvline(med, color=P.INK, linewidth=1, linestyle="--")
    ax.annotate(f"median £{med:,.0f}", (med, ax.get_ylim()[1] * 0.92), xytext=(6, 0),
                textcoords="offset points", color=P.INK, fontsize=9)
    ax.set_title("Order value distribution (log scale)")
    ax.set_xlabel("Order value")
    ax.set_ylabel("Number of orders")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    say(f"-> {P.save(fig, '04_06_order_value_distribution.png').name}")


# --------------------------------------------------------------------------
# 7. Customers: concentration
# --------------------------------------------------------------------------
def customers(df):
    section("7. CUSTOMERS (known CustomerID only)")
    c = eda.customer_summary(df)
    say(f"Customers: {len(c):,}")
    say(c[["net_sales", "orders", "avg_order_value"]].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]).to_string())
    buyers = c[c["orders"] > 0]
    one_off = (buyers["orders"] == 1).mean()
    say(f"\nCustomers with at least one real order: {len(buyers):,}  "
        f"(the other {len(c) - len(buyers)} only have returns or order-entry corrections)")
    say(f"One-time buyers: {one_off:.1%}   Repeat buyers: {1 - one_off:.1%}")
    share = eda.top_share(c["net_sales"])
    say(f"Top 20% of customers hold {share:.1%} of net sales (Pareto)")
    say(f"Top 10 customers hold {c['net_sales'].nlargest(10).sum() / c['net_sales'].clip(lower=0).sum():.1%}")
    say(f"Customers with net sales <= 0 (returned as much as they bought): {(c['net_sales'] <= 0).sum():,}")

    curve = eda.pareto_curve(c["net_sales"])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(curve["entity_share"], curve["value_share"], color=P.SERIES[0], label="Actual")
    ax.plot([0, 1], [0, 1], color=P.MUTED, linewidth=1, linestyle="--", label="Perfect equality")
    ax.scatter([0.2], [share], s=50, color=P.SERIES[0], edgecolor=P.SURFACE, linewidth=2, zorder=3)
    ax.annotate(f"Top 20% → {share:.0%} of revenue", (0.2, share), xytext=(12, -16),
                textcoords="offset points", color=P.INK, fontsize=9)
    pct = plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Share of customers (largest first)")
    ax.set_ylabel("Share of net sales")
    ax.set_title("Customer revenue concentration")
    ax.legend(loc="lower right")
    fig.tight_layout()
    say(f"-> {P.save(fig, '04_07_customer_pareto.png').name}")


# --------------------------------------------------------------------------
# 8. Price vs quantity (bivariate)
# --------------------------------------------------------------------------
def price_quantity(df):
    section("8. PRICE vs QUANTITY (sale lines)")
    s = eda.sale_lines(df)
    rho = s["UnitPrice"].corr(s["Quantity"], method="spearman")
    say(f"Spearman correlation UnitPrice vs Quantity: {rho:.2f}")
    say("(Spearman, not Pearson: both variables are extremely skewed; Spearman uses ranks.)")
    bands = pd.cut(s["UnitPrice"], [0, 1, 2, 5, 10, np.inf], labels=["<£1", "£1-2", "£2-5", "£5-10", ">£10"])
    say("Median quantity per line by price band:")
    say(s.groupby(bands, observed=True)["Quantity"].median().to_string())

    fig, ax = plt.subplots(figsize=(8, 5.5))
    hb = ax.hexbin(s["UnitPrice"], s["Quantity"], xscale="log", yscale="log", gridsize=45,
                   cmap=P.SEQUENTIAL, mincnt=1, bins="log", linewidths=0.2, edgecolors=P.SURFACE)
    fig.colorbar(hb, ax=ax, label="Number of lines (log)")
    ax.set_xlabel("Unit price (log)")
    ax.set_ylabel("Quantity per line (log)")
    ax.xaxis.set_major_formatter(P.money(2))
    ax.set_title(f"Cheaper items are bought in larger quantities  (Spearman ρ = {rho:.2f})")
    ax.grid(False)
    fig.tight_layout()
    say(f"-> {P.save(fig, '04_08_price_vs_quantity.png').name}")


# --------------------------------------------------------------------------
# 9. Country x month seasonality (multivariate)
# --------------------------------------------------------------------------
def country_seasonality(df):
    section("9. COUNTRY x MONTH (share of each country's yearly net sales)")
    t = eda.country_month_share(df, top_n=8)
    say(t.map(lambda v: f"{v:.0%}").to_string())

    t = t.copy()
    t.columns = [p.strftime("%b %y") for p in t.columns]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(t, cmap=P.SEQUENTIAL, vmin=0, linewidths=1, linecolor=P.SURFACE, annot=True, fmt=".0%",
                annot_kws={"fontsize": 8}, ax=ax, cbar_kws={"format": plt.FuncFormatter(lambda v, _: f"{v:.0%}")})
    ax.set_title("Seasonality by country: share of each country's yearly net sales")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    fig.tight_layout()
    say(f"-> {P.save(fig, '04_09_country_month_heatmap.png').name}")


def main():
    P.apply_style()
    df = eda.load_clean()
    say(f"Loaded clean data: {len(df):,} rows; revenue lines: {df['IsRevenueLine'].sum():,}")
    monthly(df)
    daily(df)
    weekday_hour(df)
    countries(df)
    products(df)
    order_values(df)
    customers(df)
    price_quantity(df)
    country_seasonality(df)

    out = REPORTS_DIR / "eda_findings.txt"
    out.write_text("\n".join(LINES), encoding="utf-8")
    print(f"\nFindings saved to {out.relative_to(PROJECT_ROOT)}; figures in reports/figures/")


if __name__ == "__main__":
    main()

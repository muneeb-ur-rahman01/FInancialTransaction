import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import charts as C
from components import data as D
from components import ui

ui.page("Product Demand", "Weekly units per product: next-week forecast and how the method performed", "📦")
rep = D.load_report("demand_forecasting.json", "demand")
panel = D.load_product_weekly()
preds = D.load_demand_predictions()
comp = pd.DataFrame(rep["test_comparison"])
champ = rep["test_metrics"]
champion_label = rep.get("champion_label", rep["model_name"])


forecast = D.demand_forecast().set_index("StockCode")          # Module 9: src/models/predict.py
next_week = pd.Timestamp(forecast["week"].iloc[0])
ui.kpi_row([
    ("Forecast method", rep["model_name"], rep["selection_decision"]),
    ("Weekly error (WAPE)", f"{champ['WAPE']:.1f}%", "Total absolute error ÷ total units over the 8 test weeks."),
    ("Product-level bias", f"{champ['ProductBias_pct']:.1f}%",
     "Σ |each product's total error| ÷ total units. Catches products that are systematically under- or over-forecast."),
    ("Products forecast", f"{len(forecast):,}", "Products with at least 4 weeks of sales history."),
    ("Units next week", f"{forecast['forecast'].sum():,.0f}", f"Sum of all product forecasts for the week of {next_week:%d %b %Y}."),
])
with st.expander("Why a simple 4-week average and not machine learning?"):
    st.markdown(
        f"{rep['selection_decision']}\n\n"
        "The ML models had lower weekly error but systematically **under-forecast the best sellers** (capping them "
        "at a fraction of real demand), which is the costliest mistake for stock planning. The 4-week average was "
        "chosen over 'last week' because it smooths one-off spikes instead of repeating them."
    )

st.subheader(f"Next-week forecast · week of {next_week:%d %b %Y}")
query = st.text_input("Search products", placeholder="e.g. HEART, CHRISTMAS, 85123A")
view = forecast
if query:
    q = query.strip().upper()
    view = forecast[forecast["product"].str.upper().str.contains(q, regex=False) | forecast.index.str.contains(q, regex=False)]
st.dataframe(view.drop(columns="week").reset_index().rename(columns={
    "StockCode": "Code", "product": "Product", "forecast": "Forecast (units)", "last_week": "Last week (units)",
    "mean_last_4_weeks": "Mean of last 4 weeks"}),
             hide_index=True, width="stretch", height=280)
ui.note("The latest week in the data (5–9 Dec 2011) is a short week, so recent averages may run slightly low.")

st.subheader("Product history and forecast")
options = forecast.index.tolist()
code = st.selectbox("Product", options, format_func=lambda c: f"{forecast.loc[c, 'product']} ({c})")
hist = panel[panel["StockCode"] == code].sort_values("week")
test = preds[preds["StockCode"] == code].sort_values("week")
ml_cols = [c for c in preds.columns if c not in ("StockCode", "product", "week", "units_sold", champion_label)
           and not c.startswith(("Last week", "Product long-run"))]
fig = go.Figure()
C.line(fig, hist["week"], hist["units_sold"], "Actual units", C.BLUE, width=2.5, markers=True, money=False)
if champion_label in test:
    C.line(fig, test["week"], test[champion_label], f"Forecast: {champion_label}", C.ORANGE, width=2, markers=True, money=False)
if ml_cols:
    best_ml = rep.get("best_ml_model", ml_cols[0])
    col = best_ml if best_ml in test else ml_cols[0]
    C.line(fig, test["week"], test[col], f"Best ML: {col}", C.AQUA, dash="dash", width=1.5, money=False)
fig.add_trace(go.Scatter(x=[next_week], y=[forecast.loc[code, "forecast"]], mode="markers", name="Next week",
                         marker=dict(size=13, color=C.ORANGE, symbol="diamond"),
                         hovertemplate="Next week forecast<br>%{x|%d %b %Y}: %{y:,.0f} units<extra></extra>"))
fig.update_yaxes(title="Units per week", rangemode="tozero")
st.plotly_chart(C.style(fig, height=380), width="stretch")
ui.note("Forecast lines cover the 8 test weeks (17 Oct – 5 Dec 2011), when the models had not seen the actual values.")

left, right = st.columns(2)
with left:
    st.subheader("Total units across all products")
    weekly = preds.groupby("week")[["units_sold", champion_label]].sum()
    fig = go.Figure()
    C.line(fig, weekly.index, weekly["units_sold"], "Actual", C.BLUE, width=2.5, markers=True, money=False)
    C.line(fig, weekly.index, weekly[champion_label], "Forecast", C.ORANGE, width=2, markers=True, money=False)
    fig.update_yaxes(rangemode="tozero", tickformat="~s")
    st.plotly_chart(C.style(fig, height=320), width="stretch")
with right:
    st.subheader("Methods compared (test weeks)")
    t = comp.sort_values("score")
    names = [m if f == "Baseline" else f"{f} ({m})" for f, m in zip(t["family"], t["model"])]
    colors = [C.GREY if f == "Baseline" else C.BLUE for f in t["family"]]
    st.plotly_chart(C.hbar([n[:42] for n in names], t["score"], money=False, height=320,
                           color=colors[::-1], text_fmt=lambda v: f"{v:.1f}"), width="stretch")
    ui.note("Score = WAPE + product-level bias (lower is better). Grey = simple rules.")

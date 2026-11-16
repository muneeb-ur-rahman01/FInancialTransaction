import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import charts as C
from components import data as D
from components import ui

ui.page("Sales Forecasting", "Next-trading-day net sales: how well the model did on 8 weeks it never saw", "🔮")
rep = D.load_report("sales_forecasting.json", "sales")
preds = D.load_sales_predictions()
comp = pd.DataFrame(rep["test_comparison"])

champion_label = f"{rep['model_family']} ({rep['model_name']})"
base = comp[comp["family"] == "Baseline"].sort_values("MAE").iloc[0]
champ = rep["test_metrics"]
gain = (1 - champ["MAE"] / base["MAE"]) * 100

ui.kpi_row([
    ("Champion model", rep["model_family"], f"Chosen by time-series cross-validation on the training period: {rep['model_name']}"),
    ("Average daily error (MAE)", ui.money(champ["MAE"]), "Mean absolute difference between forecast and actual daily net sales."),
    ("Error as % of sales (WAPE)", f"{champ['WAPE']:.1f}%", "Total absolute error ÷ total actual sales over the test period."),
    ("vs best simple rule", f"{gain:+.1f}%", f"MAE improvement over '{base['model']}' (£{base['MAE']:,.0f})."),
    ("Average bias", ui.money(champ["Bias"]), "Mean of forecast − actual. Negative = under-forecasting."),
])
st.caption(f"Test period {rep['test_period'][0]} → {rep['test_period'][1]} · production model refit on "
           f"{rep['trained_on'][0]} → {rep['trained_on'][1]}")

st.subheader("Forecast: the next trading days")
days = st.slider("Trading days to forecast", min_value=6, max_value=24, value=12, step=6,
                 help="Each forecast is fed back in as history for the next day, so uncertainty grows with distance.")
future = D.sales_forecast(days)
hist = D.load_daily_history()["net_sales"].iloc[-36:]
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(future["date"]) + list(future["date"])[::-1],
                         y=list(future["high_80"]) + list(future["low_80"])[::-1], fill="toself",
                         fillcolor="rgba(217,89,38,0.18)", line=dict(width=0), name="80% range", hoverinfo="skip"))
C.line(fig, hist.index, hist.to_numpy(), "Actual (last 6 weeks)", C.BLUE, width=2.5, markers=True)
C.line(fig, future["date"], future["forecast"], "Forecast", C.ORANGE, width=2.5, markers=True)
st.plotly_chart(C.style(fig, height=380, money_y=True), width="stretch")
c1, c2, c3 = st.columns(3)
c1.metric("Forecast total", ui.money(future["forecast"].sum()), help=f"Sum over the next {days} trading days.")
c2.metric("Average per trading day", ui.money(future["forecast"].mean()))
c3.metric("Forecast period", f"{future['date'].min():%d %b} – {future['date'].max():%d %b %Y}")
ui.note("Saturdays are skipped (the business never trades then). Public holidays are not known to the model: last "
        "year the business closed 24 Dec – 3 Jan, so treat forecasts for that period with care.")
with st.expander("Forecast table"):
    st.dataframe(future.assign(date=future["date"].dt.strftime("%a %d %b %Y")).rename(columns={
        "date": "Date", "weekday": "Day", "forecast": "Forecast", "low_80": "Low (80%)", "high_80": "High (80%)"})
        .style.format({"Forecast": "£{:,.0f}", "Low (80%)": "£{:,.0f}", "High (80%)": "£{:,.0f}"}),
        hide_index=True, width="stretch")
    st.download_button("Download forecast (CSV)", future.to_csv(index=False).encode("utf-8"),
                       file_name="sales_forecast.csv", mime="text/csv")

st.subheader("How good is the model? Actual vs forecast, test period")
others = [c for c in preds.columns if c not in ("actual", champion_label)]
default = others.index(base["model"]) if base["model"] in others else 0
compare = st.selectbox("Compare the champion with", others, index=default)
fig = go.Figure()
C.line(fig, preds.index, preds["actual"], "Actual", C.BLUE, width=2.5, markers=True)
C.line(fig, preds.index, preds[champion_label], f"Champion: {champion_label}", C.ORANGE, width=2)
C.line(fig, preds.index, preds[compare], compare, C.AQUA, dash="dash", width=1.5)
st.plotly_chart(C.style(fig, height=400, money_y=True), width="stretch")
ui.note("Every model under-forecasts the November peak: with one year of data, no model has seen an autumn ramp-up "
        "before. Single spikes (e.g. 14 Nov, £110K) come from individual wholesale orders no feature can predict.")

left, right = st.columns([3, 2])
with left:
    st.subheader("All models on the test period")
    show = comp.assign(Model=[m if f == "Baseline" else f"{f} ({m})" for f, m in zip(comp["family"], comp["model"])])
    show = show[["Model", "family", "MAE", "RMSE", "R2", "WAPE", "Bias"]].rename(columns={"family": "Type"})
    st.dataframe(show.style.format({"MAE": "£{:,.0f}", "RMSE": "£{:,.0f}", "R2": "{:.2f}", "WAPE": "{:.1f}%",
                                    "Bias": "£{:,.0f}"}), hide_index=True, width="stretch")
    ui.note("The test period is scored once. The champion was chosen by cross-validation BEFORE this table "
            "existed, so a model that happens to score better here is not switched in (that would be test-set leakage).")
with right:
    st.subheader("What drives the forecast")
    imp = pd.DataFrame(rep["permutation_importance_top10"]).head(8)
    st.plotly_chart(C.hbar(imp["feature"], imp["mae_increase"], height=330,
                           text_fmt=lambda v: f"+£{v:,.0f}"), width="stretch")
    ui.note("Increase in error when a feature is shuffled. Weekday and same-weekday-last-week matter most.")

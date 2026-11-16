import plotly.graph_objects as go
import streamlit as st

from components import charts as C
from components import data as D
from components import ui

ui.page("Financial Trends", "How revenue, returns and seasonality evolve over time", "📈")
key = D.sidebar_filters()
d = D.daily(key)
ui.empty_guard(len(d), "trading days")

st.subheader("Daily net sales")
fig = go.Figure()
C.line(fig, d.index, d["net_sales"], "Daily net sales", C.BLUE_LIGHT, width=1)
C.line(fig, d.index, d["rolling_mean"], "7-trading-day average", C.BLUE, width=2.5)
st.plotly_chart(C.style(fig, height=360, money_y=True), width="stretch")
ui.note("Single days swing a lot (one wholesale order can add £30K); the 7-day average shows the underlying trend.")

m = D.monthly(key)
labels = [p.strftime("%b %Y") for p in m.index]
left, right = st.columns(2)
with left:
    st.subheader("Net sales per trading day")
    fig = go.Figure(go.Bar(x=labels, y=m["net_per_trading_day"], marker_color=C.BLUE,
                           customdata=m["trading_days"],
                           hovertemplate="%{x}<br>£%{y:,.0f} per trading day<br>%{customdata} trading days<extra></extra>"))
    st.plotly_chart(C.style(fig, height=320, money_y=True, legend=False), width="stretch")
    ui.note("Per trading day is fair across months with different numbers of open days (e.g. December 2011).")
with right:
    st.subheader("Return rate by month")
    fig = go.Figure(go.Bar(x=labels, y=m["genuine_cancellation_rate"], marker_color=C.ORANGE,
                           hovertemplate="%{x}<br>Returns: %{y:.1%} of gross sales<extra></extra>"))
    st.plotly_chart(C.style(fig, height=320, pct_y=True, legend=False), width="stretch")
    ui.note("Order-entry corrections are excluded, so this is the rate of genuine customer returns.")

st.subheader("Seasonality by country")
t = D.country_month(key)
if len(t):
    cols = [p.strftime("%b %y") for p in t.columns]
    text = [[f"{v:.0%}" for v in row] for row in t.to_numpy()]
    st.plotly_chart(C.heatmap(t.to_numpy(), cols, list(t.index.astype(str)), money=False, height=360, text=text,
                              colorbar_title="Share"), width="stretch")
    ui.note("Each row is one country's yearly net sales split across months, so small countries can be "
            "compared with the UK. Most countries peak in September–November.")

st.subheader("Customer concentration")
cust = D.customers(key)
if len(cust) > 1 and cust["net_sales"].clip(lower=0).sum() > 0:
    from src.analysis import eda

    curve = eda.pareto_curve(cust["net_sales"])
    share = eda.top_share(cust["net_sales"])
    step = max(1, len(curve) // 400)
    c = curve.iloc[::step]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c["entity_share"], y=c["value_share"], name="Actual", line=dict(color=C.BLUE, width=2.5),
                             hovertemplate="Top %{x:.0%} of customers<br>hold %{y:.0%} of net sales<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect equality", line=dict(color=C.GREY, dash="dash", width=1),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0.2], y=[share], mode="markers+text", marker=dict(size=10, color=C.BLUE),
                             text=[f"  Top 20% → {share:.0%}"], textposition="middle right", showlegend=False,
                             hoverinfo="skip"))
    C.style(fig, height=360)
    fig.update_xaxes(tickformat=".0%", title="Share of customers (largest first)")
    fig.update_yaxes(tickformat=".0%", title="Share of net sales")
    st.plotly_chart(fig, width="stretch")

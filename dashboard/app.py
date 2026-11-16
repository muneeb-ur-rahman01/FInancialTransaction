"""
Retail Analytics Dashboard - Overview (home page)
=================================================
Run from the project root:
    streamlit run dashboard/app.py
"""

import plotly.graph_objects as go
import streamlit as st

from components import charts as C
from components import data as D
from components import ui

ui.page("Overview", "Online Retail transactions, Dec 2010 – Dec 2011 · net product sales = sales minus returns", "🏠")
key = D.sidebar_filters()
k = D.kpis(key)
ui.empty_guard(k["rows"], "transactions")

ui.kpi_row([
    ("Net product sales", ui.money(k["net_sales"]), "Product sales minus product returns. Postage, fees and "
                                                    "order-entry corrections excluded."),
    ("Orders", f"{k['orders']:,}", "Distinct sale invoices (order-entry corrections excluded)."),
    ("Known customers", f"{k['customers']:,}", "Customers with an ID who placed at least one order."),
    ("Median order value", ui.money(k["median_order"]), "The typical order. The mean is pulled up by a few "
                                                        "very large wholesale orders, so the median is more honest."),
    ("Return rate", f"{k['return_rate']:.1%}", "Value of product returns ÷ gross product sales."),
])

st.subheader("Monthly net sales")
m = D.monthly(key)
labels = [p.strftime("%b %Y") for p in m.index]
colors = [C.BLUE_LIGHT if partial else C.BLUE for partial in m["is_partial"]]
fig = go.Figure(go.Bar(
    x=labels, y=m["net_sales"], marker_color=colors,
    customdata=m[["trading_days", "net_per_trading_day"]].to_numpy(),
    hovertemplate="%{x}<br>Net sales £%{y:,.0f}<br>%{customdata[0]} trading days"
                  "<br>£%{customdata[1]:,.0f} per trading day<extra></extra>",
))
st.plotly_chart(C.style(fig, height=340, money_y=True, legend=False), width="stretch")
if m["is_partial"].any():
    ui.note("The lighter bar is a partial month (data ends on 9 Dec 2011). Compare it per trading day "
            "(hover the bar), not by total.")

left, right = st.columns(2)
with left:
    st.subheader("Top countries")
    c = D.countries(key)
    if "United Kingdom" in c.index and len(c) > 1:
        st.caption(f"The United Kingdom is **{c.loc['United Kingdom', 'share']:.0%}** of net sales; "
                   "the chart shows the next 10 countries.")
        c = c.drop("United Kingdom")
    top = c.head(10)
    st.plotly_chart(C.hbar(top.index.astype(str), top["net_sales"], height=380), width="stretch")
with right:
    st.subheader("Top products by net revenue")
    p = D.products(key).nlargest(10, "net_sales")
    st.plotly_chart(C.hbar(p["product"].str.slice(0, 32), p["net_sales"], height=380), width="stretch")

with st.expander("How to read this dashboard"):
    st.markdown(
        "- **Analysis pages** (Overview, Transaction Analysis, Financial Trends) follow the sidebar filters.\n"
        "- **Model pages** (Sales Forecasting, Product Demand, Model Performance) show how each model was "
        "evaluated on the final weeks of data it never saw during training.\n"
        "- **Customer Segmentation** and **Anomaly Detection** show the features and known cases that exist "
        "today; their machine-learning models arrive in Modules 7C and 7D.\n"
        "- Hover any chart for exact values."
    )

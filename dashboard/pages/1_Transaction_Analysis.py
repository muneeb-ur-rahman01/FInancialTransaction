import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components import charts as C
from components import data as D
from components import ui

ui.page("Transaction Analysis", "What kinds of transactions exist, how clean they are, and what a typical order looks like", "🧾")
key = D.sidebar_filters()
types, q = D.transaction_types(key)
ui.empty_guard(len(types), "transactions")

ui.kpi_row([
    ("Duplicate lines flagged", f"{q['duplicates']:,}", "Identical repeated lines. Flagged and kept (project decision), not deleted."),
    ("Order-entry corrections", f"{q['entry_corrections']:,}", "Sales cancelled with identical quantity and price within 1 hour - orders that never really happened."),
    ("Value of corrections", ui.money(q["entry_correction_value"]), "Gross value of those cancelled-by-mistake sales."),
    ("Rows without customer ID", f"{q['missing_customer_share']:.1%}", "Anonymous sales. Kept for sales and demand; excluded only for customer analysis."),
    ("Non-product rows", f"{q['non_product_rows']:,}", "Postage, fees, discounts and manual adjustments."),
])

left, right = st.columns([1, 1])
with left:
    st.subheader("Transaction types")
    names = types["TransactionType"].astype(str).str.replace("_", " ").str.capitalize()
    st.plotly_chart(C.hbar(names, types["rows"], money=False, height=300), width="stretch")
    st.dataframe(types.assign(amount=types["amount"].map(ui.money)).rename(
        columns={"TransactionType": "Type", "rows": "Rows", "amount": "Total amount"}), hide_index=True, width="stretch")
with right:
    st.subheader("Order value distribution")
    inv = D.invoices(key)
    v = inv["order_value"][inv["order_value"] > 0]
    ui.empty_guard(len(v), "orders")
    bins = np.logspace(np.log10(max(v.min(), 0.5)), np.log10(v.max()), 50)
    counts, edges = np.histogram(v, bins=bins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    fig = go.Figure(go.Bar(x=centers, y=counts, width=np.diff(edges), marker_color=C.BLUE,
                           customdata=np.c_[edges[:-1], edges[1:]],
                           hovertemplate="£%{customdata[0]:,.0f} – £%{customdata[1]:,.0f}<br>%{y:,} orders<extra></extra>"))
    med = v.median()
    fig.add_vline(x=med, line_dash="dash", line_width=1, annotation_text=f"median £{med:,.0f}",
                  annotation_position="top right")
    C.style(fig, height=300, legend=False)
    fig.update_xaxes(type="log", tickprefix="£", title="Order value (log scale)")
    fig.update_yaxes(title="Orders")
    st.plotly_chart(fig, width="stretch")
    ui.note(f"Mean £{v.mean():,.0f} vs median £{med:,.0f}: a few very large wholesale orders pull the mean up.")

st.subheader("When does revenue happen?")
mat = D.weekday_hour(key)
mat = mat.loc[:, (mat != 0).any(axis=0)]
st.plotly_chart(C.heatmap(mat.to_numpy(), [f"{h}:00" for h in mat.columns], list(mat.index), height=300,
                          colorbar_title="Avg £/day"), width="stretch")
ui.note("Average net sales per trading day in each weekday × hour cell. There are no Saturday transactions at all.")

st.subheader("Price vs quantity")
pb = D.price_bands(key)
fig = go.Figure(go.Bar(x=pb["UnitPrice"].astype(str), y=pb["median_quantity"], marker_color=C.BLUE,
                       text=pb["median_quantity"].map(lambda x: f"{x:g}"), textposition="outside",
                       hovertemplate="Unit price %{x}<br>Median quantity per line: %{y}<extra></extra>"))
fig.update_xaxes(title="Unit price band")
fig.update_yaxes(title="Median quantity per line")
st.plotly_chart(C.style(fig, height=300, legend=False), width="stretch")
ui.note("Cheaper items are bought in much larger quantities (Spearman ρ ≈ −0.40, Module 5).")

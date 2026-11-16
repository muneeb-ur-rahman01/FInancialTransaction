import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import charts as C
from components import data as D
from components import ui
from src.models.customer_segmentation import ACTIONS, LAPSING_DAYS, SEGMENT_ORDER, rule_based_segments
from src.models.predict import customer_segment, segment_for

ui.page("Customer Segmentation", "Customer groups learned from Recency, Frequency and Monetary value (K-Means)", "👥")
rep = D.load_report("customer_segmentation.json", "segments")
seg = D.load_segments()

order = [s for s in SEGMENT_ORDER if s in set(seg["segment"])] + sorted(set(seg["segment"]) - set(SEGMENT_ORDER))
color_of = {s: C.CATEGORICAL[i % len(C.CATEGORICAL)] for i, s in enumerate(order)}
prof = pd.DataFrame(rep["profiles"]).set_index("segment").reindex(order)
champ = prof.loc["Champions"] if "Champions" in prof.index else prof.iloc[0]
lapsing = seg[seg["lapsing"]]

ui.kpi_row([
    ("Customers", f"{len(seg):,}", f"Known customers with at least one real order, measured at {rep['snapshot']}."),
    ("Segments found", f"{rep['k']}", rep["k_reason"]),
    ("Champions' revenue share", f"{champ['revenue_share']:.0%}",
     f"{int(champ['customers']):,} customers ({champ['customer_share']:.0%}) bring in this share of net sales."),
    ("Separation (silhouette)", f"{rep['silhouette']:.2f}", "−1..1, higher = better separated. Rule-based segments score far lower."),
    ("Lapsing valuable customers", f"{len(lapsing):,}",
     f"Champions / Loyal regulars with no order for {LAPSING_DAYS}+ days - the win-back list below."),
])

left, right = st.columns(2)
with left:
    st.subheader("Customers vs revenue")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=prof.index, x=prof["customer_share"], name="Share of customers", orientation="h",
                         marker_color=C.GREY, hovertemplate="%{y}<br>%{x:.1%} of customers<extra></extra>"))
    fig.add_trace(go.Bar(y=prof.index, x=prof["revenue_share"], name="Share of revenue", orientation="h",
                         marker_color=C.BLUE, hovertemplate="%{y}<br>%{x:.1%} of net sales<extra></extra>"))
    C.style(fig, height=330)
    fig.update_layout(barmode="group", bargap=0.3)
    fig.update_xaxes(tickformat=".0%", showgrid=True)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")
with right:
    st.subheader("Segment shapes")
    cent = pd.DataFrame(rep["standardised_centroids"]).set_index("segment").reindex(order)
    fig = go.Figure()
    axis = ["Recency (recent = high)", "Frequency", "Monetary"]
    for s in order:
        fig.add_trace(go.Scatter(x=axis, y=cent.loc[s, ["R", "F", "M"]], name=s, mode="lines+markers",
                                 line=dict(color=color_of[s], width=2.5), marker=dict(size=9),
                                 hovertemplate=f"{s}<br>%{{x}}: %{{y:+.2f}} std from average<extra></extra>"))
    fig.add_hline(y=0, line_dash="dash", line_width=1, line_color=C.GREY)
    fig.update_yaxes(title="Standardised (0 = average customer)")
    st.plotly_chart(C.style(fig, height=330), width="stretch")

st.subheader("Recency vs spend")
fig = go.Figure()
for s in order:
    part = seg[seg["segment"] == s]
    fig.add_trace(go.Scattergl(
        x=part["recency_days"], y=part["monetary"].clip(lower=1), mode="markers", name=f"{s} ({len(part):,})",
        marker=dict(size=6, color=color_of[s], opacity=0.65, line=dict(width=0.4, color=C.SURFACE)),
        customdata=np.c_[part["CustomerID"], part["frequency"], part["country"]],
        hovertemplate=f"{s}<br>Customer %{{customdata[0]}} · %{{customdata[2]}}<br>%{{x}} days since last order"
                      "<br>%{customdata[1]} orders · £%{y:,.0f} net spend<extra></extra>"))
fig.add_vline(x=LAPSING_DAYS, line_dash="dot", line_width=1, line_color=C.GREY,
              annotation_text=f"{LAPSING_DAYS} days", annotation_position="top")
C.style(fig, height=460)
fig.update_xaxes(title="Days since last order (lower = more recent)", showgrid=True, gridwidth=0.6)
fig.update_yaxes(type="log", title="Net spend (log scale)", tickprefix="£")
st.plotly_chart(fig, width="stretch")
ui.note("Click legend entries to hide or show segments. Valuable customers to the right of the dotted line are "
        "on the win-back list.")

st.subheader("Segment profiles and actions")
table = prof[["customers", "median_recency_days", "median_orders", "median_spend", "median_order_value",
              "revenue_share", "lapsing"]].assign(action=[ACTIONS.get(s.split(" (")[0], "") for s in prof.index])
st.dataframe(table.reset_index().rename(columns={
    "segment": "Segment", "customers": "Customers", "median_recency_days": "Median days since order",
    "median_orders": "Median orders", "median_spend": "Median spend", "median_order_value": "Median order value",
    "revenue_share": "Revenue share", "lapsing": "Lapsing", "action": "Recommended action"}).style.format({
    "Median days since order": "{:.0f}", "Median orders": "{:.0f}", "Median spend": "£{:,.0f}",
    "Median order value": "£{:,.0f}", "Revenue share": "{:.1%}", "Lapsing": "{:.0f}"}),
    hide_index=True, width="stretch")

st.subheader(f"Win-back list · {len(lapsing):,} valuable customers silent for {LAPSING_DAYS}+ days")
wb = lapsing.sort_values("monetary", ascending=False)[["CustomerID", "segment", "country", "recency_days",
                                                       "frequency", "monetary", "last_purchase"]]
st.dataframe(wb.rename(columns={"segment": "Segment", "country": "Country", "recency_days": "Days silent",
                                "frequency": "Orders", "monetary": "Lifetime net spend",
                                "last_purchase": "Last order"}).style.format(
    {"Lifetime net spend": "£{:,.0f}", "Last order": "{:%d %b %Y}"}), hide_index=True, width="stretch", height=300)
st.download_button("Download win-back list (CSV)", wb.to_csv(index=False).encode("utf-8"),
                   file_name="win_back_list.csv", mime="text/csv")

st.subheader("Look up a customer")
look, whatif = st.columns(2)
with look:
    cid = st.number_input("Customer ID", min_value=0, value=int(seg["CustomerID"].iloc[0]), step=1)
    try:
        r = customer_segment(cid)
        st.markdown(f"**{r['segment']}**{' · ⚠️ lapsing - on the win-back list' if r['lapsing'] else ''}  \n"
                    f"{r['country']} · last order {r['recency_days']} days ago · {r['orders']} orders · "
                    f"net spend {ui.money(r['net_spend'])}  \n*Action:* {r['action']}")
    except KeyError as e:
        st.warning(e.args[0])
with whatif:
    st.markdown("**What if?** Segment for any profile")
    a, b, c = st.columns(3)
    rec = a.number_input("Days since last order", 0, 400, 30)
    freq = b.number_input("Orders", 1, 300, 3)
    mon = c.number_input("Net spend (£)", 0, 300_000, 900, step=100)
    r = segment_for(rec, freq, mon)
    st.markdown(f"→ **{r['segment']}**  \n*Action:* {r['action']}")

with st.expander("How the segments were chosen - and why not simple RFM rules?"):
    algos = pd.DataFrame(rep["algorithm_comparison"])
    st.markdown(f"**Number of segments.** {rep['k_reason']}")
    st.dataframe(pd.DataFrame(rep["k_selection"])[["k", "silhouette", "stability_ari", "smallest_cluster_share"]]
                 .rename(columns={"k": "Segments (k)", "silhouette": "Silhouette", "stability_ari": "Stability (ARI)",
                                  "smallest_cluster_share": "Smallest segment"})
                 .style.format({"Silhouette": "{:.3f}", "Stability (ARI)": "{:.2f}", "Smallest segment": "{:.1%}"}),
                 hide_index=True, width="stretch")
    st.markdown("**Method comparison** at the chosen number of segments (same standardised RFM space):")
    st.dataframe(algos.style.format({"silhouette": "{:.3f}", "davies_bouldin": "{:.2f}"}), hide_index=True,
                 width="stretch")
    st.markdown("Higher silhouette and lower Davies-Bouldin = better separated. Fixed RFM score rules are the "
                "baseline; K-Means separates customers far better.")
    base = rule_based_segments(seg)
    st.markdown("**How the rule-based segments map onto the K-Means segments** (customers):")
    st.dataframe(pd.crosstab(base.rename("Rule-based"), seg["segment"].rename("K-Means")).reindex(columns=order),
                 width="stretch")

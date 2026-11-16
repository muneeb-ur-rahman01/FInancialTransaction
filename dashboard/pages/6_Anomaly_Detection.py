import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import charts as C
from components import data as D
from components import ui
from src.models.predict import score_lines

ui.page("Anomaly Detection", "Order lines that look like keying errors - a review queue to check before confirming", "🚨")
rep = D.load_report("anomaly_detection.json", "anomaly")
flagged = D.load_flagged_lines()
invoices = D.load_invoice_scores()
m, prod = rep["test_metrics"], rep["production"]

ui.kpi_row([
    ("Detector", rep["method"], f"{rep['family']} - chosen on the validation period ({rep['selection']})."),
    ("Known errors caught", f"{m['recall']:.0%}",
     f"Recall on the test period: {m['tp']} of {m['tp'] + m['fn']} known entry errors were in the alert queue."),
    ("Alert precision", f"{m['precision']:.1%}",
     "Share of alerts that are KNOWN errors. A lower bound: the queue also holds real errors the answer key misses."),
    ("Better than random", f"{m['lift']:.0f}×", f"Precision ÷ base rate ({m['base_rate']:.2%} of lines are known errors)."),
    ("Lines in review queue", f"{prod['lines_flagged']:,}",
     f"Top {rep['flag_rate']:.0%} of lines across all data, in {prod['orders_flagged']:,} orders."),
])
st.info("**How to read this.** The 424 'known errors' are sales cancelled with identical quantity and price within "
        "one hour. Many are customers changing their mind on perfectly normal lines - nothing to detect. The "
        "detector catches the extreme ones (e.g. 80,995 units keyed instead of a handful) and also surfaces real "
        "errors the answer key misses, such as a £649.50 unit price on a basket that normally sells for under £10.",
        icon="💡")

st.subheader("Every order over time")
inv = invoices[invoices["order_value"] > 0]
groups = [
    ("Normal order", ~inv["flagged"] & ~inv["known_error"], C.BLUE_LIGHT, 4, 0.35),
    ("Flagged by detector", inv["flagged"] & ~inv["known_error"], C.BLUE, 6, 0.7),
    ("Known error - missed", ~inv["flagged"] & inv["known_error"], C.GREY, 8, 0.9),
    ("Known error - caught", inv["flagged"] & inv["known_error"], C.ORANGE, 11, 1.0),
]
fig = go.Figure()
for name, mask, color, size, alpha in groups:
    part = inv[mask]
    fig.add_trace(go.Scattergl(
        x=part["date"], y=part["order_value"], mode="markers", name=f"{name} ({len(part):,})",
        marker=dict(size=size, color=color, opacity=alpha, line=dict(width=0.5, color=C.SURFACE)),
        customdata=np.c_[part["InvoiceNo"], part["flagged_lines"], part["lines"]],
        hovertemplate="Invoice %{customdata[0]}<br>%{x|%d %b %Y %H:%M}<br>£%{y:,.0f}"
                      "<br>%{customdata[1]} of %{customdata[2]} lines flagged<extra>" + name + "</extra>"))
C.style(fig, height=430)
fig.update_yaxes(type="log", tickprefix="£", title="Order value (log scale)")
st.plotly_chart(fig, width="stretch")
ui.note("Click legend entries to show or hide groups. Grey dots are known errors that look like ordinary orders.")

left, right = st.columns([2, 3])
with left:
    st.subheader("Confusion matrix (test period)")
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    text = [[f"{v:,}" for v in row] for row in cm]
    fig = go.Figure(go.Heatmap(z=np.log1p(cm), x=["Not flagged", "Flagged"], y=["Normal line", "Known error"],
                               colorscale=C.SEQUENTIAL, showscale=False, text=text, texttemplate="%{text}",
                               textfont=dict(size=16), xgap=3, ygap=3, customdata=cm,
                               hovertemplate="%{y} · %{x}: %{customdata:,} lines<extra></extra>"))
    C.style(fig, height=300, legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Precision {m['precision']:.1%} · Recall {m['recall']:.1%} · F1 {m['f1']:.3f} · "
               f"ROC-AUC {m['roc_auc']:.3f} · Average precision {m['average_precision']:.4f}")
with right:
    st.subheader("Precision vs recall (test period)")
    fig = go.Figure()
    for (name, curve), color in zip(rep["pr_curves_test"].items(), C.CATEGORICAL):
        width = 3 if name == rep["method"] else 1.5
        fig.add_trace(go.Scatter(x=curve["recall"], y=curve["precision"], name=name, mode="lines",
                                 line=dict(color=color, width=width),
                                 hovertemplate=f"{name}<br>recall %{{x:.0%}}<br>precision %{{y:.2%}}<extra></extra>"))
    fig.add_hline(y=m["base_rate"], line_dash="dash", line_width=1, line_color=C.GREY,
                  annotation_text="random", annotation_position="bottom right")
    C.style(fig, height=300)
    fig.update_xaxes(tickformat=".0%", title="Recall")
    fig.update_yaxes(type="log", tickformat=".1%", title="Precision (log)")
    st.plotly_chart(fig, width="stretch")

st.subheader("Check an order line")
with st.form("check_line"):
    a, b, c, d = st.columns([2, 1, 1, 1], vertical_alignment="bottom")
    code = a.text_input("Stock code", value="85123A")
    qty = b.number_input("Quantity", min_value=1, value=5000, step=1)
    price = c.number_input("Unit price (£)", min_value=0.01, value=2.55, step=0.05, format="%.2f")
    d.form_submit_button("Check line", width="stretch")
r = score_lines(pd.DataFrame({"StockCode": [code], "Quantity": [qty], "UnitPrice": [price]})).iloc[0]
if r["flagged"]:
    st.error(f"**Suspicious - check before confirming.** {r['reason'][0].upper() + r['reason'][1:]}.", icon="🚨")
else:
    st.success(f"**No alert.** {r['reason'].capitalize()}.", icon="✅")

st.subheader("Review queue")
c1, c2 = st.columns([1, 1])
only_unknown = c1.toggle("Hide lines already known as errors", value=False)
months = sorted(flagged["InvoiceDate"].dt.to_period("M").astype(str).unique(), reverse=True)
month = c2.selectbox("Month", ["All months"] + months)
queue = flagged
if only_unknown:
    queue = queue[~queue["IsEntryCorrection"]]
if month != "All months":
    queue = queue[queue["InvoiceDate"].dt.to_period("M").astype(str) == month]
show = queue.assign(usual=queue["usual_qty"].map(lambda v: "first sales" if pd.isna(v) else f"{v:,.0f}"),
                    times=queue["qty_vs_usual"].map(lambda v: "" if pd.isna(v) else f"{v:,.1f}×"))
st.dataframe(show[["InvoiceDate", "InvoiceNo", "ProductName", "Quantity", "usual", "times", "UnitPrice", "TotalAmount",
                   "Country", "IsEntryCorrection"]].rename(columns={
    "InvoiceDate": "Date", "InvoiceNo": "Invoice", "ProductName": "Product", "usual": "Usual qty",
    "times": "× usual", "UnitPrice": "Unit price", "TotalAmount": "Line value", "IsEntryCorrection": "Known error"})
    .style.format({"Date": "{:%d %b %Y %H:%M}", "Unit price": "£{:,.2f}", "Line value": "£{:,.0f}", "Quantity": "{:,}"}),
    hide_index=True, width="stretch", height=360)
st.download_button("Download review queue (CSV)", queue.to_csv(index=False).encode("utf-8"),
                   file_name="anomaly_review_queue.csv", mime="text/csv")

with st.expander("How the detector was chosen"):
    comp = pd.DataFrame(rep["comparison"])
    st.markdown(f"Periods: fit {rep['periods']['fit']}, validation {rep['periods']['validation']}, "
                f"test {rep['periods']['test']}. The champion has the highest **validation** average precision; "
                "the test period was scored once afterwards.")
    st.dataframe(comp[["family", "method", "val_average_precision", "test_average_precision", "test_roc_auc",
                       "test_precision", "test_recall", "test_f1", "test_lift"]].style.format({
        "val_average_precision": "{:.4f}", "test_average_precision": "{:.4f}", "test_roc_auc": "{:.3f}",
        "test_precision": "{:.1%}", "test_recall": "{:.1%}", "test_f1": "{:.3f}", "test_lift": "{:.0f}×"}),
        hide_index=True, width="stretch")
    st.markdown("Simple rules beat Isolation Forest (distracted by legitimately rare lines) and supervised models "
                "(too few, too varied examples - XGBoost memorised the training errors). Detection works per "
                "**line**: at order level a single mistyped line is invisible in the totals.")

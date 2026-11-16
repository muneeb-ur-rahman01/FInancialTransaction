import pandas as pd
import streamlit as st

from components import data as D
from components import ui
from components.data import MODEL_REPORTS_DIR

ui.page("Model Performance", "How every model was evaluated, compared and chosen", "🧪")

st.markdown(
    "**Evaluation rules used for every model**\n"
    "1. Data is split **by time**: models train on the past and are tested on the most recent weeks.\n"
    "2. **Simple rules (baselines) come first.** A model is only useful if it beats them.\n"
    "3. The winner is chosen by **cross-validation on the training period only**; the test period is scored once.\n"
    "4. Preprocessing is fitted inside each training fold, so no information leaks from the future."
)

tab_over, tab_sales, tab_demand, tab_seg, tab_anom = st.tabs(["Overview", "Sales forecasting (7A)",
                                                               "Product demand (7B)", "Customer segmentation (7C)",
                                                               "Anomaly detection (7D)"])


def fmt_metric(value, metric):
    if "Recall" in metric:
        return f"{value:.1%}"
    if "£" in metric:
        return ui.money(value)
    if "Silhouette" in metric:
        return f"{value:.3f}"
    return f"{value:.1f}"


with tab_over:
    summary = D.load_report("model_evaluation_summary.json", "report")
    for t in summary["tasks"]:
        better = (t["champion_value"] < t["reference_value"]) if t["lower_is_better"] else \
            (t["champion_value"] > t["reference_value"])
        with st.container(border=True):
            left, right = st.columns([3, 2])
            left.markdown(f"#### {t['task']}\n**Champion:** {t['champion']}  \n{t['verdict']}  \n"
                          f"*Limitation:* {t['limitation']}")
            right.metric(t["metric"], fmt_metric(t["champion_value"], t["metric"]),
                         f"{'better' if better else 'worse'} than {t['reference']} "
                         f"({fmt_metric(t['reference_value'], t['metric'])})",
                         delta_color="normal" if better else "inverse", delta_arrow="off")
            right.caption(t["evaluation"])
    st.caption("Full report: reports/model_reports/model_evaluation_summary.md")

with tab_sales:
    rep = D.load_report("sales_forecasting.json", "sales")
    ui.kpi_row([
        ("Champion", f"{rep['model_family']}", rep["model_name"]),
        ("Test MAE", ui.money(rep["test_metrics"]["MAE"]), "Average daily error on the test period."),
        ("Test WAPE", f"{rep['test_metrics']['WAPE']:.1f}%", "Total absolute error ÷ total sales."),
        ("Test R²", f"{rep['test_metrics']['R2']:.2f}", "Share of day-to-day variation explained."),
    ])
    st.caption(f"Cross-validation: {rep['cv_setup']} · test period {rep['test_period'][0]} → {rep['test_period'][1]}")
    st.markdown("**Best configuration of each model family (cross-validation)**")
    st.dataframe(pd.DataFrame(rep["finalists_cv"]).style.format(
        {"cv_MAE": "£{:,.0f}", "cv_MAE_std": "£{:,.0f}", "cv_RMSE": "£{:,.0f}", "cv_WAPE": "{:.1f}%"}),
        hide_index=True, width="stretch")
    st.markdown("**Test period (scored once)**")
    st.dataframe(pd.DataFrame(rep["test_comparison"]).style.format(
        {"MAE": "£{:,.0f}", "RMSE": "£{:,.0f}", "R2": "{:.2f}", "MAPE": "{:.1f}%", "WAPE": "{:.1f}%",
         "Bias": "£{:,.0f}", "Bias_pct": "{:+.1f}%"}), hide_index=True, width="stretch")
    with st.expander("All 26 candidates (cross-validation)"):
        st.dataframe(pd.read_csv(MODEL_REPORTS_DIR / "sales_forecasting_cv_all_candidates.csv"),
                     hide_index=True, width="stretch")

with tab_demand:
    rep = D.load_report("demand_forecasting.json", "demand")
    ui.kpi_row([
        ("Champion", rep["model_name"], rep["model_family"]),
        ("Test WAPE", f"{rep['test_metrics']['WAPE']:.1f}%", "Total absolute error ÷ total units."),
        ("Product-level bias", f"{rep['test_metrics']['ProductBias_pct']:.1f}%", "Σ |each product's total error| ÷ total units."),
        ("Score", f"{rep['test_metrics']['score']:.1f}", "WAPE + product-level bias (lower is better)."),
    ])
    st.success(rep["selection_decision"], icon="⚖️")
    st.caption(f"Metric: {rep['metric']} · {rep['cv_setup']} · test weeks {rep['test_weeks'][0]} → {rep['test_weeks'][1]}")
    st.markdown("**Best configuration of each model family (cross-validation)**")
    st.dataframe(pd.DataFrame(rep["finalists_cv"]).style.format(
        {c: "{:.1f}" for c in ["cv_score", "cv_score_std", "cv_WAPE", "cv_ProductBias_pct", "cv_Bias_pct"]}),
        hide_index=True, width="stretch")
    st.markdown("**Test weeks (scored once)**")
    st.dataframe(pd.DataFrame(rep["test_comparison"]).style.format(
        {"MAE": "{:,.1f}", "RMSE": "{:,.1f}", "R2": "{:.2f}", "WAPE": "{:.1f}%", "Bias": "{:+.1f}",
         "Bias_pct": "{:+.1f}%", "ProductBias_pct": "{:.1f}%", "score": "{:.1f}"}), hide_index=True, width="stretch")
    with st.expander("All 19 candidates (cross-validation)"):
        st.dataframe(pd.read_csv(MODEL_REPORTS_DIR / "demand_forecasting_cv_all_candidates.csv"),
                     hide_index=True, width="stretch")

with tab_seg:
    rep = D.load_report("customer_segmentation.json", "segments")
    algos = pd.DataFrame(rep["algorithm_comparison"])
    baseline = algos[algos["method"].str.contains("baseline")].iloc[0]
    ui.kpi_row([
        ("Method", "K-Means", "Unsupervised clustering on log-transformed, standardised RFM."),
        ("Segments (k)", f"{rep['k']}", rep["selection_rule"]),
        ("Silhouette", f"{rep['silhouette']:.3f}", f"Separation. Rule-based baseline: {baseline['silhouette']:.3f}."),
        ("Stability (ARI)", f"{rep['stability_ari']:.2f}", "Agreement with refits on bootstrap resamples (1 = identical)."),
    ])
    st.success(rep["k_reason"], icon="⚖️")
    st.caption("No labels exist for segmentation, so models are judged on separation (silhouette, Davies-Bouldin) "
               "and on stability under resampling, not on accuracy.")
    st.markdown("**Choosing the number of segments**")
    st.dataframe(pd.DataFrame(rep["k_selection"]).style.format(
        {"inertia": "{:,.0f}", "silhouette": "{:.3f}", "davies_bouldin": "{:.2f}", "calinski_harabasz": "{:,.0f}",
         "stability_ari": "{:.3f}", "stability_ari_std": "{:.3f}", "smallest_cluster_share": "{:.1%}"}),
        hide_index=True, width="stretch")
    st.markdown("**Baseline and algorithms at the chosen k**")
    st.dataframe(algos.style.format({"silhouette": "{:.3f}", "davies_bouldin": "{:.2f}"}), hide_index=True,
                 width="stretch")

with tab_anom:
    rep = D.load_report("anomaly_detection.json", "anomaly")
    m = rep["test_metrics"]
    ui.kpi_row([
        ("Detector", rep["method"], rep["family"]),
        ("Precision", f"{m['precision']:.1%}", "Share of alerts that are known errors (lower bound)."),
        ("Recall", f"{m['recall']:.1%}", "Share of known errors that were alerted."),
        ("F1", f"{m['f1']:.3f}", "Harmonic mean of precision and recall at the alert threshold."),
        ("Lift", f"{m['lift']:.0f}×", "Precision ÷ base rate."),
    ])
    st.success(f"Selection: {rep['selection']}. Alert threshold = top {rep['flag_rate']:.0%} of training-period "
               "scores, so the test period never influenced it.", icon="⚖️")
    st.caption(f"Periods: fit {rep['periods']['fit']} · validation {rep['periods']['validation']} · "
               f"test {rep['periods']['test']}. Positives are rare ({m['base_rate']:.2%} of lines), so accuracy "
               "is not reported: flagging nothing would score 99.9%.")
    st.markdown("**Confusion matrix at the alert threshold (test period)**")
    st.dataframe(pd.DataFrame([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], index=["Normal line", "Known error"],
                              columns=["Not flagged", "Flagged"]).style.format("{:,}"), width="stretch")
    st.markdown("**All detectors**")
    st.dataframe(pd.DataFrame(rep["comparison"]).drop(columns=["test_tn"]).style.format(
        {c: "{:.4f}" for c in ["val_average_precision", "val_roc_auc", "test_average_precision", "test_roc_auc",
                               "test_precision", "test_recall", "test_f1", "test_flag_rate", "test_base_rate"]}
        | {"test_lift": "{:.1f}"}), hide_index=True, width="stretch")

"""Small layout helpers shared by all pages."""

import streamlit as st


CSS = """
<style>
  /* KPI cards */
  [data-testid="stMetric"] {
    background: #1f1f1d; border: 1px solid rgba(255,255,255,0.08); border-radius: 12px;
    padding: 14px 16px 10px 16px;
  }
  [data-testid="stMetricLabel"] p { color: #c3c2b7; font-size: 0.85rem; }
  [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 600; }
  /* page title with a thin accent bar */
  [data-testid="stHeading"] h1 { border-left: 4px solid #3987e5; padding-left: 14px; }
  /* charts and tables sit on cards too */
  [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
    background: #1a1a19; border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 6px;
  }
  [data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.06); }
  h2, h3 { letter-spacing: -0.01em; }
</style>
"""


def page(title, subtitle, icon="📊"):
    st.set_page_config(page_title=f"{title} · Retail Analytics", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def money(v, decimals=0):
    a = abs(v)
    if a >= 1e6:
        return f"£{v / 1e6:,.2f}M"
    if a >= 1e4:
        return f"£{v / 1e3:,.1f}K"
    return f"£{v:,.{decimals}f}"


def kpi_row(items):
    """items: list of (label, value, help_text)."""
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)


def note(text):
    st.caption(f"ℹ️ {text}")


def empty_guard(n, what="data"):
    if n == 0:
        st.warning(f"No {what} for the selected filters. Widen the date range or clear the country filter.")
        st.stop()

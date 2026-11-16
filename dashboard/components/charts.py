"""Plotly chart helpers with one consistent, colour-blind-safe style.

Colours follow the project palette (dark-theme steps; the static report figures use the light steps):
  categorical slots in FIXED order: blue, orange, aqua  (never cycled, never by rank)
  sequential ramp (one hue, dark = low -> bright = high on the dark surface) for magnitudes/heatmaps
  grey for baselines/de-emphasis
Backgrounds are left to Streamlit's theme (.streamlit/config.toml).
Every chart has hover tooltips (Plotly default) with formatted values.
"""

import plotly.graph_objects as go

# Dark-theme steps of the validated palette (selected for a dark surface, not an automatic flip)
BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"
GREY = "#898781"
BLUE_LIGHT = "#1c5cab"            # de-emphasised blue: recedes towards the dark surface
SURFACE = "#121211"               # page background, used for marker outlines / gaps
SEQUENTIAL = [[0.0, "#0d366b"], [0.17, "#184f95"], [0.33, "#256abf"], [0.5, "#3987e5"],
              [0.67, "#6da7ec"], [0.83, "#9ec5f4"], [1.0, "#cde2fb"]]          # dark (low) -> bright (high)
CATEGORICAL = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]


def style(fig, height=360, money_y=False, money_x=False, pct_y=False, legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        hovermode="closest",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridwidth=0.6, zeroline=False)
    if money_y:
        fig.update_yaxes(tickprefix="£", tickformat="~s")
    if money_x:
        fig.update_xaxes(tickprefix="£", tickformat="~s")
    if pct_y:
        fig.update_yaxes(tickformat=".1%")
    return fig


def hbar(labels, values, money=True, color=BLUE, height=380, text_fmt=None):
    """Horizontal bars, largest on top, value labels at the bar ends."""
    labels, values = list(labels)[::-1], list(values)[::-1]
    text = [text_fmt(v) for v in values] if text_fmt else [f"£{v:,.0f}" if money else f"{v:,.0f}" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=color, text=text, textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>" + ("£%{x:,.0f}" if money else "%{x:,.0f}") + "<extra></extra>",
    ))
    style(fig, height=height, money_x=money, legend=False)
    fig.update_xaxes(showgrid=True, gridwidth=0.6)
    fig.update_yaxes(showgrid=False)
    return fig


def heatmap(z, x, y, money=True, height=320, text=None, colorbar_title=""):
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y, colorscale=SEQUENTIAL, xgap=2, ygap=2, text=text,
        texttemplate="%{text}" if text is not None else None,
        colorbar=dict(title=colorbar_title, tickprefix="£" if money else None, tickformat="~s" if money else ".0%"),
        hovertemplate="%{y} · %{x}<br>" + ("£%{z:,.0f}" if money else "%{z:.1%}") + "<extra></extra>",
    ))
    style(fig, height=height, legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def line(fig, x, y, name, color, dash=None, width=2, markers=False, money=True):
    fig.add_trace(go.Scatter(
        x=x, y=y, name=name, mode="lines+markers" if markers else "lines",
        line=dict(color=color, width=width, dash=dash), marker=dict(size=7),
        hovertemplate=f"{name}<br>%{{x|%a %d %b %Y}}<br>" + ("£%{y:,.0f}" if money else "%{y:,.0f}") + "<extra></extra>",
    ))
    return fig

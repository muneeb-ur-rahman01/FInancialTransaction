"""One shared chart style so every figure in the project looks consistent.

Colours come from a validated, colour-blind-safe palette:
  * one blue for single-series charts,
  * a light-to-dark blue ramp for magnitude (heatmaps),
  * text and axes in neutral greys, never in series colours.
"""

import matplotlib

matplotlib.use("Agg")  # render to files; no window needed

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from src.utils.paths import FIGURES_DIR  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3
BLUE_LIGHT = "#86b6ef"                               # de-emphasised blue (step 250)
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


# Diverging (for values with a sign, e.g. correlations): red <- grey 0 -> blue.
DIVERGING = LinearSegmentedColormap.from_list(
    "div_red_blue",
    ["#a32d2c", "#e34948", "#f2a3a2", "#f0efec", "#9ec5f4", "#3987e5", "#184f95"],
)


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.edgecolor": BASELINE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False,
        "lines.linewidth": 2,
    })


def money(decimals=0):
    """Axis formatter: 1234567 -> £1.2M, 45000 -> £45K, 12.5 -> £12.50."""
    def fmt(x, _pos=None):
        a = abs(x)
        if a >= 1e6:
            return f"£{x / 1e6:.1f}M"
        if a >= 1e3:
            return f"£{x / 1e3:g}K"
        return f"£{x:,.{decimals}f}"
    return FuncFormatter(fmt)


def save(fig, filename):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

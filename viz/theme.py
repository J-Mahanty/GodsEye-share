"""Shared chart styling.

The palette is the one already used by share.html and the control room, so a
chart dropped into a slide next to a screenshot reads as the same system.

The categorical order is fixed and assigned by entity, never cycled: a filter
that drops a series must not repaint the survivors. It was checked with the
dataviz palette validator (light mode) and passes all five checks -- lightness
band, chroma floor, CVD separation (worst adjacent dE 13.7), normal-vision
floor and contrast. Re-run the validator before changing any value here.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#101720"
MUTED = "#5b6b7d"
EDGE = "#d7e0ea"
GRID = "#e8edf3"
PAPER = "#ffffff"

CATEGORICAL = ["#2563eb", "#c2410c", "#0d9488", "#a855f7", "#be123c"]

# Status colours are reserved: they mean a state, never "series 4", and every
# chart that uses them also labels the segment in words.
GOOD = "#0d9488"
WRONG = "#be123c"
NEUTRAL = "#94a3b8"

# The control room's congestion ramp, so a chart and the map agree on colour.
LEVEL = {"free": "#1f8f4e", "moderate": "#d4a72c",
         "heavy": "#d1690f", "severe": "#c92a2f"}

SEVERITY = {"critical": "#be123c", "high": "#c2410c",
            "medium": "#b58a00", "low": "#5b6b7d"}

DPI = 200


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.edgecolor": EDGE,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.35,
    })


def despine(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def headline(ax, title: str, subtitle: str) -> None:
    """Title with one or two recessive lines under it, carrying the argument in
    words. The title's pad is computed from the subtitle's line count so the two
    never overlap however long the subtitle runs."""
    lines = subtitle.count("\n") + 1
    ax.annotate(subtitle, xy=(0, 1), xytext=(0, 8), xycoords="axes fraction",
                textcoords="offset points", ha="left", va="bottom",
                fontsize=9.5, color=MUTED, linespacing=1.45)
    ax.set_title(title, pad=8 + lines * 15 + 6)


def source(fig, text: str, y: float = -0.004) -> None:
    """Provenance footer, so a judge's question traces back to a file."""
    fig.text(0.005, y, text, ha="left", va="top", fontsize=7.5, color=MUTED)


def legend_below(ax, fig, ncols: int, y: float = -0.02, **kw):
    """Legend under the plot, anchored in FIGURE coords so it and the source
    footer cannot land on each other whatever the axes height is."""
    return ax.legend(loc="upper center", ncols=ncols,
                     bbox_to_anchor=(0.5, y), bbox_transform=fig.transFigure, **kw)

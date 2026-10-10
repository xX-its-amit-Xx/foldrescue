"""Shared plotting helpers for the cookbook.

PALETTE PROVENANCE
==================
The categorical slots below are the first four of a palette validated for
colour-vision deficiency separation. Validator output on the adjacent pairlist
(light surface ``#fcfcfb``):

    Lightness band      PASS  all 4 inside L 0.43-0.77
    Chroma floor        PASS  all 4 >= 0.1
    CVD separation      PASS  worst adjacent dE 9.1 (protan), tritan 27.0
    Normal-vision floor PASS  worst adjacent dE 22.9
    Contrast vs surface WARN  aqua 2.74, yellow 2.11 -> relief required

The contrast WARN is not dismissable: it obliges either visible labels or a
table view. Both are supplied -- every figure in the cookbook is accompanied by
the printed dataframe it was drawn from, and bars carry direct value labels. So
identity is never conveyed by colour alone.

Rules followed throughout (not taste, they are the method):
* categorical hues assigned in fixed slot order, never cycled;
* one y-axis per chart, never a dual axis;
* sequential magnitude uses one hue light->dark;
* a legend whenever two or more series are present;
* recessive grid and axes, thin marks;
* every figure derived from generated data is WATERMARKED SYNTHETIC.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Select a headless backend ONLY outside a Jupyter kernel. Forcing "Agg"
# unconditionally means plt.show() renders nothing under nbclient, so the
# executed notebooks commit with text output and no figures -- which is how
# this file shipped for one iteration.
try:  # pragma: no cover - depends on execution context
    get_ipython()  # type: ignore[name-defined]  # noqa: B018
except NameError:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots, fixed order. Do not cycle; a fifth series folds into
# "other" or becomes a small multiple.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8982"
GRID = "#e3e2dd"

# Stable colour per entity, so a figure that drops a series never repaints the
# survivors.
TIER_COLOR = {
    "tier1_gbm": SERIES[0],
    "tier2_fusion": SERIES[1],
    "tier3_structure": SERIES[2],
    "baseline": INK_MUTED,
}
BLOCK_COLOR = {
    "stability": SERIES[0],
    "ligand": SERIES[1],
    "structural": SERIES[2],
    "interaction": SERIES[3],
}

FIGURE_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"


def use_style() -> None:
    """Recessive axes, light surface, readable defaults."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 9.5,
        }
    )


def watermark(fig, text: str = "SYNTHETIC DATA — pipeline demonstration, not a biological result") -> None:
    """Stamp the figure so it cannot be lifted out of context.

    A figure travels: into a slide, a grant, a thread. The notebook's caveat
    does not travel with it. The watermark does.
    """
    fig.text(
        0.5,
        0.5,
        "SYNTHETIC",
        fontsize=46,
        color="#d94f4f",
        alpha=0.11,
        ha="center",
        va="center",
        rotation=28,
        weight="bold",
        zorder=0,
    )
    fig.text(
        0.005,
        -0.035,
        text,
        fontsize=7.2,
        color="#b4453f",
        ha="left",
        va="top",
        style="italic",
    )


def save(fig, name: str, *, synthetic: bool = True) -> Path:
    """Watermark (if synthetic) and write under results/figures/."""
    if synthetic:
        watermark(fig)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path)
    print(f"wrote {path.relative_to(FIGURE_DIR.parent.parent)}")
    return path


def label_bars(ax, bars, values, *, fmt: str = "{:.3f}", dy: float = 0.004) -> None:
    """Direct value labels -- the relief the contrast WARN obliges.

    Labels wear text ink, never the series colour: a coloured mark beside the
    number carries identity, the number itself stays readable.
    """
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, bar.get_height() + dy),
            ha="center",
            va="bottom",
            fontsize=7.6,
            color=INK_SECONDARY,
        )

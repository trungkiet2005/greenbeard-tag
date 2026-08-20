r"""Publication figure style and reusable plotting helpers.

Every figure in the manuscript is saved at exactly :data:`FIG_WIDTH` inches and
included at ``\linewidth``, so all figures are reduced by the same factor on the
page.  Font sizes are therefore comparable across figures and are drawn from the
single scale in :data:`FS` rather than chosen per panel.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

#: Width in inches of every saved figure.  The text block of the manuscript is
#: 6.27 in, so figures are reduced by 0.91 uniformly.
FIG_WIDTH = 6.9

#: The only font sizes used anywhere.  Values are points before the uniform
#: 0.91 reduction, so the smallest text on the page is about 6.4 pt.
FS = {
    "title": 9.5,   # panel titles
    "label": 9.0,   # axis labels, simplex vertex labels
    "tick": 8.0,    # tick labels
    "legend": 8.0,  # legend entries
    "annot": 7.5,   # in-axes annotations
    "tiny": 7.0,    # dense in-axes annotations (cell values, threshold marks)
}

#: Colour-blind-safe qualitative palette (Okabe-Ito), with the race designs
#: ordered so that a legend reads as a gradient from safe to unsafe and the
#: identity classes keyed by their display names.
PALETTE = {
    "AS": "#0072B2",
    "CS": "#009E73",
    "CAS": "#E69F00",
    "AU": "#D55E00",
    "accent": "#CC79A7",
    "neutral": "#4D4D4D",
    "grid": "#D9D9D9",
    "unsafe": "#B2182B",
    "safe": "#2166AC",
    "certified club": "#0072B2",
    "certified aggressor": "#56B4E9",
    "falsebeard": "#D55E00",
    "unbadged cooperator": "#009E73",
    "unbadged other": "#999999",
    "club": "#0072B2",
    "mimic": "#CC79A7",
    "forger": "#D55E00",
    "anarchy": "#4D4D4D",
    "baseline": "#4D4D4D",
    "federated": "#009E73",
    "scoped": "#E69F00",
}

STRATEGY_LABEL = {
    "AS": "AS (always safe)",
    "CS": "CS (conditionally safe)",
    "CAS": "CAS (conditionally unsafe)",
    "AU": "AU (always unsafe)",
}

#: Short glosses used in legends where the full label does not fit.
STRATEGY_SHORT = {
    "AS": "AS",
    "CS": "CS",
    "CAS": "CAS",
    "AU": "AU",
}

#: Display names of the identity classes, in stacking order.
CLASS_LABEL = {
    "certified club": "certified club",
    "certified aggressor": "certified aggressor",
    "falsebeard": "falsebeard",
    "unbadged cooperator": "unbadged cooperator",
    "unbadged other": "unbadged other",
}


#: Relative luminance at which black and white text have equal WCAG contrast
#: against the same background.  Below it white reads better, above it black
#: does, and a rule keyed to the *value* of a cell rather than to the colour
#: the value maps to gets this backwards on any diverging colormap.
LUMINANCE_CROSSOVER = 0.1791

#: Smallest WCAG contrast ratio accepted for text drawn over painted content.
#: 4.5 is the AA requirement for body-sized text, which is what the in-axes
#: annotations are once the figure is reduced onto the page.
MIN_CONTRAST = 4.5

#: Background luminance above which a text counts as sitting on bare paper
#: rather than on ink.
PAPER_LUMINANCE = 0.75

#: Smallest contrast accepted for a text on bare paper.  Such a label is drawn
#: in the colour of the curve it names, so it is a graphical object in the
#: sense of WCAG 1.4.11 and 3:1 is the applicable ratio; the alternative is to
#: force every keyed label to black and lose the key.
PAPER_MIN_CONTRAST = 3.0


def relative_luminance(colour) -> float:
    """WCAG relative luminance of any matplotlib colour specification."""
    red, green, blue = mpl.colors.to_rgb(colour)

    def linear(channel: float) -> float:
        return (
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
        )

    return 0.2126 * linear(red) + 0.7152 * linear(green) + 0.0722 * linear(blue)


def contrast_ratio(first, second) -> float:
    """WCAG contrast ratio between two colours, from 1 (equal) to 21."""
    lo, hi = sorted((relative_luminance(first), relative_luminance(second)))
    return (hi + 0.05) / (lo + 0.05)


def readable_on(background) -> str:
    """Black or white, whichever the eye can actually read on `background`.

    Cell labels on a heatmap must be keyed to the colour the cell was painted,
    not to the number it encodes: a diverging colormap is lightest in the
    middle of its range and darkest at both ends, so a rule of the form "white
    in the middle band" hides exactly the cells it means to show.
    """
    return "black" if relative_luminance(background) > LUMINANCE_CROSSOVER else "white"


def badge_colour(badge: str):
    """Colour of a badge type."""
    return {"G": PALETTE["safe"], "F": PALETTE["unsafe"], "N": PALETTE["neutral"]}[badge]


def use_paper_style() -> None:
    """Apply the manuscript figure style."""
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 400,
            # a fixed saved size is what keeps the on-page scale uniform, so the
            # bounding box must not be cropped to the ink
            "savefig.bbox": None,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": FS["label"],
            "axes.titlesize": FS["title"],
            "axes.labelsize": FS["label"],
            "legend.fontsize": FS["legend"],
            "xtick.labelsize": FS["tick"],
            "ytick.labelsize": FS["tick"],
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.5,
            "grid.alpha": 0.8,
            "lines.linewidth": 1.5,
            "legend.frameon": False,
            "legend.handlelength": 2.0,
            "legend.columnspacing": 1.1,
            "legend.borderaxespad": 0.2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "xtick.major.pad": 2.0,
            "ytick.major.pad": 2.0,
            "axes.labelpad": 2.5,
        }
    )


def new_figure(height: float, *, nrows: int = 1, ncols: int = 1, **kwargs):
    """A constrained-layout figure of the standard width.

    Constrained layout fits the decorations inside a *fixed* canvas, which is
    what lets every figure keep the same saved width.
    """
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(FIG_WIDTH, height), layout="constrained", **kwargs
    )
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.03)
    return fig, axes


def save(fig: plt.Figure, path: Path | str, also_png: bool = True) -> None:
    """Write a figure as PDF (and optionally PNG) and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    if also_png:
        fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def fitted_legend(ax: plt.Axes, **kwargs):
    """Draw a legend inside `ax` and warn if it overhangs the axes box.

    A legend wider than its panel spills over the neighbouring axis, which is
    easy to miss in a multi-panel figure and impossible to see in the vector
    output until it is on the page.
    """
    kwargs.setdefault("fontsize", FS["legend"])
    legend = ax.legend(**kwargs)
    fig = ax.get_figure()
    fig.canvas.draw()
    box = legend.get_window_extent()
    frame = ax.get_window_extent()
    overhang = max(frame.x0 - box.x0, box.x1 - frame.x1)
    if overhang > 1.0:
        warnings.warn(
            f"legend overhangs its axes by {overhang:.0f} px; "
            "reduce ncol, handlelength or columnspacing",
            stacklevel=2,
        )
    return legend


def _painted_top(ax: plt.Axes) -> float:
    """Highest pixel of anything drawn inside `ax`, across its twins.

    Bars, filled bands and curves all count, and a twin axis shares the box, so
    its data is part of what the legend has to clear.
    """
    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent()
    top = -float("inf")
    for other in fig.get_axes():
        if other.bbox.bounds != ax.bbox.bounds:
            continue
        for line in other.get_lines():
            data = line.get_xydata()
            if data is None or len(data) == 0:
                continue
            pixels = line.get_transform().transform(data)
            inside = pixels[
                (pixels[:, 0] >= frame.x0) & (pixels[:, 0] <= frame.x1)
                & (pixels[:, 1] >= frame.y0) & (pixels[:, 1] <= frame.y1)
            ]
            if len(inside):
                top = max(top, float(inside[:, 1].max()))
        for artist in list(other.patches) + list(other.collections):
            try:
                box = artist.get_window_extent(renderer)
            except (RuntimeError, ValueError):  # pragma: no cover
                continue
            if box.y1 <= frame.y1 + 1.0 and box.x1 >= frame.x0:
                top = max(top, min(box.y1, frame.y1))
    return top


def fit_headroom(ax: plt.Axes, *, pad: float = 6.0, rounds: int = 8) -> None:
    """Raise the top limit exactly enough to seat the legend, and no more.

    A legend sitting on a curve and a band of empty axis under the legend are
    the same mistake made in opposite directions.  The limit is solved for
    rather than guessed: the panel is redrawn, the distance from the top of the
    data to the bottom of the legend is measured, and the limit is corrected
    until that distance is `pad` pixels.
    """
    legend = ax.get_legend()
    if legend is None:
        return
    fig = ax.get_figure()
    for _ in range(rounds):
        fig.canvas.draw()
        frame = ax.get_window_extent()
        top = _painted_top(ax)
        if top == -float("inf"):
            return
        gap = legend.get_window_extent().y0 - top
        if abs(gap - pad) <= 1.0:
            return
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi - (gap - pad) * (hi - lo) / frame.height)


#: ``gid`` carried by a legend deliberately placed outside its own axes, so the
#: layout audit checks it against the neighbouring panels and the canvas edge
#: instead of reporting the intended overhang.
OUTSIDE_LEGEND_GID = "gbtag-legend-below"


def legend_below(ax: plt.Axes, *, ncol: int | None = None, pad: float = 0.05,
                 **kwargs):
    """Draw the legend under `ax` rather than over the data it describes.

    A stackplot or a filled band paints the whole panel, so there is no empty
    region for an in-axes legend to occupy and matplotlib's own placement rule
    only avoids the *bulk* of the data.  The anchor is measured after a draw so
    the legend clears the tick labels and the axis label whatever size they take.
    """
    kwargs.setdefault("fontsize", FS["legend"])
    fig = ax.get_figure()
    fig.canvas.draw()

    frame = ax.get_window_extent()
    lowest = frame.y0
    for artist in (ax.xaxis.label, *ax.get_xticklabels()):
        if artist.get_visible() and artist.get_text().strip():
            lowest = min(lowest, artist.get_window_extent().y0)
    anchor = (lowest - frame.y0) / frame.height - pad

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, anchor),
        ncol=len(labels) if ncol is None else ncol,
        borderaxespad=0.0,
        **kwargs,
    )
    legend.set_gid(OUTSIDE_LEGEND_GID)
    fig.canvas.draw()
    box = legend.get_window_extent()
    if box.width - frame.width > 1.0:
        warnings.warn(
            f"legend below the axes is {box.width - frame.width:.0f} px wider "
            "than its panel; reduce ncol or fontsize",
            stacklevel=2,
        )
    return legend


def panel_label(ax: plt.Axes, text: str, dx: float = -0.16, dy: float = 1.06) -> None:
    """Place a bold panel label in axes coordinates."""
    ax.text(
        dx,
        dy,
        text,
        transform=ax.transAxes,
        fontsize=FS["title"],
        fontweight="bold",
        va="top",
        ha="left",
    )


def panel_title(ax: plt.Axes, letter: str, text: str = "",
                size: float | None = None, pad: float = 5.0) -> None:
    """Left-aligned title carrying its own bold panel letter.

    Keeping the letter inside the title avoids the collisions that occur when a
    separate label is placed in axes coordinates next to a centred title.
    """
    label = rf"$\bf{{{letter}}}$" + (f"   {text}" if text else "")
    ax.set_title(label, loc="left", fontsize=size or FS["title"], pad=pad)

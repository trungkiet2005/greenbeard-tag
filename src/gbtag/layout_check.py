"""Detect text collisions in a rendered figure.

A legend that sits on top of a curve, two panel titles that run into each
other, or a value label that covers the bar it annotates are all invisible
in the vector output until the figure is on the page.  This module walks a
drawn figure and reports the overlaps, so the check can run in the test
suite rather than by eye.

The rule applied is deliberately conservative.  Two pieces of text collide
when their rendered bounding boxes intersect by more than a small tolerance
in *both* axes.  Text over a filled patch is reported only when the text is
not deliberately placed inside it, which the caller declares by listing the
artists it expects to be written on.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text
from matplotlib.transforms import Bbox


@dataclass(frozen=True)
class Collision:
    """One pair of overlapping rendered elements."""

    kind: str
    first: str
    second: str
    overlap_x: float
    overlap_y: float

    def __str__(self) -> str:
        return (
            f"{self.kind}: {self.first!r} overlaps {self.second!r} "
            f"by {self.overlap_x:.1f} x {self.overlap_y:.1f} px"
        )


def _intersection(a: Bbox, b: Bbox) -> tuple[float, float]:
    """Overlap of two boxes in pixels, per axis."""
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return dx, dy


def _offscreen_tick_labels(fig: plt.Figure) -> set[int]:
    """Tick-label artists for ticks outside their axis view limits.

    Matplotlib keeps a ``Text`` for every tick the locator produced, including
    the ones beyond the current view.  Those are never painted, but they still
    report a window extent, and comparing them produces collisions that do not
    exist on the page.  They are identified here by their tick position rather
    than by their box, which is the only reliable signal.
    """
    skip: set[int] = set()
    for ax in fig.get_axes():
        for axis, (lo, hi), scale in (
            (ax.xaxis, sorted(ax.get_xlim()), ax.get_xscale()),
            (ax.yaxis, sorted(ax.get_ylim()), ax.get_yscale()),
        ):
            # on a log axis the comparison has to happen in log space: a slack
            # derived from an upper limit of 1e3 swamps a lower limit of 1e-9
            # and lets a tick several decades below the view count as inside
            log_axis = scale == "log" and lo > 0.0
            if log_axis:
                lo, hi = np.log10(lo), np.log10(hi)
            slack = 1e-9 * max(abs(lo), abs(hi), 1.0)

            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                loc = tick.get_loc()
                if log_axis:
                    if loc <= 0.0:
                        skip.add(id(tick.label1))
                        skip.add(id(tick.label2))
                        continue
                    loc = np.log10(loc)
                if not lo - slack <= loc <= hi + slack:
                    skip.add(id(tick.label1))
                    skip.add(id(tick.label2))
                # the secondary label is drawn only when its side is enabled
                if not tick.label2.get_visible():
                    skip.add(id(tick.label2))
    return skip


def _visible_texts(fig: plt.Figure) -> list[tuple[str, Bbox]]:
    """Every non-empty rendered text in the figure, with its pixel box."""
    renderer = fig.canvas.get_renderer()
    skip = _offscreen_tick_labels(fig)
    out: list[tuple[str, Bbox]] = []
    for artist in fig.findobj(Text):
        if id(artist) in skip or not artist.get_visible():
            continue
        label = artist.get_text()
        if not label.strip():
            continue
        try:
            box = artist.get_window_extent(renderer=renderer)
        except (RuntimeError, ValueError):  # pragma: no cover - unrenderable
            continue
        if box.width <= 0 or box.height <= 0:
            continue
        out.append((label, box))
    return out


def text_collisions(
    fig: plt.Figure, tolerance: float = 1.0
) -> list[Collision]:
    """Pairs of rendered texts whose boxes intersect.

    ``tolerance`` is the overlap in pixels below which an intersection is
    treated as touching rather than colliding; anti-aliasing and the padding
    matplotlib reserves around glyphs make an exact test too strict.
    """
    fig.canvas.draw()
    texts = _visible_texts(fig)
    collisions = []
    for i in range(len(texts)):
        label_i, box_i = texts[i]
        for j in range(i + 1, len(texts)):
            label_j, box_j = texts[j]
            dx, dy = _intersection(box_i, box_j)
            if dx > tolerance and dy > tolerance:
                collisions.append(
                    Collision("text/text", label_i, label_j, dx, dy)
                )
    return collisions


def legend_overhangs(fig: plt.Figure, tolerance: float = 1.0) -> list[Collision]:
    """Legends that extend beyond the axes box that owns them."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    out = []
    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend is None or not legend.get_visible():
            continue
        box = legend.get_window_extent(renderer=renderer)
        frame = ax.get_window_extent(renderer=renderer)
        dx = max(frame.x0 - box.x0, box.x1 - frame.x1)
        dy = max(frame.y0 - box.y0, box.y1 - frame.y1)
        if dx > tolerance or dy > tolerance:
            out.append(
                Collision(
                    "legend/axes",
                    f"legend of {ax.get_title(loc='left') or 'axes'}",
                    "its own axes box",
                    max(dx, 0.0),
                    max(dy, 0.0),
                )
            )
    return out


def axes_overlaps(fig: plt.Figure, tolerance: float = 1.0) -> list[Collision]:
    """Pairs of axes whose drawing areas intersect.

    Two panels that overlap are a layout failure regardless of what is drawn
    in them; twin axes share a box by construction and are excluded.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes = [ax for ax in fig.get_axes() if ax.get_visible()]
    out = []
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            a, b = axes[i], axes[j]
            if a.bbox.bounds == b.bbox.bounds:
                continue  # a twin axis, deliberately co-located
            dx, dy = _intersection(
                a.get_window_extent(renderer=renderer),
                b.get_window_extent(renderer=renderer),
            )
            if dx > tolerance and dy > tolerance:
                out.append(Collision("axes/axes", f"panel {i}", f"panel {j}", dx, dy))
    return out


def clipped_texts(fig: plt.Figure, tolerance: float = 1.0) -> list[Collision]:
    """Texts that extend past the figure canvas and are cut off when saved.

    Constrained layout fits the decorations it knows about, but an axis label
    or annotation can still be pushed over the edge, and the saved PDF simply
    loses the part that hangs over.
    """
    fig.canvas.draw()
    canvas = fig.bbox
    out = []
    for label, box in _visible_texts(fig):
        over = max(
            canvas.x0 - box.x0,
            box.x1 - canvas.x1,
            canvas.y0 - box.y0,
            box.y1 - canvas.y1,
        )
        if over > tolerance:
            out.append(Collision("text/canvas", label, "the figure edge", over, over))
    return out


def texts_over_data(
    fig: plt.Figure, min_points: int = 2, pad: float = 1.0
) -> list[Collision]:
    """Text whose box sits on top of a plotted line.

    Text laid over a curve hides the curve and is itself hard to read, which
    is the collision that costs the reader information rather than merely
    looking untidy.  Legend entries are included: the manuscript style draws
    legends without a frame, so a legend placed over a curve genuinely
    obscures it, and matplotlib's own placement rule only avoids the *bulk*
    of the data.  Tick labels and axis labels are excluded because they live
    outside the data area by construction.

    The legend box, rather than each entry's text, is what is tested for a
    legend, since its handles occupy the same region and hide as much.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    skip = _offscreen_tick_labels(fig)
    out = []

    for panel, ax in enumerate(fig.get_axes()):
        decorations = {id(ax.title), id(ax.xaxis.label), id(ax.yaxis.label)}
        legend = ax.get_legend()
        legend_texts = {id(t) for t in legend.get_texts()} if legend else set()

        annotations = []
        if legend is not None and legend.get_visible():
            try:
                annotations.append(
                    ("the legend", legend.get_window_extent(renderer=renderer))
                )
            except (RuntimeError, ValueError):  # pragma: no cover
                pass

        for artist in ax.findobj(Text):
            if id(artist) in skip | decorations | legend_texts:
                continue
            if not artist.get_visible() or not artist.get_text().strip():
                continue
            if artist in ax.get_xticklabels() or artist in ax.get_yticklabels():
                continue
            try:
                box = artist.get_window_extent(renderer=renderer)
            except (RuntimeError, ValueError):  # pragma: no cover
                continue
            if box.width > 0 and box.height > 0:
                annotations.append((artist.get_text(), box))
        if not annotations:
            continue

        for line in ax.get_lines():
            if not line.get_visible() or line.get_linestyle() == "None":
                continue
            data = line.get_xydata()
            if data is None or len(data) == 0:
                continue
            pixels = ax.transData.transform(data)
            for label, box in annotations:
                inside = (
                    (pixels[:, 0] >= box.x0 - pad)
                    & (pixels[:, 0] <= box.x1 + pad)
                    & (pixels[:, 1] >= box.y0 - pad)
                    & (pixels[:, 1] <= box.y1 + pad)
                )
                hits = int(inside.sum())
                if hits >= min_points:
                    out.append(
                        Collision(
                            "text/line",
                            f"panel {panel}: {label}",
                            line.get_label() or "an unlabelled line",
                            float(hits),
                            float(hits),
                        )
                    )
    return out


def audit(fig: plt.Figure, tolerance: float = 1.0) -> list[Collision]:
    """Every layout problem the module can detect, in one list."""
    return (
        axes_overlaps(fig, tolerance)
        + legend_overhangs(fig, tolerance)
        + clipped_texts(fig, tolerance)
        + text_collisions(fig, tolerance)
        + texts_over_data(fig)
    )

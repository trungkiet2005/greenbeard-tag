"""Detect text collisions in a rendered figure.

A legend that sits on top of a curve, two panel titles that run into each
other, or a value label that covers the bar it annotates are all invisible
in the vector output until the figure is on the page.  This module walks a
drawn figure and reports the overlaps, so the check can run in the test
suite rather than by eye.

The rule applied is deliberately conservative.  Two pieces of text collide
when their rendered bounding boxes intersect by more than a small tolerance
in *both* axes.  A text deliberately written inside a shaded region is not a
collision, so the fill detector fires on legends only; whether such a text can
still be read is a separate question, answered by :func:`low_contrast_texts`
against the pixels actually painted behind it.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from gbtag.plotting import (
    MIN_CONTRAST,
    OUTSIDE_LEGEND_GID,
    PAPER_LUMINANCE,
    PAPER_MIN_CONTRAST,
    contrast_ratio,
    relative_luminance,
)


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


def _text_box(artist: Text, renderer) -> Bbox:
    """Pixel box of a text, excluding an annotation's arrow.

    ``Annotation.get_window_extent`` unions the text with the arrow that points
    away from it, so an annotation that names a curve reports a box reaching
    all the way to that curve.  The arrow touching its target is the point of
    drawing one; only the glyphs can collide with anything.
    """
    return Text.get_window_extent(artist, renderer=renderer)


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
            box = _text_box(artist, renderer)
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
    """Legends that extend beyond the axes box that owns them.

    A legend that :func:`gbtag.plotting.legend_below` placed under its panel on
    purpose is exempt from this test and checked by
    :func:`outside_legend_collisions` instead, against the panels it could
    actually run into.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    out = []
    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend is None or not legend.get_visible():
            continue
        if legend.get_gid() == OUTSIDE_LEGEND_GID:
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


def outside_legend_collisions(
    fig: plt.Figure, tolerance: float = 1.0
) -> list[Collision]:
    """Legends placed outside their axes that run into a neighbouring panel.

    Moving a legend out of the data area trades one collision for another if
    the space it moves into belongs to the panel below, and constrained layout
    reserves room for the legend without knowing which panel it should displace.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    out = []
    for ax in fig.get_axes():
        legend = ax.get_legend()
        if legend is None or not legend.get_visible():
            continue
        if legend.get_gid() != OUTSIDE_LEGEND_GID:
            continue
        box = legend.get_window_extent(renderer=renderer)
        for panel, other in enumerate(fig.get_axes()):
            if other is ax or not other.get_visible():
                continue
            dx, dy = _intersection(
                box, other.get_window_extent(renderer=renderer)
            )
            if dx > tolerance and dy > tolerance:
                out.append(
                    Collision(
                        "legend/panel",
                        f"legend of {ax.get_title(loc='left') or 'axes'}",
                        f"panel {panel}",
                        dx,
                        dy,
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


def _segment_crossings(pixels: np.ndarray, box: Bbox, pad: float) -> int:
    """How many segments of a polyline pass through a padded box.

    Counting *vertices* inside the box misses the case that matters most on a
    sparse curve: a straight run between two widely spaced points crosses the
    whole annotation without putting a single vertex in it.  The test is the
    Liang-Barsky clip, run over every segment at once.
    """
    if len(pixels) < 2:
        return 0
    start, end = pixels[:-1], pixels[1:]
    finite = np.isfinite(start).all(axis=1) & np.isfinite(end).all(axis=1)
    if not finite.any():
        return 0
    start, end = start[finite], end[finite]
    delta = end - start

    lo = np.zeros(len(delta))
    hi = np.ones(len(delta))
    alive = np.ones(len(delta), dtype=bool)
    edges = (
        (-delta[:, 0], start[:, 0] - (box.x0 - pad)),
        (delta[:, 0], (box.x1 + pad) - start[:, 0]),
        (-delta[:, 1], start[:, 1] - (box.y0 - pad)),
        (delta[:, 1], (box.y1 + pad) - start[:, 1]),
    )
    for slope, offset in edges:
        parallel = slope == 0.0
        alive &= ~(parallel & (offset < 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing = np.where(parallel, 0.0, offset / np.where(parallel, 1.0, slope))
        lo = np.where(~parallel & (slope < 0.0), np.maximum(lo, crossing), lo)
        hi = np.where(~parallel & (slope > 0.0), np.minimum(hi, crossing), hi)

    return int((alive & (lo <= hi)).sum())


def _annotation_boxes(ax: plt.Axes, renderer, skip: set[int]):
    """Every in-axes annotation of `ax`, plus its legend box, with pixel boxes.

    The legend is reported as one box rather than as its entry texts, since its
    handles occupy the same region and hide as much.
    """
    decorations = {id(ax.title), id(ax.xaxis.label), id(ax.yaxis.label)}
    legend = ax.get_legend()
    legend_texts = {id(t) for t in legend.get_texts()} if legend else set()

    boxes = []
    if legend is not None and legend.get_visible():
        try:
            boxes.append(("the legend", legend.get_window_extent(renderer=renderer)))
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
            box = _text_box(artist, renderer)
        except (RuntimeError, ValueError):  # pragma: no cover
            continue
        if box.width > 0 and box.height > 0:
            boxes.append((artist.get_text(), box))
    return boxes


def texts_over_data(
    fig: plt.Figure, min_points: int = 1, pad: float = 1.0
) -> list[Collision]:
    """Text whose box is crossed by a plotted line.

    Text laid over a curve hides the curve and is itself hard to read, which
    is the collision that costs the reader information rather than merely
    looking untidy.  Legend entries are included: the manuscript style draws
    legends without a frame, so a legend placed over a curve genuinely
    obscures it, and matplotlib's own placement rule only avoids the *bulk*
    of the data.  Tick labels and axis labels are excluded because they live
    outside the data area by construction.

    ``min_points`` is the number of crossing *segments* below which the
    intersection is treated as touching rather than colliding.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    skip = _offscreen_tick_labels(fig)
    out = []

    for panel, ax in enumerate(fig.get_axes()):
        annotations = _annotation_boxes(ax, renderer, skip)
        if not annotations:
            continue

        # a legend belongs to the primary axes, so iterating its own lines
        # never sees the twin's curve running underneath it
        lines = [
            line
            for other in fig.get_axes()
            if other.bbox.bounds == ax.bbox.bounds
            for line in other.get_lines()
        ]
        for line in lines:
            if not line.get_visible() or line.get_linestyle() == "None":
                continue
            data = line.get_xydata()
            if data is None or len(data) == 0:
                continue
            # a span line from axhline/axvline carries a *blended* transform,
            # one axis in data coordinates and the other in axes fractions, so
            # pushing its points through transData places them off the panel
            pixels = line.get_transform().transform(data)
            for label, box in annotations:
                hits = _segment_crossings(pixels, box, pad)
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


def legends_over_fill(fig: plt.Figure, samples: int = 5) -> list[Collision]:
    """Legends drawn on top of a filled area such as a stackplot.

    A stackplot paints the whole panel, so ``texts_over_data`` sees no curve to
    cross and matplotlib's own placement rule finds every location equally
    good.  The legend still hides the composition underneath it, which on a
    stacked-share plot is the entire result.  Annotations are *not* reported:
    a label written inside a shaded band is usually deliberate, and whether it
    can still be read is :func:`low_contrast_texts`'s question.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    out = []
    for panel, ax in enumerate(fig.get_axes()):
        legend = ax.get_legend()
        if legend is None or not legend.get_visible():
            continue
        fills = [c for c in ax.collections if isinstance(c, PolyCollection)]
        if not fills:
            continue
        box = legend.get_window_extent(renderer=renderer)
        grid = np.array(
            [
                [x, y]
                for x in np.linspace(box.x0, box.x1, samples)
                for y in np.linspace(box.y0, box.y1, samples)
            ]
        )
        for fill in fills:
            if not fill.get_visible():
                continue
            transform = fill.get_transform()
            inside = np.zeros(len(grid), dtype=bool)
            for path in fill.get_paths():
                if len(path.vertices) == 0:
                    continue
                inside |= path.transformed(transform).contains_points(grid)
            if inside.any():
                out.append(
                    Collision(
                        "legend/fill",
                        f"panel {panel}: the legend",
                        fill.get_label() or "a filled area",
                        float(inside.sum()),
                        float(inside.sum()),
                    )
                )
    return out


def low_contrast_texts(
    fig: plt.Figure, minimum: float = MIN_CONTRAST
) -> list[Collision]:
    """Text the reader cannot make out against what is painted behind it.

    A heatmap cell label, a value written inside a bar and a legend entry over
    a filled band all sit on ink rather than on paper, and no bounding box
    intersects, so every geometric detector passes them.  The background is
    measured rather than assumed: every candidate text is hidden, the figure is
    redrawn once, and the median colour inside each text's box is compared with
    the colour the text is drawn in.  ``minimum`` is the WCAG contrast ratio
    below which the text is reported.

    The median is what makes one redraw enough.  A curve crossing a label tints
    a few pixels of its box and would drag a mean towards the curve's colour,
    but it cannot move the median off the background, and a curve through the
    label is :func:`texts_over_data`'s finding rather than this one's.
    """
    fig.canvas.draw()
    if not hasattr(fig.canvas, "buffer_rgba"):  # pragma: no cover - non-Agg
        return []
    # the panels have to stop moving before anything is hidden: constrained
    # layout sizes the margins from the tick labels and the axis labels, so
    # hiding the text re-flows the figure and every box measured beforehand
    # then points at the wrong pixels
    engine = fig.get_layout_engine()
    fig.set_layout_engine("none")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    skip = _offscreen_tick_labels(fig)

    candidates = []
    for ax in fig.get_axes():
        legend = ax.get_legend()
        texts = list(ax.findobj(Text))
        if legend is not None and legend.get_visible():
            texts += list(legend.get_texts())
        for artist in texts:
            if id(artist) in skip or not artist.get_visible():
                continue
            if not artist.get_text().strip():
                continue
            try:
                box = _text_box(artist, renderer)
            except (RuntimeError, ValueError):  # pragma: no cover
                continue
            if box.width <= 1 or box.height <= 1:
                continue
            candidates.append((artist, box))
    if not candidates:
        fig.set_layout_engine(engine)
        return []

    for artist, _ in candidates:
        artist.set_visible(False)
    try:
        fig.canvas.draw()
        painted = np.asarray(fig.canvas.buffer_rgba())[:, :, :3] / 255.0
    finally:
        for artist, _ in candidates:
            artist.set_visible(True)
        fig.set_layout_engine(engine)
        fig.canvas.draw()

    height, width = painted.shape[:2]
    out = []
    for artist, box in candidates:
        # the buffer is indexed from the top, the window extent from the bottom
        col0 = int(max(0, np.floor(box.x0)))
        col1 = int(min(width, np.ceil(box.x1)))
        row0 = int(max(0, np.floor(height - box.y1)))
        row1 = int(min(height, np.ceil(height - box.y0)))
        if col1 <= col0 or row1 <= row0:
            continue
        patch = painted[row0:row1, col0:col1].reshape(-1, 3)
        background = np.median(patch, axis=0)
        ratio = contrast_ratio(artist.get_color(), tuple(background))
        # a label on bare paper is keyed to the curve it names, and demanding
        # the small-text ratio of it would force every such label to black and
        # throw the key away; a label on ink had a free choice of black or
        # white and has to make the legible one
        on_paper = relative_luminance(tuple(background)) > PAPER_LUMINANCE
        needed = PAPER_MIN_CONTRAST if on_paper else minimum
        if ratio < needed:
            out.append(
                Collision(
                    "text/contrast",
                    artist.get_text(),
                    f"its background (contrast {ratio:.2f}:1, needs {needed:.1f})",
                    ratio,
                    ratio,
                )
            )
    return out


def audit(fig: plt.Figure, tolerance: float = 1.0) -> list[Collision]:
    """Every layout problem the module can detect, in one list."""
    return (
        axes_overlaps(fig, tolerance)
        + legend_overhangs(fig, tolerance)
        + outside_legend_collisions(fig, tolerance)
        + clipped_texts(fig, tolerance)
        + text_collisions(fig, tolerance)
        + texts_over_data(fig)
        + legends_over_fill(fig)
        + low_contrast_texts(fig)
    )

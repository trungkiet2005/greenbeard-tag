"""The layout checker itself, on figures with known-good and known-bad layout.

A collision detector that never fires is worthless, so each detector is
exercised on a figure built to trip it and on one built not to.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gbtag.layout_check import (
    audit,
    axes_overlaps,
    clipped_texts,
    legend_overhangs,
    legends_over_fill,
    low_contrast_texts,
    outside_legend_collisions,
    text_collisions,
    texts_over_data,
)
from gbtag.plotting import PALETTE, legend_below, use_paper_style


@pytest.fixture(autouse=True)
def _style():
    use_paper_style()
    yield
    plt.close("all")


def test_clean_figure_passes() -> None:
    fig, ax = plt.subplots(figsize=(6.9, 2.6), layout="constrained")
    ax.plot([0, 1], [0, 1], label="a line")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left")
    assert audit(fig) == []


def test_two_texts_on_the_same_spot_collide() -> None:
    fig, ax = plt.subplots(figsize=(6.9, 2.6), layout="constrained")
    ax.plot([0, 1], [0, 1])
    ax.text(0.5, 0.5, "first label here")
    ax.text(0.5, 0.5, "second label here")
    hits = text_collisions(fig)
    assert len(hits) == 1
    assert "first label here" in str(hits[0])


def test_a_legend_pushed_outside_its_axes_is_reported() -> None:
    fig, ax = plt.subplots(figsize=(6.9, 2.6), layout="constrained")
    ax.plot([0, 1], [0, 1], label="a line with a rather long label")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    assert legend_overhangs(fig)


def test_text_off_the_canvas_is_reported() -> None:
    fig, ax = plt.subplots(figsize=(6.9, 2.6), layout="constrained")
    ax.plot([0, 1], [0, 1])
    ax.text(
        0.5, 0.5, "pushed off the page", transform=fig.transFigure,
        clip_on=False, fontsize=40,
    )
    assert clipped_texts(fig)


def test_overlapping_axes_are_reported() -> None:
    fig = plt.figure(figsize=(6.9, 2.6))
    fig.add_axes((0.1, 0.1, 0.5, 0.8))
    fig.add_axes((0.3, 0.1, 0.5, 0.8))
    assert axes_overlaps(fig)


def test_twin_axes_are_not_reported_as_overlapping() -> None:
    fig, ax = plt.subplots(figsize=(6.9, 2.6), layout="constrained")
    ax.plot([0, 1], [0, 1])
    ax.twinx().plot([0, 1], [1, 0])
    assert axes_overlaps(fig) == []


def test_offview_tick_labels_are_ignored() -> None:
    """Ticks outside the view are never painted and must not count.

    Matplotlib keeps a ``Text`` for every tick the locator produced, and
    those artists report a window extent wherever the tick would have been.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6), layout="constrained")
    for ax in axes:
        ax.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
    assert text_collisions(fig) == []


def test_text_written_over_a_curve_is_reported() -> None:
    """The collision that costs the reader information rather than tidiness."""
    fig, ax = plt.subplots(figsize=(6.9, 2.6), layout="constrained")
    x = np.linspace(0, 1, 200)
    ax.plot(x, x, label="a line")
    ax.text(0.5, 0.5, "right on top of the line", ha="center", va="center")
    hits = texts_over_data(fig)
    assert hits
    assert "right on top of the line" in str(hits[0])


def test_text_beside_a_curve_is_not_reported() -> None:
    fig, ax = plt.subplots(figsize=(6.9, 2.6), layout="constrained")
    x = np.linspace(0, 1, 200)
    ax.plot(x, x, label="a line")
    ax.set_ylim(0, 1)
    ax.text(0.05, 0.92, "well clear of it", ha="left", va="top")
    assert texts_over_data(fig) == []


def test_a_legend_over_a_curve_is_reported() -> None:
    """The manuscript style draws legends unframed, so they hide what is under
    them; matplotlib's own placement rule only avoids the bulk of the data."""
    fig, ax = plt.subplots(figsize=(6.9, 2.6), layout="constrained")
    x = np.linspace(0, 1, 200)
    ax.plot(x, x, label="a line")
    ax.legend(loc="center")
    hits = texts_over_data(fig)
    assert hits
    assert "the legend" in str(hits[0])


def test_a_legend_clear_of_the_curve_is_not_reported() -> None:
    fig, ax = plt.subplots(figsize=(6.9, 2.6), layout="constrained")
    x = np.linspace(0, 1, 200)
    ax.plot(x, x, label="a line")
    ax.set_ylim(0, 2.2)
    ax.legend(loc="upper left")
    assert texts_over_data(fig) == []


def test_axis_labels_are_not_reported_as_data_overlap() -> None:
    fig, ax = plt.subplots(figsize=(6.9, 2.6), layout="constrained")
    x = np.linspace(0, 1, 200)
    ax.plot(x, x)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("a title")
    assert texts_over_data(fig) == []


def test_offview_ticks_on_a_log_axis_are_ignored() -> None:
    """The log-scale case, where an absolute slack is the wrong comparison.

    With limits ``(1e-9, 3e3)`` a tick at ``1e-11`` sits two decades below
    the view.  A tolerance scaled to the upper limit would swallow it.
    """
    fig, ax = plt.subplots(figsize=(6.9, 2.6), layout="constrained")
    ax.plot([0, 1], [1e-8, 1e2])
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 3e3)
    assert clipped_texts(fig) == []


def test_a_curve_crossing_between_two_vertices_is_reported() -> None:
    """The case a vertex test cannot see.

    A sparse curve can run straight through an annotation without putting a
    single one of its points inside the box, which is exactly what happens on
    a panel plotted from eight data points.
    """
    fig, ax = plt.subplots(figsize=(6.9, 2.6), layout="constrained")
    ax.plot([0.0, 1.0], [0.0, 1.0], label="two points only")
    ax.text(0.5, 0.5, "straight through", ha="center", va="center")
    hits = texts_over_data(fig)
    assert hits
    assert "straight through" in str(hits[0])


def test_a_span_line_is_measured_in_its_own_transform() -> None:
    """``axvline`` is blended: y is an axes fraction, not a data value.

    Pushing its points through ``transData`` sends them to data y of 0 and 1,
    which on a panel whose y axis stops at 0.5 lands the line in the title.
    """
    fig, ax = plt.subplots(figsize=(6.9, 2.6), layout="constrained")
    ax.plot(np.linspace(0, 1, 50), np.linspace(0, 0.4, 50))
    ax.set_ylim(0, 0.5)
    ax.axvline(0.5, linestyle="--")
    ax.set_title("a title the line must not reach")
    ax.text(0.05, 0.45, "clear of it", ha="left", va="top")
    assert texts_over_data(fig) == []


def test_an_arrow_reaching_its_target_is_not_reported() -> None:
    """An annotation's arrow is supposed to touch what it points at.

    ``Annotation.get_window_extent`` unions the text with the arrow, so the
    reported box otherwise stretches all the way to the curve being named.
    """
    fig, ax = plt.subplots(figsize=(6.9, 2.6), layout="constrained")
    x = np.linspace(0, 1, 200)
    ax.plot(x, x, label="a line")
    ax.set_ylim(0, 2.2)
    ax.annotate(
        "the peak", xy=(0.95, 0.95), xytext=(0.05, 2.0),
        arrowprops=dict(arrowstyle="->"),
    )
    assert texts_over_data(fig) == []


def test_a_legend_over_a_stackplot_is_reported() -> None:
    """A stackplot leaves no empty region, so no curve crosses the legend."""
    fig, ax = plt.subplots(figsize=(6.9, 2.6), layout="constrained")
    x = np.linspace(0, 1, 50)
    ax.stackplot(x, [np.full_like(x, 0.5), np.full_like(x, 0.5)],
                 labels=["lower", "upper"])
    ax.legend(loc="center left")
    assert texts_over_data(fig) == []  # nothing for the line detector to find
    assert legends_over_fill(fig)


def test_a_legend_moved_below_the_stackplot_is_not_reported() -> None:
    fig, ax = plt.subplots(figsize=(6.9, 2.6), layout="constrained")
    x = np.linspace(0, 1, 50)
    ax.stackplot(x, [np.full_like(x, 0.5), np.full_like(x, 0.5)],
                 labels=["lower", "upper"])
    ax.set_xlabel("x")
    legend_below(ax, ncol=2)
    assert legends_over_fill(fig) == []
    assert legend_overhangs(fig) == []  # the overhang is the point of it
    assert outside_legend_collisions(fig) == []


def test_a_legend_moved_below_onto_another_panel_is_reported() -> None:
    """Moving a legend out of the data trades one collision for another if the
    space it lands in belongs to the panel underneath."""
    fig, axes = plt.subplots(2, 1, figsize=(6.9, 3.4), layout="constrained")
    for ax in axes:
        ax.plot([0, 1], [0, 1], label="a line")
    legend = axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -1.2))
    legend.set_gid("gbtag-legend-below")
    assert outside_legend_collisions(fig)


def test_white_text_on_a_pale_cell_is_reported() -> None:
    """The failure no bounding box can see: nothing overlaps, nothing is read.

    ``RdYlBu`` is dark at 0.0 and pale at 0.5, so the same white label is
    legible on one cell and invisible on the other.
    """
    fig, ax = plt.subplots(figsize=(6.9, 2.6), layout="constrained")
    ax.imshow(np.array([[0.0, 0.5]]), cmap="RdYlBu", vmin=0.0, vmax=1.0,
              aspect="auto")
    ax.text(0, 0, "dark", ha="center", va="center", color="white")
    ax.text(1, 0, "pale", ha="center", va="center", color="white")
    hits = low_contrast_texts(fig)
    assert len(hits) == 1
    assert "pale" in str(hits[0])


def test_a_label_keyed_to_its_curve_is_not_reported() -> None:
    """A coloured label on bare paper is an encoding, not an accident.

    Holding it to the small-text ratio would force every such label to black
    and throw away the link between the label and the curve it names.
    """
    fig, ax = plt.subplots(figsize=(6.9, 2.6), layout="constrained")
    x = np.linspace(0, 1, 200)
    ax.plot(x, x, color=PALETTE["forger"], label="a line")
    ax.set_ylim(0, 2.2)
    ax.text(0.05, 2.0, "the orange one", color=PALETTE["forger"],
            ha="left", va="top")
    assert low_contrast_texts(fig) == []


def test_hiding_the_text_does_not_move_the_panels() -> None:
    """Constrained layout sizes the margins from the tick and axis labels.

    If the layout is left live while the text is hidden for the background
    measurement, the panels grow and every box measured beforehand points at
    the wrong pixels; black tick labels then read as low contrast.
    """
    fig, ax = plt.subplots(figsize=(6.9, 2.6), layout="constrained")
    ax.plot(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
    ax.set_xlabel("a long axis label")
    ax.set_ylabel("another long axis label")
    ax.set_title("and a title")
    assert low_contrast_texts(fig) == []


def test_a_legend_over_a_twin_axis_curve_is_reported() -> None:
    """A legend belongs to the primary axes, and the curve it hides may not.

    Iterating only ``ax.get_lines()`` never reaches the twin's data, so the
    lines of every axes sharing the same box have to be gathered together.
    """
    fig, ax = plt.subplots(figsize=(6.9, 2.6), layout="constrained")
    ax.plot(np.linspace(0, 1, 50), np.zeros(50), label="the flat one")
    ax.set_ylim(0, 1)
    twin = ax.twinx()
    twin.plot(np.linspace(0, 1, 50), np.full(50, 0.5), color="C1")
    twin.set_ylim(0, 1)
    ax.legend(loc="center")
    hits = texts_over_data(fig)
    assert hits
    assert "the legend" in str(hits[0])

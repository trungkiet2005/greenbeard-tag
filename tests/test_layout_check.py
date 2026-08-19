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
    text_collisions,
    texts_over_data,
)
from gbtag.plotting import use_paper_style


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

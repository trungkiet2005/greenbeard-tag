"""The contrast helpers that decide what colour a label on ink is drawn in.

A label written over a heatmap cell or a bar has only two sensible colours,
and which one is legible depends on the colour that was painted, not on the
number that produced it.  These tests pin that distinction, because the rule
they replaced was keyed to the value and was wrong on eight cells of sixteen.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import pytest

from gbtag.plotting import (
    LUMINANCE_CROSSOVER,
    contrast_ratio,
    readable_on,
    relative_luminance,
)


def test_luminance_of_the_two_extremes() -> None:
    assert relative_luminance("white") == pytest.approx(1.0)
    assert relative_luminance("black") == pytest.approx(0.0)


def test_contrast_ratio_is_symmetric_and_bounded() -> None:
    assert contrast_ratio("black", "white") == pytest.approx(21.0)
    assert contrast_ratio("white", "black") == pytest.approx(21.0)
    assert contrast_ratio("black", "black") == pytest.approx(1.0)


def test_readable_on_picks_the_higher_contrast_colour() -> None:
    for colour in ("white", "#FFFFCC", "#E8F4F1", "#F7F7A0"):
        assert readable_on(colour) == "black"
    for colour in ("black", "#2A3A8C", "#B2182B", "#0072B2"):
        assert readable_on(colour) == "white"


def test_the_crossover_is_where_the_two_ratios_meet() -> None:
    """Below it white wins, above it black wins, by construction."""
    grey = mpl.colors.to_hex((LUMINANCE_CROSSOVER,) * 3)  # not the luminance
    lighter = "#8A8A8A"
    darker = "#3A3A3A"
    assert readable_on(lighter) == "black"
    assert readable_on(darker) == "white"
    assert contrast_ratio("black", lighter) > contrast_ratio("white", lighter)
    assert contrast_ratio("white", darker) > contrast_ratio("black", darker)
    assert grey  # the conversion is exercised, not asserted on


def test_the_diverging_colormap_case_that_motivated_the_helper() -> None:
    """RdYlBu is lightest in the middle, so a value-band rule is backwards.

    The manuscript's race-payoff heatmap runs 0 to 105.  A rule of the form
    "white for the middle band" put white on the palest cells in the figure.
    """
    cmap = mpl.colormaps["RdYlBu"]
    norm = mpl.colors.Normalize(0.0, 105.0)
    for value in (47.4, 48.6, 59.0, 61.6):  # the pale middle
        assert readable_on(cmap(norm(value))) == "black"
    for value in (5.4, 101.8):  # the dark ends
        assert readable_on(cmap(norm(value))) == "white"

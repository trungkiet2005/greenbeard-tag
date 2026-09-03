r"""Checks on the figure style and on the rendered layout.

The manuscript includes every figure at ``\linewidth``, so two figures saved
at different widths are reduced by different factors and their text ends up
at different sizes on the page.  The first two tests pin the two things that
guarantee a uniform reduction: one saved width for all figures, and one font
scale.

The remaining tests render each figure and assert that nothing collides.
A legend sitting on a curve, two panel titles running into each other or an
axis label pushed off the canvas are invisible in the vector output until
the figure is on the page, and the collision they cause destroys exactly the
information the annotation was added to convey.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from gbtag.layout_check import audit  # noqa: E402
from gbtag.plotting import FIG_WIDTH, FS, use_paper_style  # noqa: E402
from gbtag.race import build_race_tables  # noqa: E402
from gbtag import config as cfg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "results" / "figures"
RESULTS = ROOT / "results"
MEDIABOX = re.compile(rb"/MediaBox\s*\[([^\]]*)\]")

FIGURE_FUNCTIONS = (
    "fig01",
    "fig02",
    "fig03",
    "fig04",
    "fig05",
    "fig06",
    "fig07",
    "fig08",
    "fig09",
    "fig10",
    "fig11",
)


def _width_inches(pdf: Path) -> float:
    match = MEDIABOX.search(pdf.read_bytes())
    assert match is not None, f"no MediaBox in {pdf.name}"
    box = [float(v) for v in match.group(1).split()]
    return (box[2] - box[0]) / 72.0


@pytest.mark.parametrize("pdf", sorted(FIGDIR.glob("fig*.pdf")) or [None])
def test_every_figure_has_the_standard_width(pdf: Path | None) -> None:
    if pdf is None:
        pytest.skip("figures have not been generated yet")
    assert _width_inches(pdf) == pytest.approx(FIG_WIDTH, abs=0.02)


def test_font_scale_is_ordered_and_legible() -> None:
    """Elsevier asks for 7 pt of printed text; the reduction decides that.

    The submission layout is the one that has to clear the floor, so it is
    the one asserted here.  The venue-neutral layout reduces harder and is
    checked only against the older 6 pt bar it was built to.
    """
    csf_linewidth_pt = 468.3324           # cas-sc single column, a4paper
    canvas_pt = FIG_WIDTH * 72.0          # matplotlib writes PostScript pt
    assert min(FS.values()) * csf_linewidth_pt / canvas_pt >= 7.0

    venue_neutral = 6.268 / FIG_WIDTH     # inches, article a4paper, 1 in margins
    assert min(FS.values()) * venue_neutral > 6.0

    assert FS["tiny"] <= FS["annot"] <= FS["legend"] <= FS["title"]
    assert FS["tick"] <= FS["label"]


def _load_figure_module():
    """Import ``scripts/make_figures.py`` without installing it."""
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import make_figures  # noqa: PLC0415

    return make_figures


@pytest.mark.parametrize("name", FIGURE_FUNCTIONS)
def test_rendered_figure_has_no_collisions(name: str, tmp_path: Path) -> None:
    """Render one figure and assert the layout audit is clean."""
    if not (RESULTS / "key_numbers.json").exists():
        pytest.skip("results have not been generated yet")
    if not (RESULTS / "grids.npz").exists():
        pytest.skip("robustness grids have not been generated yet")

    mf = _load_figure_module()
    use_paper_style()
    tables = build_race_tables(cfg.RACE)

    problems: list = []
    original = mf.save

    def capture(fig, path, also_png=True):
        problems.extend(audit(fig))
        original(fig, tmp_path / Path(path).name, also_png=False)

    mf.save = capture
    try:
        getattr(mf, name)(tables, RESULTS, tmp_path)
    finally:
        mf.save = original

    assert not problems, "\n".join(str(p) for p in problems)

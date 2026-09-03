"""Build the deterministic Physica A LaTeX source archive from an allowlist.

Run after ``assemble.py`` and a successful LaTeX/BibTeX build.  The allowlist
is deliberate: private submission notes, rendered PDFs, auxiliary files and the
supplement's own sources cannot enter the main archive by accident.

``elsarticle.cls`` is not bundled.  Unlike ``cas-sc.cls`` it ships with every
TeX distribution and is Elsevier's own class, so production has it; adding a
possibly older copy to the archive would only create a version conflict.

``main.bbl`` is bundled and this matters: the style is
``elsarticle-num-names.bst``, and a production tree that does not run BibTeX
cannot resolve the printed references without it.

Run from paper-physicaa/:  python package_source.py
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "PhysicaA_manuscript_source.zip"

ROOT_FILES = (
    "main.tex",
    "main.bbl",
    "refs.bib",
    "table_race_generated.tex",
)
FIGURES = (
    "figures/fig01_exclusion.pdf",
    "figures/fig02_handshake.pdf",
    "figures/fig03_invasion.pdf",
    "figures/fig04_cliff.pdf",
    "figures/fig06_instruments.pdf",
    "figures/fig10_assortment.pdf",
    "figures/fig11_phase.pdf",
)

MEMBERS = ROOT_FILES + FIGURES

# A zip records mtimes, so two archives built from identical bytes differ.
# Pin the timestamp and the external attributes and the archive becomes a
# function of its contents alone, which is what makes a checksum meaningful.
FIXED_DATE = (2026, 1, 1, 0, 0, 0)


def main() -> int:
    missing = [m for m in MEMBERS if not (HERE / m).exists()]
    if missing:
        raise SystemExit(
            "missing from the package, run assemble.py and build first: "
            + ", ".join(missing))

    # Guard against shipping a .bbl older than the .tex that cites into it.
    tex = (HERE / "main.tex").stat().st_mtime
    bbl = (HERE / "main.bbl").stat().st_mtime
    if bbl < tex:
        raise SystemExit(
            "main.bbl is older than main.tex: run pdflatex, bibtex, pdflatex "
            "twice before packaging")

    # The allowlist has to agree with main.tex in both directions.  A figure
    # main.tex uses but the archive omits compiles to a missing-image box at
    # the publisher; a figure the archive carries but main.tex no longer uses
    # is a leftover from a section that moved to the supplement.  Only the
    # first is fatal at the publisher, but the second is how the first
    # eventually happens, so both fail here.
    body = "\n".join(l for l in (HERE / "main.tex").read_text(encoding="utf-8")
                     .splitlines() if not l.lstrip().startswith("%"))

    referenced = set(re.findall(r"figures/fig[\w-]+\.pdf", body))
    listed = set(FIGURES)
    if referenced - listed:
        raise SystemExit("main.tex references figures missing from the "
                         "allowlist: " + ", ".join(sorted(referenced - listed)))
    if listed - referenced:
        raise SystemExit("the allowlist carries figures main.tex no longer "
                         "uses: " + ", ".join(sorted(listed - referenced)))

    # Same for the \input-ed generated tables.
    inputs = {name + ".tex" for name in re.findall(r"\\input\{([\w-]+)\}", body)}
    shipped = {f for f in ROOT_FILES if f.endswith("_generated.tex")}
    if inputs - shipped:
        raise SystemExit("main.tex inputs files missing from the allowlist: "
                         + ", ".join(sorted(inputs - shipped)))
    if shipped - inputs:
        raise SystemExit("the allowlist carries generated tables main.tex no "
                         "longer inputs: " + ", ".join(sorted(shipped - inputs)))

    with ZipFile(ARCHIVE, "w", ZIP_DEFLATED) as z:
        for name in MEMBERS:
            info = ZipInfo(name, date_time=FIXED_DATE)
            info.external_attr = 0o644 << 16
            info.compress_type = ZIP_DEFLATED
            z.writestr(info, (HERE / name).read_bytes())

    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    print("wrote %s (%d members, %.1f kB)"
          % (ARCHIVE.name, len(MEMBERS), ARCHIVE.stat().st_size / 1024))
    print("sha256 %s" % digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

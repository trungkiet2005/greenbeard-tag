"""Assemble, compile and package the Physica A submission in one command.

The order matters and is easy to get wrong by hand:

  assemble -> pdflatex -> bibtex -> pdflatex x2   (main)
           -> pdflatex -> bibtex -> pdflatex x2   (supplementary, needs
                                                   main.aux for xr)
           -> pdflatex                            (highlights)
           -> package_source

Skipping the second and third pdflatex passes leaves the reference numbers
unresolved; skipping bibtex ships the previous run's bibliography under a clean
log.  Both have happened before, so this script does not offer the option.

Run from paper-physicaa/:  python build.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd, log_name):
    log = HERE / log_name
    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                          errors="replace")
    log.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    return proc.returncode


def pdflatex(stem, i):
    code = run(["pdflatex", "-interaction=nonstopmode", stem + ".tex"],
               "_build_%s_%d.log" % (stem, i))
    if code != 0:
        raise SystemExit("pdflatex failed on %s pass %d; see _build_%s_%d.log"
                         % (stem, i, stem, i))


def bibtex(stem):
    if run(["bibtex", stem], "_bibtex_%s.log" % stem) != 0:
        raise SystemExit("bibtex failed on %s; see _bibtex_%s.log"
                         % (stem, stem))


def report(stem):
    """Pages, overfull boxes and unresolved references from the final pass."""
    log = (HERE / ("_build_%s_3.log" % stem))
    if not log.exists():
        log = HERE / ("_build_%s_1.log" % stem)
    text = log.read_text(encoding="utf-8", errors="replace")
    pages = re.search(r"Output written on %s\.pdf \((\d+) pages?" % stem, text)
    overfull = len(re.findall(r"Overfull \\[hv]box", text))
    undefined = len(re.findall(r"Warning: (?:Reference|Citation) .* undefined",
                               text))
    return (pages.group(1) if pages else "?"), overfull, undefined


def main() -> int:
    if run([sys.executable, "assemble.py"], "_assemble.log") != 0:
        raise SystemExit("assemble.py failed; see _assemble.log")
    print((HERE / "_assemble.log").read_text(encoding="utf-8").strip())

    # The two documents cite into each other through xr, so their passes have
    # to interleave: main writes the labels the supplement needs, the
    # supplement writes the labels main needs, and each then has to run again
    # to pick the other's up. Compiling one to completion and then the other
    # leaves every cross-document reference as "??" in whichever went first.
    pdflatex("main", 1)             # main.aux exists, supplementary.aux does not
    pdflatex("supplementary", 1)    # reads main.aux, writes supplementary.aux
    bibtex("main")
    bibtex("supplementary")
    pdflatex("main", 2)             # reads supplementary.aux
    pdflatex("supplementary", 2)
    pdflatex("main", 3)             # numbers settle
    pdflatex("supplementary", 3)
    pdflatex("highlights", 1)

    failed = False
    for stem in ("main", "supplementary", "highlights"):
        pages, overfull, undefined = report(stem)
        flag = "" if (overfull == 0 and undefined == 0) else "   <-- CHECK"
        print("%-14s %3s pages   %d overfull   %d undefined%s"
              % (stem + ".pdf", pages, overfull, undefined, flag))
        failed = failed or overfull or undefined

    if run([sys.executable, "package_source.py"], "_package.log") != 0:
        raise SystemExit((HERE / "_package.log").read_text(encoding="utf-8"))
    print((HERE / "_package.log").read_text(encoding="utf-8").strip())

    # The DOCX items are regenerated on every build rather than kept by hand,
    # so highlights.docx cannot drift away from highlights.tex the way a Word
    # file edited once always eventually does.
    if run([sys.executable, "make_submission_files.py"], "_docx.log") != 0:
        raise SystemExit((HERE / "_docx.log").read_text(encoding="utf-8"))
    print((HERE / "_docx.log").read_text(encoding="utf-8").strip())

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

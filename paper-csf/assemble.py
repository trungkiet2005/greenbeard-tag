"""Assemble the Chaos, Solitons & Fractals version of the manuscript.

Reads the venue-neutral ../paper/main.tex, keeps most of its body verbatim, and

  * swaps in an elsarticle single-column front matter (CSF requirement),
  * replaces the long venue-neutral introduction with a compact CSF version,
  * replaces Related work with a half-length subsection of the Introduction,
  * folds Limitations into the Discussion as a subsection,
  * swaps the venue-neutral data statement for the Elsevier declarations,
  * moves proofs and secondary experiments into a separate supplement,
  * switches the bibliography to the numbered Elsevier style.

The three structural changes take the manuscript from seven top-level sections
to five: 1 Introduction (1.1 Related work), 2 Model, 3 Results, 4 Discussion
(4.1 Limitations), 5 Conclusion.

Run from paper-csf/:  python assemble.py
"""

import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "paper" / "main.tex"

# cas-sc gives table and figure a KEY-VALUE optional argument, not a float
# specifier, so a bare [t] inherited from the venue-neutral manuscript is
# silently discarded with "LaTeX Warning: No positions in optional float
# specifier" and the float drifts. Rewrite every one to the key-value form.
FLOAT_RE = re.compile(r"(\\begin\{(?:figure|table)\*?\})\[([tbphH!]+)\]")


def cas_floats(text):
    return FLOAT_RE.sub(lambda m: f"{m.group(1)}[pos={m.group(2)}]", text)

SOURCE = SRC.read_text(encoding="utf-8")

# The manuscript is cut into sections by the section commands themselves rather
# than by line numbers.  An earlier version indexed ../paper/main.tex by
# hard-coded 1-indexed ranges, which made every edit to the master a two-file
# edit: add a sentence anywhere and eight boundaries move.  The asserts caught
# it, but only after the fact.  Markers cost nothing and cannot drift.
def seg(start, end=None):
    """Body from the line beginning ``start`` up to the line beginning ``end``.

    ``end=None`` runs to the end of the file.  Both markers must occur exactly
    once at the beginning of a line, so a phrase that also appears in prose is
    not a valid marker.
    """
    i = SOURCE.find(start)
    assert i != -1, f"segment marker not found: {start!r}"
    assert SOURCE.count("\n" + start) + SOURCE.startswith(start) == 1, \
        f"segment marker is not unique at line start: {start!r}"
    if end is None:
        return SOURCE[i:]
    j = SOURCE.find(end, i)
    assert j != -1, f"segment end marker not found after {start!r}: {end!r}"
    return SOURCE[i:j]


def block(name):
    return (HERE / f"_blocks_{name}.tex").read_text(encoding="utf-8")


def insert_before(text, anchor, payload, label):
    assert text.count(anchor) == 1, f"anchor not unique for {label}: {anchor!r}"
    return text.replace(anchor, payload + anchor)


def sub(text, old, new, label):
    assert text.count(old) == 1, f"substitution anchor not unique for {label}"
    return text.replace(old, new)


CITE_RE = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")
ARG_RE = re.compile(r"\\(?:cite[a-zA-Z]*|ref|eqref|autoref|label)\*?(?:\[[^\]]*\])*\{[^}]*\}")


def cite_keys(text):
    return {k.strip()
            for m in CITE_RE.finditer(text)
            for k in m.group(1).split(",") if k.strip()}


def prose_words(text):
    """Word count with citation and cross-reference arguments removed.

    Counting them, as a naive strip of backslash commands does, inflates a
    citation-dense section by a couple of hundred words and makes a length
    budget meaningless.
    """
    text = re.sub(r"^\s*%.*$", "", text, flags=re.M)
    text = ARG_RE.sub(" ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = re.sub(r"[{}$&_^%~]", " ", text)
    return len([w for w in text.split() if any(c.isalpha() for c in w)])


# ---------------------------------------------------------------- segments
front = (HERE / "_front.tex").read_text(encoding="utf-8")
INTRO = "\\section{Introduction}"
RELATED = "\\section{Related work}"
MODEL = "\\section{Model}"
RESULTS = "\\section{Results}"
DISCUSSION = "\\section{Discussion}"
LIMITATIONS = "\\section{Limitations}"
CONCLUSION = "\\section{Conclusion}"
AVAILABILITY = "\\section*{Data and code availability}"
APPENDIX = "\\appendix"

intro_full = seg(INTRO, RELATED)
related = seg(RELATED, MODEL)
model = seg(MODEL, RESULTS)              # includes \input{tables_generated}
results_full = seg(RESULTS, DISCUSSION)  # source for supplementary extensions
discussion_full = seg(DISCUSSION, LIMITATIONS)
limitations_full = seg(LIMITATIONS, CONCLUSION)
conclusion_full = seg(CONCLUSION, AVAILABILITY)
appendix = seg(APPENDIX)

CHANNELS = "\\subsection{Screening and fining are different instruments}"
PROVIDERS = "\\subsection{Provider-scoped marks make concentration a safety variable}"
MULTISTABILITY = "\\subsection{Multistability, and where the exclusion is actually paid}"
LIABILITY = "\\subsection{What identity is worth as liability varies}"
channels = seg(CHANNELS, PROVIDERS)
providers = seg(PROVIDERS, MULTISTABILITY)
extensions = seg(LIABILITY, DISCUSSION)

# The venue-neutral data statement and its \bibliography sit between the
# conclusion and the appendix and are both replaced here, so nothing from that
# stretch is carried over.  Assert that it holds no section we meant to keep.
_skipped = seg(AVAILABILITY, APPENDIX)
assert "\\section{" not in _skipped, \
    "a numbered section has appeared between the conclusion and the appendix"

# ------------------------------------------------------------------- edits
# The venue-neutral introduction previews every result in detail.  That makes
# sense when the paper has to establish its policy setting, but it repeats the
# Results section and delays the nonlinear object that CSF editors assess at
# triage.  The CSF version keeps the problem, gap, flow and main mechanisms in
# a dedicated compact introduction.
intro = (HERE / "_intro_csf.tex").read_text(encoding="utf-8")
_intro_first = next(l for l in intro.splitlines()
                    if l.strip() and not l.lstrip().startswith("%"))
assert _intro_first.startswith("\\section{Introduction}"), \
    "_intro_csf.tex must open with the Introduction section"
assert "\\label{sec:intro}" in intro, "_intro_csf.tex lost \\label{sec:intro}"
_iw = prose_words(intro)
assert 550 <= _iw <= 900, (
    f"CSF introduction is {_iw} prose words; the budget is 550--900 "
    f"(the venue-neutral introduction is {prose_words(intro_full)})")

# CSF demotes Related work from a standalone section to a subsection of the
# Introduction, and runs it at about half length.  The venue publishes compact
# papers, and four and a half pages before the model is too much of a
# twenty-five page manuscript.  The condensed text is venue-specific, so it
# lives in _related_csf.tex rather than in the venue-neutral master, which
# keeps its full-length section for the fallback venues.
#
related_full = related
related = (HERE / "_related_csf.tex").read_text(encoding="utf-8")

_first = next(l for l in related.splitlines()
              if l.strip() and not l.lstrip().startswith("%"))
assert _first.startswith("\\subsection{Related work}"), \
    "_related_csf.tex must open as a subsection, not a section"
assert "\\label{sec:related}" in related, "_related_csf.tex lost \\label{sec:related}"

# Compression is intentional, so the CSF subsection need not carry every
# citation in the venue-neutral review.  Guard the conceptual spine instead:
# evolutionary dynamics, greenbeards, forgeability, enforcement, AI races and
# recent papers from the target journal must all remain represented.
required_related = {
    "taylor1978ess", "hofbauer1998evolutionary", "perc2017statphys",
    "ren2021tolerance", "zhang2023expulsion", "yue2025reputation",
    "hamilton1964genetical1", "riolo2001evolution", "jansen2006altruism",
    "gardner2010greenbeards", "robson1990efficiency", "becker1968crime",
    "han2020regulate", "chan2025infrastructure", "otsuka2026aiidentity",
}
dropped_core = sorted(required_related - cite_keys(related))
assert not dropped_core, (
    "condensed Related work lost core references: " + ", ".join(dropped_core))

_rw = prose_words(related)
assert 300 <= _rw <= 550, (
    f"condensed Related work is {_rw} prose words; the budget is 300--550 "
    f"(the full-length section is {prose_words(related_full) + prose_words(block('related'))})")

model = insert_before(
    model,
    "The distinction between the two dynamics is not a robustness check",
    block("model"),
    "model",
)
model = sub(
    model,
    "\\input{tables_generated}",
    "\\input{table_race_generated}",
    "main-table-split",
)
model = sub(
    model,
    "Section~\\ref{sec:channels}\nseparates the two.",
    "Supplementary Section S2 separates the two.",
    "supplement-channels-reference",
)
model = sub(
    model,
    "Section~\\ref{sec:liability} reopens it and measures what\nidentity is worth as a function of $L$.",
    "Supplementary Section S4 reopens it and measures what identity is worth\n"
    "as a function of $L$.",
    "supplement-liability-reference",
)

# The CSF main paper now has a dedicated results narrative.  It preserves the
# exact equations and reported values but gives priority to the six nonlinear
# mechanisms promised by the title.  Secondary policy experiments and all
# derivations remain available in supplementary.tex.
results = (HERE / "_results_csf.tex").read_text(encoding="utf-8")
assert results.count("% BIFURCATION_TABLE") == 1
results = results.replace("% BIFURCATION_TABLE", block("results"))
assert 2000 <= prose_words(results) <= 3200, (
    f"compact Results is {prose_words(results)} prose words")
for required in (
    "prop:reciprocity", "prop:spoof", "prop:fines", "prop:nucleation",
    "prop:mimicry", "prop:bistable", "prop:exclusion", "cor:rescue",
):
    assert f"\\label{{{required}}}" in results, f"Results lost {required}"

discussion = (HERE / "_discussion_csf.tex").read_text(encoding="utf-8")
conclusion = (HERE / "_conclusion_csf.tex").read_text(encoding="utf-8")
assert 550 <= prose_words(discussion) <= 1000, (
    f"compact Discussion is {prose_words(discussion)} prose words")
assert 120 <= prose_words(conclusion) <= 250, (
    f"compact Conclusion is {prose_words(conclusion)} prose words")

# -------------------------------------------------- back matter and supplement
# Elsevier asks for the generative-AI declaration in a dedicated section at
# the end of the manuscript ahead of the references.  Proofs and secondary
# experiments are delivered as a separately compiled supplementary PDF.
declarations = block("declarations") + "\n\\printcredits\n\n"
bibliography = "\n\\bibliographystyle{elsarticle-num-names}\n\\bibliography{refs}\n\n"

# the appendix segment runs to the end of the master, so it carries
# \end{document} with it; that has to come back out and go last
END = "\\end{document}"
assert appendix.rstrip().endswith(END), "master no longer ends with \\end{document}"
appendix = appendix.rstrip()[:-len(END)].rstrip() + "\n\n"

out = "".join([
    front,
    intro,
    related,
    model,
    results,
    discussion,
    conclusion,
    declarations,
    bibliography,
    END + "\n",
])

out = cas_floats(out)

bad = [l for l in out.splitlines()
       if "---" in l and not l.lstrip().startswith("%")]
assert not bad, "em-dash found in assembled manuscript: " + repr(bad[:3])

stray = FLOAT_RE.findall(out)
assert not stray, f"bare float specifier left in main.tex: {stray[:3]}"

(HERE / "main.tex").write_text(out, encoding="utf-8")

# Split deterministic generated tables by destination.  This keeps one model
# table in the main paper and puts the parameter grids in the supplement while
# retaining the build pipeline as their single source of truth.
def generated_table(text, label):
    matches = [m.group(0) for m in re.finditer(
        r"\\begin\{table\}(?:\[[^]]*\])?.*?\\end\{table\}", text, re.S)
        if f"\\label{{{label}}}" in m.group(0)]
    assert len(matches) == 1, f"expected one generated table {label}"
    return matches[0] + "\n"


tables_text = (SRC.parent / "tables_generated.tex").read_text(encoding="utf-8")
robustness_text = (SRC.parent / "robustness_generated.tex").read_text(encoding="utf-8")
table_files = {
    "table_race_generated.tex": cas_floats(
        generated_table(tables_text, "tab:race")),
    "table_thresholds_generated.tex": generated_table(tables_text, "tab:thresholds"),
    "table_pools_generated.tex": generated_table(tables_text, "tab:pools"),
    "table_providers_generated.tex": generated_table(tables_text, "tab:providers"),
    "table_robustness_generated.tex": generated_table(robustness_text, "tab:robustness"),
}

# The main-table caption identifies the matrix and points to the threshold;
# the Results section, rather than the caption, carries the interpretation.
table_files["table_race_generated.tex"] = re.sub(
    r"\\caption\{.*?\}\s*\\label\{tab:race\}",
    lambda _: "\\caption{Exact interaction layer: expected race payoff $A$ "
    "(left) and expected unsafe actions $M$ (right) for row against column "
    "conduct.}\n\\label{tab:race}",
    table_files["table_race_generated.tex"],
    count=1,
    flags=re.S,
)
for name, text in table_files.items():
    (HERE / name).write_text(text, encoding="utf-8")

# Assemble a self-contained supplementary document.  Cross-references to the
# main paper are resolved through main.aux at build time; the submitted PDF is
# final-form supplementary material and does not depend on that file at read
# time.
channels_supp = channels.replace(CHANNELS, "\\section{Screening and fining channels}", 1)
providers_supp = providers.replace(PROVIDERS, "\\section{Provider-scoped credentials}", 1)
providers_supp = insert_before(
    providers_supp,
    "\\begin{figure}[t]",
    "\\input{table_providers_generated}\n\n",
    "supplement-provider-table",
)
extensions_supp = extensions.replace(
    LIABILITY, "\\section{Liability and robustness}", 1)
extensions_supp = extensions_supp.replace(
    "\\label{sec:robustness}", "\\label{sec:supp-robustness}", 1)
extensions_supp = sub(
    extensions_supp,
    "\\paragraph{Design pools.}",
    "\\input{table_pools_generated}\n\n\\paragraph{Design pools.}",
    "supplement-pools-table",
)
extensions_supp = sub(
    extensions_supp,
    "\\input{robustness_generated}",
    "\\input{table_robustness_generated}",
    "supplement-robustness-table",
)
proofs = sub(
    appendix,
    "\\appendix\n\n\\section{Proofs}\n\\label{app:proofs}",
    "\\section{Analytical derivations}\n\\label{app:proofs}",
    "supplement-proof-heading",
)
supplement = "".join([
    (HERE / "_supplement_front.tex").read_text(encoding="utf-8"),
    "\\section{Threshold sweeps}\n"
    "Supplementary Table~\\ref{tab:thresholds} reports how dues and fines move "
    "the closed-form invasion and nucleation boundaries.\n\n"
    "\\input{table_thresholds_generated}\n\n",
    channels_supp,
    providers_supp,
    extensions_supp,
    proofs,
    "\\bibliographystyle{unsrtnat}\n\\bibliography{refs}\n\n",
    END + "\n",
])
(HERE / "supplementary.tex").write_text(supplement, encoding="utf-8")

# The figures were copied here once, by hand, and nothing refreshed them: a
# re-rendered figure left the package silently showing the old one, with a
# caption describing the new. Restage them from results/ on every assembly,
# and fail loudly rather than build against a figure that is not there.
figdir = HERE / "figures"
figdir.mkdir(exist_ok=True)
rendered = sorted((SRC.parents[1] / "results" / "figures").glob("fig*.pdf"))
assert rendered, "no rendered figures: run scripts/make_figures.py first"
for pdf in rendered:
    shutil.copy2(pdf, figdir / pdf.name)
stale = sorted(p.name for p in figdir.glob("fig*.pdf")
               if p.name not in {q.name for q in rendered})
assert not stale, f"figures in the package that results/ no longer renders: {stale}"

print(f"wrote main.tex ({out.count(chr(10))} lines) and "
      f"supplementary.tex ({supplement.count(chr(10))} lines); "
      f"{len(table_files)} generated tables split, "
      f"{len(rendered)} figures restaged")

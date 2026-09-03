"""Assemble the Physica A version of the manuscript.

Reads the venue-neutral ../paper/main.tex, keeps most of its body verbatim, and

  * swaps in an elsarticle single-column front matter (Physica A's class is
    elsarticle, not the cas-sc used by the CSF package next door),
  * replaces the long venue-neutral introduction with a compact one,
  * replaces the Related work section with a half-length subsection of the
    Introduction that cites this journal's own work on the same mechanisms,
  * folds Limitations into the Discussion as a subsection,
  * swaps the venue-neutral data statement for the Elsevier declarations,
    including the CRediT statement that elsarticle, unlike cas-sc, does not
    generate,
  * moves proofs and secondary experiments into a separate supplement,
  * switches the bibliography to elsarticle-num.

The structural changes take the manuscript from seven top-level sections to
five: 1 Introduction (1.1 Related work), 2 Model, 3 Results, 4 Discussion
(4.1 Limitations), 5 Conclusion.

Unlike the CSF package this one keeps the multistability section whole. It
carries Theorem 1, the (sigma, r) regime map and the basin-measure analysis,
which are the results a statistical-physics readership is here for, and the
CSF package predates the theorem entirely.

Run from paper-physicaa/:  python assemble.py
"""

import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "paper" / "main.tex"

SOURCE = SRC.read_text(encoding="utf-8")


def seg(start, end=None):
    """Body from the line beginning ``start`` up to the line beginning ``end``.

    Markers rather than line numbers, so that adding a sentence to the master
    does not silently shift eight boundaries at once.
    """
    i = SOURCE.find(start)
    assert i != -1, "segment marker not found: %r" % (start,)
    assert SOURCE.count("\n" + start) + SOURCE.startswith(start) == 1, \
        "segment marker is not unique at line start: %r" % (start,)
    if end is None:
        return SOURCE[i:]
    j = SOURCE.find(end, i)
    assert j != -1, "segment end marker not found after %r: %r" % (start, end)
    return SOURCE[i:j]


def part(name):
    return (HERE / ("_%s.tex" % name)).read_text(encoding="utf-8")


def sub(text, old, new, label):
    assert text.count(old) == 1, \
        "substitution anchor not unique for %s: %r" % (label, old[:60])
    return text.replace(old, new)


CITE_RE = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")
ARG_RE = re.compile(
    r"\\(?:cite[a-zA-Z]*|ref|eqref|autoref|label)\*?(?:\[[^\]]*\])*\{[^}]*\}")


def cite_keys(text):
    return {k.strip()
            for m in CITE_RE.finditer(text)
            for k in m.group(1).split(",") if k.strip()}


def prose_words(text):
    """Word count with citation and cross-reference arguments removed."""
    text = re.sub(r"^\s*%.*$", "", text, flags=re.M)
    text = ARG_RE.sub(" ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = re.sub(r"[{}$&_^%~]", " ", text)
    return len([w for w in text.split() if any(c.isalpha() for c in w)])


# ---------------------------------------------------------------- segments
INTRO = "\\section{Introduction}"
RELATED = "\\section{Related work}"
MODEL = "\\section{Model}"
RESULTS = "\\section{Results}"
NUCLEATION = "\\subsection{No badge starts a club by itself}"
ABLATION = "\\subsection{What the badge is doing, and what it is not}"
MIMICRY = "\\subsection{Collapse hollows out the mark before it degrades the conduct}"
CHANNELS = "\\subsection{Screening and fining are different instruments}"
PROVIDERS = "\\subsection{Provider-scoped marks make concentration a safety variable}"
MULTISTABILITY = "\\subsection{Multistability, and where the exclusion is actually paid}"
LIABILITY = "\\subsection{What identity is worth as liability varies}"
DISCUSSION = "\\section{Discussion}"
LIMITATIONS = "\\section{Limitations}"
CONCLUSION = "\\section{Conclusion}"
AVAILABILITY = "\\section*{Data and code availability}"
APPENDIX = "\\appendix"

intro_full = seg(INTRO, RELATED)
related_full = seg(RELATED, MODEL)
model = seg(MODEL, RESULTS)
# Results is cut into four so that the two instrument results can move.  They
# are the two that cost the most page for the least argument: the main line
# runs threshold -> spoof -> fines -> mimicry -> multistability, and neither
# nucleation nor the ablation scan is on it.  Measured saving, two printed
# pages of twenty-two.
results_a = seg(RESULTS, NUCLEATION)       # threshold, spoof, fines
nucleation = seg(NUCLEATION, ABLATION)     # -> supplement
ablation = seg(ABLATION, MIMICRY)          # -> supplement
results_b = seg(MIMICRY, CHANNELS)         # mimicry
channels = seg(CHANNELS, PROVIDERS)        # -> supplement
providers = seg(PROVIDERS, MULTISTABILITY)  # -> supplement
exclusion = seg(MULTISTABILITY, LIABILITY)
extensions = seg(LIABILITY, DISCUSSION)    # -> supplement
discussion = seg(DISCUSSION, LIMITATIONS)
limitations = seg(LIMITATIONS, CONCLUSION)
conclusion = seg(CONCLUSION, AVAILABILITY)
appendix = seg(APPENDIX)

# Nothing between the conclusion and the appendix is carried over: it is the
# venue-neutral data statement and \bibliography, both replaced below.
_skipped = seg(AVAILABILITY, APPENDIX)
assert "\\section{" not in _skipped, \
    "a numbered section has appeared between the conclusion and the appendix"

# ---------------------------------------------------- front matter budgets
# The guide-for-authors page is behind a bot wall, so these two caps come from
# Elsevier house style rather than from a fetched copy of the Physica A guide.
# They are checked here so that a later edit cannot quietly push the abstract
# past a limit nobody rechecked, and SUBMISSION_CHECKLIST.md item 1 asks for
# the live page to be confirmed by eye before submitting.
front = part("front")
_abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", front, re.S)
assert _abstract, "_front.tex has no abstract"
_aw = prose_words(_abstract.group(1))
assert 150 <= _aw <= 250, \
    "abstract is %d words; the working budget is 150--250" % _aw

_keyword = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", front, re.S)
assert _keyword, "_front.tex has no keyword block"
_kw = [k for k in _keyword.group(1).split("\\sep") if k.strip()]
assert 3 <= len(_kw) <= 6, \
    "%d keywords; Elsevier asks for at most six" % len(_kw)

# ------------------------------------------------------------------- edits
intro = part("intro_physa")
_first = next(l for l in intro.splitlines()
              if l.strip() and not l.lstrip().startswith("%"))
assert _first.startswith("\\section{Introduction}"), \
    "_intro_physa.tex must open with the Introduction section"
assert "\\label{sec:intro}" in intro, "_intro_physa.tex lost \\label{sec:intro}"
_iw = prose_words(intro)
assert 900 <= _iw <= 1500, (
    "Physica A introduction is %d prose words; the budget is 900--1500 "
    "(the venue-neutral introduction is %d)" % (_iw, prose_words(intro_full)))

related = part("related_physa")
_first = next(l for l in related.splitlines()
              if l.strip() and not l.lstrip().startswith("%"))
assert _first.startswith("\\subsection{Related work}"), \
    "_related_physa.tex must open as a subsection, not a section"
assert "\\label{sec:related}" in related, \
    "_related_physa.tex lost \\label{sec:related}"

# Compression is intentional, so the condensed subsection need not carry every
# citation in the venue-neutral review.  Guard the conceptual spine instead,
# plus the three Physica A papers that are the whole point of rewriting it.
required_related = {
    "taylor1978ess", "hofbauer1998evolutionary", "perc2017statphys",
    "hamilton1964genetical1", "riolo2001evolution", "jansen2006altruism",
    "gardner2010greenbeards", "garcia2014evilbeards",
    "hammond2006ethnocentrism", "choi2007parochial",
    "robson1990efficiency", "becker1968crime", "dranove2010quality",
    "domingos2026falling", "han2020regulate", "chan2025infrastructure",
    "otsuka2026aiidentity",
    "quan2019exclusion", "zhu2025adaptive", "zhang2026trust",
}
dropped_core = sorted(required_related - cite_keys(related))
assert not dropped_core, \
    "condensed Related work lost core references: " + ", ".join(dropped_core)
assert not cite_keys(related) & {"ren2021tolerance", "zhang2023expulsion",
                                 "yue2025reputation"}, \
    "the CSF courtesy citations are still in the Physica A Related work"

_rw = prose_words(related)
assert 550 <= _rw <= 900, (
    "condensed Related work is %d prose words; the budget is 550--900 "
    "(the full-length section is %d)" % (_rw, prose_words(related_full)))

# Cross-references from kept text into the sections that move.  Every one goes
# through \ref rather than a typed "Section S4": xr resolves it against
# supplementary.aux, so reordering the supplement below cannot leave the main
# text pointing at the wrong section.  What the rewrite adds is the word
# "Supplementary", without which "Section S4" reads as a typo.
model = sub(
    model,
    "Section~\\ref{sec:channels}\nseparates the two.",
    "Supplementary Section~\\ref{sec:channels} separates the two.",
    "model-channels-pointer")
model = sub(
    model,
    "Section~\\ref{sec:liability} reopens it and measures what\n"
    "identity is worth as a function of $L$.",
    "Supplementary Section~\\ref{sec:liability} reopens it and measures what\n"
    "identity is worth as a function of $L$.",
    "model-liability-pointer")
model = sub(
    model,
    "Section~\\ref{sec:robustness} and the parameter-resolved experiments of",
    "Supplementary Section~\\ref{sec:supp-robustness} and the "
    "parameter-resolved experiments of",
    "model-robustness-pointer")
#% The master's other reference to prop:futility sits at line 540, inside the
#% Related work *section*, which this package replaces wholesale with
#% _related_physa.tex. It therefore needs no rewrite: it leaves with the text
#% around it. The assertion in sub() caught an attempt to patch it here.
model = sub(
    model,
    "\\input{tables_generated}",
    "\\input{table_race_generated}",
    "main-table-split")

results_a = sub(
    results_a,
    "Proofs of all propositions appear after the main text.",
    "Proofs of all propositions appear in Supplementary "
    "Section~\\ref{app:proofs}.",
    "results-proof-pointer")
results_a = sub(
    results_a,
    "the work by itself, and Section~\\ref{sec:liability} shows the identity layer\n"
    "becoming worthless there.",
    "the work by itself, and Supplementary Section~\\ref{sec:liability} shows\n"
    "the identity layer becoming worthless there.",
    "results-liability-pointer")
results_a = sub(
    results_a,
    "Table~\\ref{tab:thresholds} sweeps the dues against the fine and shows the\n"
    "two moving in opposite directions across the whole grid.",
    "Supplementary Table~\\ref{tab:thresholds} sweeps the dues against the fine\n"
    "and shows the two moving in opposite directions across the whole grid.",
    "results-thresholds-pointer")
results_a = sub(
    results_a,
    "Section~\\ref{sec:nucleation} shows a club also needs.",
    "Supplementary Section~\\ref{sec:nucleation} shows a club also needs.",
    "results-nucleation-pointer-a")

results_b = sub(
    results_b,
    "Section~\\ref{sec:nucleation}. Clustering is what makes clubs possible at all,",
    "Supplementary Section~\\ref{sec:nucleation}. Clustering is what makes\n"
    "clubs possible at all,",
    "results-nucleation-pointer-b")

exclusion = sub(
    exclusion,
    "The reason is Proposition~\\ref{prop:futility}.",
    "The reason is Supplementary Proposition~\\ref{prop:futility}.",
    "exclusion-futility-pointer")

# The master delivered every generated table through one \input in the Model
# section. Splitting them means each surviving table is re-inserted at its
# point of use, and tab:ablation now travels with the subsection that cites it.
ablation = sub(
    ablation,
    "Table~\\ref{tab:ablation} scores four arms.",
    "\\input{table_ablation_generated}\n\n"
    "Table~\\ref{tab:ablation} scores four arms.",
    "supplement-ablation-table")

# elsarticle's preprint measure is narrower than the master's one-inch-margin
# 11pt page, and tab:outgroup overflows it by 27pt at \small. Six columns of
# short numbers do not need the default column separation.
exclusion = sub(
    exclusion,
    "\\label{tab:outgroup}\n\\small\n\\begin{tabular}{lrrrrr}",
    "\\label{tab:outgroup}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
    "\\begin{tabular}{lrrrrr}",
    "outgroup-table-width")

discussion = sub(
    discussion,
    "\\paragraph{Measure attestation integrity, not conduct.} Section~\\ref{sec:channels}\n"
    "shows a regime",
    "\\paragraph{Measure attestation integrity, not conduct.} Supplementary\n"
    "Section S2 shows a regime",
    "discussion-channels-pointer")

# Limitations becomes a subsection of the Discussion rather than a section of
# its own: five top-level sections read better at this length, and Elsevier
# reviewers expect limitations inside the discussion.
limitations = sub(
    limitations,
    "\\section{Limitations}",
    "\\subsection{Limitations}",
    "limitations-demotion")

# ------------------------------------------- back matter and main assembly
declarations = part("declarations_physa")
bibliography = "\n\\bibliographystyle{elsarticle-num-names}\n\\bibliography{refs}\n\n"

END = "\\end{document}"
assert appendix.rstrip().endswith(END), \
    "master no longer ends with \\end{document}"
appendix = appendix.rstrip()[:-len(END)].rstrip() + "\n\n"

out = "".join([
    front,
    intro,
    related,
    model,
    results_a,
    results_b,
    exclusion,
    discussion,
    limitations,
    conclusion,
    declarations,
    bibliography,
    END + "\n",
])

bad = [l for l in out.splitlines()
       if "---" in l and not l.lstrip().startswith("%")]
assert not bad, "em-dash found in assembled manuscript: " + repr(bad[:3])

# Every reference from the main text into the supplement must be worded
# "Supplementary Section 5" and not "Section 5", or the reader looks for a
# section of the main paper that is not there.  xr makes the number correct on
# its own; nothing makes the wording correct on its own, so check it here.
# The check is on the rendered word order, so it tolerates the line break that
# LaTeX source puts between "Supplementary" and the reference.
moved_labels = ["sec:nucleation", "sec:ablation", "sec:channels",
                "sec:providers", "sec:liability", "sec:supp-robustness",
                "app:proofs", "prop:futility", "prop:nucleation",
                "tab:ablation", "tab:thresholds", "tab:pools",
                "tab:providers", "tab:robustness", "fig:nucleation",
                "fig:channels", "fig:providers", "fig:robustness"]
flat = re.sub(r"\s+", " ", out)
unmarked = sorted({
    lab for lab in moved_labels
    for m in re.finditer(r"\\ref\{%s\}" % re.escape(lab), flat)
    if "Supplementary" not in flat[max(0, m.start() - 40):m.start()]
})
assert not unmarked, (
    "main text references supplement content without saying "
    "\"Supplementary\": %s" % unmarked)

# The reverse mistake: a label that no longer exists anywhere is a silent "??".
defined_here = set(re.findall(r"\\label\{([^}]*)\}", out))
for lab in moved_labels:
    assert lab not in defined_here, \
        "%s is still defined in the main text but treated as moved" % lab

(HERE / "main.tex").write_text(out, encoding="utf-8")

# ------------------------------------------------------- generated tables
def generated_table(text, label):
    matches = [m.group(0) for m in re.finditer(
        r"\\begin\{table\}(?:\[[^]]*\])?.*?\\end\{table\}", text, re.S)
        if "\\label{%s}" % label in m.group(0)]
    assert len(matches) == 1, "expected one generated table %s" % label
    return matches[0] + "\n"


tables_text = (SRC.parent / "tables_generated.tex").read_text(encoding="utf-8")
robustness_text = (SRC.parent / "robustness_generated.tex").read_text(encoding="utf-8")
table_files = {
    "table_race_generated.tex": generated_table(tables_text, "tab:race"),
    "table_ablation_generated.tex": generated_table(tables_text, "tab:ablation"),
    "table_thresholds_generated.tex": generated_table(tables_text, "tab:thresholds"),
    "table_pools_generated.tex": generated_table(tables_text, "tab:pools"),
    "table_providers_generated.tex": generated_table(tables_text, "tab:providers"),
    "table_robustness_generated.tex": generated_table(robustness_text, "tab:robustness"),
}
for name, text in table_files.items():
    (HERE / name).write_text(text, encoding="utf-8")

# ------------------------------------------------------------- supplement
# The two demoted results keep their own headings; they were subsections of
# Results and become sections of the supplement, so the promotion is a plain
# heading swap.  They go first, before the mechanisms that were always
# supplementary, because the main text sends the reader to them by name.
nucleation_supp = nucleation.replace(
    NUCLEATION, "\\section{No badge starts a club by itself}", 1)
ablation_supp = ablation.replace(
    ABLATION, "\\section{What the badge is doing, and what it is not}", 1)

channels_supp = channels.replace(
    CHANNELS, "\\section{Screening and fining channels}", 1)
providers_supp = providers.replace(
    PROVIDERS, "\\section{Provider-scoped credentials}", 1)
providers_supp = sub(
    providers_supp,
    "\\begin{figure}[t]",
    "\\input{table_providers_generated}\n\n\\begin{figure}[t]",
    "supplement-provider-table")
extensions_supp = extensions.replace(
    LIABILITY, "\\section{Liability and robustness}", 1)
extensions_supp = extensions_supp.replace(
    "\\label{sec:robustness}", "\\label{sec:supp-robustness}", 1)
extensions_supp = sub(
    extensions_supp,
    "\\paragraph{Design pools.}",
    "\\input{table_pools_generated}\n\n\\paragraph{Design pools.}",
    "supplement-pools-table")
extensions_supp = sub(
    extensions_supp,
    "\\input{robustness_generated}",
    "\\input{table_robustness_generated}",
    "supplement-robustness-table")
proofs = sub(
    appendix,
    "\\appendix\n\n\\section{Proofs}\n\\label{app:proofs}",
    "\\section{Analytical derivations}\n\\label{app:proofs}",
    "supplement-proof-heading")

supplement = "".join([
    part("supplement_front"),
    "\\section{Threshold sweeps}\n"
    "Supplementary Table~\\ref{tab:thresholds} reports how dues and fines move "
    "the closed-form invasion and nucleation boundaries.\n\n"
    "\\input{table_thresholds_generated}\n\n",
    nucleation_supp,
    ablation_supp,
    channels_supp,
    providers_supp,
    extensions_supp,
    proofs,
    "\\bibliographystyle{elsarticle-num-names}\n\\bibliography{refs}\n\n",
    END + "\n",
])

bad = [l for l in supplement.splitlines()
       if "---" in l and not l.lstrip().startswith("%")]
assert not bad, "em-dash found in the supplement: " + repr(bad[:3])

(HERE / "supplementary.tex").write_text(supplement, encoding="utf-8")

# ------------------------------------------------------------------ figures
# Restage from results/ on every assembly rather than trusting a hand copy: a
# re-rendered figure otherwise leaves the package showing the old one under a
# caption describing the new.
figdir = HERE / "figures"
figdir.mkdir(exist_ok=True)
rendered = sorted((SRC.parents[1] / "results" / "figures").glob("fig*.pdf"))
assert rendered, "no rendered figures: run scripts/make_figures.py first"
for pdf in rendered:
    shutil.copy2(pdf, figdir / pdf.name)
stale = sorted(p.name for p in figdir.glob("fig*.pdf")
               if p.name not in {q.name for q in rendered})
assert not stale, \
    "figures in the package that results/ no longer renders: %s" % stale

# --------------------------------------------------------------- highlights
# Elsevier caps a highlight at 85 characters including spaces and allows three
# to five of them.  Checked here rather than in a reviewer's rejection e-mail.
hl = (HERE / "highlights.tex").read_text(encoding="utf-8")
bullets = re.findall(r"^\s*\\item\s+(.*\S)\s*$", hl, re.M)
assert 3 <= len(bullets) <= 5, \
    "Elsevier wants three to five highlights; found %d" % len(bullets)
over = [(len(b), b) for b in bullets if len(b) > 85]
assert not over, "highlights over the 85-character cap: %r" % over

print("wrote main.tex (%d lines, %d prose words) and supplementary.tex "
      "(%d lines); %d generated tables split, %d figures restaged, "
      "%d highlights within the cap"
      % (out.count("\n"), prose_words(out), supplement.count("\n"),
         len(table_files), len(rendered), len(bullets)))

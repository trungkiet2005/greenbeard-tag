"""Assemble the Chaos, Solitons & Fractals version of the manuscript.

Reads the venue-neutral ../paper/main.tex, keeps most of its body verbatim, and

  * swaps in an elsarticle single-column front matter (CSF requirement),
  * inserts the nonlinear-dynamics framing that the venue's scope asks for,
  * replaces Related work with a half-length subsection of the Introduction,
  * folds Limitations into the Discussion as a subsection,
  * swaps the venue-neutral data statement for the Elsevier declarations,
  * emits the appendix ahead of the reference list,
  * switches the bibliography to the numbered Elsevier style.

The three structural changes take the manuscript from seven top-level sections
to five: 1 Introduction (1.1 Related work), 2 Model, 3 Results, 4 Discussion
(4.1 Limitations), 5 Conclusion.

Run from paper-csf/:  python assemble.py
"""

import re
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

lines = SRC.read_text(encoding="utf-8").splitlines(keepends=True)


def seg(a, b):
    """1-indexed inclusive line range of the source manuscript."""
    return "".join(lines[a - 1:b])


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
intro = seg(101, 274)        # Introduction
related = seg(275, 398)      # Related work
model = seg(399, 544)        # Model, including \input{tables_generated}
results = seg(545, 1206)     # Results, including \input{robustness_generated}
discussion = seg(1207, 1270)
limitations = seg(1271, 1318)
conclusion = seg(1319, 1350)
appendix = seg(1369, len(lines))

assert intro.startswith(r"\section{Introduction}")
assert related.startswith(r"\section{Related work}")
assert model.startswith(r"\section{Model}")
assert results.startswith(r"\section{Results}")
assert discussion.startswith(r"\section{Discussion}")
assert limitations.startswith(r"\section{Limitations}")
assert conclusion.startswith(r"\section{Conclusion}")
assert appendix.startswith(r"\appendix")

# ------------------------------------------------------------------- edits
intro = insert_before(intro, "\\paragraph{Results.}", block("intro"), "intro")

# CSF demotes Related work from a standalone section to a subsection of the
# Introduction, and runs it at about half length.  The venue publishes compact
# papers, and four and a half pages before the model is too much of a
# twenty-five page manuscript.  The condensed text is venue-specific, so it
# lives in _related_csf.tex rather than in the venue-neutral master, which
# keeps its full-length section for the fallback venues.
#
# Twenty-eight of the references in that section are cited nowhere else in the
# paper, so a careless compression deletes them from the bibliography without
# any visible symptom.  The assertion below is what makes that impossible: the
# condensed text has to carry every key the full-length text carried.
related_full = related
related = (HERE / "_related_csf.tex").read_text(encoding="utf-8")

_first = next(l for l in related.splitlines()
              if l.strip() and not l.lstrip().startswith("%"))
assert _first.startswith("\\subsection{Related work}"), \
    "_related_csf.tex must open as a subsection, not a section"
assert "\\label{sec:related}" in related, "_related_csf.tex lost \\label{sec:related}"

# _blocks_related.tex is no longer inserted into the manuscript: its content is
# folded into the condensed subsection.  It is kept because it is the record of
# what the CSF-specific framing paragraph cited, and the two assertions below
# hold the condensed text to that record.
dropped = sorted((cite_keys(related_full) | cite_keys(block("related")))
                 - cite_keys(related))
assert not dropped, (
    f"condensing Related work dropped {len(dropped)} reference(s) that the "
    f"full-length section cited: {', '.join(dropped)}")

added = sorted(cite_keys(related) - cite_keys(related_full) - cite_keys(block("related")))
assert not added, f"_related_csf.tex cites keys the source never did: {', '.join(added)}"

_rw = prose_words(related)
assert 620 <= _rw <= 790, (
    f"condensed Related work is {_rw} prose words; the budget is about 700 "
    f"(the full-length section is {prose_words(related_full) + prose_words(block('related'))})")

model = insert_before(
    model,
    "The distinction between the two is not a robustness check",
    block("model"),
    "model",
)

results = sub(
    results,
    "\\section{Results}\n\\label{sec:results}\n\n",
    "\\section{Results}\n\\label{sec:results}\n\n" + block("results"),
    "results",
)

results = sub(
    results,
    "They cross at $r = 0.077$\n(Figure~\\ref{fig:invasion}C).",
    "They cross at $r = 0.077$\n(Figure~\\ref{fig:invasion}C). At that point both transversal eigenvalues of\n"
    "Equation~\\eqref{eq:eigen} at the club vertex vanish together, which makes\n"
    "$(r, \\sigma) = (0.077, 0.938)$ a codimension-two organising centre for the\n"
    "identity layer: it is the parameter pair at which the ecosystem switches which\n"
    "of its two failure modes it meets first.",
    "codim-two",
)

results = sub(
    results,
    "buy its way out with penalties. Figure~\\ref{fig:instruments}B shows the\ndivergence.",
    "buy its way out with penalties. The fine here is levied by an authority rather\n"
    "than by peers, so it escapes the second-order free-rider problem that\n"
    "constrains punishment in evolutionary settings where the punishing itself is\n"
    "costly and voluntary~\\citep{szolnoki2017second}; what constrains it instead is\n"
    "detection alone. Figure~\\ref{fig:instruments}B shows the divergence.",
    "peer-punishment-contrast",
)

discussion = sub(
    discussion,
    "Multistable evolutionary dynamics are\ncommon~\\citep{hofbauer1998evolutionary,sandholm2010population}",
    "Multistable evolutionary dynamics are\n"
    "common~\\citep{hofbauer1998evolutionary,hofbauer2003evolutionary,sandholm2010population},"
    "\nand so is the practice of summarising a population by a single averaged\n"
    "state~\\citep{perc2017statphys}",
    "multistability-cites",
)

# A standalone Limitations section is a convention of the AI and social-science
# venues, not of a nonlinear-dynamics journal, where it belongs inside the
# discussion.  Demoting it also takes the manuscript from seven top-level
# sections to five.  It stays a subsection rather than a run-in \paragraph so
# that it keeps its own entry in the table of contents and the PDF bookmarks:
# at four hundred words a referee should be able to find it.
limitations = sub(
    limitations,
    "\\section{Limitations}\n\\label{sec:limitations}",
    "\\subsection{Limitations}\n\\label{sec:limitations}",
    "limitations-demote",
)

limitations = sub(
    limitations,
    "inherited\ndeliberately so that our results are comparable with the sister studies, and it\nis not a model of any particular deployment.",
    "inherited\ndeliberately and without modification, so that every effect we report is\nattributable to the identity layer rather than to a change in the underlying\ngame, and it is not a model of any particular deployment.",
    "sister-studies",
)

limitations = sub(
    limitations,
    "reputation carried between encounters, and forgers who invest in better",
    "reputation carried between encounters in the manner of indirect\n"
    "reciprocity~\\citep{santos2018social}, and forgers who invest in better",
    "reputation-cite",
)

# -------------------------------------------------- back matter and appendix
# Order follows the elsarticle convention and the sibling AMC manuscript:
# declarations, then the CRediT statement that cas-sc builds from the
# \credit{} commands in the front matter, then the appendix, then the
# references.  The appendix used to be emitted after \bibliography, which put
# the proofs behind the reference list; every Elsevier template puts \appendix
# ahead of it.  Elsevier asks for the generative-AI declaration in a dedicated
# section at the end of the manuscript ahead of the references, which this
# order still satisfies.
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
    limitations,
    conclusion,
    declarations,
    appendix,
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

# the generated tables come from the same build as the results, so refresh
# them from ../paper/ every time rather than letting a stale copy drift
for name in ("tables_generated", "robustness_generated"):
    text = (SRC.parent / f"{name}.tex").read_text(encoding="utf-8")
    (HERE / f"{name}.tex").write_text(cas_floats(text), encoding="utf-8")

print(f"wrote main.tex: {out.count(chr(10))} lines, "
      f"tables_generated.tex and robustness_generated.tex refreshed")

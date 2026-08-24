# Chaos, Solitons & Fractals submission package

This directory contains the submission-ready *Chaos, Solitons & Fractals*
(CSF) version of the greenbeard-tag manuscript. It uses Elsevier's CAS bundle
(`cas-sc.cls`, single column), numbered citations, journal-specific front
matter, declarations, highlights and a separate graphical abstract.

## Build

```text
python assemble.py
python make_submission_files.py
latexmk -pdf main.tex
latexmk -pdf highlights.tex
latexmk -pdf graphical_abstract.tex  # editable vector backup
```

`assemble.py` is the source of truth for `main.tex`. It combines the reusable
body in `../paper/main.tex` with the CSF-specific front matter and editorial
blocks in this directory. Its guarded substitutions fail if an expected source
passage changes, if the compact introduction leaves its word budget, or if a
required core citation is lost.

## Verified state (24 August 2026)

- 25 PDF pages: one unnumbered CAS highlights sheet and 24 numbered manuscript
  pages.
- 0 LaTeX errors, 0 undefined references/citations and 59 printed references.
- One 117 pt overfull-box report generated inside the CAS front-matter macro;
  visual inspection confirms that no content protrudes from the page.
- Abstract: 221 words. Highlights: five items, 65--73 characters each.
- Introduction plus Related work: 977 words.
- Five top-level sections, followed by declarations, Appendix A and references.
- Nine manuscript figures and the graphical abstract are vector PDFs with
  embedded fonts; no Type 3 fonts occur in `main.pdf`.

## Submission files

- `main.tex`, `refs.bib`, the CAS class/style files and `figures/`: editable
  LaTeX source package.
- `main.pdf`: compiled manuscript for final visual review.
- `highlights.docx` (or `highlights.tex`/`highlights.pdf`): separate editable
  highlights file.
- `graphical_abstract_ai.png`: primary graphical abstract (1752 x 898 px),
  rendered from the author-specified four-panel layout.
- `graphical_abstract.tex` and `graphical_abstract.pdf`: editable vector backup;
  `graphical_abstract_caption.txt` explains the concept and AI provenance.
- `declaration_of_interest.docx`: separate declaration requested by Elsevier.
- `cover_letter.md`: cover letter draft.
- `SUBMISSION_CHECKLIST.md`: author actions that cannot be completed from the
  repository, especially the archival data DOI and reviewer suggestions.

## Journal-specific editorial changes

The CSF version uses the title *Forgeable greenbeards: multistability and
hollow collapse in certified agent populations*. The abstract, introduction
and related-work section foreground nonlinear population flow, local stability,
transcritical edge bifurcations, separatrices and basin structure. The compact
introduction is in `_intro_csf.tex`; the compact literature synthesis is in
`_related_csf.tex`.

`_blocks_model.tex` makes the replicator flow and transversal eigenvalue
explicit. `_blocks_results.tex` provides a single inventory of the qualitative
transitions and distinguishes local edge bifurcations from global basin
statistics. `_blocks_declarations.tex` contains competing-interest, funding,
data-availability and generative-AI disclosures plus CRediT roles.

The revision also removes repeated previews of every result, corrects the
uniform-initialisation basin shares to 0.388 and 0.612, avoids presenting a rare
mixed attractor as a third globally important regime, and removes duplicated
limitations text. These are editorial and consistency corrections; the model
and computed results are unchanged.

## Template notes

The package includes `cas-sc.cls`, `cas-common.sty` and `thumbnails/` from CAS
bundle v2.4. Numbered citations use `elsarticle-num-names`, which keeps
`\citet` author names while printing square-bracket numbers. `_front.tex` also
contains a guarded compatibility shim for current expl3 releases; leave it in
place when submitting the source archive.

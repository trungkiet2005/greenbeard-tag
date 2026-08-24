# Chaos, Solitons & Fractals submission package

This directory contains the *Chaos, Solitons & Fractals*
(CSF) version of the greenbeard-tag manuscript. It uses Elsevier's CAS bundle
(`cas-sc.cls`, single column), numbered citations, journal-specific front
matter, declarations and highlights.

This package is a non-peer-reviewed preprint/development source. It has not been
formally published and is not a version of record.

## Build

```text
python assemble.py
python make_submission_files.py
latexmk -pdf main.tex
latexmk -pdf highlights.tex
python package_source.py
```

`assemble.py` is the source of truth for `main.tex`. It combines the reusable
body in `../paper/main.tex` with the CSF-specific front matter and editorial
blocks in this directory. Its guarded substitutions fail if an expected source
passage changes, if the compact introduction leaves its word budget, or if a
required core citation is lost.

## Verified state (24 August 2026)

`CITATION.cff` records release metadata for version 1.0.0 dated 24 August
2026. The `v1.0.0` tag and GitHub release identify the exact tested submission
snapshot. They provide immutable source archives but do not mint a DOI.

All three authors confirmed the final spelling and order, equal-contribution
note, CRediT roles, corresponding-author address, funding and competing-interest
statements, generative-AI declaration, companion-paper disclosure, and final
manuscript on 24 August 2026.

- 26 PDF pages: one unnumbered CAS highlights sheet and 25 numbered manuscript
  pages.
- 0 LaTeX errors, 0 undefined references/citations and 59 printed references.
- One 117 pt overfull-box report generated inside the CAS front-matter macro;
  visual inspection confirms that no content protrudes from the page.
- Abstract: 227 words. Highlights: five items, 65--73 characters each.
- Introduction plus Related work: 994 words.
- Five top-level sections, followed by declarations, Appendix A and references.
- Nine manuscript figures are vector PDFs with embedded fonts; no Type 3 fonts
  occur in `main.pdf`.

## Submission files

Use either the deterministic ZIP or the corresponding individual LaTeX source
files, depending on what Editorial Manager accepts; do not upload both source
routes blindly. The ZIP already contains the manuscript figures.

- `main.tex`, `refs.bib`, the CAS class/style files and `figures/`: editable
  LaTeX source package.
- `CSF_manuscript_source.zip`: deterministic allowlisted source archive with an
  internal SHA-256 manifest; it excludes the graphical-abstract artifacts.
- `main.pdf`: compiled manuscript for final visual review.
- `highlights.docx`: the highlights upload. The generated
  `highlights.tex`/`highlights.pdf` pair is retained only as a local visual-QA
  rendering; do not upload it instead of the DOCX.
- `declaration_of_interest.docx`: separate declaration requested by Elsevier.
- `cover_letter.md`: private cover-letter authoring draft; do not upload the
  Markdown file. After the date, status and wording receive final author
  approval, paste its body into the portal or use the generated
  `cover_letter.docx`/an accepted PDF when the portal permits file upload.
- `SUBMISSION_CHECKLIST.md`: author actions that cannot be completed from the
  repository, especially publishing the exact tested snapshot and confirming
  reviewer suggestions. An immutable archival DOI remains recommended.

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
data-availability and generative-AI disclosures. The latter records assistance
with manuscript, scientific-code, validation, visualization, build, packaging
and reproducibility work while distinguishing deterministic scientific outputs
from generative imagery. CRediT roles are in `_front.tex`.

The revision also removes repeated previews of every result, corrects the
uniform-initialisation basin shares to 0.388 and 0.612, reports the targeted
unsafe mixed rest face separately from that sample without treating it as a
third globally important regime, and removes duplicated limitations text. These
are editorial and consistency corrections; the model and computed results are
unchanged.

## Graphical abstract: do not submit

The journal-specific guide requires Highlights but does not require a graphical
abstract. The graphical-abstract files already tracked in this repository are
retained only as historical working artifacts. They are excluded from the
submission plan and source archive and **must not be uploaded**. In particular,
Elsevier's current generative-AI policy prohibits a general-purpose generative
AI tool from creating a graphical abstract; no replacement is needed for this
journal.

## Template notes

The package includes `cas-sc.cls`, `cas-common.sty` and `thumbnails/` from CAS
bundle v2.4. Numbered citations use `elsarticle-num-names`, which keeps
`\citet` author names while printing square-bracket numbers. `_front.tex` also
contains a guarded compatibility shim for current expl3 releases; leave it in
place when submitting the source archive.

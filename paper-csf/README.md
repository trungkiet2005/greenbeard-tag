# Chaos, Solitons & Fractals submission package

This directory holds the *Chaos, Solitons & Fractals* (CSF) version of the
greenbeard-tag manuscript, typeset with Elsevier's current **CAS bundle**
(`cas-sc.cls`, single column). The venue-neutral version stays untouched in
[`../paper/`](../paper/); nothing here modifies it, and nothing here modifies
the model, the results or the figures.

## Build

```
python assemble.py                 # rebuilds main.tex from ../paper/main.tex + the blocks
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex highlights                # optional, the standalone highlights page
python make_submission_files.py    # highlights.docx + declaration_of_interest.docx
```

Current state: 26 pages, 0 errors, 0 undefined references, 1 overfull box of
117pt inside the CAS front-matter box (a class artefact, nothing protrudes on
the page), 79 references printed.

`assemble.py` is the single source of truth. It copies the body of
`../paper/main.tex` and applies a short, asserted list of changes, so
any edit to the underlying study propagates here by re-running it. Every
substitution is guarded by a uniqueness assertion, so the build fails loudly
rather than silently skipping an edit. It also fails if an em-dash appears in
non-comment text.

One part of the body is not copied but replaced: Related work, which CSF gets
at half length as a subsection of the Introduction (see below). Two assertions
police that replacement. The first fails if the condensed text drops any
reference the full-length section cited, which matters because 28 of those
references are cited nowhere else in the paper and would leave the
bibliography without any visible symptom. The second fails if the condensed
text runs outside a 620 to 790 word budget, counted with citation and
cross-reference arguments stripped, since counting those inflates a
citation-dense section by about a hundred words.

## Template and class files

`cas-sc.cls`, `cas-common.sty` and `thumbnails/` are copied here from
`../els-cas-templates.zip` (CAS bundle v2.4) so the source builds anywhere,
including on a machine where MiKTeX or TeX Live has no CAS installation.

Two things about the CAS bundle are worth knowing:

- **Numbered references.** The bundle ships only the author-year style
  `cas-model2-names.bst`. Its own template points at `model1-num-names` for the
  numbered case, which the bundle does not include either. We use Elsevier's
  `elsarticle-num-names.bst`, which is the same numbered-with-names style and
  ships with every TeX distribution. That gives CSF's required `[1]` citations
  while keeping `\citet` resolving to author names.
- **An expl3 incompatibility.** `cas-common.sty` v2.4 still calls
  `\vbox_unpack_clear:N`, which current expl3 releases have removed in favour of
  `\vbox_unpack_drop:N`. Without a shim, `\maketitle` fails outright on an
  up-to-date TeX installation. `_front.tex` carries a guarded shim that defines
  the old name only when it is missing, so the source builds on old and new
  installations alike. Leave it in place when you submit.

## What differs from the venue-neutral version

| | venue-neutral | CSF |
|---|---|---|
| sections | 7 top level: Introduction, Related work, Model, Results, Discussion, Limitations, Conclusion | 5 top level: Related work demoted to 1.1 and halved, Limitations folded into the Discussion as 4.1 |
| back matter | references, then appendix | declarations, CRediT, appendix, then references, as elsarticle and the sibling AMC manuscript do |
| class | `article`, 11pt, a4 | `cas-sc`, `a4paper,fleqn,longmktitle` (single column, as CSF requires of LaTeX submissions) |
| references | `unsrtnat` | `elsarticle-num-names` (numbered, `\citet` still resolves to author names) |
| abstract | 248 words, JTB framing | 246 words, nonlinear-dynamics framing |
| highlights | none | 5 bullets, each <= 85 characters, in the front matter and as separate files |
| keywords | 7, inline | 7, in the CAS `keywords` environment, none joined by "and" or "of" |
| CRediT | none | `\credit{}` per author, emitted by `\printcredits` |
| title | "Certified clubs: agent identity is a forgeable greenbeard..." | "Forgeable greenbeards: bistability and hollow collapse..." |
| declarations | data statement only | competing interest, funding, data availability, generative AI |

Four blocks of new prose were written for this version, all in
`_blocks_*.tex`, all framing rather than new results:

- `_blocks_intro.tex` states plainly that the object of study is a nonlinear
  flow, that most results are transversal eigenvalues changing sign and that
  the rest concern the global basin structure.
- `_blocks_related.tex` situates the model in evolutionary game dynamics and
  the statistical physics of cooperation, and says what distinguishes this
  system from that literature. It is no longer inserted into the manuscript:
  its content is folded into `_related_csf.tex`, and the file is kept because
  it is the record of what that framing paragraph cited, which is what the
  no-dropped-reference assertion checks against.
- `_blocks_model.tex` writes the replicator flow, the transversal eigenvalue
  `lambda_{i|j}`, and the observation that every threshold in the paper is a
  simple root of one such eigenvalue, hence a transcritical bifurcation of the
  corresponding edge flow.
- `_blocks_results.tex` adds Table 5, an inventory of every qualitative
  transition with its control parameter, its closed form and its value.

Four sentences were inserted into the inherited body: one naming
`(r, sigma) = (0.077, 0.938)` as a codimension-two point, one contrasting an
authority-levied fine with peer punishment, one citing indirect reciprocity
where the limitations already mention reputation carried between encounters,
and one rewriting a reference to unnamed "sister studies" so it stands alone.

Ten references were appended to `refs.bib`, each verified against Crossref, and
each cited exactly where it supports the sentence. No existing entry was
changed.

## Files to upload to Editorial Manager

| file | submission item |
|---|---|
| `main.tex`, `refs.bib`, `tables_generated.tex`, `robustness_generated.tex`, `cas-sc.cls`, `cas-common.sty`, `thumbnails/` | manuscript source |
| `main.pdf` | built manuscript, for reference |
| `figures/fig01..fig09.pdf` | figures, one file each, vector PDF |
| `highlights.docx` | Highlights |
| `declaration_of_interest.docx` | Declaration of Interest |
| `cover_letter.md` | cover letter, after filling in the bracketed fields |

## Before you submit

1. **Author block.** `_front.tex` carries the three authors, their real
   ORCIDs, the shared Faculty of Information Technology affiliation and
   Trung-Kiet Huynh as corresponding author; all three contributed equally,
   carried by the `\fnmark[1]`/`\fntext[1]` note. CSF runs a single-anonymized
   review, so authors are named on the submitted manuscript. Confirm the
   postal address is the campus you want printed. Rerun `assemble.py` after
   any change: `main.tex` is generated.
2. **CRediT roles.** These live in the `\credit{}` commands in `_front.tex`,
   not in the declarations block. All three carry identical roles, which is
   what equal contribution means; correct them if that is not what happened.
3. **Generative AI statement.** `_blocks_declarations.tex` carries a statement
   describing use of a language-model assistant for drafting, code scaffolding
   and number checking. Confirm it matches your use, or edit it. Elsevier asks
   for it in a dedicated section at the end of the manuscript ahead of the
   reference list; `assemble.py` puts it last among the declarations, followed
   by the CRediT statement and the appendix, which is the order the published
   Elsevier layout uses.
4. **Data deposit.** CSF applies Option C of the Elsevier research data policy:
   data should live in a repository, be cited and be linked. The GitHub link is
   in the data availability statement; archiving the repository on Zenodo to
   obtain a DOI and citing that DOI would satisfy the policy more squarely.
5. **Reviewer suggestions.** The cover letter has a slot for them.

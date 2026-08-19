# Model identity as a spoofable greenbeard tag

Evolutionary dynamics of badge-conditioned trust in the reduced AI race of
Fernandez Domingos and Han (2026).  A seat in the race carries an identity
badge -- a cryptographic attestation, a certification mark, or a stylistic
signature -- that is verifiable but forgeable, and conditions the design it
executes on the badge its opponent presents.  The study asks when a
certification mark can sustain safe play that neither liability nor
within-race reciprocity can, how good verification has to be for the mark
to survive forgery, and what it takes to rebuild trust once it collapses.

The interaction layer (`src/gbtag/race.py`) is unmodified with respect to
the sister studies (`deployment-layer-selection`, `delegation-cascade`), so
every identity effect is attributable to the new layer alone.  Everything
is evaluated exactly; there is no simulation anywhere in the results.

## Layout

```
src/gbtag/          the model
  race.py           exact evaluation of the reduced AI race (shared engine)
  identity.py       badges, verification, the handshake law, design classes
  functionals.py    pi_P, pi_S, and the population observables
  dynamics.py       replicator flow and finite-population process (shared)
  theory.py         the propositions in closed form
  interventions.py  verification, fines, dues, forgery cost, assortment
  robustness.py     planes, pool ablations, process sensitivity
  plotting.py       the manuscript figure style
  config.py         the baseline parameterisation
scripts/            run_analysis, run_robustness, make_figures, build_paper
tests/              exactness, propositions, instruments, figure style
results/            tables/*.csv, key_numbers.json, grids.npz, figures/
paper/              the manuscript
```

## Reproduce

The order matters: `make_figures.py` reads the tables written by
`run_analysis.py` *and* the grids written by `run_robustness.py`, and
`build_paper.py` reads the rendered figures.

```
pip install -e .
python scripts/run_analysis.py     # results/tables/*.csv, results/key_numbers.json
python scripts/run_robustness.py   # results/grids.npz, robustness tables
python scripts/make_figures.py     # results/figures/fig*.{pdf,png}
python scripts/check_numbers.py    # every scalar the paper quotes
python scripts/build_paper.py      # requires pdflatex
pytest                             # the full suite, including figure layout
```

Two safeguards are worth knowing about. `scripts/check_numbers.py` pairs each
number quoted in the manuscript with the results entry it came from
and exits non-zero if any has drifted, so a parameter change cannot silently
leave a stale figure in the text. And `gbtag.layout_check`, exercised by
`tests/test_figures.py`, renders every figure and fails if any text collides
with other text, with a plotted curve, or with the canvas edge.

Running `pytest` before the scripts is fine: the figure-layout tests skip when
`results/` has not been generated yet.

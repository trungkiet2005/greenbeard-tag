"""The baseline parameterisation used everywhere in the manuscript.

Every script, figure and test reads its defaults from here, so a change of
baseline propagates to the whole study and cannot leave one figure behind.

The values are chosen as follows.

``RACE``
    the protocol of the source experiment, unmodified.

``LIABILITY = 0``
    the identity layer is analysed in the regime where harm cannot be traced
    back to the seat that caused it, so the private liability channel is
    closed.  This is the regime that motivates the study: the reciprocity
    threshold of the race is ``L = 0.551`` (below it, a first-strike design
    profitably exploits a conditional cooperator), so with no enforceable
    liability neither deterrence nor within-race reciprocity can stabilise
    safe play, and identity is the only channel left.  Values up to 8 are
    swept, which brackets the population-level critical liability 6.52 of
    the sister studies.

``SOCIAL_HARM = 20``
    the harm of one Unsafe action in the social ledger, as in the sister
    studies.  It never enters the dynamics.

``IDENTITY``
    ``sigma = 0.5`` (a forged badge is a coin flip), ``kappa_g = 2`` (about
    3.4 per cent of the safe in-club payoff of 57), ``kappa_f = 0`` (forgery
    is free, the worst case), ``rho = 0`` (no fines; fines are an
    instrument, not part of the world), ``r = 0.1`` (one encounter in ten is
    within one's own provider cluster).  Every one of these is swept.

``POPULATION = 100``, ``BETA = 0.05``
    the finite-population process of the sister studies, unchanged.
"""

from __future__ import annotations

import numpy as np

from .identity import IdentityParams
from .race import RaceParams

#: Interaction layer, exactly as in the source experiment.
RACE = RaceParams()

#: Identity layer baseline.
IDENTITY = IdentityParams(
    sigma=0.5, kappa_g=2.0, kappa_f=0.0, rho=0.0, r=0.1, protection=True
)

#: Private liability per Unsafe action (closed channel at baseline) and the
#: social harm used in the social ledger.
LIABILITY = 0.0
SOCIAL_HARM = 20.0

#: Finite population used for the stationary analysis.
POPULATION = 100
BETA = 0.05

#: Number of interior starts for the basin-averaged replicator attractor.
REPLICATOR_STARTS = 200
SEED = 20260819

#: Grids shared by every sweep, so cross-sections always agree.
SIGMA_GRID = np.round(np.linspace(0.0, 1.0, 101), 4)
R_GRID = np.round(np.linspace(0.0, 0.5, 51), 4)
L_GRID = np.round(np.linspace(0.0, 8.0, 81), 4)
RHO_GRID = np.round(np.linspace(0.0, 60.0, 61), 4)
KAPPA_GRID = np.round(np.linspace(0.0, 12.0, 61), 4)

#: The canonical designs the propositions are stated for.
CLUB = ("G", "CS", "CAS")
"""The certified reciprocator: genuine badge, reciprocate inside the club,
open unsafe against the unverified."""

TRUSTING_CLUB = ("G", "AS", "CAS")
"""The certified unconditional trustor: same badge, but unconditionally safe
towards whoever passes the check."""

FALSEBEARD = ("F", "CAS", "CAS")
"""The minimal-deviation forger: wears the mark, opens unsafe to steal the
race, then mirrors."""

MIMIC = ("F", "CS", "CAS")
"""The parasitic mimic: wears the mark and behaves like the club, paying
neither the dues nor the conduct."""

ANARCHY = ("N", "CAS", "CAS")
"""The unbadged first-striker that rules the uncertified world at L = 0."""

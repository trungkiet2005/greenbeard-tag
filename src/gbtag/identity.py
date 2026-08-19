"""The identity layer: badges, verification, and badge-conditioned play.

A seat in the race is operated by an agent that carries an *identity badge*
and conditions the design it executes on the badge its opponent presents.
The badge is the model-identity signal of an agent economy: a cryptographic
attestation of provenance, a certification mark, or merely the stylistic
signature by which one model recognises another.  What makes the layer a
game, rather than a label, is that the badge can be forged.

Designs
-------
A design is a triple ``(badge, s_in, s_out)``:

``badge``
    ``G`` carries a *genuine* badge, issued by the attestation infrastructure
    at a per-race cost ``kappa_g``.  It passes verification always.

    ``F`` carries a *forged* badge, produced without the issuer at cost
    ``kappa_f``.  It passes verification with the *spoof success probability*
    ``sigma``; ``1 - sigma`` is the detection power of the verification
    technology.

    ``N`` carries no badge.  It never passes verification, and there is
    nothing to detect.

``s_in``
    the reduced race design executed against an opponent whose badge
    *passes* the focal agent's check;

``s_out``
    the design executed against an opponent that presents no badge or whose
    badge fails the check.

With the four reduced designs of the interaction layer this gives
``3 x 4 x 4 = 48`` designs.  The classical figures of the greenbeard
literature are corners of this space: a *greenbeard* is ``(G, safe, harsh)``,
a *falsebeard* is ``(F, exploit, .)``, and an unconditional design is any
``(N, s, s)``.

The handshake
-------------
Verification happens once, at the start of a race, and the outcome holds for
the whole race; this matches the interaction layer, in which the executed
design of a seat is fixed before the first round.  The two checks of an
encounter concern two different badges, so they are independent by
construction rather than by assumption, and every pairwise expectation is a
four-term sum -- the layer stays exactly evaluable, with no simulation
anywhere.

Detection has two separable consequences, and the model keeps them separate
because they are different policy instruments:

*screening* (``protection = True``)
    the checking agent treats a failed badge as out-group and plays
    ``s_out``.  This is real-time verification: the response happens inside
    the encounter.

*fines* (``rho > 0``)
    a detected forgery is charged the penalty ``rho``.  With
    ``protection = False`` this is retrospective auditing: the fine is
    levied, but the counterparty could not act on the detection in time.

Assortment
----------
With probability ``r`` an agent interacts within its own provider cluster --
modelled, as is standard, as meeting its own design -- and with probability
``1 - r`` it meets a uniform draw from the population.  ``r`` is the one
parameter the identity layer does not create: it is the pre-existing
network structure of the agent economy (agents of one provider co-occur on
one platform).  The nucleation results show it is also the parameter nothing
else can substitute for.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .race import STRATEGIES

#: Badge types: genuine, forged, none.
BADGES: tuple[str, ...] = ("G", "F", "N")

#: Number of identity designs.
N_DESIGNS = len(BADGES) * len(STRATEGIES) * len(STRATEGIES)

#: The reduced designs counted as safe conduct inside a club: they never
#: initiate an Unsafe action against a partner that plays safe.
SAFE_DESIGNS: tuple[str, ...] = ("AS", "CS")


@dataclass(frozen=True)
class IdentityParams:
    """Parameters of the identity layer.

    Attributes
    ----------
    sigma:
        Spoof success probability: the probability that a forged badge
        passes one verification.  ``1 - sigma`` is the detection power of
        the attestation technology.
    kappa_g:
        Per-race cost of carrying a genuine badge (issuance, attestation,
        compliance).  Paid whether or not the partner checks.
    kappa_f:
        Per-race cost of carrying a forged badge.  The interesting regime is
        ``kappa_f < kappa_g``: forging is cheaper than compliance.
    rho:
        Fine charged to a forger each time a check detects its badge.  The
        expected fine per encounter is ``(1 - sigma) * rho``.
    r:
        Assortment: probability of interacting within one's own provider
        cluster instead of with a uniform draw from the population.
    protection:
        Whether a failed check changes the checker's behaviour (real-time
        screening).  ``False`` models retrospective auditing: fines are
        still levied but the checker plays ``s_in`` as if the badge had
        passed.
    """

    sigma: float = 0.5
    kappa_g: float = 2.0
    kappa_f: float = 0.0
    rho: float = 0.0
    r: float = 0.1
    protection: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.sigma <= 1.0:
            raise ValueError(f"sigma must lie in [0, 1], got {self.sigma}")
        if self.kappa_g < 0.0 or self.kappa_f < 0.0:
            raise ValueError("badge costs must be non-negative")
        if self.rho < 0.0:
            raise ValueError("rho must be non-negative")
        if not 0.0 <= self.r <= 1.0:
            raise ValueError(f"r must lie in [0, 1], got {self.r}")

    def pass_rate(self, badge: str) -> float:
        """Probability that ``badge`` passes one verification."""
        if badge == "G":
            return 1.0
        if badge == "F":
            return self.sigma
        if badge == "N":
            return 0.0
        raise ValueError(f"unknown badge {badge!r}")

    def behavioural_pass_rate(self, badge: str) -> float:
        """Probability that the checker responds to ``badge`` with ``s_in``.

        Equal to :meth:`pass_rate` under screening.  Under retrospective
        auditing (``protection = False``) a forged badge always elicits the
        in-group response, because the detection arrives too late to act on:
        the fine still binds, the behaviour does not.
        """
        if badge == "F" and not self.protection:
            return 1.0
        return self.pass_rate(badge)

    def badge_cost(self, badge: str) -> float:
        """Per-race cost of carrying ``badge``."""
        return {"G": self.kappa_g, "F": self.kappa_f, "N": 0.0}[badge]

    def expected_fine(self, badge: str) -> float:
        """Expected fine per encounter for carrying ``badge``.

        Detection is a property of the badge check, not of the checker's
        response, so the fine is ``(1 - sigma) * rho`` for a forger under
        screening and auditing alike.
        """
        if badge == "F":
            return (1.0 - self.sigma) * self.rho
        return 0.0


def design_space() -> tuple[tuple[str, str, str], ...]:
    """Every ``(badge, s_in, s_out)`` design, badge-major."""
    return tuple(
        (b, si, so) for b, si, so in product(BADGES, STRATEGIES, STRATEGIES)
    )


def design_labels() -> tuple[str, ...]:
    """Compact labels ``badge:in/out`` for the designs of the space."""
    return tuple(f"{b}:{si}/{so}" for b, si, so in design_space())


def design_index(badge: str, s_in: str, s_out: str) -> int:
    """Position of one design in the space."""
    return design_space().index((badge, s_in, s_out))


# --------------------------------------------------------------------------
# interpretable classes of the 48 designs
# --------------------------------------------------------------------------

#: Class names in display order.
CLASSES: tuple[str, ...] = (
    "certified club",
    "certified aggressor",
    "falsebeard",
    "unbadged cooperator",
    "unbadged other",
)


def classify(badge: str, s_in: str, s_out: str) -> str:
    """Interpretable class of one design.

    ``certified club``
        genuine badge and safe conduct towards verified partners.  These are
        the designs for which the badge means what it claims.
    ``certified aggressor``
        genuine badge, unsafe conduct towards verified partners: the badge
        is honest about provenance but not about behaviour.
    ``falsebeard``
        any forged badge, whatever the conduct.
    ``unbadged cooperator``
        no badge, safe conduct towards everyone.
    ``unbadged other``
        no badge, some unsafe conduct.
    """
    if badge == "F":
        return "falsebeard"
    if badge == "G":
        return "certified club" if s_in in SAFE_DESIGNS else "certified aggressor"
    if s_in in SAFE_DESIGNS and s_out in SAFE_DESIGNS:
        return "unbadged cooperator"
    return "unbadged other"


def class_of_each_design() -> tuple[str, ...]:
    """Class of every design, in the order of :func:`design_space`."""
    return tuple(classify(*d) for d in design_space())


def class_masses(x: np.ndarray) -> dict[str, float]:
    """Mass of each class under the design distribution ``x``."""
    classes = class_of_each_design()
    x = np.asarray(x, dtype=float)
    return {c: float(x[[i for i, k in enumerate(classes) if k == c]].sum()) for c in CLASSES}


def badge_masses(x: np.ndarray) -> dict[str, float]:
    """Mass of each badge type under the design distribution ``x``."""
    designs = design_space()
    x = np.asarray(x, dtype=float)
    return {
        b: float(x[[i for i, (bi, _, _) in enumerate(designs) if bi == b]].sum())
        for b in BADGES
    }


# --------------------------------------------------------------------------
# the executed-design law of one encounter
# --------------------------------------------------------------------------


def encounter_terms(
    focal: tuple[str, str, str],
    partner: tuple[str, str, str],
    params: IdentityParams,
) -> list[tuple[float, str, str]]:
    """The four-term law of the executed pair for one ordered encounter.

    Returns ``(weight, focal_design, partner_design)`` triples summing to
    one.  The focal agent plays ``s_in`` with the *behavioural* pass rate of
    the partner's badge, and the partner symmetrically; the two checks
    concern different badges and are independent.
    """
    b_f, in_f, out_f = focal
    b_p, in_p, out_p = partner
    q_partner = params.behavioural_pass_rate(b_p)  # governs the focal response
    q_focal = params.behavioural_pass_rate(b_f)  # governs the partner response
    terms = []
    for w_f, s_f in ((q_partner, in_f), (1.0 - q_partner, out_f)):
        for w_p, s_p in ((q_focal, in_p), (1.0 - q_focal, out_p)):
            weight = w_f * w_p
            if weight > 0.0:
                terms.append((weight, s_f, s_p))
    return terms


def pairwise_expectation(
    matrix: np.ndarray,
    params: IdentityParams,
) -> np.ndarray:
    """Expectation of a race-layer matrix over the badge handshake.

    ``matrix`` is a ``4 x 4`` race quantity indexed by executed designs
    (payoff, unsafe count, unsafe frequency); the result is the ``48 x 48``
    matrix of its expectations over the handshake outcomes of every ordered
    pair of identity designs.
    """
    matrix = np.asarray(matrix, dtype=float)
    designs = design_space()
    n = len(designs)
    idx = {s: k for k, s in enumerate(STRATEGIES)}

    # the focal executed design depends only on (own design, partner badge);
    # collect the two possible rows and mix them with the behavioural rates
    out = np.empty((n, n))
    q = np.array([params.behavioural_pass_rate(b) for b, _, _ in designs])
    in_row = np.array([idx[si] for _, si, _ in designs])
    out_row = np.array([idx[so] for _, _, so in designs])

    for i in range(n):
        # focal plays in_row[i] w.p. q[j], out_row[i] w.p. 1 - q[j]
        # partner plays in_row[j] w.p. q[i], out_row[j] w.p. 1 - q[i]
        m_ii = matrix[in_row[i], in_row]  # focal in, partner in
        m_io = matrix[in_row[i], out_row]
        m_oi = matrix[out_row[i], in_row]
        m_oo = matrix[out_row[i], out_row]
        qi = q[i]
        out[i] = q * (qi * m_ii + (1.0 - qi) * m_io) + (1.0 - q) * (
            qi * m_oi + (1.0 - qi) * m_oo
        )
    return out


def apply_assortment(matrix: np.ndarray, r: float) -> np.ndarray:
    """Assortment-adjusted pairwise matrix ``r * diag + (1 - r) * matrix``.

    With probability ``r`` the partner is one's own design, so every row is
    mixed with its diagonal entry.  Applied to the private payoff it yields
    the fitness a design earns in a population where interaction is
    provider-clustered; applied to an observable it yields the population
    average of that observable under the same encounter law.
    """
    if not 0.0 <= r <= 1.0:
        raise ValueError(f"r must lie in [0, 1], got {r}")
    matrix = np.asarray(matrix, dtype=float)
    return r * np.diag(matrix)[:, None] * np.ones_like(matrix) + (1.0 - r) * matrix

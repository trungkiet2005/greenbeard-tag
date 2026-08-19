"""Payoff functionals of the identity layer.

A design is a triple ``(badge, s_in, s_out)``.  Two designs meeting in the
race produce, in expectation over the handshake and the horizon,

.. math::

    a(i, j) &= \\mathbb{E}\\,[A(s_i, s_j)] - c(b_i) - f(b_i), \\\\
    m(i, j) &= \\mathbb{E}\\,[M(s_i, s_j)],

where ``A`` and ``M`` are the task-payoff and unsafe-action matrices of the
interaction layer, the expectation runs over the two independent badge
checks, ``c`` is the badge cost and ``f`` the expected fine.  Everything is
a four-term exact sum; nothing is simulated.

Two functionals are built on those primitives.

``pi_P`` (private / selection functional)
    what the operator of the seat receives: the race payoff, minus the
    private liability ``L`` per Unsafe action, minus its badge cost, minus
    its expected fines.  This is what drives the dynamics.  The baseline of
    the study is ``L = 0``: harms that cannot be attributed cannot be
    charged, which is the regime in which identity is the only instrument
    left.

``pi_S`` (social functional)
    the same race payoff with the full social harm ``h`` per Unsafe action
    and the real resource costs (badge issuance and forgery both burn real
    resources).  Fines are transfers and are excluded.  It never enters the
    dynamics; it is the yardstick.

Assortment enters as the standard interaction structure: with probability
``r`` a design meets itself, with probability ``1 - r`` a uniform draw.  The
matrix handed to the dynamics is the assortment-adjusted private functional,
and every population observable is averaged under the same encounter law.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .identity import (
    IdentityParams,
    SAFE_DESIGNS,
    apply_assortment,
    design_labels,
    design_space,
    pairwise_expectation,
)
from .race import STRATEGIES, RaceTables


@dataclass(frozen=True)
class IdentityFunctionals:
    """Payoff, harm and behaviour matrices over the 48-design identity space."""

    designs: tuple[tuple[str, str, str], ...]
    labels: tuple[str, ...]
    badge: tuple[str, ...]
    """Badge of each design."""

    s_in: tuple[str, ...]
    """Design executed against a partner whose badge passes."""

    s_out: tuple[str, ...]
    """Design executed against everyone else."""

    task: np.ndarray
    """``a``: expected race payoff of the focal seat, before costs."""

    harm: np.ndarray
    """``m``: expected number of Unsafe actions of the focal seat."""

    unsafe_frequency: np.ndarray
    """``u``: expected fraction of Unsafe rounds of the focal seat."""

    badge_cost: np.ndarray
    """Per-race badge cost of each design, shape ``(48,)``."""

    expected_fine: np.ndarray
    """Expected fine per encounter of each design, shape ``(48,)``."""

    pi_P: np.ndarray
    """Private functional before assortment; ``fitness`` is what evolves."""

    pi_S: np.ndarray
    """Social functional; fines excluded, resource costs included."""

    fitness: np.ndarray
    """Assortment-adjusted private functional; this drives the dynamics."""

    params: IdentityParams
    liability: float
    social_harm: float

    @property
    def n(self) -> int:
        return len(self.labels)

    def index(self, badge: str, s_in: str, s_out: str) -> int:
        """Position of design ``(badge, s_in, s_out)`` in the design space."""
        return self.designs.index((badge, s_in, s_out))

    def badge_block(self, badge: str) -> np.ndarray:
        """Indices of every design carrying a given badge."""
        return np.array([i for i, b in enumerate(self.badge) if b == badge])

    def subspace(self, keep: np.ndarray) -> "IdentityFunctionals":
        """Restriction of the functionals to a subset of designs.

        Used for the uncertified baseline (badge ``N`` only), for
        no-falsebeard counterfactuals, and for the small design spaces on
        which the full finite-population chain is tractable.
        """
        keep = np.asarray(keep, dtype=int)
        grid = np.ix_(keep, keep)
        return IdentityFunctionals(
            designs=tuple(self.designs[i] for i in keep),
            labels=tuple(self.labels[i] for i in keep),
            badge=tuple(self.badge[i] for i in keep),
            s_in=tuple(self.s_in[i] for i in keep),
            s_out=tuple(self.s_out[i] for i in keep),
            task=self.task[grid],
            harm=self.harm[grid],
            unsafe_frequency=self.unsafe_frequency[grid],
            badge_cost=self.badge_cost[keep],
            expected_fine=self.expected_fine[keep],
            pi_P=self.pi_P[grid],
            pi_S=self.pi_S[grid],
            fitness=apply_assortment(self.pi_P[grid], self.params.r),
            params=self.params,
            liability=self.liability,
            social_harm=self.social_harm,
        )


def build_functionals(
    tables: RaceTables,
    params: IdentityParams,
    liability: float = 0.0,
    social_harm: float = 20.0,
) -> IdentityFunctionals:
    """Assemble the identity-layer functionals.

    Parameters
    ----------
    tables:
        Output of :func:`gbtag.race.build_race_tables`.
    params:
        Identity-layer parameters.
    liability:
        Private liability ``L`` charged per Unsafe action of one's own seat.
        The baseline of the study is zero: the identity layer is analysed in
        the regime where harm cannot be traced to the seat that caused it.
    social_harm:
        Social harm ``h`` of one Unsafe action, used only in ``pi_S``.
    """
    if tables.strategies != STRATEGIES:
        raise ValueError("the race tables must use the design order of STRATEGIES")
    if liability < 0.0:
        raise ValueError(f"liability must be non-negative, got {liability}")
    if social_harm < 0.0:
        raise ValueError(f"social_harm must be non-negative, got {social_harm}")

    designs = design_space()
    task = pairwise_expectation(tables.payoff, params)
    harm = pairwise_expectation(tables.unsafe_count, params)
    unsafe = pairwise_expectation(tables.unsafe_frequency, params)

    badge = tuple(b for b, _, _ in designs)
    cost = np.array([params.badge_cost(b) for b in badge])
    fine = np.array([params.expected_fine(b) for b in badge])

    pi_P = task - liability * harm - cost[:, None] - fine[:, None]
    pi_S = task - social_harm * harm - cost[:, None]

    return IdentityFunctionals(
        designs=designs,
        labels=design_labels(),
        badge=badge,
        s_in=tuple(si for _, si, _ in designs),
        s_out=tuple(so for _, _, so in designs),
        task=task,
        harm=harm,
        unsafe_frequency=unsafe,
        badge_cost=cost,
        expected_fine=fine,
        pi_P=pi_P,
        pi_S=pi_S,
        fitness=apply_assortment(pi_P, params.r),
        params=params,
        liability=float(liability),
        social_harm=float(social_harm),
    )


def unbadged_subspace(fun: IdentityFunctionals) -> IdentityFunctionals:
    """The uncertified world: the 16 designs with no badge."""
    return fun.subspace(fun.badge_block("N"))


def honest_subspace(fun: IdentityFunctionals) -> IdentityFunctionals:
    """The forgery-free counterfactual: genuine and absent badges only."""
    keep = np.array([i for i, b in enumerate(fun.badge) if b != "F"])
    return fun.subspace(keep)


def plain_subspace(fun: IdentityFunctionals) -> IdentityFunctionals:
    """The four unconditional unbadged designs, i.e. the raw race game."""
    keep = np.array(
        [
            i
            for i, (b, si, so) in enumerate(fun.designs)
            if b == "N" and si == so
        ]
    )
    return fun.subspace(keep)


# --------------------------------------------------------------------------
# population observables
# --------------------------------------------------------------------------


def population_average(
    x: np.ndarray, matrix: np.ndarray, r: float, monomorphic: bool = False
) -> float:
    """Average of a pairwise observable under the encounter law.

    In the small-mutation limit the population is monomorphic almost always,
    so the long-run average of an observable is its self-play value weighted
    by the time spent in each design (``monomorphic = True``); assortment is
    then irrelevant, every partner being one's own design anyway.  In a
    mixed population (the replicator reading) the average runs over the
    assortative pair law ``r * diag + (1 - r) * outer``.
    """
    x = np.asarray(x, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if monomorphic:
        return float(x @ np.diag(matrix))
    return float(x @ (r * np.diag(matrix) + (1.0 - r) * (matrix @ x)))


def aggregate_unsafe_frequency(
    x: np.ndarray, fun: IdentityFunctionals, monomorphic: bool = False
) -> float:
    """Population-level Unsafe frequency under the encounter law.

    The primary welfare-relevant observable: it needs no welfare weights, so
    it does not depend on the (necessarily arbitrary) choice of ``h``.
    """
    return population_average(x, fun.unsafe_frequency, fun.params.r, monomorphic)


def mean_social_payoff(
    x: np.ndarray, fun: IdentityFunctionals, monomorphic: bool = False
) -> float:
    """Population-average social payoff under the encounter law."""
    return population_average(x, fun.pi_S, fun.params.r, monomorphic)


def _safe_indicator(fun: IdentityFunctionals) -> tuple[np.ndarray, np.ndarray]:
    """Whether ``s_in`` / ``s_out`` of each design is safe conduct."""
    safe_in = np.array([s in SAFE_DESIGNS for s in fun.s_in], dtype=float)
    safe_out = np.array([s in SAFE_DESIGNS for s in fun.s_out], dtype=float)
    return safe_in, safe_out


def _encounter_weights(
    x: np.ndarray, r: float, monomorphic: bool = False
) -> np.ndarray:
    """Weight of the ordered pair ``(i, j)`` under the encounter law.

    ``monomorphic = True`` puts all mass on the diagonal: the population is
    monomorphic almost always (the small-mutation limit), so an encounter is
    a self-encounter of whichever design currently rules, weighted by the
    time the process spends there.
    """
    x = np.asarray(x, dtype=float)
    if monomorphic:
        return np.diag(x)
    w = (1.0 - r) * np.outer(x, x)
    w[np.diag_indices_from(w)] += r * x
    return w


def mark_lift(
    x: np.ndarray, fun: IdentityFunctionals, monomorphic: bool = False
) -> float:
    """Behavioural value of a passed check: the safety lift of the mark.

    ``P(partner plays safe conduct towards you | partner's badge passed
    your check) - P(safe conduct | no badge passed)``, in the stationary
    population, under the encounter law.  This is what a passed check is
    worth to the agent that performed it.  Returns ``0`` when one of the
    two conditioning events has probability zero.
    """
    params = fun.params
    x = np.asarray(x, dtype=float)
    w = _encounter_weights(x, params.r, monomorphic)
    safe_in, safe_out = _safe_indicator(fun)

    # partner j's badge passes i's check with the *verification* rate;
    # j's conduct towards i mixes on i's behavioural rate, independently
    pass_j = np.array([params.pass_rate(b) for b in fun.badge])
    q_beh = np.array([params.behavioural_pass_rate(b) for b in fun.badge])
    conduct = q_beh[:, None] * safe_in[None, :] + (1.0 - q_beh[:, None]) * safe_out[None, :]
    # conduct[i, j] = P(j plays safe conduct towards i)

    d_pass = float((w * pass_j[None, :]).sum())
    d_fail = float((w * (1.0 - pass_j)[None, :]).sum())
    if d_pass <= 0.0 or d_fail <= 0.0:
        return 0.0
    n_pass = float((w * pass_j[None, :] * conduct).sum())
    n_fail = float((w * (1.0 - pass_j)[None, :] * conduct).sum())
    return n_pass / d_pass - n_fail / d_fail


def attestation_integrity(
    x: np.ndarray, fun: IdentityFunctionals, monomorphic: bool = False
) -> float:
    """Provenance value of a passed check.

    ``P(badge genuine | badge passed)`` in the stationary population under
    the encounter law: the posterior confidence that a passed badge was
    issued rather than forged.  Behavioural conduct does not enter; this is
    the epistemic content of the attestation itself.  Returns ``1`` when no
    badge ever passes (an empty conditioning event reads as an unspoiled
    mark).
    """
    params = fun.params
    x = np.asarray(x, dtype=float)
    w = _encounter_weights(x, params.r, monomorphic)
    pass_j = np.array([params.pass_rate(b) for b in fun.badge])
    genuine = np.array([b == "G" for b in fun.badge], dtype=float)
    d = float((w * pass_j[None, :]).sum())
    if d <= 0.0:
        return 1.0
    n = float((w * (pass_j * genuine)[None, :]).sum())
    return n / d


def unsafe_split(
    x: np.ndarray,
    fun: IdentityFunctionals,
    tables: RaceTables,
    monomorphic: bool = False,
) -> dict[str, float]:
    """Unsafe frequency inside and outside the verified handshake.

    Splits the focal seat's Unsafe frequency by whether the focal seat's own
    check of its partner passed, exactly, under the encounter law.  The
    difference between the two entries is the behavioural wall the identity
    layer builds between the club interior and the open market.
    """
    params = fun.params
    x = np.asarray(x, dtype=float)
    w = _encounter_weights(x, params.r, monomorphic)
    q_beh = np.array([params.behavioural_pass_rate(b) for b in fun.badge])
    u_in, u_out = conditional_unsafe(tables, fun)

    # the focal check of partner j passes with j's behavioural rate, so the
    # pair weight splits as w_ij q_j into "verified" and w_ij (1 - q_j) out
    d_pass = float((w * q_beh[None, :]).sum())
    d_fail = float((w * (1.0 - q_beh)[None, :]).sum())
    return {
        "verified": (
            float((w * q_beh[None, :] * u_in).sum()) / d_pass if d_pass > 0 else 0.0
        ),
        "unverified": (
            float((w * (1.0 - q_beh)[None, :] * u_out).sum()) / d_fail
            if d_fail > 0
            else 0.0
        ),
    }


def conditional_unsafe(
    tables: RaceTables, fun: IdentityFunctionals
) -> tuple[np.ndarray, np.ndarray]:
    """Focal unsafe frequency conditional on the focal check passing/failing."""
    idx = {s: k for k, s in enumerate(STRATEGIES)}
    u = np.asarray(tables.unsafe_frequency, dtype=float)
    q_beh = np.array([fun.params.behavioural_pass_rate(b) for b in fun.badge])
    in_row = np.array([idx[s] for s in fun.s_in])
    out_row = np.array([idx[s] for s in fun.s_out])
    n = fun.n
    u_in = np.empty((n, n))
    u_out = np.empty((n, n))
    for i in range(n):
        qi = q_beh[i]
        # partner mixes on its own check of the focal badge (rate q_beh[i])
        u_in[i] = qi * u[in_row[i], in_row] + (1.0 - qi) * u[in_row[i], out_row]
        u_out[i] = qi * u[out_row[i], in_row] + (1.0 - qi) * u[out_row[i], out_row]
    return u_in, u_out

"""Robustness of the identity results.

Five families of checks:

* **planes** -- the ``(sigma, r)`` and ``(sigma, L)`` grids on which the
  phase structure lives.  Cross-sections in the manuscript are always taken
  from these shared grids, so curves and planes cannot disagree.
* **pool ablations** -- the same dynamics on restricted design spaces
  (no forgery, no badges, unconditional designs only), which is what the
  certification value of the identity layer is measured against.
* **process sensitivity** -- population size, selection intensity, and the
  replicator flow as an independent dynamic.
* **payoff reading** -- the ``setback_scope = "prize"`` variant of the race,
  under which a setback removes only the prize.
* **threshold agreement** -- the closed-form thresholds of
  :mod:`gbtag.theory` against brute-force invasion checks on the exact
  matrices, over a grid of parameter draws.
* **global experiments** -- the parameter-resolved replicator experiments
  the global claims rest on: the basin share under a family of start
  measures, the basin structure along the assortment axis, the
  ``(sigma, r)`` regime map of the flow itself, the badge counterfactuals
  including a badge that carries no information, and the club's out-group
  policy as a continuous axis rather than four rungs.  Each of these
  replaces a claim that used to rest on one sampled experiment at one
  parameter point.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from . import config as cfg
from .dynamics import (
    basin_classification,
    replicator_attractors,
    stationary_analysis_sml,
    stratified_alpha,
)
from .functionals import (
    IdentityFunctionals,
    aggregate_unsafe_frequency,
    attestation_integrity,
    build_functionals,
    honest_subspace,
    mean_social_payoff,
    plain_subspace,
    random_badge_functionals,
    unbadged_subspace,
)
from .identity import IdentityParams
from .race import STRATEGIES, RaceParams, RaceTables, build_race_tables
from .theory import (
    Equilibrium,
    entry_barrier,
    equilibrium,
    fitness_against_resident,
    pair_payoff,
    spoof_threshold,
    spoof_threshold_closed_form,
    # the solver behind spoof_threshold; the mixed-conduct sweep has to use
    # the same one, or its agreement with the discrete scan at the rungs
    # would be an agreement between two root finders and not a check of the
    # interpolation
    _threshold_by_root,
)


# --------------------------------------------------------------------------
# planes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Plane:
    """Equilibrium observables on a two-parameter grid."""

    x_name: str
    y_name: str
    x_values: np.ndarray
    y_values: np.ndarray
    unsafe: np.ndarray
    club: np.ndarray
    falsebeard: np.ndarray
    integrity: np.ndarray
    """All arrays have shape ``(len(y_values), len(x_values))``."""


def sigma_r_plane(
    tables: RaceTables,
    params: IdentityParams,
    sigmas: np.ndarray,
    rs: np.ndarray,
    liability: float = 0.0,
    social_harm: float = 20.0,
    method: str = "sml",
    **kwargs,
) -> Plane:
    """Equilibrium observables over spoof success and assortment."""
    shape = (len(rs), len(sigmas))
    unsafe = np.empty(shape)
    club = np.empty(shape)
    falsebeard = np.empty(shape)
    integrity = np.empty(shape)
    for a, r in enumerate(rs):
        for b, s in enumerate(sigmas):
            fun = build_functionals(
                tables,
                replace(params, sigma=float(s), r=float(r)),
                liability,
                social_harm,
            )
            eq = equilibrium(fun, tables, method=method, **kwargs)
            unsafe[a, b] = eq.unsafe_frequency
            club[a, b] = eq.class_distribution["certified club"]
            falsebeard[a, b] = eq.class_distribution["falsebeard"]
            integrity[a, b] = eq.attestation_integrity
    return Plane(
        x_name="sigma",
        y_name="r",
        x_values=np.asarray(sigmas, dtype=float),
        y_values=np.asarray(rs, dtype=float),
        unsafe=unsafe,
        club=club,
        falsebeard=falsebeard,
        integrity=integrity,
    )


@dataclass(frozen=True)
class CertificationValuePlane:
    """The certification value ``U_uncertified - U_certified`` on a grid."""

    sigmas: np.ndarray
    liabilities: np.ndarray
    certified: np.ndarray
    uncertified: np.ndarray
    value: np.ndarray
    """Shape ``(len(liabilities), len(sigmas))``."""


def sigma_liability_plane(
    tables: RaceTables,
    params: IdentityParams,
    sigmas: np.ndarray,
    liabilities: np.ndarray,
    social_harm: float = 20.0,
    method: str = "sml",
    **kwargs,
) -> CertificationValuePlane:
    """What the identity layer is worth at each liability level.

    At every ``(sigma, L)`` the certified world (full design space) is
    compared with the uncertified world (badge ``N`` only) under the same
    encounter structure.  Where liability alone stabilises the race, the
    value collapses to zero: identity is the poor regulator's instrument.
    """
    shape = (len(liabilities), len(sigmas))
    certified = np.empty(shape)
    uncertified = np.empty(shape)
    for a, L in enumerate(liabilities):
        for b, s in enumerate(sigmas):
            fun = build_functionals(
                tables, replace(params, sigma=float(s)), float(L), social_harm
            )
            eq = equilibrium(fun, tables, method=method, **kwargs)
            certified[a, b] = eq.unsafe_frequency
            sub = unbadged_subspace(fun)
            uncertified[a, b] = equilibrium(
                sub, tables, method=method, **kwargs
            ).unsafe_frequency
    return CertificationValuePlane(
        sigmas=np.asarray(sigmas, dtype=float),
        liabilities=np.asarray(liabilities, dtype=float),
        certified=certified,
        uncertified=uncertified,
        value=uncertified - certified,
    )


# --------------------------------------------------------------------------
# pool ablations
# --------------------------------------------------------------------------


def pool_ablation(
    tables: RaceTables,
    params: IdentityParams,
    liability: float = 0.0,
    social_harm: float = 20.0,
    method: str = "sml",
    **kwargs,
) -> dict[str, Equilibrium]:
    """The same dynamics on the nested design pools.

    ``full``
        all 48 designs;
    ``honest``
        forgery removed (32 designs): the upper bound the attestation
        infrastructure could reach if badges were unforgeable;
    ``uncertified``
        badges removed (16 designs): the world the layer is measured
        against;
    ``plain``
        unconditional unbadged designs only (4 designs): the raw race.
    """
    fun = build_functionals(tables, params, liability, social_harm)
    return {
        "full": equilibrium(fun, tables, method=method, **kwargs),
        "honest": equilibrium(honest_subspace(fun), tables, method=method, **kwargs),
        "uncertified": equilibrium(
            unbadged_subspace(fun), tables, method=method, **kwargs
        ),
        "plain": equilibrium(plain_subspace(fun), tables, method=method, **kwargs),
    }


# --------------------------------------------------------------------------
# process sensitivity
# --------------------------------------------------------------------------


def process_sensitivity(
    tables: RaceTables,
    params: IdentityParams,
    liability: float = 0.0,
    social_harm: float = 20.0,
    population_sizes: tuple[int, ...] = (50, 100, 200),
    betas: tuple[float, ...] = (0.01, 0.05, 0.2),
) -> list[dict[str, float]]:
    """Headline observables across population size and selection intensity."""
    fun = build_functionals(tables, params, liability, social_harm)
    rows = []
    for z in population_sizes:
        for beta in betas:
            eq = equilibrium(fun, tables, method="sml", population_size=z, beta=beta)
            rows.append(
                {
                    "population_size": float(z),
                    "beta": float(beta),
                    "unsafe": eq.unsafe_frequency,
                    "club": eq.class_distribution["certified club"],
                    "falsebeard": eq.class_distribution["falsebeard"],
                    "integrity": eq.attestation_integrity,
                }
            )
    return rows


def replicator_agreement(
    tables: RaceTables,
    params: IdentityParams,
    sigmas: np.ndarray,
    liability: float = 0.0,
    social_harm: float = 20.0,
    n_starts: int = 60,
    seed: int = 20260819,
    **kwargs,
) -> list[dict[str, float]]:
    """SML and replicator answers side by side along the sigma sweep.

    The two dynamics answer different questions -- where the finite process
    spends its time, and which mixtures are stable -- so agreement is
    checked on the direction and location of the collapse, not on levels.
    """
    rows = []
    for s in sigmas:
        fun = build_functionals(
            tables, replace(params, sigma=float(s)), liability, social_harm
        )
        sml = equilibrium(fun, tables, method="sml", **kwargs)
        rep = equilibrium(fun, tables, method="replicator", n_starts=n_starts, seed=seed)
        rows.append(
            {
                "sigma": float(s),
                "sml_unsafe": sml.unsafe_frequency,
                "sml_club": sml.class_distribution["certified club"],
                "sml_integrity": sml.attestation_integrity,
                "rep_unsafe": rep.unsafe_frequency,
                "rep_club": rep.class_distribution["certified club"],
                "rep_integrity": rep.attestation_integrity,
            }
        )
    return rows


# --------------------------------------------------------------------------
# payoff reading
# --------------------------------------------------------------------------


def setback_scope_check(
    race: RaceParams,
    params: IdentityParams,
    liability: float = 0.0,
    social_harm: float = 20.0,
    method: str = "sml",
    **kwargs,
) -> dict[str, dict[str, float]]:
    """Headline numbers under both readings of the setback clause."""
    out: dict[str, dict[str, float]] = {}
    for scope in ("total", "prize"):
        tables = build_race_tables(replace(race, setback_scope=scope))
        fun = build_functionals(tables, params, liability, social_harm)
        eq = equilibrium(fun, tables, method=method, **kwargs)
        sub = equilibrium(unbadged_subspace(fun), tables, method=method, **kwargs)
        out[scope] = {
            "unsafe": eq.unsafe_frequency,
            "club": eq.class_distribution["certified club"],
            "integrity": eq.attestation_integrity,
            "uncertified_unsafe": sub.unsafe_frequency,
        }
    return out


# --------------------------------------------------------------------------
# threshold agreement
# --------------------------------------------------------------------------


def threshold_agreement(
    tables: RaceTables,
    club: tuple[str, str, str],
    forger: tuple[str, str, str],
    kappa_gs: tuple[float, ...] = (0.5, 2.0, 4.0),
    rhos: tuple[float, ...] = (0.0, 5.0, 20.0),
    liabilities: tuple[float, ...] = (0.0, 0.3),
) -> list[dict[str, float]]:
    """Closed-form spoof thresholds against numeric root-finding.

    The closed form covers the well-mixed unconditional-forger case; the
    numeric root uses the generic machinery.  Both are exact, so agreement
    is to solver precision.
    """
    rows = []
    _, club_in, club_out = club
    _, w_in, w_out = forger
    for kg in kappa_gs:
        for rho in rhos:
            for L in liabilities:
                params = IdentityParams(
                    sigma=0.5, kappa_g=kg, kappa_f=0.0, rho=rho, r=0.0
                )
                numeric = spoof_threshold(tables, params, L, club, forger)
                closed = None
                if w_in == w_out:
                    closed = spoof_threshold_closed_form(
                        tables, L, club_in, club_out, w_in, kg, 0.0, rho
                    )
                rows.append(
                    {
                        "kappa_g": kg,
                        "rho": rho,
                        "liability": L,
                        "numeric": np.nan if numeric is None else numeric,
                        "closed_form": np.nan if closed is None else closed,
                    }
                )
    return rows


# --------------------------------------------------------------------------
# global experiments
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StartMeasureAblation:
    """Basin observables of one flow under several start measures.

    A basin share is not a property of a flow.  It is the measure of the set
    of starts that reach one attractor, so it is a property of the flow *and*
    of the measure the starts are drawn from, and quoting one without the
    other is quoting half a number.  This object is that other half: the same
    flow, the same classifier, the same seed, and a family of start measures
    wide enough that the reader can see how far the share moves.
    """

    measures: tuple[str, ...]
    """Name of each start measure, in the order the arrays are indexed."""

    concentration: np.ndarray
    """Dirichlet concentration of each measure, shape ``(len(measures), n)``."""

    badged_share: np.ndarray
    mixed_count: np.ndarray
    unsafe: np.ndarray
    social_payoff: np.ndarray
    """Shape ``(len(measures),)``.  The two observables are evaluated at each
    attractor and averaged afterwards, never at the mean end state."""

    share_range: tuple[float, float]
    """Smallest and largest badged share over the measures.

    This is the honest uncertainty on the basin claim.  A Wilson interval on
    a single measure answers "how well is this sample proportion resolved",
    which is not the question anyone asked: the proportion is exactly
    resolved by more draws and still says nothing about the claim, because
    the measure was never argued for.  This range answers "how much of the
    claim survives the choice of measure", and it is wider.
    """

    n_starts: int
    seed: int


def start_measure_ablation(
    tables: RaceTables,
    params: IdentityParams,
    liability: float,
    social_harm: float,
    alphas: tuple[float, ...] = (0.5, 1.0, 5.0),
    n_starts: int = 400,
    seed: int = cfg.SEED,
    stratified: bool = True,
    stratified_weights: Mapping[str, float] | None = None,
    t_end: float = 3000.0,
) -> StartMeasureAblation:
    """Basin share of the badged face under a family of start measures.

    Each measure is a Dirichlet on the 48-design simplex.  Small
    concentrations pile the starts near the vertices, which is the market
    that has already settled on a handful of designs and is being perturbed;
    large ones concentrate the starts at the barycentre, which is the market
    in which every design is equally present.  Neither is the truth, and the
    flat ``alpha = 1`` of the published run is not either: it is the measure
    that makes the *algebra* simplest, not the one a market is drawn from.

    The badge-stratified measure is included because a flat Dirichlet gives
    each badge class an expected mass proportional to the number of designs
    that happen to carry that badge.  On the full space the three classes
    hold sixteen designs each, so equal thirds and the flat measure coincide
    exactly and the stratified row reproduces the ``alpha = 1`` row to the
    last bit.  That coincidence is worth printing rather than hiding: it says
    the published measure is badge-balanced, so the spread reported here is
    produced by the concentration alone and not by a hidden badge prior.
    Pass ``stratified_weights`` to move the prior off equal thirds, which is
    the way to ask what a badge-poor world would have found.

    Parameters
    ----------
    tables, params, liability, social_harm:
        The world whose flow is being sampled.
    alphas:
        Scalar Dirichlet concentrations, one measure each.
    n_starts:
        Starts per measure.  Every measure uses the same ``seed``, so the
        differences between rows are differences of measure and not of draw.
    stratified:
        Whether to append the badge-stratified measure.
    stratified_weights:
        Target mass of each badge class, defaulting to equal thirds.
    t_end:
        Integration horizon handed to the flow.
    """
    if n_starts < 1:
        raise ValueError(f"n_starts must be positive, got {n_starts}")
    if not alphas and not stratified:
        raise ValueError("no start measure was requested")

    fun = build_functionals(tables, params, liability, social_harm)
    carries = np.array([b != "N" for b in fun.badge], dtype=float)

    names: list[str] = []
    concentrations: list[np.ndarray] = []
    for a in alphas:
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError(
                f"every Dirichlet concentration must be finite and positive, got {a}"
            )
        names.append(f"alpha={float(a):g}")
        concentrations.append(np.full(fun.n, float(a)))
    if stratified:
        names.append("badge-stratified")
        concentrations.append(stratified_alpha(fun.badge, stratified_weights))

    shape = (len(names),)
    badged_share = np.empty(shape)
    mixed_count = np.empty(shape, dtype=int)
    unsafe = np.empty(shape)
    social = np.empty(shape)
    for k, alpha in enumerate(concentrations):
        ends = replicator_attractors(
            fun.fitness, n_starts=n_starts, seed=seed, t_end=t_end, alpha=alpha
        )
        split = basin_classification(ends, carries)
        badged_share[k] = split["badged_share"]
        mixed_count[k] = split["mixed_count"]
        unsafe[k] = np.mean([aggregate_unsafe_frequency(e, fun) for e in ends])
        social[k] = np.mean([mean_social_payoff(e, fun) for e in ends])

    return StartMeasureAblation(
        measures=tuple(names),
        concentration=np.array(concentrations),
        badged_share=badged_share,
        mixed_count=mixed_count,
        unsafe=unsafe,
        social_payoff=social,
        share_range=(float(badged_share.min()), float(badged_share.max())),
        n_starts=int(n_starts),
        seed=int(seed),
    )


@dataclass(frozen=True)
class AssortmentBasinSweep:
    """Basin structure of the replicator flow along the assortment axis.

    The badged and unbadged endpoint groups are scored separately, which is
    the only way to see the content of the face reduction in the dynamics:
    the two groups are two entries into one conduct game, so their conduct
    observables have to agree wherever both groups are populated, at every
    assortment and without any parameter being tuned to make them.
    """

    rs: np.ndarray
    badged_share: np.ndarray
    mixed_count: np.ndarray
    unsafe_certified: np.ndarray
    unsafe_uncertified: np.ndarray
    social_certified: np.ndarray
    social_uncertified: np.ndarray
    """Shape ``(len(rs),)``; the four group observables are ``nan`` at an
    assortment where the corresponding group holds no endpoint."""

    max_face_gap: float
    """Largest ``|U_certified - U_uncertified|`` over the assortments at
    which both groups are populated.  The face reduction predicts zero, so
    this number is the dynamical residual of Theorem 1 and is quotable as
    such; ``nan`` if no assortment populates both groups."""

    n_starts: int
    seed: int


def assortment_basin_sweep(
    tables: RaceTables,
    params: IdentityParams,
    liability: float,
    social_harm: float,
    rs: np.ndarray,
    n_starts: int = 400,
    seed: int = cfg.SEED,
    t_end: float = 3000.0,
) -> AssortmentBasinSweep:
    """Replicator basins along the assortment axis, split by face.

    Assortment is the one parameter of the identity layer that the layer
    does not create, so it is the axis on which the basin claim has to be
    resolved rather than sampled at one point.  Per assortment the flow is
    started from ``n_starts`` interior points of the flat measure, the end
    states are split into the badged and unbadged faces by
    :func:`gbtag.dynamics.basin_classification`, and each face's conduct
    observables are averaged over its own endpoints.

    Every assortment uses the same seed, so the start set is the same set of
    simplex points throughout and a change along the axis is a change of the
    flow, not of the draw.
    """
    rs = np.asarray(rs, dtype=float)
    if rs.ndim != 1 or rs.size < 1:
        raise ValueError(f"rs must be a non-empty one-dimensional grid, got {rs.shape}")
    if rs.min() < 0.0 or rs.max() > 1.0:
        raise ValueError(f"every assortment must lie in [0, 1], got {rs.min()}..{rs.max()}")
    if n_starts < 1:
        raise ValueError(f"n_starts must be positive, got {n_starts}")

    shape = (rs.size,)
    badged_share = np.empty(shape)
    mixed_count = np.empty(shape, dtype=int)
    unsafe = {True: np.full(shape, np.nan), False: np.full(shape, np.nan)}
    social = {True: np.full(shape, np.nan), False: np.full(shape, np.nan)}
    for k, r in enumerate(rs):
        fun = build_functionals(
            tables, replace(params, r=float(r)), liability, social_harm
        )
        carries = np.array([b != "N" for b in fun.badge], dtype=float)
        ends = replicator_attractors(
            fun.fitness, n_starts=n_starts, seed=seed, t_end=t_end
        )
        split = basin_classification(ends, carries)
        badged_share[k] = split["badged_share"]
        mixed_count[k] = split["mixed_count"]
        is_badged = split["badged_mass"] > 0.5
        for face in (True, False):
            group = ends[is_badged if face else ~is_badged]
            if group.size == 0:
                continue
            unsafe[face][k] = np.mean(
                [aggregate_unsafe_frequency(e, fun) for e in group]
            )
            social[face][k] = np.mean([mean_social_payoff(e, fun) for e in group])

    gap = np.abs(unsafe[True] - unsafe[False])
    return AssortmentBasinSweep(
        rs=rs,
        badged_share=badged_share,
        mixed_count=mixed_count,
        unsafe_certified=unsafe[True],
        unsafe_uncertified=unsafe[False],
        social_certified=social[True],
        social_uncertified=social[False],
        max_face_gap=float(np.nanmax(gap)) if np.any(np.isfinite(gap)) else float("nan"),
        n_starts=int(n_starts),
        seed=int(seed),
    )


#: Regimes of the global phase plane, in the order the integer codes index.
REGIMES: tuple[str, ...] = (
    "badged-safe",
    "hollow",
    "mixed-unsafe",
    "unbadged-safe",
)

#: Population unsafe frequency below which a cell counts as safe.  Five per
#: cent of rounds is two orders of magnitude above the safe attractors of this
#: model and two orders below the anarchic ones, so no cell sits near it.
REGIME_UNSAFE_TOLERANCE: float = 0.05

#: Badged basin share above which a badged regime counts as *reachable* in a
#: cell.  This is deliberately not a majority test.  A basin share is a
#: property of the flow and of the start measure together, and this study
#: measures that the badged share moves from 0.04 to 0.47 under nothing but a
#: change of Dirichlet concentration, so a label gated on "which basin is
#: bigger" would encode the very artefact the study reports.  What the label
#: should say is what the badged attractors *are* where they are reached, so
#: the gate only asks whether they are reached at all, well below the smallest
#: share any measure we tried produces.
REGIME_SHARE_PRESENT: float = 0.02

#: Posterior confidence in a passed badge below which a safe badged cell is
#: hollow: the conduct holds, the attestation no longer means anything.
REGIME_INTEGRITY_FLOOR: float = 0.5


def _regime_code(share: float, unsafe: float, integrity: float) -> int:
    """Index into :data:`REGIMES` for one cell of the global plane.

    The unsafe test comes first because the other three regimes are all
    readings of a *safe* cell: asking whether a mark is hollow in a world
    that is unsafe anyway confuses a property of the attestation with a
    property of the outcome.  ``integrity`` is only consulted on the branch
    where a badged endpoint exists, so a ``nan`` integrity (no badged
    endpoint at all) can never decide a label.

    The badged branch is gated on reachability rather than on a majority of
    the basin, for the reason given at :data:`REGIME_SHARE_PRESENT`: the share
    is measure-dependent and the label should not be.
    """
    if not unsafe < REGIME_UNSAFE_TOLERANCE:
        return REGIMES.index("mixed-unsafe")
    if np.isnan(integrity) or share <= REGIME_SHARE_PRESENT:
        return REGIMES.index("unbadged-safe")
    if integrity >= REGIME_INTEGRITY_FLOOR:
        return REGIMES.index("badged-safe")
    return REGIMES.index("hollow")


@dataclass(frozen=True)
class GlobalPlane:
    """The replicator flow's own map of the ``(sigma, r)`` plane.

    The small-mutation plane of :func:`sigma_r_plane` answers where the
    finite process spends its time, which is a statement about one design at
    a time.  This one answers which regimes are reachable and how much of the
    simplex reaches them, which is the statement the headline claim of the
    manuscript is actually about, and it is resolved over the whole plane
    rather than sampled at the baseline point.
    """

    sigmas: np.ndarray
    rs: np.ndarray
    badged_share: np.ndarray
    unsafe: np.ndarray
    integrity: np.ndarray
    """Shape ``(len(rs), len(sigmas))``.  ``integrity`` is the mean over the
    *badged* endpoints only, and is ``nan`` in a cell that has none."""

    regime: np.ndarray
    """Integer code into :attr:`regime_names`, shape ``(len(rs), len(sigmas))``.

    Coded rather than stored as strings so that the plane travels in an
    ``npz`` beside the three float arrays it summarises.
    """

    regime_names: tuple[str, ...]
    n_starts: int
    seed: int


def global_phase_plane(
    tables: RaceTables,
    params: IdentityParams,
    liability: float,
    social_harm: float,
    sigmas: np.ndarray,
    rs: np.ndarray,
    n_starts: int = 200,
    seed: int = cfg.SEED,
    t_end: float = 3000.0,
) -> GlobalPlane:
    """Basin share, safety, integrity and regime over the ``(sigma, r)`` plane.

    Per cell the replicator flow is integrated from ``n_starts`` interior
    starts of the flat measure; the badged share is the share of end states
    on the badged face, the unsafe frequency is averaged over every end state
    and the attestation integrity over the badged end states alone.  Both
    observables are bilinear in the state, so they are evaluated at each
    attractor and averaged afterwards; evaluating either at the mean end
    state of a multistable flow manufactures encounters between designs that
    never meet.

    Integrity is conditioned on the badged endpoints because it is the
    posterior confidence in a *passed* badge, and in an unbadged end state
    nothing passes: the unconditional average would report the convention
    that an empty conditioning event reads as an unspoiled mark, and would
    make an unbadged world look like a world of honest marks.

    Every cell uses the same seed, so neighbouring cells differ by their
    flow and not by their draw, and the regime boundaries are boundaries of
    the model rather than of the sample.

    Cost is ``len(sigmas) * len(rs) * n_starts`` integrations of a
    48-dimensional flow and is the dominant cost of the whole robustness
    suite; a 24 x 24 grid at 200 starts is about forty minutes here.
    """
    sigmas = np.asarray(sigmas, dtype=float)
    rs = np.asarray(rs, dtype=float)
    if sigmas.ndim != 1 or sigmas.size < 1:
        raise ValueError(f"sigmas must be a non-empty grid, got shape {sigmas.shape}")
    if rs.ndim != 1 or rs.size < 1:
        raise ValueError(f"rs must be a non-empty grid, got shape {rs.shape}")
    if sigmas.min() < 0.0 or sigmas.max() > 1.0:
        raise ValueError(
            f"every spoof success must lie in [0, 1], got {sigmas.min()}..{sigmas.max()}"
        )
    if rs.min() < 0.0 or rs.max() > 1.0:
        raise ValueError(
            f"every assortment must lie in [0, 1], got {rs.min()}..{rs.max()}"
        )
    if n_starts < 1:
        raise ValueError(f"n_starts must be positive, got {n_starts}")

    shape = (rs.size, sigmas.size)
    badged_share = np.empty(shape)
    unsafe = np.empty(shape)
    integrity = np.full(shape, np.nan)
    regime = np.empty(shape, dtype=int)
    for a, r in enumerate(rs):
        for b, s in enumerate(sigmas):
            fun = build_functionals(
                tables,
                replace(params, sigma=float(s), r=float(r)),
                liability,
                social_harm,
            )
            carries = np.array([bb != "N" for bb in fun.badge], dtype=float)
            ends = replicator_attractors(
                fun.fitness, n_starts=n_starts, seed=seed, t_end=t_end
            )
            split = basin_classification(ends, carries)
            badged = ends[split["badged_mass"] > 0.5]
            badged_share[a, b] = split["badged_share"]
            unsafe[a, b] = np.mean([aggregate_unsafe_frequency(e, fun) for e in ends])
            if badged.size:
                integrity[a, b] = np.mean(
                    [attestation_integrity(e, fun) for e in badged]
                )
            regime[a, b] = _regime_code(
                badged_share[a, b], unsafe[a, b], integrity[a, b]
            )

    return GlobalPlane(
        sigmas=sigmas,
        rs=rs,
        badged_share=badged_share,
        unsafe=unsafe,
        integrity=integrity,
        regime=regime,
        regime_names=REGIMES,
        n_starts=int(n_starts),
        seed=int(seed),
    )


#: Arms of the badge ablation, in the order the arrays are indexed.
BADGE_ABLATION_ARMS: tuple[str, ...] = (
    "full",
    "no-badge",
    "unforgeable",
    "random-badge",
)


@dataclass(frozen=True)
class BadgeAblation:
    """What the mark contributes, against three counterfactual marks.

    ``full``
        the unmodified 48-design space;
    ``no-badge``
        the four unconditional unbadged designs, i.e. conduct alone;
    ``unforgeable``
        the same 48 designs with ``sigma = 0``, so a forged badge never
        passes and the mark is a perfect signal of provenance;
    ``random-badge``
        the sixteen conduct pairs under a badge drawn independently of
        conduct, so the mark is a perfect *non*-signal: it still splits
        encounters into two streams, still costs dues, still runs the
        handshake, and carries no information about the seat wearing it.

    The last arm is the one that tests the greenbeard reading.  If safety
    survives a badge that says nothing, the mechanism was an assortment
    device that happened to be wearing a badge, and the greenbeard language
    is decoration.
    """

    arms: tuple[str, ...]
    rs: np.ndarray
    sml_unsafe: np.ndarray
    basin_unsafe: np.ndarray
    """Shape ``(len(arms), len(rs))``: equilibrium unsafe frequency under the
    small-mutation chain and under the basin average of the replicator flow."""

    badged_share: np.ndarray
    """Shape ``(len(arms), len(rs))``, ``nan`` on the arms in which no design
    chooses a mark (``no-badge`` has none, ``random-badge`` gives every design
    the same drawn one)."""

    random_badge_sml_gap: float
    random_badge_basin_gap: float
    """Largest gap between the ``random-badge`` and ``no-badge`` arms over the
    swept assortments, under each dynamic.  Small means the informative badge
    is doing the work; large means the two streams are."""

    n_starts: int
    seed: int
    population_size: int
    beta: float


def badge_ablation(
    tables: RaceTables,
    params: IdentityParams,
    liability: float,
    social_harm: float,
    rs: np.ndarray,
    n_starts: int = 300,
    seed: int = cfg.SEED,
    population_size: int = cfg.POPULATION,
    beta: float = cfg.BETA,
    p_genuine: float = 1.0 / 3.0,
    p_forged: float = 1.0 / 3.0,
    t_end: float = 3000.0,
) -> BadgeAblation:
    """Sweep the four badge counterfactuals in assortment, under both dynamics.

    Each arm is scored by the same two formulas: the stationary unsafe
    frequency of the embedded small-mutation chain, and the basin average of
    the aggregate unsafe frequency over the replicator end states.  Both are
    read straight off the arm's own matrices rather than through
    :func:`gbtag.theory.equilibrium`, because the ``random-badge`` arm wears
    a sentinel badge and the badge-derived observables of that routine
    (attestation integrity, mark lift, the verified/unverified split) have no
    meaning on a mark nobody chose.  The two formulas used here are the ones
    that routine applies, so the arms remain comparable with the rest of the
    study.

    ``p_genuine`` and ``p_forged`` default to equal thirds over the three
    badge types, the draw that treats the sentinel badge as the population
    average of the badges of the full space.  They reach the arm through two
    channels only, the common pass rate ``qbar`` and the common dues, and
    ``qbar`` is the one that matters: at ``qbar = 0`` no check ever passes,
    every design executes its out-group conduct in every race, and the arm
    collapses onto the badgeless conduct game; at ``qbar = 1`` every check
    passes and it collapses onto the in-group conduct game less the dues.
    Equal thirds sits between them, at ``qbar = 1/2`` for the baseline
    ``sigma``, which is the most uninformative lottery the badge can be.
    """
    rs = np.asarray(rs, dtype=float)
    if rs.ndim != 1 or rs.size < 1:
        raise ValueError(f"rs must be a non-empty one-dimensional grid, got {rs.shape}")
    if rs.min() < 0.0 or rs.max() > 1.0:
        raise ValueError(
            f"every assortment must lie in [0, 1], got {rs.min()}..{rs.max()}"
        )
    if n_starts < 1:
        raise ValueError(f"n_starts must be positive, got {n_starts}")

    shape = (len(BADGE_ABLATION_ARMS), rs.size)
    sml_unsafe = np.empty(shape)
    basin_unsafe = np.empty(shape)
    badged_share = np.full(shape, np.nan)
    for k, r in enumerate(rs):
        at_r = replace(params, r=float(r))
        full = build_functionals(tables, at_r, liability, social_harm)
        arms = {
            "full": full,
            "no-badge": plain_subspace(full),
            "unforgeable": build_functionals(
                tables, replace(at_r, sigma=0.0), liability, social_harm
            ),
            "random-badge": random_badge_functionals(full, p_genuine, p_forged),
        }
        for j, name in enumerate(BADGE_ABLATION_ARMS):
            arm = arms[name]
            sml_unsafe[j, k] = stationary_analysis_sml(
                arm.fitness, arm.unsafe_frequency, population_size, beta
            ).unsafe_frequency
            ends = replicator_attractors(
                arm.fitness, n_starts=n_starts, seed=seed, t_end=t_end
            )
            basin_unsafe[j, k] = np.mean(
                [aggregate_unsafe_frequency(e, arm) for e in ends]
            )
            if name in ("full", "unforgeable"):
                carries = np.array([b != "N" for b in arm.badge], dtype=float)
                badged_share[j, k] = basin_classification(ends, carries)["badged_share"]

    plain = BADGE_ABLATION_ARMS.index("no-badge")
    random = BADGE_ABLATION_ARMS.index("random-badge")
    return BadgeAblation(
        arms=BADGE_ABLATION_ARMS,
        rs=rs,
        sml_unsafe=sml_unsafe,
        basin_unsafe=basin_unsafe,
        badged_share=badged_share,
        random_badge_sml_gap=float(
            np.abs(sml_unsafe[random] - sml_unsafe[plain]).max()
        ),
        random_badge_basin_gap=float(
            np.abs(basin_unsafe[random] - basin_unsafe[plain]).max()
        ),
        n_starts=int(n_starts),
        seed=int(seed),
        population_size=int(population_size),
        beta=float(beta),
    )


def _severity_rungs(severity: float) -> tuple[str, str, float]:
    """Adjacent rungs of the erosion ladder, and the weight on the upper one.

    :data:`gbtag.race.STRATEGIES` is already the erosion order -- each design
    carries one fewer safety clause than the one before it -- so the ladder
    of out-group severity is that tuple and not a second ordering that could
    drift from it.  ``severity`` is mapped affinely onto the three gaps
    between the four rungs, so the rungs sit at ``0, 1/3, 2/3, 1`` and each
    of them returns weight zero on the upper rung, i.e. the deterministic
    conduct itself.
    """
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must lie in [0, 1], got {severity}")
    steps = len(STRATEGIES) - 1
    position = float(severity) * steps
    # the top rung lands on position == steps, which has no upper neighbour;
    # it is expressed as the upper end of the last gap instead
    lower = min(int(np.floor(position)), steps - 1)
    return STRATEGIES[lower], STRATEGIES[lower + 1], position - lower


def _club_pool(
    fun: IdentityFunctionals, club: tuple[str, str, str]
) -> IdentityFunctionals:
    """One certified design against every unbadged and forged rival.

    The same pool :func:`gbtag.theory.out_group_policy_scan` scores a club
    in, so the discrete rungs of the severity sweep reproduce that scan
    rather than merely resembling it.
    """
    keep = np.array([i for i, d in enumerate(fun.designs) if d[0] != "G" or d == club])
    return fun.subspace(keep)


@dataclass(frozen=True)
class SeveritySweep:
    """The club's out-group policy as a continuous axis.

    Table 3 of the manuscript reads the exclusion trade-off off four
    discrete out-group conducts, which leaves it open whether the jump from
    a tolerant club to a viable one is a threshold of the model or an
    artefact of the ladder having only four rungs.  The sweep answers that
    by interpolating between the rungs exactly.
    """

    severities: np.ndarray
    lower_rung: tuple[str, ...]
    upper_rung: tuple[str, ...]
    weight: np.ndarray
    """Weight on the upper rung, shape ``(len(severities),)``."""

    spoof_threshold: np.ndarray
    entry_penalty: np.ndarray
    boundary_unsafe: np.ndarray
    club_share: np.ndarray
    certified_basin_share: np.ndarray
    """Shape ``(len(severities),)``.  ``spoof_threshold`` is ``nan`` where the
    forger invades at no spoof success in ``[0, 1]``; ``club_share`` is the
    monomorphic (small-mutation) share of the club design in its pool."""

    club_in: str
    forger: tuple[str, str, str]
    n_starts: int
    seed: int
    population_size: int
    beta: float


def out_group_severity_sweep(
    tables: RaceTables,
    params: IdentityParams,
    liability: float,
    social_harm: float,
    severities: np.ndarray,
    n_starts: int = 200,
    seed: int = cfg.SEED,
    club_in: str = "CS",
    forger: tuple[str, str, str] | None = None,
    population_size: int = cfg.POPULATION,
    beta: float = cfg.BETA,
    t_end: float = 3000.0,
) -> SeveritySweep:
    """Forgery tolerance and entry toll along a continuous out-group severity.

    A design may randomise *which whole race* it plays against the
    unverified: at the start of a race it draws one of the two adjacent
    rungs of the erosion ladder, with probability ``weight`` on the harsher
    one, and executes that deterministic conduct for every round of that
    race.  This is what makes the interpolation exact.  The race payoff ``A``
    is already an expectation over the horizon law of a *deterministic* pair,
    so the payoff of a race in which one seat's conduct was drawn at the
    start is the corresponding convex combination of the rows of ``A``, and
    likewise for the unsafe counts and frequencies.

    A per-round mixture is a different object and would not be a convex
    combination of ``A``.  Three of the four conducts condition on the
    partner's last action, so redrawing every round changes the action path
    itself: the opponent's reply to a round in which the coin came up harsh
    is carried into the next round, and the expectation of the path is not
    the mixture of the expectations of the two deterministic paths.  Nothing
    in this sweep is a per-round mixture, and nothing in it may be read as
    one.

    Everything the sweep reports is a convex combination of the two rungs'
    own quantities, exactly and for the same reason.  The mixed conduct is
    executed only against partners whose badge failed the club's check, and
    a club member's check of another club member never fails, so no encounter
    in the pool ever puts two draws of the mixture against one another and
    every matrix entry is affine in the weight.  The *thresholds* are not
    affine, because a root of an affine family is not the affine combination
    of the roots; the spoof threshold is therefore solved on the mixed
    advantage function, by the root finder that solves the discrete case.

    Parameters
    ----------
    severities:
        Values in ``[0, 1]``.  ``0, 1/3, 2/3, 1`` are the four rungs
        ``AS, CS, CAS, AU`` themselves, at which every reported quantity
        reproduces the discrete scan.
    club_in:
        The club's in-group conduct, held fixed while the out-group conduct
        sweeps.
    forger:
        The invader the spoof threshold is measured against; defaults to the
        canonical minimal-deviation forger.
    n_starts, seed:
        Interior starts and seed for the basin share of the club's pool.
    """
    severities = np.asarray(severities, dtype=float)
    if severities.ndim != 1 or severities.size < 1:
        raise ValueError(
            f"severities must be a non-empty grid, got shape {severities.shape}"
        )
    if club_in not in STRATEGIES:
        raise ValueError(f"club_in must be one of {STRATEGIES}, got {club_in!r}")
    if forger is None:
        forger = cfg.FALSEBEARD

    fun = build_functionals(tables, params, liability, social_harm)
    shape = (severities.size,)
    lower_rung: list[str] = []
    upper_rung: list[str] = []
    weight = np.empty(shape)
    threshold = np.full(shape, np.nan)
    penalty = np.empty(shape)
    boundary = np.empty(shape)
    club_share = np.empty(shape)
    basin_share = np.empty(shape)

    for k, s in enumerate(severities):
        lower, upper, w = _severity_rungs(float(s))
        lower_rung.append(lower)
        upper_rung.append(upper)
        weight[k] = w
        clubs = (("G", club_in, lower), ("G", club_in, upper))

        def advantage(sigma: float, clubs: tuple = clubs, w: float = w) -> float:
            """Invasion advantage of the forger against the mixed club."""
            at_sigma = replace(params, sigma=float(sigma))
            return sum(
                weight_of * (
                    fitness_against_resident(
                        tables, at_sigma, liability, forger, club
                    )
                    - pair_payoff(tables, at_sigma, liability, club, club)
                )
                for club, weight_of in zip(clubs, (1.0 - w, w))
            )

        root = _threshold_by_root(advantage, increasing=True)
        if root is not None:
            threshold[k] = root

        barriers = [entry_barrier(tables, params, liability, club_in, c[2]) for c in clubs]
        penalty[k] = (1.0 - w) * barriers[0][0] + w * barriers[1][0]
        boundary[k] = (1.0 - w) * barriers[0][1] + w * barriers[1][1]

        pools = [_club_pool(fun, club) for club in clubs]
        index = pools[0].designs.index(clubs[0])
        if pools[1].designs[index] != clubs[1] or (
            pools[0].designs[:index] + pools[0].designs[index + 1 :]
            != pools[1].designs[:index] + pools[1].designs[index + 1 :]
        ):
            raise ValueError(
                "the two rung pools are not aligned, so their matrices cannot "
                f"be mixed entry by entry: {clubs[0]} against {clubs[1]}"
            )
        fitness = (1.0 - w) * pools[0].fitness + w * pools[1].fitness
        unsafe = (1.0 - w) * pools[0].unsafe_frequency + w * pools[1].unsafe_frequency
        club_share[k] = stationary_analysis_sml(
            fitness, unsafe, population_size, beta
        ).strategy_frequencies[index]
        ends = replicator_attractors(
            fitness, n_starts=n_starts, seed=seed, t_end=t_end
        )
        carries = np.array([b != "N" for b in pools[0].badge], dtype=float)
        basin_share[k] = basin_classification(ends, carries)["badged_share"]

    return SeveritySweep(
        severities=severities,
        lower_rung=tuple(lower_rung),
        upper_rung=tuple(upper_rung),
        weight=weight,
        spoof_threshold=threshold,
        entry_penalty=penalty,
        boundary_unsafe=boundary,
        club_share=club_share,
        certified_basin_share=basin_share,
        club_in=club_in,
        forger=tuple(forger),
        n_starts=int(n_starts),
        seed=int(seed),
        population_size=int(population_size),
        beta=float(beta),
    )

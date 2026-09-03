"""Closed-form results and the structural quantities of the identity model.

Four quantities organise everything.

``L_R``
    the *reciprocity threshold* of the race: the liability below which a
    first-strike design profitably exploits the safe opening move of a
    conditional cooperator.  Below it the uncertified world cannot hold any
    safe population, however the designs reciprocate, because retaliation
    in kind cannot claw back a stolen first-move lead before the horizon.

``sigma*``
    the *spoof threshold* of a club: the spoof success probability above
    which a forged badge invades a monomorphic certified club.

``r*``
    the *nucleation threshold*: the assortment below which no certified
    club can invade the anarchic equilibrium of the uncertified world, at
    any verification quality whatsoever.

``sigma_m``
    the *mimicry threshold*: the spoof success above which a behaviourally
    identical forger displaces the genuine club, hollowing out the
    attestation while leaving conduct intact.

Everything below is computed from the exact matrices of
:mod:`gbtag.functionals`; nothing is estimated by simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .dynamics import (
    replicator_attractors,
    replicator_mutator_equilibrium,
    stationary_analysis_sml,
)
from .functionals import (
    IdentityFunctionals,
    aggregate_unsafe_frequency,
    attestation_integrity,
    mark_lift,
    mean_social_payoff,
    unsafe_split,
)
from .identity import IdentityParams, encounter_terms
from .race import STRATEGIES, RaceTables

_IDX = {s: i for i, s in enumerate(STRATEGIES)}


# --------------------------------------------------------------------------
# race-level primitives
# --------------------------------------------------------------------------


def race_private(tables: RaceTables, liability: float) -> np.ndarray:
    """``P = A - L M``: the private race payoff at liability ``L``."""
    return tables.payoff - liability * tables.unsafe_count


def pair_payoff(
    tables: RaceTables,
    params: IdentityParams,
    liability: float,
    focal: tuple[str, str, str],
    partner: tuple[str, str, str],
) -> float:
    """Exact expected private payoff of ``focal`` against ``partner``.

    The four-term handshake expectation of ``A - L M``, minus the focal
    badge cost and expected fine.  This is the scalar the propositions
    manipulate; it agrees entry by entry with the matrices of
    :func:`gbtag.functionals.build_functionals`.
    """
    p = race_private(tables, liability)
    value = sum(
        w * p[_IDX[s_f], _IDX[s_p]]
        for w, s_f, s_p in encounter_terms(focal, partner, params)
    )
    return float(
        value - params.badge_cost(focal[0]) - params.expected_fine(focal[0])
    )


def fitness_against_resident(
    tables: RaceTables,
    params: IdentityParams,
    liability: float,
    mutant: tuple[str, str, str],
    resident: tuple[str, str, str],
) -> float:
    """Fitness of a rare mutant in a monomorphic resident population.

    With assortment ``r`` the mutant meets its own design with probability
    ``r`` and the resident otherwise; the resident meets the resident either
    way.
    """
    own = pair_payoff(tables, params, liability, mutant, mutant)
    cross = pair_payoff(tables, params, liability, mutant, resident)
    return params.r * own + (1.0 - params.r) * cross


def invades(
    tables: RaceTables,
    params: IdentityParams,
    liability: float,
    mutant: tuple[str, str, str],
    resident: tuple[str, str, str],
    tol: float = 1e-9,
) -> bool:
    """Whether a rare ``mutant`` has a selective advantage in ``resident``."""
    mutant_fitness = fitness_against_resident(tables, params, liability, mutant, resident)
    resident_fitness = pair_payoff(tables, params, liability, resident, resident)
    return bool(mutant_fitness > resident_fitness + tol)


# --------------------------------------------------------------------------
# Proposition 1: the reciprocity threshold of the race
# --------------------------------------------------------------------------


def reciprocity_threshold(tables: RaceTables) -> float:
    r"""Liability below which the first-striker exploits the reciprocator.

    ``CAS`` against a ``CS`` resident steals the opening round, converts the
    half-step lead into the prize, and is punished only in kind, so its
    advantage is affine in the liability and vanishes at

    .. math:: L_R = \frac{A(\mathrm{CAS}, \mathrm{CS}) - A(\mathrm{CS},
        \mathrm{CS})}{M(\mathrm{CAS}, \mathrm{CS}) - M(\mathrm{CS},
        \mathrm{CS})} .

    ``CS`` is the best unbadged safe resident (the corresponding threshold
    for an ``AS`` resident is two orders of magnitude larger), so below
    ``L_R`` *no* unbadged design holds a safe population:
    :func:`uncertified_safety_is_impossible` checks the full statement.
    """
    a = tables.payoff
    m = tables.unsafe_count
    i, j = _IDX["CAS"], _IDX["CS"]
    return float((a[i, j] - a[j, j]) / (m[i, j] - m[j, j]))


def first_strike_threshold(tables: RaceTables, resident: str) -> float | None:
    """Liability at which ``CAS`` stops invading a safe unbadged resident."""
    a = tables.payoff
    m = tables.unsafe_count
    i, j = _IDX["CAS"], _IDX[resident]
    gain_m = m[i, j] - m[j, j]
    if abs(gain_m) < 1e-12:
        return None
    return float((a[i, j] - a[j, j]) / gain_m)


def first_strike_assortment_bound(tables: RaceTables) -> float:
    r"""Assortment above which reciprocity alone holds a safe unbadged population.

    The first-striker's transversal eigenvalue at a ``(N, *, CS)`` resident is
    ``A(CAS,CS) - A(CS,CS) - r [A(CAS,CS) - A(CAS,CAS)]``, which is affine and
    decreasing in ``r``.  Above its root the striker meets enough of its own
    aggression in self-encounters to give back the half-step it steals, and the
    safe unbadged residents survive.  The ``(N, *, AS)`` residents have the
    larger root (0.574), so this one binds.

    This is why the uncertified regime of the bistability result is safe at
    ``L = 0``: the baseline ``r = 0.1`` is above it.
    """
    p = tables.payoff
    i = _IDX
    return float(
        (p[i["CAS"], i["CS"]] - p[i["CS"], i["CS"]])
        / (p[i["CAS"], i["CS"]] - p[i["CAS"], i["CAS"]])
    )


def uncertified_safety_is_impossible(
    tables: RaceTables, liability: float, r: float = 0.0, tol: float = 1e-9
) -> bool:
    """Below ``L_R`` *and* below ``r_dagger``, every safe unbadged population is
    invadable.

    A safe unbadged resident plays a safe design in self-play, so its
    ``s_out`` is ``AS`` or ``CS`` (badges are absent, every check fails, and
    only ``s_out`` is ever executed).  The statement is that the plain
    first-striker ``(N, CAS, CAS)`` invades every such resident.

    Both bounds are needed.  The assortment default is ``0.0`` because that is
    the case the closed form ``L_R`` describes; at the manuscript's baseline
    ``r = 0.1`` the four ``(N, *, CS)`` residents resist and this returns
    ``False``, which is the honest answer and not a bug.
    """
    params = IdentityParams(sigma=0.0, kappa_g=0.0, kappa_f=0.0, rho=0.0, r=r)
    striker = ("N", "CAS", "CAS")
    for s_in in STRATEGIES:
        for s_out in ("AS", "CS"):
            resident = ("N", s_in, s_out)
            if not invades(tables, params, liability, striker, resident, tol):
                return False
    return True


# --------------------------------------------------------------------------
# Proposition 2: the spoof threshold of a certified club
# --------------------------------------------------------------------------


def spoof_threshold(
    tables: RaceTables,
    params: IdentityParams,
    liability: float,
    club: tuple[str, str, str],
    forger: tuple[str, str, str],
) -> float | None:
    """Spoof success above which ``forger`` invades a monomorphic ``club``.

    The forger's fitness in the club is affine in ``sigma`` whenever the
    forger is unconditional (``s_in == s_out``) and quadratic otherwise
    (its assorted self-play mixes two independent checks).  Both cases are
    solved exactly; the returned value is the smallest root in ``[0, 1]``
    above which the invasion condition holds, ``None`` if the forger
    invades nowhere in ``[0, 1]``, and ``0.0`` if it invades everywhere.
    """
    return _threshold_by_root(
        lambda sigma: fitness_against_resident(
            tables, replace(params, sigma=sigma), liability, forger, club
        )
        - pair_payoff(
            tables, replace(params, sigma=sigma), liability, club, club
        ),
        increasing=True,
    )


def spoof_threshold_closed_form(
    tables: RaceTables,
    liability: float,
    club_in: str,
    club_out: str,
    exploit: str,
    kappa_g: float,
    kappa_f: float,
    rho: float,
) -> float | None:
    r"""The well-mixed spoof threshold in closed form.

    For an unconditional forger ``(F, w, w)`` against a club ``(G, u, v)``
    at ``r = 0`` the invasion condition is affine in ``sigma`` and the
    threshold is

    .. math:: \sigma^{*} = \frac{P(u,u) - P(w,v) - \kappa_g + \kappa_f +
        \rho}{P(w,u) - P(w,v) + \rho} ,

    where ``P = A - L M``.  ``None`` when the denominator vanishes.
    """
    p = race_private(tables, liability)
    u, v, w = _IDX[club_in], _IDX[club_out], _IDX[exploit]
    denominator = p[w, u] - p[w, v] + rho
    if abs(denominator) < 1e-12:
        return None
    sigma = (p[u, u] - p[w, v] - kappa_g + kappa_f + rho) / denominator
    return float(sigma)


def _threshold_by_root(
    advantage,
    increasing: bool,
    lo: float = 0.0,
    hi: float = 1.0,
    grid: int = 2001,
) -> float | None:
    """Smallest ``sigma`` in ``[lo, hi]`` at which ``advantage`` crosses zero.

    The advantage functions of this model are polynomials of degree at most
    two in ``sigma``, so a fine grid plus bisection recovers the crossing to
    machine precision.  Returns ``None`` if the advantage never becomes
    positive on the interval, ``lo`` if it is positive everywhere.
    """
    xs = np.linspace(lo, hi, grid)
    values = np.array([advantage(x) for x in xs])
    positive = values > 0.0
    if not positive.any():
        return None
    if positive.all():
        return float(lo)
    k = int(np.argmax(positive)) if increasing else int(np.argmax(~positive))
    if increasing:
        if k == 0:
            return float(lo)
        a, b = xs[k - 1], xs[k]
    else:
        a, b = xs[k - 1], xs[k]
    for _ in range(80):
        mid = 0.5 * (a + b)
        if (advantage(mid) > 0.0) == increasing:
            b = mid
        else:
            a = mid
    return float(0.5 * (a + b))


# --------------------------------------------------------------------------
# Proposition 3: fines, dues, and the Becker limit
# --------------------------------------------------------------------------


def required_fine(
    tables: RaceTables,
    liability: float,
    sigma: float,
    club_in: str,
    club_out: str,
    exploit: str,
    kappa_g: float,
    kappa_f: float,
) -> float:
    r"""Break-even fine against the unconditional forger, well-mixed.

    Setting the invasion advantage to zero and solving for ``rho``:

    .. math:: \rho^{*}(\sigma) = \frac{\sigma\,[P(w,u) - P(w,v)] -
        [P(u,u) - P(w,v)] + \kappa_g - \kappa_f}{1 - \sigma} ,

    which diverges as ``sigma`` approaches one whenever exploiting the club
    at full pass beats club membership: no finite fine substitutes for
    detection in the spoof-proof limit.  Negative values mean no fine is
    needed at this ``sigma`` and the smallest feasible fine is
    ``max(0, rho_star)``; ``inf`` is returned at ``sigma = 1`` when the exploit
    pays.
    """
    p = race_private(tables, liability)
    u, v, w = _IDX[club_in], _IDX[club_out], _IDX[exploit]
    numerator = sigma * (p[w, u] - p[w, v]) - (p[u, u] - p[w, v]) + kappa_g - kappa_f
    if sigma >= 1.0:
        exploit_pays = p[w, u] - kappa_f > p[u, u] - kappa_g
        return float("inf") if exploit_pays else 0.0
    return float(numerator / (1.0 - sigma))


def dues_gradient(
    tables: RaceTables,
    liability: float,
    club_in: str,
    club_out: str,
    exploit: str,
    rho: float,
) -> float:
    r"""``d sigma* / d kappa_g``: what one unit of dues costs in tolerance.

    From the closed form, ``-1 / [P(w,u) - P(w,v) + \rho]``: every unit of
    certification cost lowers the spoof threshold of the club, because dues
    burden the genuine member and never the forger.
    """
    p = race_private(tables, liability)
    u, v, w = _IDX[club_in], _IDX[club_out], _IDX[exploit]
    return float(-1.0 / (p[w, u] - p[w, v] + rho))


# --------------------------------------------------------------------------
# Propositions 4 and 5: nucleation, and the futility of the badge alone
# --------------------------------------------------------------------------


def nucleation_threshold(
    tables: RaceTables,
    liability: float,
    club_in: str,
    club_out: str,
    resident: str,
    kappa_g: float,
) -> float:
    r"""Assortment below which the club cannot invade the anarchic resident.

    Against an unconditional unbadged resident ``(N, w, w)`` a club mutant
    earns its dues back only in its assorted self-encounters, so it invades
    exactly when

    .. math:: r > r^{*} = \frac{P(w,w) - P(v,w) + \kappa_g}
        {P(u,u) - P(v,w)} .

    The direction of that inequality is the sign of the denominator, and the
    sign is not always positive: 36 of the 64 ``(u, v, w)`` triples make it
    non-positive.  For ``(CS, CAS, CS)`` -- the baseline club against a *safe*
    unbadged resident, which is the transition between the two regimes of the
    bistability result -- it is ``-2.631``, so the club invades *below*
    ``r = 0.240`` and assortment obstructs it rather than enabling it.  Reading
    the returned number as a lower bound in that case inverts the result, so
    the sign is returned with it rather than left for the caller to rediscover.

    Returns ``(r_star, direction)`` where ``direction`` is ``+1`` when the club
    invades above ``r_star``, ``-1`` when it invades below, and ``0`` when the
    denominator vanishes and no assortment lets the club in.
    """
    p = race_private(tables, liability)
    u, v, w = _IDX[club_in], _IDX[club_out], _IDX[resident]
    denominator = p[u, u] - p[v, w]
    if abs(denominator) < 1e-12:
        return float("nan"), 0
    r_star = float((p[w, w] - p[v, w] + kappa_g) / denominator)
    return r_star, 1 if denominator > 0 else -1


def badge_is_futile_against_nonconditioners(
    tables: RaceTables,
    params: IdentityParams,
    liability: float,
    tol: float = 1e-9,
) -> bool:
    """A genuine badge is a pure cost against non-conditioning residents.

    For every resident with ``s_in == s_out`` and every conduct pair, the
    badged design earns exactly the unbadged twin's payoff minus the dues,
    at every ``sigma``.  Consequently, at ``r = 0`` and ``kappa_g > 0`` no
    certified design can invade any unconditional resident that its
    unbadged twin could not already invade: verification quality is
    irrelevant to nucleation.

    The proposition is stated for *any* resident whose conduct does not
    condition on badges, which includes badged and forged non-conditioners as
    well as unbadged ones, so all three resident badge types are checked here.
    """
    wm = replace(params, r=0.0)
    for resident_badge in ("N", "G", "F"):
        for s_in in STRATEGIES:
            for s_out in STRATEGIES:
                for w in STRATEGIES:
                    resident = (resident_badge, w, w)
                    badged = pair_payoff(
                        tables, wm, liability, ("G", s_in, s_out), resident
                    )
                    plain = pair_payoff(
                        tables, wm, liability, ("N", s_in, s_out), resident
                    )
                    if abs(badged - (plain - wm.kappa_g)) > tol:
                        return False
    return True


# --------------------------------------------------------------------------
# Proposition 6: mimicry and the two collapse modes
# --------------------------------------------------------------------------


def mimic_threshold_closed_form(
    tables: RaceTables,
    liability: float,
    club_in: str,
    club_out: str,
    kappa_g: float,
    kappa_f: float,
    rho: float,
) -> float:
    r"""Spoof success above which the behavioural mimic displaces the club.

    The mimic ``(F, u, v)`` plays exactly the club's conduct; its only
    difference is the forged badge.  Well-mixed, it invades when the dues
    saved exceed the expected cost of failing checks:

    .. math:: \sigma_m = 1 - \frac{\kappa_g - \kappa_f}
        {P(u,u) - P(u,v) + \rho} .

    Above ``sigma_m`` the club is displaced by forgers with identical
    conduct: the population still behaves, but a passed badge no longer
    certifies provenance.  Attestation collapses before safety does.
    """
    p = race_private(tables, liability)
    u, v = _IDX[club_in], _IDX[club_out]
    return float(1.0 - (kappa_g - kappa_f) / (p[u, u] - p[u, v] + rho))


def invader_ordering_flip(
    tables: RaceTables,
    params: IdentityParams,
    liability: float,
    club: tuple[str, str, str],
    exploiter: tuple[str, str, str],
    mimic: tuple[str, str, str],
    r_hi: float = 0.5,
    iterations: int = 60,
) -> float | None:
    """Assortment at which the mimic overtakes the exploiter as first invader.

    Assortment penalises the exploiter and not the mimic, because a mimic
    plays the club's own conduct in its self-encounters while an exploiter
    plays its aggressive conduct against itself.  The exploiter's threshold
    therefore climbs steeply in ``r`` and the mimic's is nearly flat, and
    they cross.  Below the crossing the club fails visibly, by being
    exploited; above it the club fails invisibly, by being impersonated.

    Returns the crossing assortment, or ``None`` if the mimic is never the
    first invader on ``[0, r_hi]``.
    """

    def order(r: float) -> bool:
        """Whether the mimic threshold is the lower of the two at ``r``."""
        p = replace(params, r=r)
        s_e = spoof_threshold(tables, p, liability, club, exploiter)
        s_m = spoof_threshold(tables, p, liability, club, mimic)
        if s_m is None:
            return False
        if s_e is None:
            return True  # the exploiter cannot invade at all, so the mimic leads
        return s_m < s_e

    if order(0.0):
        return 0.0
    if not order(r_hi):
        return None
    lo, hi = 0.0, r_hi
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if order(mid):
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


def exploiter_immunity(
    tables: RaceTables,
    params: IdentityParams,
    liability: float,
    club: tuple[str, str, str],
    exploiter: tuple[str, str, str],
    r_hi: float = 0.5,
    iterations: int = 60,
) -> float | None:
    """Assortment above which the exploiter cannot invade at any ``sigma``.

    Beyond this level of clustering the club is immune to behavioural
    exploitation even by a forgery that always passes, and the only invasion
    route left is impersonation by a design that behaves.
    """

    def invadable(r: float) -> bool:
        p = replace(params, r=r)
        return spoof_threshold(tables, p, liability, club, exploiter) is not None

    if not invadable(0.0):
        return 0.0
    if invadable(r_hi):
        return None
    lo, hi = 0.0, r_hi
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if invadable(mid):
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def reentry_threshold(
    tables: RaceTables,
    params: IdentityParams,
    liability: float,
    club: tuple[str, str, str],
    resident: tuple[str, str, str],
) -> float | None:
    """Smallest sigma at which the club re-invades a collapsed resident.

    ``None`` when the club cannot re-enter at any verification quality --
    the generic case at ``r = 0``, which is the hysteresis result: the
    conditions for keeping a certified club are strictly weaker than the
    conditions for rebuilding one.
    """
    return _threshold_by_root(
        lambda sigma: fitness_against_resident(
            tables, replace(params, sigma=sigma), liability, club, resident
        )
        - pair_payoff(
            tables, replace(params, sigma=sigma), liability, resident, resident
        ),
        increasing=False,
    )


# --------------------------------------------------------------------------
# Proposition 7: providers, concentration, and federation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderFrontier:
    """The symmetric ``K``-provider world under provider-scoped marks."""

    n_providers: int
    hhi: float
    unsafe_frequency: float
    member_payoff: float
    entrant_payoff: float
    exclusion_rent: float
    federated_unsafe_frequency: float


def provider_frontier(
    tables: RaceTables,
    liability: float,
    club_in: str,
    club_out: str,
    kappa_g: float,
    n_providers: int,
) -> ProviderFrontier:
    r"""Exact outcome of ``K`` symmetric provider clubs with scoped marks.

    Every provider runs the club design with its own mark; cross-provider
    checks fail, so within-provider play is ``u`` and cross-provider play is
    ``v``.  With symmetric shares the within-provider encounter probability
    equals the Herfindahl index ``1/K`` and

    .. math:: U = \mathrm{HHI}\,U(u,u) + (1 - \mathrm{HHI})\,U(v,v) .

    The exclusion rent is the payoff gap between a member and a compliant
    unbadged entrant that plays the club conduct ``u`` towards everyone and
    is treated as an outsider by all ``K`` clubs.  Under federation (mutual
    recognition of marks) every cross-provider check passes and the
    ecosystem plays ``u`` throughout, at any concentration.
    """
    p = race_private(tables, liability)
    uf = tables.unsafe_frequency
    u, v = _IDX[club_in], _IDX[club_out]
    hhi = 1.0 / n_providers
    unsafe = hhi * uf[u, u] + (1.0 - hhi) * uf[v, v]
    member = hhi * p[u, u] + (1.0 - hhi) * p[v, v] - kappa_g
    entrant = p[u, v]  # every club treats it as an outsider, so it meets v throughout
    rent = member - entrant
    return ProviderFrontier(
        n_providers=n_providers,
        hhi=hhi,
        unsafe_frequency=float(unsafe),
        member_payoff=float(member),
        entrant_payoff=float(entrant),
        exclusion_rent=float(rent),
        federated_unsafe_frequency=float(uf[u, u]),
    )


# --------------------------------------------------------------------------
# Proposition 8: the exclusion dividend
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HarmDecomposition:
    """Where the harm of a mixed population actually happens.

    Every ordered encounter is assigned to one of four blocks by the badge
    status of the two seats, and the population unsafe frequency is split
    across them exactly.  The blocks sum to the reported total by
    construction, so the shares are an accounting identity rather than an
    approximation.
    """

    mass: dict[str, float]
    """Probability of each block under the encounter law."""

    contribution: dict[str, float]
    """Additive contribution of each block to the population unsafe frequency."""

    conditional: dict[str, float]
    """Unsafe frequency conditional on being in each block."""

    total: float
    interface_share: float
    """Share of all harm produced at the badged/unbadged interface."""


#: The four blocks of the harm decomposition.
BLOCKS: tuple[str, ...] = (
    "badged with badged",
    "badged with unbadged",
    "unbadged with badged",
    "unbadged with unbadged",
)


def harm_decomposition(
    x: np.ndarray, fun: IdentityFunctionals, monomorphic: bool = False
) -> HarmDecomposition:
    """Split the population unsafe frequency by the badge status of the pair.

    A badge is *carried* if the design presents one at all, genuine or
    forged; the split is by what an observer of the encounter would see,
    not by what the check concluded.
    """
    from .functionals import _encounter_weights

    x = np.asarray(x, dtype=float)
    w = _encounter_weights(x, fun.params.r, monomorphic)
    u = fun.unsafe_frequency
    carries = np.array([b != "N" for b in fun.badge])
    masks = {
        "badged with badged": np.outer(carries, carries),
        "badged with unbadged": np.outer(carries, ~carries),
        "unbadged with badged": np.outer(~carries, carries),
        "unbadged with unbadged": np.outer(~carries, ~carries),
    }
    mass = {k: float((w * m).sum()) for k, m in masks.items()}
    contribution = {k: float((w * m * u).sum()) for k, m in masks.items()}
    conditional = {
        k: contribution[k] / mass[k] if mass[k] > 1e-15 else 0.0 for k in masks
    }
    total = float(sum(contribution.values()))
    interface = contribution["badged with unbadged"] + contribution["unbadged with badged"]
    return HarmDecomposition(
        mass=mass,
        contribution=contribution,
        conditional=conditional,
        total=total,
        interface_share=interface / total if total > 1e-15 else 0.0,
    )


def entry_barrier(
    tables: RaceTables,
    params: IdentityParams,
    liability: float,
    club_in: str,
    club_out: str,
) -> tuple[float, float]:
    """What a compliant outsider forfeits at the boundary of a club.

    The entrant carries no badge and plays the club's own in-group conduct
    towards everybody, so it is behaviourally beyond reproach.  Every club
    member's check of it nonetheless fails, so the member executes its
    out-group conduct and the entrant meets ``club_out`` in every encounter.

    Returns ``(penalty, boundary_unsafe)``: the payoff gap between a club
    member in its own club and this entrant, and the entrant's unsafe
    frequency at the boundary.  Both are exact closed forms with no dynamics
    in them, which is what makes them the right place to measure the
    exclusion: they do not depend on the two populations coexisting.
    """
    p = race_private(tables, liability)
    u, v = _IDX[club_in], _IDX[club_out]
    member = float(p[u, u] - params.kappa_g)
    entrant = float(p[u, v])
    return member - entrant, float(tables.unsafe_frequency[u, v])


@dataclass(frozen=True)
class OutGroupPolicy:
    """What one out-group policy buys the club, and what it costs.

    ``spoof_threshold`` is the forgery the policy lets the club survive;
    ``entry_penalty`` is what it costs a behaviourally compliant outsider.
    Proposition 8 is that the two cannot be separated: the policies with a
    positive threshold are exactly the policies with a positive penalty.
    """

    s_out: str
    spoof_threshold: float | None
    mimic_threshold: float | None
    nucleation_threshold: float
    entry_penalty: float
    boundary_unsafe: float
    certified_basin_share: float
    """Share of interior replicator starts that reach a badged regime."""

    club_share_monomorphic: float
    unsafe_basin_mean: float
    """Unsafe frequency averaged over attractors, not evaluated at their mean."""

    unsafe_monomorphic: float


def out_group_policy_scan(
    tables: RaceTables,
    params: IdentityParams,
    club_in: str = "CS",
    liability: float = 0.0,
    social_harm: float = 20.0,
    population_size: int = 100,
    beta: float = 0.05,
    n_starts: int = 200,
    seed: int = 20260819,
) -> list[OutGroupPolicy]:
    """Scan the club's out-group policy from unconditionally safe to harsh.

    For each candidate ``s_out`` the design pool keeps that one certified
    design plus every unbadged and forged design, so the club is scored
    against exactly the invaders it would actually face.
    """
    from .functionals import build_functionals

    fun_full = build_functionals(tables, params, liability, social_harm)
    out = []
    for s_out in STRATEGIES:
        club = ("G", club_in, s_out)
        keep = np.array(
            [i for i, d in enumerate(fun_full.designs) if d[0] != "G" or d == club]
        )
        sub = fun_full.subspace(keep)
        ci = sub.designs.index(club)
        basins = equilibrium(sub, tables, "replicator", n_starts=n_starts, seed=seed)
        mono = equilibrium(sub, tables, "sml", population_size, beta)
        penalty, boundary = entry_barrier(
            tables, params, liability, club_in, s_out
        )
        out.append(
            OutGroupPolicy(
                s_out=s_out,
                spoof_threshold=spoof_threshold(
                    tables, params, liability, club, ("F", "CAS", "CAS")
                ),
                mimic_threshold=spoof_threshold(
                    tables, params, liability, club, ("F", club_in, s_out)
                ),
                nucleation_threshold=nucleation_threshold(
                    tables, liability, club_in, s_out, "CAS", params.kappa_g
                )[0],
                entry_penalty=penalty,
                boundary_unsafe=boundary,
                certified_basin_share=basins.certified_basin_share,
                club_share_monomorphic=float(mono.frequencies[ci]),
                unsafe_basin_mean=basins.unsafe_frequency,
                unsafe_monomorphic=mono.unsafe_frequency,
            )
        )
    return out


@dataclass(frozen=True)
class GentleClubRescue:
    """Whether a fine can make a club that is safe towards outsiders viable.

    A fine on detected forgery restores the *barrier*: it charges an
    impostor for the badge it wears, so the gentle club's spoof threshold
    rises from zero.  It does not restore the club's *purpose*.  By
    Proposition 4 a badge earns nothing against residents that do not
    condition on badges, and a gentle club is behaviourally identical to the
    unbadged cooperator, which delivers the same conduct without the dues.
    Excludability against forgers and a reason to exist are different
    requirements, and only out-group harshness supplies both.
    """

    rho: float
    spoof_threshold: float | None
    club_share_mixed: float
    club_share_monomorphic: float
    unsafe_mixed: float
    unsafe_monomorphic: float
    attestation_integrity: float


def gentle_club_rescue(
    tables: RaceTables,
    params: IdentityParams,
    rhos: tuple[float, ...],
    club_in: str = "CS",
    liability: float = 0.0,
    social_harm: float = 20.0,
    population_size: int = 100,
    beta: float = 0.05,
    n_starts: int = 200,
    seed: int = 20260819,
) -> list[GentleClubRescue]:
    """Sweep the fine for a club whose out-group conduct equals its in-group.

    The design pool keeps the gentle certified design plus every unbadged
    and forged design, so the club is scored against the invaders it would
    actually face, exactly as in :func:`out_group_policy_scan`.
    """
    from .functionals import build_functionals

    club = ("G", club_in, club_in)
    exploiter = ("F", "CAS", "CAS")
    out = []
    for rho in rhos:
        p = replace(params, rho=float(rho))
        fun = build_functionals(tables, p, liability, social_harm)
        keep = np.array(
            [i for i, d in enumerate(fun.designs) if d[0] != "G" or d == club]
        )
        sub = fun.subspace(keep)
        ci = sub.designs.index(club)
        mixed = equilibrium(sub, tables, "replicator", n_starts=n_starts, seed=seed)
        mono = equilibrium(sub, tables, "sml", population_size, beta)
        out.append(
            GentleClubRescue(
                rho=float(rho),
                spoof_threshold=spoof_threshold(
                    tables, replace(p, r=0.0), liability, club, exploiter
                ),
                club_share_mixed=float(mixed.frequencies[ci]),
                club_share_monomorphic=float(mono.frequencies[ci]),
                unsafe_mixed=mixed.unsafe_frequency,
                unsafe_monomorphic=mono.unsafe_frequency,
                attestation_integrity=mono.attestation_integrity,
            )
        )
    return out


# --------------------------------------------------------------------------
# equilibrium summaries
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Equilibrium:
    """Long-run outcome of one parameterisation.

    Every scalar field is an average of the observable over whatever the
    method's long-run object is, never the observable evaluated at an
    average state.  The distinction is not pedantic: the replicator flow of
    this game is multistable, its attractors are disjoint faces of the
    simplex, and the arithmetic mean of two such faces is a state the flow
    never occupies.  Scoring a bilinear observable there manufactures
    cross-terms between designs that never actually meet.
    """

    frequencies: np.ndarray
    """The long-run state, or the basin-weighted mean of the attractors.

    For a multistable method this is a summary of *composition* only, and
    the scalar fields below are not functions of it.
    """

    unsafe_frequency: float
    class_distribution: dict[str, float]
    badge_distribution: dict[str, float]
    mark_lift: float
    attestation_integrity: float
    unsafe_verified: float
    unsafe_unverified: float
    social_payoff: float
    method: str
    harm_blocks: HarmDecomposition
    n_attractors: int = 1
    """Number of distinct attractors the basin sample reached."""

    certified_basin_share: float = float("nan")
    """Share of interior starts reaching a state in which badges are worn."""


def _score_state(
    x: np.ndarray,
    fun: IdentityFunctionals,
    tables: RaceTables,
    monomorphic: bool,
) -> dict[str, object]:
    """Every observable evaluated at one population state."""
    split = unsafe_split(x, fun, tables, monomorphic=monomorphic)
    return {
        "unsafe_frequency": aggregate_unsafe_frequency(x, fun, monomorphic=monomorphic),
        "mark_lift": mark_lift(x, fun, monomorphic=monomorphic),
        "attestation_integrity": attestation_integrity(
            x, fun, monomorphic=monomorphic
        ),
        "unsafe_verified": split["verified"],
        "unsafe_unverified": split["unverified"],
        "social_payoff": mean_social_payoff(x, fun, monomorphic=monomorphic),
        "harm_blocks": harm_decomposition(x, fun, monomorphic=monomorphic),
    }


def _average_harm_blocks(blocks: list[HarmDecomposition]) -> HarmDecomposition:
    """Basin average of the harm decomposition, block by block."""
    mass = {b: float(np.mean([d.mass[b] for d in blocks])) for b in BLOCKS}
    contribution = {
        b: float(np.mean([d.contribution[b] for d in blocks])) for b in BLOCKS
    }
    total = float(sum(contribution.values()))
    conditional = {
        b: contribution[b] / mass[b] if mass[b] > 1e-15 else 0.0 for b in BLOCKS
    }
    interface = contribution["badged with unbadged"] + contribution["unbadged with badged"]
    return HarmDecomposition(
        mass=mass,
        contribution=contribution,
        conditional=conditional,
        total=total,
        interface_share=interface / total if total > 1e-15 else 0.0,
    )


def equilibrium(
    fun: IdentityFunctionals,
    tables: RaceTables,
    method: str = "sml",
    population_size: int = 100,
    beta: float = 0.05,
    # 200 is what every script passes and what the paper reports; a lower
    # default here only produces basin shares that match nothing published.
    n_starts: int = 200,
    seed: int = 20260819,
    mu: float = 0.01,
    ends: np.ndarray | None = None,
) -> Equilibrium:
    """Long-run design distribution under the private functional.

    Three methods, answering three different questions.

    ``sml``
        the small-mutation limit of the finite-population process: which
        single design the market spends its time in.
    ``replicator``
        the basin distribution of the replicator flow: which pure regimes
        are stable, and how large each basin is.  Observables are evaluated
        at each attractor and averaged over basins, never at the mean state.
    ``mutator``
        the interior rest point of the replicator-mutator flow: a market in
        which every design is continually reintroduced and therefore
        genuinely coexists.  This is the only one of the three that produces
        encounters across the badged/unbadged boundary.
    """
    if method == "sml":
        x = stationary_analysis_sml(
            fun.fitness, fun.unsafe_frequency, population_size, beta
        ).strategy_frequencies
        scored = _score_state(x, fun, tables, monomorphic=True)
        return Equilibrium(
            frequencies=x,
            class_distribution=_class_masses_for(fun, x),
            badge_distribution=_badge_masses_for(fun, x),
            method=method,
            **scored,
        )

    if method == "mutator":
        x = replicator_mutator_equilibrium(fun.fitness, mu)
        scored = _score_state(x, fun, tables, monomorphic=False)
        return Equilibrium(
            frequencies=x,
            class_distribution=_class_masses_for(fun, x),
            badge_distribution=_badge_masses_for(fun, x),
            method=method,
            **scored,
        )

    if method != "replicator":
        raise ValueError(f"unknown method {method!r}")

    # A caller that also needs the raw end states can pass them in rather than
    # pay for the integration twice.  At the basin sample size that is the
    # difference between eight minutes and sixteen.
    if ends is None:
        ends = replicator_attractors(fun.fitness, n_starts=n_starts, seed=seed)
    scored = [_score_state(e, fun, tables, monomorphic=False) for e in ends]
    mean_state = ends.mean(axis=0)
    carries = np.array([b != "N" for b in fun.badge], dtype=float)
    unique = np.unique(np.round(ends, 6), axis=0)
    return Equilibrium(
        frequencies=mean_state,
        unsafe_frequency=float(np.mean([s["unsafe_frequency"] for s in scored])),
        class_distribution=_class_masses_for(fun, mean_state),
        badge_distribution=_badge_masses_for(fun, mean_state),
        mark_lift=float(np.mean([s["mark_lift"] for s in scored])),
        attestation_integrity=float(
            np.mean([s["attestation_integrity"] for s in scored])
        ),
        unsafe_verified=float(np.mean([s["unsafe_verified"] for s in scored])),
        unsafe_unverified=float(np.mean([s["unsafe_unverified"] for s in scored])),
        social_payoff=float(np.mean([s["social_payoff"] for s in scored])),
        method=method,
        harm_blocks=_average_harm_blocks([s["harm_blocks"] for s in scored]),
        n_attractors=int(len(unique)),
        certified_basin_share=float(np.mean(ends @ carries > 0.5)),
    )


def _class_masses_for(fun: IdentityFunctionals, x: np.ndarray) -> dict[str, float]:
    """Class masses that work on subspaces as well as the full space."""
    from .identity import CLASSES, classify

    out = {c: 0.0 for c in CLASSES}
    for k, design in enumerate(fun.designs):
        out[classify(*design)] += float(x[k])
    return out


def _badge_masses_for(fun: IdentityFunctionals, x: np.ndarray) -> dict[str, float]:
    from .identity import BADGES

    return {
        b: float(sum(x[k] for k, bb in enumerate(fun.badge) if bb == b))
        for b in BADGES
    }


# --------------------------------------------------------------------------
# Theorem 1: the invariant-face reduction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FaceReduction:
    """Certificate that the two badge-pure faces carry the same conduct game.

    Every field is measured off the assembled matrices rather than asserted,
    so the object is the proof and not a restatement of it.
    """

    dues: float
    """``kappa_g``: the additive constant that separates the two faces."""

    certified_deviation: float
    """``max |pi_P on the all-G face - (P(s_in) - kappa_g)|`` over the block."""

    unbadged_deviation: float
    """``max |pi_P on the all-N face - P(s_out)|`` over the block."""

    conduct_payoff: np.ndarray
    """The shared conduct game ``P = A - L M``, shape ``(4, 4)``."""

    social_gap: float
    """Social payoff on the unbadged face minus the certified face.

    Constant over all sixteen ordered conduct pairs and equal to the dues:
    the certified face buys nothing in conduct and pays for the badge.  It
    does not depend on the social harm ``h`` either, because ``h`` multiplies
    the same expected Unsafe count on both faces.
    """

    exact: bool
    """Whether both deviations sit at or below the requested tolerance."""


def face_reduction(
    tables: RaceTables,
    params: IdentityParams,
    liability: float = 0.0,
    tol: float = 1e-12,
) -> FaceReduction:
    r"""Reduce the 48-design game on its two badge-pure faces.

    A face of the simplex on which a set of designs carries zero mass is
    invariant under the replicator flow, so the all-genuine face and the
    all-unbadged face are each a closed subsystem.  On the all-genuine face
    every badge passes, so ``q_i = q_j = 1``, the four handshake weights
    collapse to ``w_in = 1`` and ``w_out = 0``, and

    .. math:: \pi_P(i, j) = A(s^{\mathrm{in}}_i, s^{\mathrm{in}}_j)
        - L\,M(s^{\mathrm{in}}_i, s^{\mathrm{in}}_j) - \kappa_g .

    On the all-unbadged face every check fails, ``q = 0``, and the same
    identity holds with ``s^{\mathrm{out}}`` in place of ``s^{\mathrm{in}}``
    and no dues.  Adding a constant to every entry of a payoff matrix leaves
    the replicator field unchanged, so the two faces carry *the same*
    replicator system on the four conducts: identical rest points, identical
    stability, identical conduct attractors, identical unsafe frequency, and
    social payoffs differing by exactly ``kappa_g``.

    This is the reduction the bistability result actually needs.  The two
    attractors of the full game are not two behavioural worlds, one safe
    because it is certified and one safe by accident: they are one conduct
    game entered twice, and the whole equilibrium effect of certification on
    a settled market is its price.  Establishing that by exhibiting the
    identity costs nothing and holds at every ``(sigma, r, kappa_g, L)``,
    where a sampled basin experiment at one parameter point establishes it
    nowhere.

    The two blocks are *not* aligned with one another, and that is the trap
    in computing this.  Designs are ordered badge-major, then ``s_in``, then
    ``s_out``, so inside the ``G`` block the executed conduct ``s_in`` is
    constant along runs of four while inside the ``N`` block the executed
    conduct ``s_out`` cycles every one.  The prediction is therefore indexed
    by each block's own executed conduct, read back off the block, never by
    position: indexing both blocks the same way yields a plausible-looking
    matrix that is wrong in twelve of its sixteen conduct pairs.

    Parameters
    ----------
    tables:
        Race tables supplying ``A`` and ``M``.
    params:
        Identity-layer parameters.  Only ``kappa_g`` enters the statement;
        ``sigma``, ``rho`` and ``r`` cannot, because no forged badge is
        present on either face and assortment shifts every row of a block by
        its own diagonal on both faces alike.
    liability:
        Private liability ``L`` per Unsafe action.
    tol:
        Tolerance for ``exact``.  The default is ``1e-12`` rather than zero
        because the caller decides how much slack a claim of exactness may
        carry; the deviations themselves are returned unrounded, and at every
        parameterisation tried here they are ``0.0``.
    """
    from .functionals import build_functionals

    if liability < 0.0:
        raise ValueError(f"liability must be non-negative, got {liability}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")

    fun = build_functionals(tables, params, liability)
    conduct = race_private(tables, liability)

    certified = fun.badge_block("G")
    unbadged = fun.badge_block("N")
    # the executed conduct of each design *of that block*, in the block's own
    # order; this is the indexing the docstring warns about
    certified_conduct = [_IDX[fun.s_in[i]] for i in certified]
    unbadged_conduct = [_IDX[fun.s_out[i]] for i in unbadged]

    predicted_certified = (
        conduct[np.ix_(certified_conduct, certified_conduct)] - params.kappa_g
    )
    predicted_unbadged = conduct[np.ix_(unbadged_conduct, unbadged_conduct)]
    certified_deviation = float(
        np.abs(fun.pi_P[np.ix_(certified, certified)] - predicted_certified).max()
    )
    unbadged_deviation = float(
        np.abs(fun.pi_P[np.ix_(unbadged, unbadged)] - predicted_unbadged).max()
    )

    # One representative design per executed conduct on each face.  The other
    # three per conduct are payoff-identical on their own face -- their
    # unexecuted branch never runs -- so the choice cannot move the gap, and
    # the constancy of the gap over the sixteen pairs is checked below rather
    # than assumed.
    representative_certified = [
        int(next(i for i in certified if fun.s_in[i] == c)) for c in STRATEGIES
    ]
    representative_unbadged = [
        int(next(i for i in unbadged if fun.s_out[i] == c)) for c in STRATEGIES
    ]
    gap = (
        fun.pi_S[np.ix_(representative_unbadged, representative_unbadged)]
        - fun.pi_S[np.ix_(representative_certified, representative_certified)]
    )
    spread = float(gap.max() - gap.min())
    if spread > tol:
        raise ValueError(
            "the social gap between the faces is not constant across the "
            f"conduct pairs, spread {spread}"
        )

    return FaceReduction(
        dues=float(params.kappa_g),
        certified_deviation=certified_deviation,
        unbadged_deviation=unbadged_deviation,
        conduct_payoff=conduct,
        social_gap=float(gap.flat[0]),
        exact=bool(certified_deviation <= tol and unbadged_deviation <= tol),
    )


# --------------------------------------------------------------------------
# Theorem 2: the macroscopic factorisation
# --------------------------------------------------------------------------


def _macro_rows(
    fun: IdentityFunctionals,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """The sixteen ``T`` and ``Q`` rows of one design space.

    ``Q`` weights each design by the pass rate its *partner* responds to,
    which is the behavioural rate and not the verification rate.  The two
    coincide under screening (the baseline), and under retrospective auditing
    they do not: there a forged badge elicits the in-group response whatever
    ``sigma`` is, so a ``Q`` row built from ``pass_rate`` would leave the span
    and the factorisation would be false for a reason that has nothing to do
    with the algebra.
    """
    q = np.array([fun.params.behavioural_pass_rate(b) for b in fun.badge])
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for kind in ("T", "Q"):
        for branch, executed in (("in", fun.s_in), ("out", fun.s_out)):
            for conduct in STRATEGIES:
                indicator = np.array(
                    [1.0 if s == conduct else 0.0 for s in executed]
                )
                rows.append(indicator if kind == "T" else indicator * q)
                labels.append(f"{kind}[{branch},{conduct}]")
    return np.vstack(rows), tuple(labels)


@dataclass(frozen=True)
class MacroFactorisation:
    """Low-rank certificate of the 48-design game.

    The design space is 48-dimensional but the game played on it is not: a
    design enters another design's payoff only through the conduct it
    executes on each branch of the handshake and the rate at which its badge
    is believed.
    """

    functionals: np.ndarray
    """The ``T`` and ``Q`` rows, shape ``(16, n_designs)``."""

    rank: int
    """``matrix_rank(pi_P)``: the true dimension of the game."""

    span_residual: float
    """``max |pi_P - C.T @ Phi|`` from the least-squares fit."""

    free_dimensions: int
    """Rank of the functional block, minus the simplex constraint."""

    row_labels: tuple[str, ...]
    """Name of each row of ``functionals``, e.g. ``Q[out,CAS]``."""


def macro_factorisation(
    tables: RaceTables,
    params: IdentityParams,
    liability: float = 0.0,
) -> MacroFactorisation:
    r"""Factor the private functional through sixteen macroscopic observables.

    For each branch ``y`` in ``{in, out}`` and each conduct ``c`` define the
    linear functionals

    .. math:: T_{y,c}(x) = \sum_j x_j\,\mathbb{1}[s^y_j = c], \qquad
        Q_{y,c}(x) = \sum_j x_j\,q_j\,\mathbb{1}[s^y_j = c] ,

    with ``q = 1, sigma, 0`` for a genuine, forged and absent badge.
    Expanding the four-term handshake and collecting the terms that depend on
    the partner shows every row of ``pi_P`` to be a linear combination of
    these sixteen: the ``q_j`` and ``1 - q_j`` weights multiply race payoffs
    that depend on the partner only through ``s^{in}_j`` or ``s^{out}_j``,
    and the focal design's own dues and fines ride on the constant row, which
    is ``sum_c T_{in,c}``.

    Three linear relations tie the sixteen together: ``sum_c T_{y,c} = 1`` on
    each branch (two rows, one relation, since both equal the all-ones row)
    and ``sum_c Q_{in,c} = sum_c Q_{out,c} = q`` (one more).  Fourteen
    independent rows remain, and one of those is the constant that the
    simplex pins, leaving thirteen free directions -- exactly the rank of
    ``pi_P``.

    This is what makes a parameter-resolved statement about the 48-design
    game tractable rather than a 47-dimensional fishing expedition, and it is
    why the conduct marginals of :func:`face_reduction` are the right
    coordinates to report a trajectory in: the game cannot distinguish two
    states that agree on all sixteen functionals.

    Parameters
    ----------
    tables:
        Race tables supplying ``A`` and ``M``.
    params:
        Identity-layer parameters; ``sigma`` and ``protection`` set ``q``.
    liability:
        Private liability ``L`` per Unsafe action.  It changes the entries of
        ``pi_P`` and not its rank: liability is charged on a race quantity
        that factors through the same sixteen rows.
    """
    from .functionals import build_functionals

    if liability < 0.0:
        raise ValueError(f"liability must be non-negative, got {liability}")

    fun = build_functionals(tables, params, liability)
    phi, labels = _macro_rows(fun)

    # pi_P = C.T @ Phi, i.e. Phi.T @ C = pi_P.T; lstsq rather than a solve
    # because Phi is rank deficient by construction (the three relations above)
    coefficients, *_ = np.linalg.lstsq(phi.T, fun.pi_P.T, rcond=None)
    residual = float(np.abs(fun.pi_P - coefficients.T @ phi).max())

    return MacroFactorisation(
        functionals=phi,
        rank=int(np.linalg.matrix_rank(fun.pi_P)),
        span_residual=residual,
        free_dimensions=int(np.linalg.matrix_rank(phi)) - 1,
        row_labels=labels,
    )


# --------------------------------------------------------------------------
# Theorem 3: the invader-exchange locus
# --------------------------------------------------------------------------

#: Monomials of an invasion advantage as ``(power of sigma, power of r)``.
#: The advantage is a polynomial of degree at most two in ``sigma`` (two
#: independent badge checks) and affine in ``r`` (one assorted self-encounter
#: against one uniform draw), so this basis is complete, not a truncation.
_ADVANTAGE_MONOMIALS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
)

#: Human-readable names of :data:`_ADVANTAGE_MONOMIALS`, in the same order.
ADVANTAGE_BASIS: tuple[str, ...] = (
    "1",
    "r",
    "sigma",
    "r*sigma",
    "sigma^2",
    "r*sigma^2",
)

#: ``(sigma, r)`` design points used to identify the six coefficients.  Two
#: assortments times three spoof successes makes the design matrix a Kronecker
#: product of two Vandermonde blocks, which is invertible by construction.
_ADVANTAGE_DESIGN: tuple[tuple[float, float], ...] = tuple(
    (sigma, r) for r in (0.0, 0.5) for sigma in (0.0, 0.5, 1.0)
)


def advantage_coefficients(
    tables: RaceTables,
    params: IdentityParams,
    liability: float,
    mutant: tuple[str, str, str],
    resident: tuple[str, str, str],
    tol: float = 1e-10,
    grid: tuple[int, int] = (41, 21),
    r_hi: float = 0.5,
) -> tuple[float, ...]:
    r"""Exact coefficients of an invasion advantage in ``(sigma, r)``.

    The advantage of a rare ``mutant`` in a monomorphic ``resident``,

    .. math:: a(\sigma, r) = r\,\pi_P(\text{mutant}, \text{mutant})
        + (1 - r)\,\pi_P(\text{mutant}, \text{resident})
        - \pi_P(\text{resident}, \text{resident}),

    is a polynomial in ``sigma`` of degree at most two -- the self-encounter
    mixes two independent checks and nothing mixes three -- and affine in
    ``r``.  Six coefficients therefore determine it everywhere, and they are
    recovered by evaluating the advantage at six design points and solving,
    which is exact arithmetic on the payoff algebra rather than a regression.

    The fit is then *checked* on a dense grid and a failure raises, so the
    returned coefficients are proved and not guessed.  That matters here
    because the natural guess is wrong: the quadratic term of a conditional
    forger sits on ``r * sigma^2`` and not on ``sigma^2``.  Only the assorted
    self-encounter puts two forged badges in one interaction, so a
    well-mixed population never sees the quadratic at all, and a basis that
    offers ``sigma^2`` without ``r * sigma^2`` misses it entirely while
    fitting the well-mixed cross-section perfectly.

    Returns the coefficients in the order of :data:`ADVANTAGE_BASIS`.
    """
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")
    if not 0.0 < r_hi <= 1.0:
        raise ValueError(f"r_hi must lie in (0, 1], got {r_hi}")
    n_sigma, n_r = grid
    if n_sigma < 2 or n_r < 2:
        raise ValueError(f"the verification grid must be at least 2x2, got {grid}")

    def advantage(sigma: float, r: float) -> float:
        p = replace(params, sigma=sigma, r=r)
        return fitness_against_resident(
            tables, p, liability, mutant, resident
        ) - pair_payoff(tables, p, liability, resident, resident)

    design = np.array(
        [
            [sigma**a * r**b for a, b in _ADVANTAGE_MONOMIALS]
            for sigma, r in _ADVANTAGE_DESIGN
        ]
    )
    observed = np.array([advantage(sigma, r) for sigma, r in _ADVANTAGE_DESIGN])
    coefficients = np.linalg.solve(design, observed)

    worst = 0.0
    for sigma in np.linspace(0.0, 1.0, n_sigma):
        for r in np.linspace(0.0, r_hi, n_r):
            predicted = float(
                sum(
                    c * sigma**a * r**b
                    for c, (a, b) in zip(coefficients, _ADVANTAGE_MONOMIALS)
                )
            )
            worst = max(worst, abs(predicted - advantage(float(sigma), float(r))))
    if worst > tol:
        raise ValueError(
            f"the advantage of {mutant} in {resident} is not spanned by "
            f"{ADVANTAGE_BASIS}: worst residual {worst}"
        )
    return tuple(float(c) for c in coefficients)


def _advantage_in_sigma(
    coefficients: tuple[float, ...], r: float
) -> tuple[float, float, float]:
    """The advantage at one assortment, as ``(a0, a1, a2)`` in ``sigma``."""
    return tuple(
        float(coefficients[2 * k] + coefficients[2 * k + 1] * r) for k in range(3)
    )


def _smallest_upcrossing(
    quadratic: tuple[float, float, float],
    lo: float = 0.0,
    hi: float = 1.0,
    tiny: float = 1e-12,
) -> float | None:
    """Smallest ``sigma`` in ``[lo, hi]`` above which the quadratic is positive.

    The closed-form counterpart of :func:`_threshold_by_root`, and it returns
    the same number for the same advantage: ``lo`` when the advantage is
    already positive there, ``None`` when it is positive nowhere on the
    interval, and otherwise the first up-crossing.  Solving the quadratic
    instead of scanning a grid removes the only place where a *threshold*
    was previously found by search, which is what the exchange locus needs:
    a locus assembled from bisected roots is a plot, not a condition.
    """
    a0, a1, a2 = quadratic

    def value(s: float) -> float:
        return a0 + a1 * s + a2 * s * s

    roots: list[float] = []
    if abs(a2) > tiny:
        discriminant = a1 * a1 - 4.0 * a2 * a0
        if discriminant >= 0.0:
            root = float(np.sqrt(discriminant))
            roots = [(-a1 - root) / (2.0 * a2), (-a1 + root) / (2.0 * a2)]
    elif abs(a1) > tiny:
        roots = [-a0 / a1]

    breaks = sorted({lo, hi} | {r for r in roots if lo < r < hi})
    for left, right in zip(breaks, breaks[1:]):
        if value(0.5 * (left + right)) > 0.0:
            return float(left)
    return None


def _polynomial_determinant(rows: list[list[np.ndarray]]) -> np.ndarray:
    """Determinant of a matrix whose entries are polynomials in ``r``.

    Entries and result are coefficient arrays in ascending powers.  A
    cofactor expansion is used rather than a numerical determinant because
    the entries are polynomials and not numbers: expanding keeps the answer
    exact in the coefficients, which is the whole point of returning a locus
    rather than a number found by bisection.
    """
    n = len(rows)
    if n == 1:
        return np.asarray(rows[0][0], dtype=float)
    total = np.zeros(1)
    for j in range(n):
        minor = [[row[k] for k in range(n) if k != j] for row in rows[1:]]
        term = np.polynomial.polynomial.polymul(
            rows[0][j], _polynomial_determinant(minor)
        )
        total = np.polynomial.polynomial.polyadd(
            total, term if j % 2 == 0 else -term
        )
    return total


def _sigma_resultant(
    first: tuple[float, ...], second: tuple[float, ...]
) -> np.ndarray:
    """Resultant in ``sigma`` of two advantages, as a polynomial in ``r``.

    Two quadratics share a root exactly where the determinant of their
    Sylvester matrix vanishes.  Every entry of that matrix is affine in
    ``r``, so the determinant is a quartic in ``r`` whose real roots are the
    complete candidate set for the exchange locus -- no interval, no scan.
    An advantage that is only affine in ``sigma`` (an unconditional forger)
    makes the formal degree-two Sylvester matrix singular for reasons that
    have nothing to do with a shared root, so the candidates are filtered
    afterwards by checking that the two thresholds really do coincide.
    """
    zero = np.zeros(1)
    p = [np.array([first[2 * k], first[2 * k + 1]]) for k in range(3)]
    q = [np.array([second[2 * k], second[2 * k + 1]]) for k in range(3)]
    sylvester = [
        [p[2], p[1], p[0], zero],
        [zero, p[2], p[1], p[0]],
        [q[2], q[1], q[0], zero],
        [zero, q[2], q[1], q[0]],
    ]
    return _polynomial_determinant(sylvester)


@dataclass(frozen=True)
class ExchangeLocus:
    """Where the mimic overtakes the exploiter, as a condition not a number.

    ``exploiter_coeffs`` and ``mimic_coeffs`` are six-tuples in the order of
    :data:`ADVANTAGE_BASIS`.  The brief for this object asked for five
    coefficients, ``(1, r, sigma, r*sigma, sigma^2)``; the sixth is not
    optional.  A conditional forger meets two forged badges only in its
    assorted self-encounter, so its quadratic term is ``r * sigma^2`` and its
    pure ``sigma^2`` coefficient is exactly zero -- the mirror image of the
    five-term guess.  Fitting in the five-term basis reproduces every
    well-mixed cross-section and misrepresents the whole ``r > 0`` plane,
    which is the plane the exchange happens in.
    """

    r_star: float | None
    """Assortment at which the two thresholds coincide, ``None`` if never."""

    exploiter_coeffs: tuple[float, ...]
    """Exploiter advantage in the :data:`ADVANTAGE_BASIS` monomials."""

    mimic_coeffs: tuple[float, ...]
    """Mimic advantage in the same basis."""

    condition: str
    """Plain-text statement of the general condition."""

    mimic_first_below: bool
    """Whether the mimic threshold is the lower one *below* ``r_star``."""

    basis: tuple[str, ...] = ADVANTAGE_BASIS
    """Monomial names of the two coefficient tuples."""


def invader_exchange_locus(
    tables: RaceTables,
    params: IdentityParams,
    liability: float = 0.0,
    club: tuple[str, str, str] | None = None,
    exploiter: tuple[str, str, str] | None = None,
    mimic: tuple[str, str, str] | None = None,
    r_hi: float = 0.5,
    tol: float = 1e-9,
) -> ExchangeLocus:
    r"""The assortment at which the club stops failing visibly.

    :func:`invader_ordering_flip` answers this by bisecting an ordering
    predicate, which returns one number at one parameterisation and settles
    nothing in general.  Both invasion advantages are polynomials of degree
    at most two in ``sigma`` with coefficients affine in ``r``
    (:func:`advantage_coefficients`), so both thresholds are explicit
    functions of ``r`` and the exchange is an algebraic condition:

        the exploiter threshold and the mimic threshold coincide exactly at
        the assortments where the resultant in ``sigma`` of the two advantage
        polynomials vanishes and the shared root is the threshold selected by
        both, that is, the first up-crossing of each inside ``[0, 1]``.

    Below the crossing the club fails visibly, by being exploited; above it
    the club fails invisibly, by being impersonated.  The mechanism is the
    asymmetry the resultant makes explicit: assortment multiplies the
    exploiter's own aggressive conduct against itself and leaves the mimic
    playing the club's conduct, so the exploiter's coefficient on ``r`` is
    large and negative while the mimic's is small.

    Parameters
    ----------
    tables, params, liability:
        As elsewhere.  ``params.sigma`` and ``params.r`` are swept out by the
        fit and only the remaining fields (dues, forgery cost, fine,
        screening) enter the coefficients.
    club, exploiter, mimic:
        The three designs.  ``None`` selects the canonical trio of
        :mod:`gbtag.config`.
    r_hi:
        Upper end of the assortment range searched for the locus.
    tol:
        Slack for accepting a resultant root as a genuine coincidence of the
        two thresholds, and for the coefficient fit.
    """
    from .config import CLUB, FALSEBEARD, MIMIC

    if liability < 0.0:
        raise ValueError(f"liability must be non-negative, got {liability}")
    if not 0.0 < r_hi <= 1.0:
        raise ValueError(f"r_hi must lie in (0, 1], got {r_hi}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")

    club = CLUB if club is None else club
    exploiter = FALSEBEARD if exploiter is None else exploiter
    mimic = MIMIC if mimic is None else mimic

    exploiter_coeffs = advantage_coefficients(
        tables, params, liability, exploiter, club, r_hi=r_hi
    )
    mimic_coeffs = advantage_coefficients(
        tables, params, liability, mimic, club, r_hi=r_hi
    )

    def thresholds(r: float) -> tuple[float | None, float | None]:
        return (
            _smallest_upcrossing(_advantage_in_sigma(exploiter_coeffs, r)),
            _smallest_upcrossing(_advantage_in_sigma(mimic_coeffs, r)),
        )

    resultant = _sigma_resultant(exploiter_coeffs, mimic_coeffs)
    candidates: list[float] = []
    if np.abs(resultant).max() > 0.0:
        for root in np.roots(resultant[::-1]):
            if abs(root.imag) > tol:
                continue
            r = float(np.clip(root.real, 0.0, r_hi))
            if not -tol <= root.real <= r_hi + tol:
                continue
            exploiter_threshold, mimic_threshold = thresholds(r)
            if exploiter_threshold is None or mimic_threshold is None:
                continue
            if abs(exploiter_threshold - mimic_threshold) > tol:
                continue
            candidates.append(r)
    r_star = min(candidates) if candidates else None

    # "below" the locus means strictly below it; halving is enough because the
    # thresholds are continuous and cross only at the roots just enumerated
    probe = 0.5 * r_star if r_star else 0.0
    exploiter_threshold, mimic_threshold = thresholds(probe)
    mimic_first_below = mimic_threshold is not None and (
        exploiter_threshold is None or mimic_threshold < exploiter_threshold
    )

    first, second = (
        ("mimic", "exploiter") if mimic_first_below else ("exploiter", "mimic")
    )
    condition = (
        "the two invasion advantages are quadratics in sigma whose three "
        "coefficients are affine in r, so each threshold is the first "
        "up-crossing of an explicit quadratic; the exploiter and the mimic "
        "exchange places exactly at the assortments where the resultant in "
        "sigma of the two advantage polynomials vanishes and the shared root "
        "is the selected threshold of both. Below that locus the "
        f"{first} threshold is the lower one, above it the {second} "
        "threshold is, so the club fails visibly on one side of it and "
        "invisibly on the other."
    )

    return ExchangeLocus(
        r_star=r_star,
        exploiter_coeffs=exploiter_coeffs,
        mimic_coeffs=mimic_coeffs,
        condition=condition,
        mimic_first_below=bool(mimic_first_below),
    )

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
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .functionals import (
    build_functionals,
    honest_subspace,
    plain_subspace,
    unbadged_subspace,
)
from .identity import IdentityParams
from .race import RaceParams, RaceTables, build_race_tables
from .theory import (
    Equilibrium,
    equilibrium,
    spoof_threshold,
    spoof_threshold_closed_form,
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

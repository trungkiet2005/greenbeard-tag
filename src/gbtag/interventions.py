"""Six ways to govern an identity layer, and what each one buys.

The instruments act at different points of the handshake and are not
substitutes:

``verification``
    harden the attestation technology, lowering the spoof success
    ``sigma``.  Acts on the *screen* itself.

``fines``
    charge a detected forgery the penalty ``rho``.  Acts on the *incentive*
    to forge, but only through the detections the screen already makes.

``dues``
    change what a genuine badge costs its carrier, ``kappa_g``.  Acts on
    the *club's* side of the ledger -- and, by the dues gradient, always
    against it.

``forgery_cost``
    raise what a forged badge costs to produce, ``kappa_f``.  The mirror
    image of dues: it burdens only the forger.

``assortment``
    engineer the interaction structure, raising ``r``: provider-scoped
    platforms, federated defaults, procurement that routes within trusted
    pools.  The nucleation theorem says this is the one axis on which the
    other instruments cannot substitute.

``liability``
    reopen the attribution channel, charging ``L`` per Unsafe action.  The
    bridge to the sister studies; identity matters exactly where this
    channel is closed.

Each instrument is applied to the same baseline and scored on the same
numbers, so the comparison is like for like.  The screening-versus-audit
decomposition separates the two consequences of one detection -- the
behavioural response of the checking counterparty and the fine levied on
the forger -- by switching each off in turn.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .functionals import build_functionals
from .identity import IdentityParams
from .race import RaceTables
from .theory import Equilibrium, equilibrium


@dataclass(frozen=True)
class InterventionOutcome:
    """Long-run outcome of one instrument at one setting."""

    instrument: str
    setting: float
    unsafe_frequency: float
    club_share: float
    falsebeard_share: float
    mark_lift: float
    attestation_integrity: float
    social_payoff: float
    equilibrium: Equilibrium


def _score(
    tables: RaceTables,
    params: IdentityParams,
    instrument: str,
    setting: float,
    liability: float,
    social_harm: float,
    method: str,
    **kwargs,
) -> InterventionOutcome:
    fun = build_functionals(tables, params, liability, social_harm)
    eq = equilibrium(fun, tables, method=method, **kwargs)
    return InterventionOutcome(
        instrument=instrument,
        setting=float(setting),
        unsafe_frequency=eq.unsafe_frequency,
        club_share=eq.class_distribution["certified club"],
        falsebeard_share=eq.class_distribution["falsebeard"],
        mark_lift=eq.mark_lift,
        attestation_integrity=eq.attestation_integrity,
        social_payoff=eq.social_payoff,
        equilibrium=eq,
    )


def verification_sweep(
    tables: RaceTables,
    params: IdentityParams,
    sigmas: np.ndarray,
    liability: float = 0.0,
    social_harm: float = 20.0,
    method: str = "sml",
    **kwargs,
) -> list[InterventionOutcome]:
    """Sweep the spoof success probability."""
    return [
        _score(
            tables,
            replace(params, sigma=float(s)),
            "verification",
            float(s),
            liability,
            social_harm,
            method,
            **kwargs,
        )
        for s in sigmas
    ]


def fine_sweep(
    tables: RaceTables,
    params: IdentityParams,
    rhos: np.ndarray,
    liability: float = 0.0,
    social_harm: float = 20.0,
    method: str = "sml",
    **kwargs,
) -> list[InterventionOutcome]:
    """Sweep the fine charged on a detected forgery."""
    return [
        _score(
            tables,
            replace(params, rho=float(rho)),
            "fines",
            float(rho),
            liability,
            social_harm,
            method,
            **kwargs,
        )
        for rho in rhos
    ]


def dues_sweep(
    tables: RaceTables,
    params: IdentityParams,
    kappas: np.ndarray,
    liability: float = 0.0,
    social_harm: float = 20.0,
    method: str = "sml",
    **kwargs,
) -> list[InterventionOutcome]:
    """Sweep the cost of carrying a genuine badge."""
    return [
        _score(
            tables,
            replace(params, kappa_g=float(k)),
            "dues",
            float(k),
            liability,
            social_harm,
            method,
            **kwargs,
        )
        for k in kappas
    ]


def forgery_cost_sweep(
    tables: RaceTables,
    params: IdentityParams,
    kappas: np.ndarray,
    liability: float = 0.0,
    social_harm: float = 20.0,
    method: str = "sml",
    **kwargs,
) -> list[InterventionOutcome]:
    """Sweep the cost of producing a forged badge."""
    return [
        _score(
            tables,
            replace(params, kappa_f=float(k)),
            "forgery_cost",
            float(k),
            liability,
            social_harm,
            method,
            **kwargs,
        )
        for k in kappas
    ]


def assortment_sweep(
    tables: RaceTables,
    params: IdentityParams,
    rs: np.ndarray,
    liability: float = 0.0,
    social_harm: float = 20.0,
    method: str = "sml",
    **kwargs,
) -> list[InterventionOutcome]:
    """Sweep the within-cluster interaction probability."""
    return [
        _score(
            tables,
            replace(params, r=float(r)),
            "assortment",
            float(r),
            liability,
            social_harm,
            method,
            **kwargs,
        )
        for r in rs
    ]


def liability_sweep(
    tables: RaceTables,
    params: IdentityParams,
    liabilities: np.ndarray,
    social_harm: float = 20.0,
    method: str = "sml",
    **kwargs,
) -> list[InterventionOutcome]:
    """Sweep the private liability, reopening the attribution channel."""
    return [
        _score(
            tables,
            params,
            "liability",
            float(L),
            float(L),
            social_harm,
            method,
            **kwargs,
        )
        for L in liabilities
    ]


# --------------------------------------------------------------------------
# the two consequences of one detection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionDecomposition:
    """What screening and fining each contribute, alone and together.

    One detection has two consequences: the checker plays defensively
    (*screening*, the behavioural channel) and the forger is charged
    (*fines*, the incentive channel).  The 2x2 switches each channel off in
    turn at the same detection power.  ``neither`` is the placebo in which
    badges are checked but nothing follows from a failed check.
    """

    neither: Equilibrium
    screening_only: Equilibrium
    fines_only: Equilibrium
    both: Equilibrium
    interaction: float
    """``U(both) - U(screening) - U(fines) + U(neither)``."""

    screening_effect: float
    fines_effect: float
    total_effect: float


def detection_decomposition(
    tables: RaceTables,
    params: IdentityParams,
    rho: float,
    liability: float = 0.0,
    social_harm: float = 20.0,
    method: str = "sml",
    **kwargs,
) -> DetectionDecomposition:
    """Run the 2x2: screening on/off crossed with fines on/off.

    ``params`` supplies the detection power and the baseline; ``rho`` is
    the fine used in the cells that fine.
    """

    def cell(protection: bool, fine: float) -> Equilibrium:
        fun = build_functionals(
            tables,
            replace(params, protection=protection, rho=fine),
            liability,
            social_harm,
        )
        return equilibrium(fun, tables, method=method, **kwargs)

    neither = cell(False, 0.0)
    screening = cell(True, 0.0)
    fines = cell(False, rho)
    both = cell(True, rho)

    u0 = neither.unsafe_frequency
    us = screening.unsafe_frequency
    uf = fines.unsafe_frequency
    ub = both.unsafe_frequency
    return DetectionDecomposition(
        neither=neither,
        screening_only=screening,
        fines_only=fines,
        both=both,
        interaction=ub - us - uf + u0,
        screening_effect=us - u0,
        fines_effect=uf - u0,
        total_effect=ub - u0,
    )

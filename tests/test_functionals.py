"""The identity functionals: exactness, embeddings, and observables."""

from __future__ import annotations

import numpy as np
import pytest

from gbtag.functionals import (
    aggregate_unsafe_frequency,
    attestation_integrity,
    build_functionals,
    conditional_unsafe,
    honest_subspace,
    mark_lift,
    mean_social_payoff,
    plain_subspace,
    population_average,
    unbadged_subspace,
    unsafe_split,
)
from gbtag.identity import IdentityParams, design_space
from gbtag.race import STRATEGIES, RaceParams, build_race_tables

TABLES = build_race_tables(RaceParams())


def _fun(**kwargs):
    defaults = dict(sigma=0.5, kappa_g=2.0, kappa_f=0.0, rho=0.0, r=0.1)
    defaults.update(kwargs)
    return build_functionals(TABLES, IdentityParams(**defaults), 0.0, 20.0)


def test_shapes_and_indexing() -> None:
    fun = _fun()
    assert fun.n == 48
    assert fun.pi_P.shape == (48, 48)
    assert fun.index("G", "CS", "CAS") == design_space().index(("G", "CS", "CAS"))
    assert len(fun.badge_block("F")) == 16


def test_private_functional_accounting() -> None:
    """``pi_P = a - L m - dues - expected fines``, entry by entry."""
    params = IdentityParams(sigma=0.4, kappa_g=2.0, kappa_f=0.5, rho=10.0, r=0.0)
    fun = build_functionals(TABLES, params, 0.7, 20.0)
    i = fun.index("F", "CAS", "AU")
    j = fun.index("G", "CS", "CAS")
    expected = (
        fun.task[i, j]
        - 0.7 * fun.harm[i, j]
        - params.badge_cost("F")
        - (1.0 - 0.4) * 10.0
    )
    assert fun.pi_P[i, j] == pytest.approx(expected, abs=1e-12)


def test_social_functional_excludes_fines_includes_costs() -> None:
    params = IdentityParams(sigma=0.4, kappa_g=2.0, kappa_f=0.5, rho=10.0, r=0.0)
    fun = build_functionals(TABLES, params, 0.7, 20.0)
    i = fun.index("F", "CAS", "AU")
    j = fun.index("G", "CS", "CAS")
    expected = fun.task[i, j] - 20.0 * fun.harm[i, j] - params.badge_cost("F")
    assert fun.pi_S[i, j] == pytest.approx(expected, abs=1e-12)


def test_plain_subspace_embeds_the_raw_race() -> None:
    """The four unconditional unbadged designs replay the race exactly."""
    fun = _fun(r=0.0)
    sub = plain_subspace(fun)
    assert sub.n == 4
    order = [sub.designs.index(("N", s, s)) for s in STRATEGIES]
    grid = np.ix_(order, order)
    assert np.allclose(sub.task[grid], TABLES.payoff)
    assert np.allclose(sub.harm[grid], TABLES.unsafe_count)
    assert np.allclose(sub.unsafe_frequency[grid], TABLES.unsafe_frequency)


def test_liability_enters_like_the_sister_studies() -> None:
    """At depth zero of the sister models, ``pi_P = A - L M``; same here."""
    fun = build_functionals(TABLES, IdentityParams(sigma=0.0, kappa_g=0.0, r=0.0), 3.0, 20.0)
    sub = plain_subspace(fun)
    order = [sub.designs.index(("N", s, s)) for s in STRATEGIES]
    grid = np.ix_(order, order)
    assert np.allclose(sub.pi_P[grid], TABLES.payoff - 3.0 * TABLES.unsafe_count)


def test_subspace_masks() -> None:
    fun = _fun()
    assert unbadged_subspace(fun).n == 16
    assert honest_subspace(fun).n == 32
    assert plain_subspace(fun).n == 4
    assert all(b == "N" for b in unbadged_subspace(fun).badge)
    assert all(b != "F" for b in honest_subspace(fun).badge)


def test_subspace_preserves_entries() -> None:
    fun = _fun()
    sub = unbadged_subspace(fun)
    i_full = fun.index("N", "CS", "CAS")
    j_full = fun.index("N", "AU", "AU")
    i_sub = sub.designs.index(("N", "CS", "CAS"))
    j_sub = sub.designs.index(("N", "AU", "AU"))
    assert sub.pi_P[i_sub, j_sub] == fun.pi_P[i_full, j_full]
    assert sub.fitness.shape == (16, 16)


def test_fitness_is_the_assorted_private_functional() -> None:
    fun = _fun(r=0.3)
    expected = 0.3 * np.diag(fun.pi_P)[:, None] + 0.7 * fun.pi_P
    assert np.allclose(fun.fitness, expected)


def test_population_average_monomorphic_reads_the_diagonal() -> None:
    rng = np.random.default_rng(1)
    x = rng.dirichlet(np.ones(48))
    m = rng.normal(size=(48, 48))
    assert population_average(x, m, r=0.4, monomorphic=True) == pytest.approx(
        float(x @ np.diag(m))
    )


def test_population_average_mixed_uses_the_encounter_law() -> None:
    rng = np.random.default_rng(2)
    x = rng.dirichlet(np.ones(48))
    m = rng.normal(size=(48, 48))
    r = 0.25
    expected = float(x @ (r * np.diag(m) + (1 - r) * m @ x))
    assert population_average(x, m, r=r) == pytest.approx(expected)


def test_aggregate_unsafe_of_a_pure_club_is_zero() -> None:
    fun = _fun()
    x = np.zeros(48)
    x[fun.index("G", "CS", "CAS")] = 1.0
    assert aggregate_unsafe_frequency(x, fun, monomorphic=True) == pytest.approx(0.0)
    assert aggregate_unsafe_frequency(x, fun, monomorphic=False) == pytest.approx(0.0)


def test_mark_lift_of_a_pure_world() -> None:
    """A half-club half-anarchy mix: passed badges predict safe conduct."""
    fun = _fun(sigma=0.0)
    x = np.zeros(48)
    x[fun.index("G", "CS", "CAS")] = 0.5
    x[fun.index("N", "CAS", "CAS")] = 0.5
    lift = mark_lift(x, fun, monomorphic=False)
    # every passed badge is the club's and elicits CS (safe); the fail side
    # mixes club-out (CAS) and anarchy (CAS), so the lift is positive
    assert lift > 0.5


def test_attestation_integrity_bounds() -> None:
    fun = _fun(sigma=0.8)
    x = np.full(48, 1.0 / 48)
    val = attestation_integrity(x, fun, monomorphic=False)
    assert 0.0 <= val <= 1.0
    # a world with only genuine badges has full integrity
    y = np.zeros(48)
    y[fun.index("G", "CS", "CAS")] = 1.0
    assert attestation_integrity(y, fun, monomorphic=True) == pytest.approx(1.0)
    # a world with only forged badges has zero integrity (some pass at 0.8)
    z = np.zeros(48)
    z[fun.index("F", "CS", "CAS")] = 1.0
    assert attestation_integrity(z, fun, monomorphic=True) == pytest.approx(0.0)


def test_conditional_unsafe_mixture_recovers_the_marginal() -> None:
    """``u = q u_in + (1 - q) u_out`` row by row."""
    fun = _fun(sigma=0.37)
    u_in, u_out = conditional_unsafe(TABLES, fun)
    q = np.array([fun.params.behavioural_pass_rate(b) for b in fun.badge])
    mixed = q[None, :] * u_in + (1.0 - q)[None, :] * u_out
    assert np.allclose(mixed, fun.unsafe_frequency, atol=1e-12)


def test_unsafe_split_of_a_pure_club_world() -> None:
    fun = _fun(sigma=0.5)
    x = np.zeros(48)
    x[fun.index("G", "CS", "CAS")] = 1.0
    split = unsafe_split(x, fun, TABLES, monomorphic=True)
    # all encounters are club-club: every check passes, verified play is CS
    # against CS, which is entirely safe; the unverified side is empty
    assert split["verified"] == pytest.approx(0.0)
    assert split["unverified"] == pytest.approx(0.0)


def test_unsafe_split_on_a_mixed_population_matches_a_hand_computation() -> None:
    """The only non-degenerate check of :func:`unsafe_split`.

    Half a harsh certified club, half an unbadged design that is safe to
    everyone, well mixed, with forgery impossible so that every badge check
    is deterministic.  The four ordered pairs are then enumerable by hand:

    ==========================  =========  ==================  ======
    focal, partner              check      executed pair       unsafe
    ==========================  =========  ==================  ======
    (G,CS,CAS), (G,CS,CAS)      passes     (CS, CS)            0
    (G,CS,CAS), (N,AS,AS)       fails      (CAS, AS)           u
    (N,AS,AS),  (G,CS,CAS)      passes     (AS, CAS)           0
    (N,AS,AS),  (N,AS,AS)       fails      (AS, AS)            0
    ==========================  =========  ==================  ======

    Each pair carries weight 1/4, so the verified side is exactly zero and
    the unverified side is ``u / 2`` with ``u = U(CAS, AS)``.
    """
    fun = build_functionals(
        TABLES, IdentityParams(sigma=0.0, kappa_g=2.0, r=0.0), 0.0, 20.0
    )
    x = np.zeros(fun.n)
    x[fun.index("G", "CS", "CAS")] = 0.5
    x[fun.index("N", "AS", "AS")] = 0.5

    idx = {s: i for i, s in enumerate(STRATEGIES)}
    expected = TABLES.unsafe_frequency[idx["CAS"], idx["AS"]] / 2.0

    split = unsafe_split(x, fun, TABLES, monomorphic=False)
    assert split["verified"] == pytest.approx(0.0, abs=1e-12)
    assert split["unverified"] == pytest.approx(expected, abs=1e-12)
    assert split["unverified"] > 0.06  # the hand value, so a zero cannot pass


def test_unsafe_split_is_sensitive_to_the_screening_switch() -> None:
    """Turning screening off must change the split, not just the fines."""
    mixed = np.zeros(48)
    with_screen = build_functionals(
        TABLES, IdentityParams(sigma=0.8, kappa_g=2.0, r=0.0, protection=True),
        0.0, 20.0,
    )
    without = build_functionals(
        TABLES, IdentityParams(sigma=0.8, kappa_g=2.0, r=0.0, protection=False),
        0.0, 20.0,
    )
    mixed[with_screen.index("F", "CAS", "CAS")] = 0.5
    mixed[with_screen.index("G", "CS", "CAS")] = 0.5
    a = unsafe_split(mixed, with_screen, TABLES, monomorphic=False)
    b = unsafe_split(mixed, without, TABLES, monomorphic=False)
    assert a != b


def test_mean_social_payoff_of_a_pure_club() -> None:
    fun = _fun()
    x = np.zeros(48)
    x[fun.index("G", "CS", "CAS")] = 1.0
    # CS against CS is all-safe: task 59, no harm, dues 2
    assert mean_social_payoff(x, fun, monomorphic=True) == pytest.approx(57.0)

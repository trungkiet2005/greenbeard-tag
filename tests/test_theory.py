"""The propositions: closed forms against the exact matrices and dynamics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gbtag import config as cfg
from gbtag import theory as th
from gbtag.functionals import build_functionals, unbadged_subspace
from gbtag.identity import IdentityParams
from gbtag.race import RaceParams, build_race_tables
from gbtag.theory import (
    badge_is_futile_against_nonconditioners,
    dues_gradient,
    equilibrium,
    fitness_against_resident,
    first_strike_threshold,
    invades,
    mimic_threshold_closed_form,
    nucleation_threshold,
    pair_payoff,
    provider_frontier,
    reciprocity_threshold,
    reentry_threshold,
    required_fine,
    spoof_threshold,
    spoof_threshold_closed_form,
    uncertified_safety_is_impossible,
)

TABLES = build_race_tables(RaceParams())
WM = IdentityParams(sigma=0.5, kappa_g=2.0, kappa_f=0.0, rho=0.0, r=0.0)


# ------------------------------------------------------------------ pair payoffs


def test_pair_payoff_matches_functionals() -> None:
    params = IdentityParams(sigma=0.62, kappa_g=1.7, kappa_f=0.2, rho=4.0, r=0.0)
    fun = build_functionals(TABLES, params, 0.9, 20.0)
    rng = np.random.default_rng(5)
    for _ in range(40):
        i, j = rng.integers(0, 48, size=2)
        got = pair_payoff(TABLES, params, 0.9, fun.designs[i], fun.designs[j])
        assert got == pytest.approx(fun.pi_P[i, j], abs=1e-10)


def test_fitness_against_resident_is_the_assorted_row() -> None:
    params = replace(WM, r=0.2)
    club, forger = cfg.CLUB, cfg.FALSEBEARD
    own = pair_payoff(TABLES, params, 0.0, forger, forger)
    cross = pair_payoff(TABLES, params, 0.0, forger, club)
    expected = 0.2 * own + 0.8 * cross
    got = fitness_against_resident(TABLES, params, 0.0, forger, club)
    assert got == pytest.approx(expected, abs=1e-12)


# ------------------------------------------------- Proposition 1: reciprocity


def test_reciprocity_threshold_value() -> None:
    """The number quoted throughout the manuscript."""
    assert reciprocity_threshold(TABLES) == pytest.approx(0.5506, abs=2e-4)


def test_reciprocity_threshold_is_the_invasion_boundary() -> None:
    lr = reciprocity_threshold(TABLES)
    striker, resident = ("N", "CAS", "CAS"), ("N", "CS", "CS")
    assert invades(TABLES, WM, lr - 0.01, striker, resident)
    assert not invades(TABLES, WM, lr + 0.01, striker, resident)


def test_first_strike_thresholds() -> None:
    assert first_strike_threshold(TABLES, "CS") == pytest.approx(0.5506, abs=2e-4)
    # the benchmark shared with the sister studies
    assert first_strike_threshold(TABLES, "AS") == pytest.approx(42.765, abs=1e-2)


def test_uncertified_safety_is_impossible_below_lr() -> None:
    lr = reciprocity_threshold(TABLES)
    assert uncertified_safety_is_impossible(TABLES, 0.0)
    assert uncertified_safety_is_impossible(TABLES, lr - 0.01)
    assert not uncertified_safety_is_impossible(TABLES, lr + 0.01)


# ---------------------------------------------- Proposition 2: spoof threshold


def test_spoof_threshold_closed_form_values() -> None:
    got_cs = spoof_threshold_closed_form(TABLES, 0.0, "CS", "CAS", "CAS", 2.0, 0.0, 0.0)
    got_as = spoof_threshold_closed_form(TABLES, 0.0, "AS", "CAS", "CAS", 2.0, 0.0, 0.0)
    assert got_cs == pytest.approx(0.8655, abs=2e-4)
    assert got_as == pytest.approx(0.3996, abs=2e-4)
    # the reciprocating club tolerates over twice the spoof success
    assert got_cs / got_as > 2.0


def test_spoof_threshold_numeric_agrees_with_closed_form() -> None:
    for kg in (0.5, 2.0, 4.0):
        for rho in (0.0, 5.0):
            params = replace(WM, kappa_g=kg, rho=rho)
            numeric = spoof_threshold(TABLES, params, 0.0, cfg.CLUB, cfg.FALSEBEARD)
            closed = spoof_threshold_closed_form(
                TABLES, 0.0, "CS", "CAS", "CAS", kg, 0.0, rho
            )
            assert numeric == pytest.approx(closed, abs=1e-6)


def test_spoof_threshold_is_the_invasion_boundary() -> None:
    s = spoof_threshold_closed_form(TABLES, 0.0, "CS", "CAS", "CAS", 2.0, 0.0, 0.0)
    below = replace(WM, sigma=s - 0.01)
    above = replace(WM, sigma=s + 0.01)
    assert not invades(TABLES, below, 0.0, cfg.FALSEBEARD, cfg.CLUB)
    assert invades(TABLES, above, 0.0, cfg.FALSEBEARD, cfg.CLUB)


def test_assortment_raises_the_spoof_threshold() -> None:
    """Self-play hurts the exploitative forger, so clustering protects."""
    wm = spoof_threshold(TABLES, WM, 0.0, cfg.CLUB, cfg.FALSEBEARD)
    clustered = spoof_threshold(
        TABLES, replace(WM, r=0.1), 0.0, cfg.CLUB, cfg.FALSEBEARD
    )
    assert clustered > wm


# --------------------------------------- Proposition 3: fines and the Becker limit


def test_required_fine_holds_the_line() -> None:
    sigma = 0.95
    rho = required_fine(TABLES, 0.0, sigma, "CS", "CAS", "CAS", 2.0, 0.0)
    params = replace(WM, sigma=sigma)
    assert invades(TABLES, replace(params, rho=rho * 0.99), 0.0, cfg.FALSEBEARD, cfg.CLUB)
    assert not invades(TABLES, replace(params, rho=rho * 1.01), 0.0, cfg.FALSEBEARD, cfg.CLUB)


def test_required_fine_diverges_at_the_spoof_proof_limit() -> None:
    r95 = required_fine(TABLES, 0.0, 0.95, "CS", "CAS", "CAS", 2.0, 0.0)
    r99 = required_fine(TABLES, 0.0, 0.99, "CS", "CAS", "CAS", 2.0, 0.0)
    assert r99 > 4.0 * r95
    assert required_fine(TABLES, 0.0, 1.0, "CS", "CAS", "CAS", 2.0, 0.0) == float("inf")


def test_no_fine_needed_below_the_free_threshold() -> None:
    s0 = spoof_threshold_closed_form(TABLES, 0.0, "CS", "CAS", "CAS", 2.0, 0.0, 0.0)
    assert required_fine(TABLES, 0.0, s0 - 0.05, "CS", "CAS", "CAS", 2.0, 0.0) < 0.0


def test_dues_gradient_is_negative() -> None:
    g = dues_gradient(TABLES, 0.0, "CS", "CAS", "CAS", 0.0)
    assert g < 0.0
    # finite-difference check on the closed form
    s1 = spoof_threshold_closed_form(TABLES, 0.0, "CS", "CAS", "CAS", 2.0, 0.0, 0.0)
    s2 = spoof_threshold_closed_form(TABLES, 0.0, "CS", "CAS", "CAS", 3.0, 0.0, 0.0)
    assert s2 - s1 == pytest.approx(g, rel=1e-9)


# --------------------------------- Propositions 4 and 5: nucleation and futility


def test_nucleation_threshold_value_and_boundary() -> None:
    r_star, direction = nucleation_threshold(TABLES, 0.0, "CS", "CAS", "CAS", 2.0)
    assert r_star == pytest.approx(0.0629, abs=2e-4)
    assert direction == 1
    below = replace(WM, r=r_star - 0.005)
    above = replace(WM, r=r_star + 0.005)
    assert not invades(TABLES, below, 0.0, cfg.CLUB, cfg.ANARCHY)
    assert invades(TABLES, above, 0.0, cfg.CLUB, cfg.ANARCHY)


def test_badge_futility() -> None:
    assert badge_is_futile_against_nonconditioners(TABLES, WM, 0.0)
    assert badge_is_futile_against_nonconditioners(
        TABLES, replace(WM, sigma=0.9, rho=7.0), 1.5
    )


def test_no_reentry_at_zero_assortment() -> None:
    """The hysteresis result: collapsed trust does not rebuild, at any sigma."""
    assert reentry_threshold(TABLES, WM, 0.0, cfg.CLUB, cfg.FALSEBEARD) is None
    assert reentry_threshold(TABLES, WM, 0.0, cfg.CLUB, cfg.ANARCHY) is None


def test_reentry_possible_with_assortment() -> None:
    clustered = replace(WM, r=0.1)
    got = reentry_threshold(TABLES, clustered, 0.0, cfg.CLUB, cfg.ANARCHY)
    assert got is not None


# ------------------------------------------------- Proposition 6: mimicry


def test_mimic_threshold_value() -> None:
    got = mimic_threshold_closed_form(TABLES, 0.0, "CS", "CAS", 2.0, 0.0, 0.0)
    assert got == pytest.approx(0.9382, abs=2e-4)


def test_mimic_threshold_is_the_invasion_boundary() -> None:
    s = mimic_threshold_closed_form(TABLES, 0.0, "CS", "CAS", 2.0, 0.0, 0.0)
    assert not invades(TABLES, replace(WM, sigma=s - 0.01), 0.0, cfg.MIMIC, cfg.CLUB)
    assert invades(TABLES, replace(WM, sigma=s + 0.01), 0.0, cfg.MIMIC, cfg.CLUB)


def test_with_assortment_the_mimic_strikes_before_the_exploiter() -> None:
    """The graceful-collapse ordering at the baseline interaction structure."""
    clustered = replace(WM, r=0.1)
    s_mimic = spoof_threshold(TABLES, clustered, 0.0, cfg.CLUB, cfg.MIMIC)
    s_forger = spoof_threshold(TABLES, clustered, 0.0, cfg.CLUB, cfg.FALSEBEARD)
    assert s_mimic < s_forger


def test_assortment_reverses_which_invader_arrives_first() -> None:
    """Clustering penalises the exploiter and leaves the mimic untouched.

    An exploiter meets its own aggressive conduct in a self-encounter; a
    mimic meets the club's conduct, which is what it plays. So assortment
    raises the exploiter's threshold steeply and the mimic's barely at all,
    and the two cross.
    """
    flip = th.invader_ordering_flip(
        TABLES, cfg.IDENTITY, 0.0, cfg.CLUB, cfg.FALSEBEARD, cfg.MIMIC
    )
    assert flip is not None
    assert flip == pytest.approx(0.077, abs=2e-3)

    below = replace(cfg.IDENTITY, r=flip - 0.02)
    above = replace(cfg.IDENTITY, r=flip + 0.02)
    for params, first in ((below, "exploiter"), (above, "mimic")):
        s_e = th.spoof_threshold(TABLES, params, 0.0, cfg.CLUB, cfg.FALSEBEARD)
        s_m = th.spoof_threshold(TABLES, params, 0.0, cfg.CLUB, cfg.MIMIC)
        if first == "exploiter":
            assert s_e < s_m
        else:
            assert s_m < s_e


def test_the_mimic_threshold_is_nearly_flat_in_assortment() -> None:
    """The quantitative asymmetry behind the reversal."""
    thresholds = {}
    for r in (0.0, 0.1, 0.2):
        params = replace(cfg.IDENTITY, r=r)
        thresholds[r] = (
            th.spoof_threshold(TABLES, params, 0.0, cfg.CLUB, cfg.FALSEBEARD),
            th.spoof_threshold(TABLES, params, 0.0, cfg.CLUB, cfg.MIMIC),
        )
    # the mimic threshold moves by less than one part in a hundred
    mimics = [m for _, m in thresholds.values()]
    assert max(mimics) - min(mimics) < 0.01
    # the exploiter is driven out of the unit interval entirely
    assert thresholds[0.0][0] == pytest.approx(0.866, abs=2e-3)
    assert thresholds[0.2][0] is None


def test_exploiter_immunity_threshold() -> None:
    """Above a modest assortment no forgery quality lets an exploiter in."""
    r_immune = th.exploiter_immunity(
        TABLES, cfg.IDENTITY, 0.0, cfg.CLUB, cfg.FALSEBEARD
    )
    assert r_immune is not None
    assert 0.1 < r_immune < 0.2
    just_above = replace(cfg.IDENTITY, r=r_immune + 0.01)
    assert th.spoof_threshold(TABLES, just_above, 0.0, cfg.CLUB, cfg.FALSEBEARD) is None
    # but the mimic still gets in
    assert th.spoof_threshold(TABLES, just_above, 0.0, cfg.CLUB, cfg.MIMIC) is not None


def test_mimicry_leaves_conduct_intact() -> None:
    """A mimic-ruled world keeps near-club conduct.

    The only residual unsafety is failed-check friction among the mimics
    themselves, of order ``2 sigma (1 - sigma)``, which vanishes exactly as
    the forgeries get good.
    """
    fun = build_functionals(TABLES, replace(WM, sigma=0.97, r=0.1), 0.0, 20.0)
    i = fun.index(*cfg.MIMIC)
    assert fun.unsafe_frequency[i, i] < 0.05
    perfect = build_functionals(TABLES, replace(WM, sigma=1.0, r=0.1), 0.0, 20.0)
    assert perfect.unsafe_frequency[i, i] == pytest.approx(0.0)


# ------------------------------------------ Proposition 7: providers


def test_provider_frontier_is_linear_in_hhi() -> None:
    """The baseline club, where the mixture collapses to ``1 - HHI``."""
    for k in (1, 2, 4, 8):
        f = provider_frontier(TABLES, 0.0, "CS", "CAS", 2.0, k)
        assert f.hhi == pytest.approx(1.0 / k)
        assert f.unsafe_frequency == pytest.approx(1.0 - f.hhi)
        assert f.federated_unsafe_frequency == pytest.approx(0.0)


@pytest.mark.parametrize(
    "club_in,club_out", [("CS", "AU"), ("AS", "CAS"), ("CS", "CS"), ("AS", "CS")]
)
def test_provider_frontier_general_mixture(club_in: str, club_out: str) -> None:
    """The general mixture, where ``1 - HHI`` is *not* the answer.

    At the baseline pair ``u[v, v] = 1`` exactly, so the formula degenerates and
    a wrong implementation would still pass. These pairs exercise it.
    """
    idx = {s: i for i, s in enumerate(("AS", "CS", "CAS", "AU"))}
    uf = TABLES.unsafe_frequency
    for k in (1, 2, 5):
        f = provider_frontier(TABLES, 0.0, club_in, club_out, 2.0, k)
        hhi = 1.0 / k
        expected = hhi * uf[idx[club_in], idx[club_in]] + (1.0 - hhi) * uf[
            idx[club_out], idx[club_out]
        ]
        assert f.unsafe_frequency == pytest.approx(expected, abs=1e-12)
        assert f.federated_unsafe_frequency == pytest.approx(
            uf[idx[club_in], idx[club_in]], abs=1e-12
        )


def test_exclusion_rent_falls_with_fragmentation() -> None:
    rents = [
        provider_frontier(TABLES, 0.0, "CS", "CAS", 2.0, k).exclusion_rent
        for k in (1, 2, 4, 8)
    ]
    assert all(a > b for a, b in zip(rents, rents[1:]))
    assert rents[0] == pytest.approx(59.0 - 2.0 - 26.644, abs=1e-2)


# ------------------------------------------ Proposition 8: exclusion dividend


def test_harm_decomposition_is_an_accounting_identity() -> None:
    """The four blocks reproduce the reported unsafe frequency exactly."""
    from gbtag.functionals import aggregate_unsafe_frequency

    fun = build_functionals(TABLES, cfg.IDENTITY, cfg.LIABILITY, cfg.SOCIAL_HARM)
    rng = np.random.default_rng(11)
    for mono in (False, True):
        x = rng.dirichlet(np.ones(48))
        dec = th.harm_decomposition(x, fun, monomorphic=mono)
        assert sum(dec.mass.values()) == pytest.approx(1.0, abs=1e-12)
        assert dec.total == pytest.approx(
            aggregate_unsafe_frequency(x, fun, monomorphic=mono), abs=1e-12
        )
        assert 0.0 <= dec.interface_share <= 1.0


def test_monomorphic_decomposition_has_no_interface() -> None:
    """A monomorphic population never meets a differently badged partner."""
    fun = build_functionals(TABLES, cfg.IDENTITY, cfg.LIABILITY, cfg.SOCIAL_HARM)
    rng = np.random.default_rng(12)
    x = rng.dirichlet(np.ones(48))
    dec = th.harm_decomposition(x, fun, monomorphic=True)
    assert dec.mass["badged with unbadged"] == pytest.approx(0.0)
    assert dec.interface_share == pytest.approx(0.0)


def test_the_flow_is_bistable_with_no_badge_mixed_attractor() -> None:
    """The correct object: two disjoint regimes, not one mixed population.

    Averaging the attractors and scoring a bilinear observable at the mean
    manufactures encounters between designs that never meet, so the state
    average and the observable average must be kept apart.
    """
    from gbtag.dynamics import replicator_attractors
    from gbtag.functionals import aggregate_unsafe_frequency

    fun = build_functionals(TABLES, cfg.IDENTITY, cfg.LIABILITY, cfg.SOCIAL_HARM)
    ends = replicator_attractors(fun.fitness, n_starts=120, seed=cfg.SEED)
    carries = np.array([b != "N" for b in fun.badge], dtype=float)
    badged_mass = ends @ carries

    # every attractor is a pure regime
    assert np.sum((badged_mass > 1e-6) & (badged_mass < 1 - 1e-6)) == 0
    assert 0.2 < float(np.mean(badged_mass > 0.5)) < 0.8

    # and every one of them is internally safe
    per_attractor = [aggregate_unsafe_frequency(e, fun) for e in ends]
    assert max(per_attractor) < 1e-15

    # the mean state, by contrast, looks catastrophic; that is the artefact
    assert aggregate_unsafe_frequency(ends.mean(axis=0), fun) > 0.3


def test_the_equilibrium_averages_observables_not_states() -> None:
    """The regression guard on the bug this replaced."""
    fun = build_functionals(TABLES, cfg.IDENTITY, cfg.LIABILITY, cfg.SOCIAL_HARM)
    eq = th.equilibrium(fun, TABLES, "replicator", n_starts=120, seed=cfg.SEED)
    assert eq.unsafe_frequency < 1e-15
    assert eq.harm_blocks.total < 1e-15
    assert eq.n_attractors > 1
    assert 0.2 < eq.certified_basin_share < 0.8


def test_certification_costs_its_dues_in_a_settled_market() -> None:
    """Both regimes are safe, so the club's whole equilibrium effect is cost."""
    from gbtag.dynamics import replicator_attractors
    from gbtag.functionals import mean_social_payoff

    fun = build_functionals(TABLES, cfg.IDENTITY, cfg.LIABILITY, cfg.SOCIAL_HARM)
    ends = replicator_attractors(fun.fitness, n_starts=120, seed=cfg.SEED)
    carries = np.array([b != "N" for b in fun.badge], dtype=float)
    certified = ends[ends @ carries > 0.5]
    uncertified = ends[ends @ carries < 0.5]

    social_c = np.mean([mean_social_payoff(e, fun) for e in certified])
    social_u = np.mean([mean_social_payoff(e, fun) for e in uncertified])
    assert social_u - social_c == pytest.approx(cfg.IDENTITY.kappa_g, abs=1e-6)


def test_the_exclusion_is_an_entry_barrier() -> None:
    """Proposition 8, measured where it does not need coexistence."""
    penalty, boundary = th.entry_barrier(TABLES, cfg.IDENTITY, 0.0, "CS", "CAS")
    assert penalty == pytest.approx(30.356, abs=2e-3)
    assert boundary == pytest.approx(0.461, abs=2e-3)

    # a gentle club imposes a *negative* penalty: the outsider is better off
    # than the member, by exactly the dues, which is why it cannot survive
    gentle_penalty, gentle_boundary = th.entry_barrier(
        TABLES, cfg.IDENTITY, 0.0, "CS", "CS"
    )
    assert gentle_penalty == pytest.approx(-cfg.IDENTITY.kappa_g, abs=1e-9)
    assert gentle_boundary == pytest.approx(0.0)


def test_out_group_harshness_orders_both_defence_and_exclusion() -> None:
    """The policies with a positive spoof threshold are exactly those with
    a positive entry penalty: the defence and the barrier are one policy."""
    scan = th.out_group_policy_scan(
        TABLES, cfg.IDENTITY, "CS", cfg.LIABILITY, cfg.SOCIAL_HARM,
        cfg.POPULATION, cfg.BETA, n_starts=40, seed=cfg.SEED,
    )
    by_out = {p.s_out: p for p in scan}
    soft = [by_out["AS"], by_out["CS"]]
    hard = [by_out["CAS"], by_out["AU"]]

    assert all(p.spoof_threshold == pytest.approx(0.0) for p in soft)
    assert all(p.spoof_threshold > 0.9 for p in hard)

    # the barrier orders the same way, and by a wide margin: a gentle club
    # pays its outsiders a bonus, a harsh one charges them a large toll
    assert max(p.entry_penalty for p in soft) <= 0.0
    assert min(p.entry_penalty for p in hard) > 25.0
    assert all(p.boundary_unsafe < 1e-9 for p in soft)
    assert all(p.boundary_unsafe > 0.4 for p in hard)


def test_a_gentle_club_has_a_degenerate_spoof_threshold() -> None:
    """With no fine, a club that treats outsiders as members is defenceless.

    The closed-form denominator $P(w,u) - P(w,v) + \\rho$ collapses to $\\rho$
    when the club's two conducts coincide, so at $\\rho = 0$ the invasion
    advantage does not depend on the spoof success at all, and it is positive
    because the forger saves the dues.
    """
    from gbtag.race import STRATEGIES

    p0 = th.race_private(TABLES, 0.0)
    idx = {s: i for i, s in enumerate(STRATEGIES)}
    u = v = idx["CS"]
    w = idx["CAS"]
    assert p0[w, u] - p0[w, v] == pytest.approx(0.0, abs=1e-12)

    wm = replace(WM, rho=0.0)
    gentle = ("G", "CS", "CS")
    assert th.spoof_threshold(TABLES, wm, 0.0, gentle, cfg.FALSEBEARD) == pytest.approx(0.0)
    assert th.spoof_threshold(TABLES, wm, 0.0, gentle, ("F", "CS", "CS")) == pytest.approx(0.0)


def test_a_fine_restores_the_barrier_but_not_the_purpose() -> None:
    """Corollary 9: fining forgery cannot make a gentle club worth joining.

    The fine raises the gentle club's spoof threshold from zero, so forgery
    stops defeating it, and the club still never gains share, because a
    badge earns nothing against residents that do not read badges and the
    unbadged cooperator supplies identical conduct without the dues.
    """
    rows = th.gentle_club_rescue(
        TABLES, cfg.IDENTITY, (0.0, 20.0, 60.0), "CS",
        cfg.LIABILITY, cfg.SOCIAL_HARM, cfg.POPULATION, cfg.BETA,
        n_starts=60, seed=cfg.SEED,
    )
    by_rho = {row.rho: row for row in rows}

    # the barrier is restored
    assert by_rho[0.0].spoof_threshold == pytest.approx(0.0)
    assert by_rho[60.0].spoof_threshold > 0.9

    # the club is still selectively dead at every fine
    assert all(row.club_share_mixed < 0.02 for row in rows)
    assert all(row.club_share_monomorphic < 0.05 for row in rows)

    # and with no harsh club there is no interface, so no harm in the mix
    assert all(row.unsafe_mixed < 1e-9 for row in rows)


def test_the_fine_still_buys_attestation_integrity() -> None:
    """What the fine does purchase, so the result is not read as futility."""
    rows = th.gentle_club_rescue(
        TABLES, cfg.IDENTITY, (0.0, 60.0), "CS",
        cfg.LIABILITY, cfg.SOCIAL_HARM, cfg.POPULATION, cfg.BETA,
        n_starts=40, seed=cfg.SEED,
    )
    assert rows[1].attestation_integrity > rows[0].attestation_integrity + 0.5


# ------------------------------------------------- equilibrium plumbing


def test_equilibrium_methods_run_and_normalise() -> None:
    fun = build_functionals(TABLES, cfg.IDENTITY, cfg.LIABILITY, cfg.SOCIAL_HARM)
    eq = equilibrium(fun, TABLES, "sml", population_size=50, beta=0.05)
    assert eq.frequencies.sum() == pytest.approx(1.0)
    assert sum(eq.class_distribution.values()) == pytest.approx(1.0)
    assert 0.0 <= eq.unsafe_frequency <= 1.0
    rep = equilibrium(fun, TABLES, "replicator", n_starts=8, seed=1)
    assert rep.frequencies.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        equilibrium(fun, TABLES, "nonsense")


def test_certification_beats_the_uncertified_world_at_baseline() -> None:
    """The headline comparison at the baseline parameterisation."""
    fun = build_functionals(TABLES, cfg.IDENTITY, cfg.LIABILITY, cfg.SOCIAL_HARM)
    cert = equilibrium(fun, TABLES, "sml", cfg.POPULATION, cfg.BETA)
    uncert = equilibrium(unbadged_subspace(fun), TABLES, "sml", cfg.POPULATION, cfg.BETA)
    assert cert.unsafe_frequency < uncert.unsafe_frequency
    assert cert.class_distribution["certified club"] > 0.3



def test_harm_decomposition_orientation_is_pinned() -> None:
    """Which interface block is which, on a deliberately asymmetric population.

    The two directed interface blocks carry near-identical mass at the
    baseline, so a transpose of the masks would survive every other test.
    Here the badged design is harsh out-group and the unbadged one is safe
    towards everyone, which separates the two conditionals by a wide margin.
    """
    fun = build_functionals(TABLES, cfg.IDENTITY, cfg.LIABILITY, cfg.SOCIAL_HARM)
    x = np.zeros(fun.n)
    x[fun.index("G", "CS", "AU")] = 0.5   # badged, maximally harsh outwards
    x[fun.index("N", "AS", "AS")] = 0.5   # unbadged, safe to everyone
    dec = th.harm_decomposition(x, fun, monomorphic=False)

    # the badged seat attacks across the boundary; the unbadged seat never does
    assert dec.conditional["badged with unbadged"] > 0.9
    assert dec.conditional["unbadged with badged"] == pytest.approx(0.0, abs=1e-12)
    # and both interiors stay clean
    assert dec.conditional["badged with badged"] == pytest.approx(0.0, abs=1e-12)
    assert dec.conditional["unbadged with unbadged"] == pytest.approx(0.0, abs=1e-12)


def test_nucleation_threshold_reports_the_sign_of_its_denominator() -> None:
    """Against a *safe* unbadged resident the inequality runs the other way.

    P(u,u) - P(v,w) is negative there, so a caller that reads the returned
    number as a lower bound on assortment inverts the result.  This is the case
    that separates nucleation from anarchy from nucleation from a safe
    uncertified regime, so it is the one worth pinning.
    """
    r_star, direction = nucleation_threshold(TABLES, 0.0, "CS", "CAS", "CS", 2.0)
    assert direction == -1
    assert r_star == pytest.approx(0.2397, abs=5e-4)
    below = replace(WM, r=r_star - 0.01)
    above = replace(WM, r=r_star + 0.01)
    safe_resident = ("N", "CS", "CS")
    assert invades(TABLES, below, 0.0, cfg.CLUB, safe_resident)
    assert not invades(TABLES, above, 0.0, cfg.CLUB, safe_resident)


def test_nucleation_threshold_flags_the_degenerate_denominator() -> None:
    """When P(u,u) equals P(v,w) no assortment lets the club in."""
    import math

    r_star, direction = nucleation_threshold(TABLES, 0.0, "CAS", "CAS", "CAS", 2.0)
    assert direction == 0
    assert math.isnan(r_star)


def test_first_strike_assortment_bound_is_where_proposition_one_stops() -> None:
    """Proposition 1 is false at the manuscript's own baseline assortment.

    The uncertified attractor of the bistability result is a safe unbadged
    population at L = 0, which is only consistent with Proposition 1 because
    the baseline r = 0.1 sits above this bound.
    """
    r_dagger = th.first_strike_assortment_bound(TABLES)
    assert r_dagger == pytest.approx(0.0764, abs=5e-4)
    assert uncertified_safety_is_impossible(TABLES, 0.0, r_dagger - 0.005)
    assert not uncertified_safety_is_impossible(TABLES, 0.0, r_dagger + 0.005)
    assert not uncertified_safety_is_impossible(TABLES, 0.0, cfg.IDENTITY.r)

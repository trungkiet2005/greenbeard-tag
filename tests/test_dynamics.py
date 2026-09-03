"""Evolutionary dynamics, including the closed form used in place of EGTtools."""

from __future__ import annotations

import numpy as np
import pytest

from egttools.analytical import PairwiseComparison
from egttools.games import Matrix2PlayerGameHolder

from gbtag.dynamics import (
    replicator_attractors,
    basin_classification,
    fixation_probability,
    integrate_replicator,
    interior_starts,
    invades,
    neutrally_stable_strategies,
    replicator_attractor,
    replicator_field,
    sml_transition_matrix,
    stationary_analysis,
    stationary_analysis_sml,
    stratified_alpha,
    strict_nash_strategies,
)


def _random_game(seed: int, n: int = 4) -> np.ndarray:
    return np.ascontiguousarray(np.random.default_rng(seed).normal(size=(n, n)) * 3.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("beta", [0.05, 0.2])
def test_fixation_probability_matches_egttools(seed: int, beta: float) -> None:
    """The closed form replaces the generic routine, so it must agree with it."""
    payoff = _random_game(seed)
    size = 60
    # the evolver holds a raw pointer to the game, so the game must outlive it
    game = Matrix2PlayerGameHolder(4, payoff)
    evolver = PairwiseComparison(size, game)
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            mine = fixation_probability(payoff, i, j, size, beta)
            theirs = float(evolver.calculate_fixation_probability(i, j, beta))
            # EGTtools truncates strongly disadvantaged mutants to exactly zero;
            # see test_closed_form_resolves_probabilities_egttools_truncates
            assert mine == pytest.approx(theirs, abs=1e-6)


def test_closed_form_resolves_probabilities_egttools_truncates() -> None:
    """A rare mutant always has a positive fixation probability at finite beta.

    The generic EGTtools routine returns exactly zero for the case below; the
    closed form used here returns 3.287091609e-8, which agrees with a 60-digit
    evaluation of the same sum to ten significant figures.
    """
    payoff = _random_game(0)
    size, beta = 60, 0.05
    game = Matrix2PlayerGameHolder(4, payoff)
    evolver = PairwiseComparison(size, game)
    assert float(evolver.calculate_fixation_probability(2, 1, beta)) == 0.0
    assert fixation_probability(payoff, 2, 1, size, beta) == pytest.approx(
        3.287091609e-8, rel=1e-6
    )


def test_neutral_game_gives_the_neutral_fixation_probability() -> None:
    payoff = np.zeros((3, 3))
    assert fixation_probability(payoff, 0, 1, 50, 0.1) == pytest.approx(1 / 50)


def test_fixation_survives_a_large_selection_pressure() -> None:
    """The exponentials are shifted, so a strongly dominated mutant is not a NaN."""
    payoff = np.array([[0.0, 0.0], [500.0, 500.0]])
    value = fixation_probability(payoff, 0, 1, 200, 1.0)
    assert np.isfinite(value)
    assert 0.0 <= value < 1e-6


def test_sml_transition_matrix_is_row_stochastic() -> None:
    p = sml_transition_matrix(_random_game(3, 6), 80, 0.05)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= -1e-15).all()


def test_sml_stationary_distribution_is_a_distribution() -> None:
    payoff = _random_game(4, 8)
    result = stationary_analysis_sml(payoff, np.zeros((8, 8)), 100, 0.05)
    assert result.strategy_frequencies.sum() == pytest.approx(1.0)
    assert (result.strategy_frequencies >= -1e-12).all()


def test_sml_is_uniform_on_a_neutral_game() -> None:
    payoff = np.zeros((5, 5))
    result = stationary_analysis_sml(payoff, np.zeros((5, 5)), 60, 0.1)
    assert result.strategy_frequencies == pytest.approx(np.full(5, 0.2), abs=1e-9)


def test_sml_scales_to_the_full_design_space() -> None:
    """28 designs are out of reach for the state-space route but fine here."""
    payoff = _random_game(5, 28)
    result = stationary_analysis_sml(payoff, np.zeros((28, 28)), 100, 0.05)
    assert result.strategy_frequencies.sum() == pytest.approx(1.0)


def test_full_chain_and_small_mutation_limit_agree_as_mutation_vanishes() -> None:
    payoff = _random_game(6, 3)
    unsafe = np.abs(_random_game(7, 3)) / 10.0
    limit = stationary_analysis_sml(payoff, unsafe, 40, 0.05)
    coarse = stationary_analysis(payoff, unsafe, 40, 0.05, mu=0.05)
    fine = stationary_analysis(payoff, unsafe, 40, 0.05, mu=0.002)
    gap_coarse = np.abs(coarse.strategy_frequencies - limit.strategy_frequencies).max()
    gap_fine = np.abs(fine.strategy_frequencies - limit.strategy_frequencies).max()
    assert gap_fine < gap_coarse


def test_replicator_stays_on_the_simplex() -> None:
    payoff = _random_game(8, 5)
    x0 = np.full(5, 0.2)
    _, traj = integrate_replicator(payoff, x0, t_end=50.0, n_points=25)
    assert np.allclose(traj.sum(axis=1), 1.0)
    assert (traj >= -1e-9).all()


def test_replicator_field_sums_to_zero() -> None:
    payoff = _random_game(9, 4)
    x = np.array([0.1, 0.2, 0.3, 0.4])
    assert replicator_field(x, payoff).sum() == pytest.approx(0.0, abs=1e-12)


def test_dominant_strategy_takes_over() -> None:
    payoff = np.array([[5.0, 5.0], [0.0, 0.0]])
    end = replicator_attractors(payoff, n_starts=8, t_end=500.0).mean(axis=0)
    assert end[0] == pytest.approx(1.0, abs=1e-3)


def test_interior_starts_reproduce_the_published_draws_bit_for_bit() -> None:
    """The flat measure must be the *same* draws the published share came from.

    Spelled out as the old inline code rather than as a stored digest, so the
    test says which stream is meant and fails loudly if numpy ever changes it.
    """
    n, n_starts, seed = 48, 64, 20260819
    rng = np.random.default_rng(seed)
    published = np.array([rng.dirichlet(np.ones(n)) for _ in range(n_starts)])
    assert np.array_equal(interior_starts(n, n_starts, seed), published)
    assert np.array_equal(interior_starts(n, n_starts, seed, alpha=1.0), published)


def test_interior_starts_are_interior_points_of_the_simplex() -> None:
    for alpha in (0.1, 1.0, 50.0):
        starts = interior_starts(12, 200, 7, alpha=alpha)
        assert starts.shape == (200, 12)
        assert starts.sum(axis=1) == pytest.approx(1.0)
        assert (starts > 0.0).all()


def test_a_vector_alpha_sets_the_expected_mass_of_each_design() -> None:
    alpha = np.array([0.5, 2.0, 4.0, 1.0])
    starts = interior_starts(4, 40000, 11, alpha=alpha)
    assert starts.mean(axis=0) == pytest.approx(alpha / alpha.sum(), abs=5e-3)


def test_interior_starts_rejects_a_bad_concentration() -> None:
    with pytest.raises(ValueError, match="positive"):
        interior_starts(4, 10, 0, alpha=0.0)
    with pytest.raises(ValueError, match="length 4"):
        interior_starts(4, 10, 0, alpha=np.ones(3))
    with pytest.raises(ValueError, match="n_starts"):
        interior_starts(4, 0, 0)


def test_stratified_alpha_equalises_unequal_badge_classes() -> None:
    """Three designs of one badge and one of another still split the mass."""
    badge = np.array(["G", "G", "G", "F", "N", "N"])
    alpha = stratified_alpha(badge)
    assert alpha.sum() == pytest.approx(len(badge))

    draws = interior_starts(len(badge), 20000, 20260819, alpha=alpha)
    mass = draws.mean(axis=0)
    for letter in ("G", "F", "N"):
        block = np.array([b == letter for b in badge])
        assert mass[block].sum() == pytest.approx(1.0 / 3.0, abs=0.01)


def test_stratified_alpha_honours_unequal_weights() -> None:
    badge = np.array(["G", "G", "F", "N", "N", "N"])
    weights = {"G": 0.5, "F": 0.2, "N": 0.3}
    alpha = stratified_alpha(badge, weights)
    assert alpha.sum() == pytest.approx(len(badge))

    draws = interior_starts(len(badge), 20000, 20260819, alpha=alpha)
    mass = draws.mean(axis=0)
    for letter, target in weights.items():
        block = np.array([b == letter for b in badge])
        assert mass[block].sum() == pytest.approx(target, abs=0.01)


def test_stratified_alpha_reproduces_the_flat_measure_on_size_proportions() -> None:
    """The published Dirichlet(1) is the member of the family that weights by size."""
    badge = np.array(["G", "G", "G", "F", "N", "N"])
    weights = {"G": 3 / 6, "F": 1 / 6, "N": 2 / 6}
    assert stratified_alpha(badge, weights) == pytest.approx(np.ones(6))


def test_stratified_alpha_rejects_unrealisable_targets() -> None:
    badge = np.array(["N", "N", "N"])
    with pytest.raises(ValueError, match="no design carries"):
        stratified_alpha(badge)
    with pytest.raises(ValueError, match="sum to one"):
        stratified_alpha(np.array(["G", "N"]), {"G": 0.5, "N": 0.4})
    with pytest.raises(ValueError, match="no target mass"):
        stratified_alpha(np.array(["G", "N"]), {"G": 1.0})


def test_replicator_attractors_default_alpha_is_the_pre_change_result() -> None:
    """The keyword must be inert at its default: same seed, same end states."""
    payoff = _random_game(11, 4)
    rng = np.random.default_rng(20260819)
    published = np.array(
        [
            replicator_attractor(payoff, rng.dirichlet(np.ones(4)), t_end=3000.0)
            for _ in range(50)
        ]
    )
    assert np.array_equal(replicator_attractors(payoff, n_starts=50), published)
    assert np.array_equal(
        replicator_attractors(payoff, n_starts=50, alpha=1.0), published
    )


def test_replicator_attractors_start_measure_changes_the_sample() -> None:
    """A different measure is a different experiment, not a different seed."""
    payoff = np.array([[1.0, 0.0], [0.0, 1.0]])
    flat = replicator_attractors(payoff, n_starts=40, t_end=200.0)
    edge = replicator_attractors(payoff, n_starts=40, t_end=200.0, alpha=0.05)
    assert not np.allclose(flat, edge)


def test_basin_classification_matches_the_inline_logic() -> None:
    """Agreement with the code in run_analysis.py the published share came from."""
    carries = np.array([1.0, 1.0, 0.0, 0.0])
    interior = np.random.default_rng(3).dirichlet(np.ones(4), size=64)
    faces = np.array(
        [
            [0.4, 0.6, 0.0, 0.0],  # pure badged face
            [0.0, 0.0, 0.25, 0.75],  # pure unbadged face
            [0.3, 0.3, 0.4, 0.0],  # badge-mixed, and on the badged side
        ]
    )
    ends = np.vstack([interior, faces])

    # verbatim from scripts/run_analysis.py
    is_certified = ends @ carries > 0.5
    badged_mass = ends @ carries
    mixed = int(((badged_mass > 1e-6) & (badged_mass < 1 - 1e-6)).sum())

    out = basin_classification(ends, carries)
    assert out["badged_share"] == float(is_certified.mean())
    assert out["unbadged_share"] == float((~is_certified).mean())
    assert out["mixed_count"] == mixed
    assert np.array_equal(out["badged_mass"], badged_mass)
    assert out["badged_share"] + out["unbadged_share"] == pytest.approx(1.0)


def test_basin_classification_counts_only_the_genuinely_mixed() -> None:
    carries = np.array([1.0, 0.0, 0.0])
    ends = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.5, 0.0]])
    out = basin_classification(ends, carries)
    assert out["mixed_count"] == 1
    assert out["badged_share"] == pytest.approx(1 / 3)


def test_basin_classification_rejects_a_non_indicator() -> None:
    with pytest.raises(ValueError, match="0/1 indicator"):
        basin_classification(np.eye(3), np.array([1.0, 0.5, 0.0]))
    with pytest.raises(ValueError, match="one entry per design"):
        basin_classification(np.eye(3), np.ones(2))


def test_equilibrium_notions_on_a_known_game() -> None:
    # a prisoner's dilemma: defection is the only strict equilibrium
    payoff = np.array([[3.0, 0.0], [5.0, 1.0]])
    assert strict_nash_strategies(payoff) == [1]
    assert neutrally_stable_strategies(payoff) == [1]
    assert invades(payoff, 1, 0)
    assert not invades(payoff, 0, 1)


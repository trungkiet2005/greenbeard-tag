"""Robustness machinery: planes, ablations, and threshold agreement."""

from __future__ import annotations

import numpy as np
import pytest

from gbtag import config as cfg
from gbtag.dynamics import stationary_analysis_sml
from gbtag.functionals import (
    build_functionals,
    plain_subspace,
    random_badge_functionals,
)
from gbtag.race import RaceParams, build_race_tables
from gbtag.robustness import (
    assortment_basin_sweep,
    badge_ablation,
    global_phase_plane,
    out_group_severity_sweep,
    pool_ablation,
    process_sensitivity,
    replicator_agreement,
    setback_scope_check,
    sigma_liability_plane,
    sigma_r_plane,
    start_measure_ablation,
    threshold_agreement,
)
from gbtag.theory import out_group_policy_scan

TABLES = build_race_tables(RaceParams())
FAST = dict(population_size=50, beta=0.05)


def test_sigma_r_plane_shapes_and_nucleation() -> None:
    plane = sigma_r_plane(
        TABLES,
        cfg.IDENTITY,
        sigmas=np.array([0.2, 0.8]),
        rs=np.array([0.0, 0.15]),
        method="sml",
        **FAST,
    )
    assert plane.unsafe.shape == (2, 2)
    # below the nucleation threshold the world is anarchic, above it is not
    assert plane.unsafe[0, 0] > 0.5
    assert plane.unsafe[1, 0] < 0.15


def test_sigma_liability_plane_value_vanishes_under_liability() -> None:
    plane = sigma_liability_plane(
        TABLES,
        cfg.IDENTITY,
        sigmas=np.array([0.5]),
        liabilities=np.array([0.0, 6.0]),
        method="sml",
        **FAST,
    )
    assert plane.value.shape == (2, 1)
    assert plane.value[0, 0] > 0.03
    assert abs(plane.value[1, 0]) < 0.02


def test_pool_ablation_ordering() -> None:
    pools = pool_ablation(TABLES, cfg.IDENTITY, method="sml", **FAST)
    assert set(pools) == {"full", "honest", "uncertified", "plain"}
    # forgery can only cost safety relative to the honest world
    assert pools["honest"].unsafe_frequency <= pools["full"].unsafe_frequency + 1e-9
    # and the certified world beats the uncertified one at baseline
    assert pools["full"].unsafe_frequency < pools["uncertified"].unsafe_frequency


def test_process_sensitivity_rows() -> None:
    rows = process_sensitivity(
        TABLES,
        cfg.IDENTITY,
        population_sizes=(50,),
        betas=(0.05, 0.2),
    )
    assert len(rows) == 2
    assert all(0.0 <= row["unsafe"] <= 1.0 for row in rows)


def test_replicator_agreement_runs() -> None:
    rows = replicator_agreement(
        TABLES, cfg.IDENTITY, sigmas=np.array([0.5]), n_starts=6, seed=2, **FAST
    )
    assert len(rows) == 1
    assert 0.0 <= rows[0]["rep_unsafe"] <= 1.0


def test_setback_scope_check_has_both_scopes() -> None:
    out = setback_scope_check(RaceParams(), cfg.IDENTITY, method="sml", **FAST)
    assert set(out) == {"total", "prize"}
    for scope in out.values():
        assert 0.0 <= scope["unsafe"] <= 1.0
        assert scope["unsafe"] < scope["uncertified_unsafe"] + 1e-9


def test_threshold_agreement_matches() -> None:
    rows = threshold_agreement(
        TABLES,
        cfg.CLUB,
        cfg.FALSEBEARD,
        kappa_gs=(2.0,),
        rhos=(0.0, 5.0),
        liabilities=(0.0,),
    )
    for row in rows:
        if not np.isnan(row["closed_form"]) and not np.isnan(row["numeric"]):
            if 0.0 <= row["closed_form"] <= 1.0:
                assert row["numeric"] == pytest.approx(row["closed_form"], abs=1e-6)


# --------------------------------------------------------------------------
# global structure: the experiments that put the published basin count in
# context.  Each one exists because the published study measured the global
# outcome at one point of a five-parameter space under one start measure.
# --------------------------------------------------------------------------


def test_start_measure_ablation_moves_the_share_it_is_asked_about() -> None:
    """The badged basin share is a property of the measure, not of the flow.

    This is the point of the experiment, so the test asserts the swing rather
    than any particular share: a concentrated Dirichlet and a diffuse one must
    disagree by far more than either one's sampling error.
    """
    out = start_measure_ablation(
        TABLES,
        cfg.IDENTITY,
        cfg.LIABILITY,
        cfg.SOCIAL_HARM,
        alphas=(0.1, 1.0, 50.0),
        n_starts=60,
        seed=7,
    )
    assert len(out.measures) == 4  # three concentrations plus the stratified one
    assert out.concentration.shape == (4, 48)
    lo, hi = out.share_range
    assert lo == pytest.approx(out.badged_share.min())
    assert hi == pytest.approx(out.badged_share.max())
    # a sample proportion at 60 draws has a standard error near 0.06; the swing
    # across measures is several times that, which is the whole finding
    assert hi - lo > 0.25
    # the equal-thirds stratified measure is exactly Dirichlet(1) on this design
    # space, because the three badge classes hold sixteen designs each
    assert out.concentration[-1] == pytest.approx(np.ones(48))


def test_assortment_basin_sweep_reproduces_the_published_baseline() -> None:
    out = assortment_basin_sweep(
        TABLES,
        cfg.IDENTITY,
        cfg.LIABILITY,
        cfg.SOCIAL_HARM,
        np.array([0.0, 0.1]),
        n_starts=60,
        seed=cfg.SEED,
    )
    assert out.rs.shape == out.badged_share.shape == (2,)
    # at zero assortment nothing separates the badge classes and the flow ends
    # badge-mixed almost everywhere; at the baseline it never does
    assert out.mixed_count[0] > out.mixed_count[1]
    assert out.mixed_count[1] == 0
    # both faces are unsafe below the first-strike bound and safe above it
    assert out.unsafe_certified[0] > 0.9 and out.unsafe_uncertified[0] > 0.9
    assert out.unsafe_certified[1] < 1e-6 and out.unsafe_uncertified[1] < 1e-6
    # where the conduct flow is single-valued the two faces agree exactly, which
    # is the dynamical content of the face-reduction theorem
    assert abs(out.unsafe_certified[1] - out.unsafe_uncertified[1]) < 1e-12


def test_assortment_basin_sweep_is_deterministic() -> None:
    kw = dict(n_starts=24, seed=11)
    rs = np.array([0.1])
    first = assortment_basin_sweep(
        TABLES, cfg.IDENTITY, cfg.LIABILITY, cfg.SOCIAL_HARM, rs, **kw
    )
    second = assortment_basin_sweep(
        TABLES, cfg.IDENTITY, cfg.LIABILITY, cfg.SOCIAL_HARM, rs, **kw
    )
    assert first.badged_share == pytest.approx(second.badged_share)
    assert first.unsafe_certified == pytest.approx(second.unsafe_certified)


def test_global_phase_plane_labels_every_cell() -> None:
    plane = global_phase_plane(
        TABLES,
        cfg.IDENTITY,
        cfg.LIABILITY,
        cfg.SOCIAL_HARM,
        np.array([0.5, 0.97]),
        np.array([0.0, 0.15]),
        n_starts=24,
        seed=3,
    )
    assert plane.regime.shape == (2, 2)
    assert plane.regime.min() >= 0
    assert plane.regime.max() < len(plane.regime_names)
    # the anarchic corner is unsafe, the settled corner is not
    assert plane.regime_names[plane.regime[0, 0]] == "mixed-unsafe"
    assert plane.regime_names[plane.regime[1, 0]] != "mixed-unsafe"


def test_badge_ablation_separates_the_four_arms() -> None:
    out = badge_ablation(
        TABLES,
        cfg.IDENTITY,
        cfg.LIABILITY,
        cfg.SOCIAL_HARM,
        np.array([0.1]),
        n_starts=24,
        seed=5,
        **FAST,
    )
    assert out.arms == ("full", "no-badge", "unforgeable", "random-badge")
    assert out.sml_unsafe.shape == (4, 1)
    full, plain, honest, random = out.sml_unsafe[:, 0]
    # an unforgeable badge is the best case and forgery can only cost safety
    assert honest <= full + 1e-9
    # the badge earns its keep against no badge at all
    assert full < plain
    # a badge nobody can read is worse than no badge, because conduct still
    # conditions on it and the condition has become a coin flip
    assert random > plain


def test_random_badge_degenerate_rates_reduce_to_the_faces() -> None:
    """A non-informative badge at qbar = 0 or 1 must reproduce the pure faces.

    This is the face-reduction theorem reached by a different code path: with
    no check ever passing every seat executes ``s_out``, and with every check
    passing every seat executes ``s_in``, so both collapse onto the same
    four-conduct race and must score the unsafe frequency of the plain
    subspace.  The dues differ between the two, and the unsafe frequency does
    not, which is the theorem in one line.
    """
    fun = build_functionals(TABLES, cfg.IDENTITY, cfg.LIABILITY, cfg.SOCIAL_HARM)
    plain = plain_subspace(fun)
    reference = stationary_analysis_sml(
        plain.fitness, plain.unsafe_frequency, 50, 0.05
    ).unsafe_frequency
    for p_genuine, p_forged in ((0.0, 0.0), (1.0, 0.0)):
        arm = random_badge_functionals(fun, p_genuine, p_forged)
        got = stationary_analysis_sml(
            arm.fitness, arm.unsafe_frequency, 50, 0.05
        ).unsafe_frequency
        assert got == pytest.approx(reference, abs=1e-9)


def test_out_group_severity_sweep_reproduces_the_discrete_policies() -> None:
    """The continuous ladder must pass exactly through the four conduct rules.

    Severity 0, 1/3, 2/3 and 1 are pure AS, CS, CAS and AU, so the sweep has to
    reproduce the discrete out-group scan there.  That agreement is what proves
    the interpolation is a real convex combination of the race matrix rather
    than a plausible-looking curve drawn through the same four endpoints.
    """
    severities = np.array([0.0, 1 / 3, 2 / 3, 1.0])
    out = out_group_severity_sweep(
        TABLES,
        cfg.IDENTITY,
        cfg.LIABILITY,
        cfg.SOCIAL_HARM,
        severities,
        n_starts=12,
        seed=2,
        **FAST,
    )
    scan = out_group_policy_scan(
        TABLES,
        cfg.IDENTITY,
        liability=cfg.LIABILITY,
        social_harm=cfg.SOCIAL_HARM,
        n_starts=12,
        seed=2,
        **FAST,
    )
    by_conduct = {row.s_out: row for row in scan}
    for i, conduct in enumerate(("AS", "CS", "CAS", "AU")):
        reference = by_conduct[conduct]
        expected = reference.spoof_threshold
        assert out.spoof_threshold[i] == pytest.approx(
            0.0 if expected is None else expected, abs=1e-6
        )
        assert out.entry_penalty[i] == pytest.approx(
            reference.entry_penalty, abs=1e-6
        )
        assert out.boundary_unsafe[i] == pytest.approx(
            reference.boundary_unsafe, abs=1e-6
        )


def test_out_group_severity_frontier_is_convex_in_the_toll() -> None:
    """Most of the forgery tolerance is bought by a small part of the toll.

    The published table reports four points and reads as a binary choice
    between a gentle club that cannot resist forgery at all and a harsh one
    that can.  On the continuous axis the frontier is strictly convex, so the
    harsh corner is inefficient, and that is a designable trade-off rather than
    a dichotomy.
    """
    out = out_group_severity_sweep(
        TABLES,
        cfg.IDENTITY,
        cfg.LIABILITY,
        cfg.SOCIAL_HARM,
        np.linspace(1 / 3, 1.0, 9),
        n_starts=12,
        seed=2,
        **FAST,
    )
    resisting = out.spoof_threshold > 0.0
    assert resisting.any(), "no severity on the ladder resists forgery"
    first = int(np.argmax(resisting))
    best = int(np.argmax(out.spoof_threshold))
    assert out.entry_penalty[best] > out.entry_penalty[first] > 0.0
    tolerance_fraction = out.spoof_threshold[first] / out.spoof_threshold[best]
    toll_fraction = out.entry_penalty[first] / out.entry_penalty[best]
    assert tolerance_fraction > 2.0 * toll_fraction

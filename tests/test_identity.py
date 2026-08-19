"""The identity layer: badges, perception, and the handshake law."""

from __future__ import annotations

import numpy as np
import pytest

from gbtag.identity import (
    BADGES,
    CLASSES,
    IdentityParams,
    N_DESIGNS,
    apply_assortment,
    badge_masses,
    class_masses,
    class_of_each_design,
    classify,
    design_index,
    design_labels,
    design_space,
    encounter_terms,
    pairwise_expectation,
)
from gbtag.race import STRATEGIES, RaceParams, build_race_tables


def test_design_space_has_48_designs_badge_major() -> None:
    space = design_space()
    assert len(space) == N_DESIGNS == 48
    assert space[0] == ("G", "AS", "AS")
    assert space[16] == ("F", "AS", "AS")
    assert space[-1] == ("N", "AU", "AU")
    assert len(set(space)) == 48


def test_labels_round_trip() -> None:
    labels = design_labels()
    for k, (b, si, so) in enumerate(design_space()):
        assert labels[k] == f"{b}:{si}/{so}"
        assert design_index(b, si, so) == k


def test_pass_rates() -> None:
    p = IdentityParams(sigma=0.3)
    assert p.pass_rate("G") == 1.0
    assert p.pass_rate("F") == 0.3
    assert p.pass_rate("N") == 0.0


def test_behavioural_pass_rate_under_auditing() -> None:
    """Without screening, a forged badge always elicits the in-group response."""
    p = IdentityParams(sigma=0.3, protection=False)
    assert p.behavioural_pass_rate("F") == 1.0
    assert p.behavioural_pass_rate("G") == 1.0
    assert p.behavioural_pass_rate("N") == 0.0


def test_expected_fine_is_detection_times_rho_regardless_of_protection() -> None:
    for protection in (True, False):
        p = IdentityParams(sigma=0.3, rho=10.0, protection=protection)
        assert p.expected_fine("F") == pytest.approx(7.0)
        assert p.expected_fine("G") == 0.0
        assert p.expected_fine("N") == 0.0


def test_parameter_validation() -> None:
    with pytest.raises(ValueError):
        IdentityParams(sigma=1.2)
    with pytest.raises(ValueError):
        IdentityParams(kappa_g=-1.0)
    with pytest.raises(ValueError):
        IdentityParams(rho=-0.1)
    with pytest.raises(ValueError):
        IdentityParams(r=1.5)


def test_encounter_terms_sum_to_one() -> None:
    params = IdentityParams(sigma=0.37)
    for focal in (("G", "CS", "CAS"), ("F", "CAS", "AU"), ("N", "AS", "AS")):
        for partner in (("G", "AS", "CS"), ("F", "CS", "CAS"), ("N", "AU", "AU")):
            terms = encounter_terms(focal, partner, params)
            assert sum(w for w, _, _ in terms) == pytest.approx(1.0)


def test_encounter_terms_are_the_product_law() -> None:
    """The two checks concern different badges, so the law factorises."""
    params = IdentityParams(sigma=0.4)
    focal = ("F", "AS", "CAS")
    partner = ("F", "CS", "AU")
    law = {}
    for w, sf, sp in encounter_terms(focal, partner, params):
        law[(sf, sp)] = law.get((sf, sp), 0.0) + w
    # focal plays AS iff partner's forged badge passes (0.4), partner plays
    # CS iff the focal forged badge passes (0.4), independently
    assert law[("AS", "CS")] == pytest.approx(0.16)
    assert law[("AS", "AU")] == pytest.approx(0.24)
    assert law[("CAS", "CS")] == pytest.approx(0.24)
    assert law[("CAS", "AU")] == pytest.approx(0.36)


def test_pairwise_expectation_agrees_with_enumeration() -> None:
    """The vectorised handshake expectation equals the four-term sum."""
    tables = build_race_tables(RaceParams())
    params = IdentityParams(sigma=0.61)
    idx = {s: k for k, s in enumerate(STRATEGIES)}
    got = pairwise_expectation(tables.payoff, params)
    space = design_space()
    rng = np.random.default_rng(0)
    for _ in range(60):
        i, j = rng.integers(0, len(space), size=2)
        expected = sum(
            w * tables.payoff[idx[sf], idx[sp]]
            for w, sf, sp in encounter_terms(space[i], space[j], params)
        )
        assert got[i, j] == pytest.approx(expected, abs=1e-12)


def test_sigma_one_makes_forged_equal_genuine() -> None:
    """At full spoof success a forged badge is behaviourally a genuine one."""
    tables = build_race_tables(RaceParams())
    params = IdentityParams(sigma=1.0)
    a = pairwise_expectation(tables.payoff, params)
    space = design_space()
    for si in STRATEGIES:
        for so in STRATEGIES:
            g = space.index(("G", si, so))
            f = space.index(("F", si, so))
            assert np.allclose(a[g], a[f])
            assert np.allclose(a[:, g], a[:, f])


def test_sigma_zero_makes_forged_equal_unbadged() -> None:
    """At zero spoof success a forged badge is behaviourally no badge."""
    tables = build_race_tables(RaceParams())
    params = IdentityParams(sigma=0.0)
    a = pairwise_expectation(tables.payoff, params)
    space = design_space()
    for si in STRATEGIES:
        for so in STRATEGIES:
            f = space.index(("F", si, so))
            n = space.index(("N", si, so))
            assert np.allclose(a[f], a[n])
            assert np.allclose(a[:, f], a[:, n])


def test_unconditional_designs_ignore_the_partner_badge() -> None:
    tables = build_race_tables(RaceParams())
    params = IdentityParams(sigma=0.5)
    a = pairwise_expectation(tables.payoff, params)
    space = design_space()
    i = space.index(("N", "CS", "CS"))
    # an unconditional design earns the same against G, F, N partners with
    # identical conduct pairs, because its own play never branches and the
    # partner's play branches only on the focal (absent) badge
    for si in STRATEGIES:
        g = space.index(("G", si, si))
        f = space.index(("F", si, si))
        n = space.index(("N", si, si))
        assert a[i, g] == pytest.approx(a[i, f], abs=1e-12)
        assert a[i, f] == pytest.approx(a[i, n], abs=1e-12)


def test_apply_assortment_mixes_towards_the_diagonal() -> None:
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    mixed = apply_assortment(m, 0.25)
    assert mixed[0, 0] == pytest.approx(1.0)
    assert mixed[0, 1] == pytest.approx(0.25 * 1.0 + 0.75 * 2.0)
    assert mixed[1, 0] == pytest.approx(0.25 * 4.0 + 0.75 * 3.0)
    assert np.allclose(apply_assortment(m, 0.0), m)
    full = apply_assortment(m, 1.0)
    assert np.allclose(full, np.array([[1.0, 1.0], [4.0, 4.0]]))


def test_classes_partition_the_design_space() -> None:
    classes = class_of_each_design()
    assert len(classes) == 48
    assert set(classes) == set(CLASSES)
    assert classify("G", "CS", "CAS") == "certified club"
    assert classify("G", "AS", "AU") == "certified club"
    assert classify("G", "CAS", "AS") == "certified aggressor"
    assert classify("F", "CS", "CAS") == "falsebeard"
    assert classify("N", "AS", "CS") == "unbadged cooperator"
    assert classify("N", "CS", "CAS") == "unbadged other"


def test_class_and_badge_masses_sum_to_one() -> None:
    rng = np.random.default_rng(3)
    x = rng.dirichlet(np.ones(48))
    assert sum(class_masses(x).values()) == pytest.approx(1.0)
    assert sum(badge_masses(x).values()) == pytest.approx(1.0)
    assert set(class_masses(x)) == set(CLASSES)
    assert set(badge_masses(x)) == set(BADGES)

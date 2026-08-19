"""The instruments and the detection decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from gbtag import config as cfg
from gbtag.identity import IdentityParams
from gbtag.interventions import (
    assortment_sweep,
    detection_decomposition,
    dues_sweep,
    fine_sweep,
    forgery_cost_sweep,
    liability_sweep,
    verification_sweep,
)
from gbtag.race import RaceParams, build_race_tables

TABLES = build_race_tables(RaceParams())
FAST = dict(population_size=50, beta=0.05)


def test_verification_sweep_shapes_and_labels() -> None:
    out = verification_sweep(
        TABLES, cfg.IDENTITY, np.array([0.0, 0.5, 1.0]), method="sml", **FAST
    )
    assert [o.setting for o in out] == [0.0, 0.5, 1.0]
    assert all(o.instrument == "verification" for o in out)
    assert all(0.0 <= o.unsafe_frequency <= 1.0 for o in out)


def test_attestation_integrity_collapses_along_the_sweep() -> None:
    out = verification_sweep(
        TABLES, cfg.IDENTITY, np.array([0.3, 0.99]), method="sml", **FAST
    )
    assert out[0].attestation_integrity > 0.8
    assert out[1].attestation_integrity < 0.3


def test_assortment_sweep_crosses_the_nucleation_threshold() -> None:
    out = assortment_sweep(
        TABLES, cfg.IDENTITY, np.array([0.0, 0.15]), method="sml", **FAST
    )
    assert out[0].unsafe_frequency > 0.5
    assert out[1].unsafe_frequency < 0.15
    assert out[1].club_share > out[0].club_share


def test_fines_suppress_forgers_when_detection_exists() -> None:
    params = IdentityParams(sigma=0.5, kappa_g=2.0, kappa_f=0.0, rho=0.0, r=0.1)
    out = fine_sweep(TABLES, params, np.array([0.0, 30.0]), method="sml", **FAST)
    assert out[1].falsebeard_share < out[0].falsebeard_share


def test_dues_hurt_the_club() -> None:
    out = dues_sweep(TABLES, cfg.IDENTITY, np.array([0.5, 8.0]), method="sml", **FAST)
    assert out[1].club_share < out[0].club_share


def test_forgery_cost_helps_the_club() -> None:
    params = IdentityParams(sigma=0.95, kappa_g=2.0, kappa_f=0.0, rho=0.0, r=0.1)
    out = forgery_cost_sweep(
        TABLES, params, np.array([0.0, 4.0]), method="sml", **FAST
    )
    assert out[1].falsebeard_share <= out[0].falsebeard_share + 1e-9


def test_liability_sweep_restores_the_sister_regime() -> None:
    """With liability well above L_R the identity layer stops mattering."""
    out = liability_sweep(
        TABLES, cfg.IDENTITY, np.array([0.0, 6.0]), method="sml", **FAST
    )
    assert out[1].unsafe_frequency < out[0].unsafe_frequency
    assert out[1].unsafe_frequency < 0.05


def test_detection_decomposition_accounting() -> None:
    params = IdentityParams(sigma=0.9, kappa_g=2.0, kappa_f=0.0, rho=0.0, r=0.1)
    dec = detection_decomposition(TABLES, params, rho=20.0, method="sml", **FAST)
    total = dec.both.unsafe_frequency - dec.neither.unsafe_frequency
    assert dec.total_effect == pytest.approx(total, abs=1e-12)
    assert dec.interaction == pytest.approx(
        dec.total_effect - dec.screening_effect - dec.fines_effect, abs=1e-12
    )


def test_placebo_cell_has_no_screening() -> None:
    """With protection off and no fine, verification is inert for conduct."""
    params = IdentityParams(sigma=0.5, kappa_g=2.0, kappa_f=0.0, rho=0.0, r=0.1)
    dec = detection_decomposition(TABLES, params, rho=20.0, method="sml", **FAST)
    # in the placebo world a forged badge behaves exactly like a genuine one,
    # so falsebeards can only be *more* prevalent than under screening
    fb_placebo = dec.neither.class_distribution["falsebeard"]
    fb_screen = dec.screening_only.class_distribution["falsebeard"]
    assert fb_placebo >= fb_screen - 1e-9

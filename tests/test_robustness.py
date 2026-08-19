"""Robustness machinery: planes, ablations, and threshold agreement."""

from __future__ import annotations

import numpy as np
import pytest

from gbtag import config as cfg
from gbtag.race import RaceParams, build_race_tables
from gbtag.robustness import (
    pool_ablation,
    process_sensitivity,
    replicator_agreement,
    setback_scope_check,
    sigma_liability_plane,
    sigma_r_plane,
    threshold_agreement,
)

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

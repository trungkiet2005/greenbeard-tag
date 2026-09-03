"""Compute the robustness grids and checks.

Usage::

    python scripts/run_robustness.py [--outdir results]

Writes ``results/grids.npz``, ``results/tables/robustness_*.csv`` and
``results/robustness_summary.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gbtag import config as cfg
from gbtag import robustness as rb
from gbtag import theory as th
from gbtag.race import build_race_tables

ROOT = Path(__file__).resolve().parents[1]


def main(outdir: Path) -> None:
    tables_dir = outdir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}

    race = build_race_tables(cfg.RACE)
    base = cfg.IDENTITY
    sml = dict(population_size=cfg.POPULATION, beta=cfg.BETA)

    # ---------------------------------------------------------------- planes
    sigmas = cfg.SIGMA_GRID[::2]
    rs = cfg.R_GRID[::2]
    plane = rb.sigma_r_plane(race, base, sigmas, rs, cfg.LIABILITY,
                             cfg.SOCIAL_HARM, "sml", **sml)

    liabilities = cfg.L_GRID[::2]
    value_plane = rb.sigma_liability_plane(
        race, base, sigmas, liabilities, cfg.SOCIAL_HARM, "sml", **sml
    )

    # Accumulated rather than written here: the global phase plane below is
    # computed after this point and belongs in the same archive, and npz has no
    # append.  The file is written once, at the end of main().
    grids = {
        "plane_sigmas": plane.x_values,
        "plane_rs": plane.y_values,
        "plane_unsafe": plane.unsafe,
        "plane_club": plane.club,
        "plane_falsebeard": plane.falsebeard,
        "plane_integrity": plane.integrity,
        "value_sigmas": value_plane.sigmas,
        "value_liabilities": value_plane.liabilities,
        "value_certified": value_plane.certified,
        "value_uncertified": value_plane.uncertified,
        "value": value_plane.value,
    }
    summary["value_plane_max"] = float(value_plane.value.max())
    summary["value_plane_min"] = float(value_plane.value.min())
    # the wedge where identity moves the outcome by more than one point
    summary["value_share_above_001"] = float((value_plane.value > 0.01).mean())
    lr = th.reciprocity_threshold(race)
    below = value_plane.liabilities <= lr
    above = value_plane.liabilities > lr
    summary["mean_value_below_lr"] = float(value_plane.value[below].mean())
    summary["mean_value_above_lr"] = float(value_plane.value[above].mean())

    # ---------------------------------------------------------------- pools
    pools = rb.pool_ablation(race, base, cfg.LIABILITY, cfg.SOCIAL_HARM, "sml", **sml)
    summary["pool_unsafe"] = {
        name: eq.unsafe_frequency for name, eq in pools.items()
    }

    # ---------------------------------------------------------------- process
    proc = rb.process_sensitivity(
        race, base, cfg.LIABILITY, cfg.SOCIAL_HARM,
        population_sizes=(50, 100, 200), betas=(0.01, 0.05, 0.2),
    )
    pd.DataFrame(proc).to_csv(tables_dir / "robustness_process.csv", index=False)
    unsafe_range = [row["unsafe"] for row in proc]
    summary["process_unsafe_min"] = float(min(unsafe_range))
    summary["process_unsafe_max"] = float(max(unsafe_range))

    agree = rb.replicator_agreement(
        race, base, cfg.SIGMA_GRID[::10], cfg.LIABILITY, cfg.SOCIAL_HARM,
        n_starts=cfg.REPLICATOR_STARTS // 4, seed=cfg.SEED, **sml
    )
    pd.DataFrame(agree).to_csv(tables_dir / "robustness_replicator.csv", index=False)
    integ_sml = np.array([row["sml_integrity"] for row in agree])
    integ_rep = np.array([row["rep_integrity"] for row in agree])
    summary["replicator_integrity_correlation"] = float(
        np.corrcoef(integ_sml, integ_rep)[0, 1]
    )

    # ------------------------------------------------------------ global structure
    # The five experiments below all interrogate the *replicator basins* rather
    # than the small-mutation chain.  They exist because the published study
    # measured the global outcome at exactly one point of a five-parameter space
    # and under exactly one start measure, which is not enough to tell a robust
    # phenomenon from a baseline coincidence.
    #
    # ``r`` is swept over [0, 0.25] rather than the whole of ``R_GRID``: every
    # locus the paper reports (r* = 0.063, r_dagger = 0.076, the invader
    # exchange at 0.077, the exploiter lock-out at 0.134) lies inside it, and
    # the flow is featureless above it.
    r_global = cfg.R_GRID[cfg.R_GRID <= 0.25]

    measures = rb.start_measure_ablation(
        race, base, cfg.LIABILITY, cfg.SOCIAL_HARM,
        alphas=(0.1, 0.5, 1.0, 5.0, 50.0), n_starts=400, seed=cfg.SEED,
    )
    # ``concentration`` is a full per-design vector for each measure, so the
    # table records its range: the two agree for a scalar Dirichlet and bracket
    # the stratified measure, which is exactly the distinction worth seeing.
    pd.DataFrame({
        "measure": measures.measures,
        "n_starts": measures.n_starts,
        "concentration_min": measures.concentration.min(axis=1),
        "concentration_max": measures.concentration.max(axis=1),
        "badged_share": measures.badged_share,
        "mixed_count": measures.mixed_count,
        "unsafe": measures.unsafe,
        "social_payoff": measures.social_payoff,
    }).to_csv(tables_dir / "start_measure.csv", index=False)
    # The honest uncertainty on the basin claim is this range, not the Wilson
    # interval of any one measure inside it.
    summary["start_measure_share_range"] = list(measures.share_range)
    summary["start_measure_share_spread"] = float(
        measures.share_range[1] - measures.share_range[0]
    )

    basins = rb.assortment_basin_sweep(
        race, base, cfg.LIABILITY, cfg.SOCIAL_HARM, r_global,
        n_starts=400, seed=cfg.SEED,
    )
    # ``n_starts`` travels with the table: a mixed-endpoint count is unreadable
    # without the denominator, and a downstream figure should not have to
    # hard-code it.
    pd.DataFrame({
        "r": basins.rs,
        "n_starts": basins.n_starts,
        "badged_share": basins.badged_share,
        "mixed_count": basins.mixed_count,
        "unsafe_certified": basins.unsafe_certified,
        "unsafe_uncertified": basins.unsafe_uncertified,
        "social_certified": basins.social_certified,
        "social_uncertified": basins.social_uncertified,
    }).to_csv(tables_dir / "assortment_basins.csv", index=False)
    # The face-reduction theorem makes the two faces carry the same conduct
    # flow, so their attractor sets coincide exactly.  It does NOT make these
    # two sampled means coincide: a Dirichlet start on 48 designs projects onto
    # the two faces as two different measures, so wherever the conduct flow has
    # more than one attractor the two sampled means can differ.  Record both the
    # gap and the gap restricted to the region where the conduct flow is
    # single-valued, which is where the theorem is directly visible.
    summary["face_gap_max"] = float(basins.max_face_gap)
    settled = basins.unsafe_certified < 1e-6
    summary["face_gap_max_where_safe"] = float(
        np.abs(basins.unsafe_certified[settled]
               - basins.unsafe_uncertified[settled]).max()
    ) if settled.any() else float("nan")
    unsafe_both = basins.unsafe_certified > 0.05
    summary["both_faces_unsafe_below_r"] = float(
        basins.rs[unsafe_both].max()
    ) if unsafe_both.any() else float("nan")

    # 100 starts per cell rather than the 400 the one-dimensional sweeps use.
    # The regime label turns on the unsafe frequency and on the attestation
    # integrity of the badged endpoints, both of which are stable across
    # starts, and on whether badged endpoints are reached at all, which needs
    # only a couple of them; the basin share itself is reported from the
    # dedicated sweeps, where it is resolved properly.
    global_plane = rb.global_phase_plane(
        race, base, cfg.LIABILITY, cfg.SOCIAL_HARM,
        cfg.SIGMA_GRID[::4], r_global, n_starts=100, seed=cfg.SEED,
    )
    grids |= {
        "global_sigmas": global_plane.sigmas,
        "global_rs": global_plane.rs,
        "global_badged_share": global_plane.badged_share,
        "global_unsafe": global_plane.unsafe,
        "global_integrity": global_plane.integrity,
        "global_regime": global_plane.regime,
        "global_regime_names": np.array(global_plane.regime_names),
    }
    summary["global_regime_fractions"] = {
        name: float((global_plane.regime == k).mean())
        for k, name in enumerate(global_plane.regime_names)
    }

    ablation = rb.badge_ablation(
        race, base, cfg.LIABILITY, cfg.SOCIAL_HARM, r_global[::2],
        n_starts=200, seed=cfg.SEED, population_size=cfg.POPULATION, beta=cfg.BETA,
    )
    pd.DataFrame(
        {"r": ablation.rs}
        | {f"sml_{a}": ablation.sml_unsafe[j] for j, a in enumerate(ablation.arms)}
        | {f"basin_{a}": ablation.basin_unsafe[j] for j, a in enumerate(ablation.arms)}
    ).to_csv(tables_dir / "badge_ablation.csv", index=False)
    summary["badge_ablation_random_sml_gap"] = float(ablation.random_badge_sml_gap)
    summary["badge_ablation_random_basin_gap"] = float(ablation.random_badge_basin_gap)

    severity = rb.out_group_severity_sweep(
        race, base, cfg.LIABILITY, cfg.SOCIAL_HARM,
        np.round(np.linspace(0.0, 1.0, 25), 4), n_starts=200, seed=cfg.SEED,
        population_size=cfg.POPULATION, beta=cfg.BETA,
    )
    pd.DataFrame({
        "severity": severity.severities,
        "lower_rung": severity.lower_rung,
        "upper_rung": severity.upper_rung,
        "weight": severity.weight,
        "spoof_threshold": severity.spoof_threshold,
        "entry_penalty": severity.entry_penalty,
        "boundary_unsafe": severity.boundary_unsafe,
        "club_share": severity.club_share,
    }).to_csv(tables_dir / "out_group_severity.csv", index=False)
    # The frontier is strictly convex, so the harsh corner is inefficient: read
    # off how much of the maximum tolerance the cheapest resisting policy buys,
    # and at what fraction of the maximum toll.
    resists = severity.spoof_threshold > 0.0
    if resists.any():
        first = int(np.argmax(resists))
        best = int(np.argmax(severity.spoof_threshold))
        summary["severity_first_resisting"] = float(severity.severities[first])
        summary["severity_first_tolerance"] = float(severity.spoof_threshold[first])
        summary["severity_first_toll"] = float(severity.entry_penalty[first])
        summary["severity_max_tolerance"] = float(severity.spoof_threshold[best])
        summary["severity_max_toll"] = float(severity.entry_penalty[best])
        summary["severity_toll_fraction_for_first"] = float(
            severity.entry_penalty[first] / severity.entry_penalty[best]
        )

    # ---------------------------------------------------------------- payoff reading
    scope = rb.setback_scope_check(
        cfg.RACE, base, cfg.LIABILITY, cfg.SOCIAL_HARM, "sml", **sml
    )
    pd.DataFrame(
        [{"scope": k, **v} for k, v in scope.items()]
    ).to_csv(tables_dir / "robustness_setback_scope.csv", index=False)
    summary["setback_scope"] = scope

    # ---------------------------------------------------------------- thresholds
    thr = rb.threshold_agreement(race, cfg.CLUB, cfg.FALSEBEARD)
    pd.DataFrame(thr).to_csv(tables_dir / "robustness_thresholds.csv", index=False)
    diffs = [
        abs(row["numeric"] - row["closed_form"])
        for row in thr
        if not (np.isnan(row["numeric"]) or np.isnan(row["closed_form"]))
        and 0.0 <= row["closed_form"] <= 1.0
    ]
    summary["threshold_max_abs_gap"] = float(max(diffs)) if diffs else 0.0

    np.savez(outdir / "grids.npz", **grids)
    (outdir / "robustness_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(f"wrote {outdir / 'grids.npz'} ({len(grids)} arrays) and "
          f"{outdir / 'robustness_summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=ROOT / "results")
    main(parser.parse_args().outdir)

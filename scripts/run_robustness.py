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

    np.savez(
        outdir / "grids.npz",
        plane_sigmas=plane.x_values,
        plane_rs=plane.y_values,
        plane_unsafe=plane.unsafe,
        plane_club=plane.club,
        plane_falsebeard=plane.falsebeard,
        plane_integrity=plane.integrity,
        value_sigmas=value_plane.sigmas,
        value_liabilities=value_plane.liabilities,
        value_certified=value_plane.certified,
        value_uncertified=value_plane.uncertified,
        value=value_plane.value,
    )
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

    (outdir / "robustness_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(f"wrote {outdir / 'robustness_summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=ROOT / "results")
    main(parser.parse_args().outdir)

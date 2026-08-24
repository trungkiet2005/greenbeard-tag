"""Compute every table and every number quoted in the manuscript.

Usage::

    python scripts/run_analysis.py [--outdir results]

Writes ``results/tables/*.csv`` and ``results/key_numbers.json``, which holds
every scalar the text quotes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from gbtag import config as cfg
from gbtag import interventions as iv
from gbtag import theory as th
from gbtag.dynamics import (
    focal_mass_start,
    full_dimensional_perturbations,
    integrate_replicator,
    replicator_attractor,
    replicator_attractors,
    replicator_field,
)
from gbtag.functionals import (
    aggregate_unsafe_frequency,
    build_functionals,
    honest_subspace,
    mean_social_payoff,
    plain_subspace,
    unbadged_subspace,
)
from gbtag.identity import design_labels
from gbtag.race import STRATEGIES, build_race_tables

ROOT = Path(__file__).resolve().parents[1]


def _wilson(successes: int, trials: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    The normal approximation is not usable here: it is symmetric, and a basin
    share near a boundary would get an interval running outside ``[0, 1]``.
    """
    p = successes / trials
    den = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / den
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / den
    return float(centre - half), float(centre + half)


def _frame(matrix: np.ndarray, labels) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=list(labels), columns=list(labels))


def _sweep_frame(outcomes) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "setting": o.setting,
                "unsafe": o.unsafe_frequency,
                "club": o.club_share,
                "falsebeard": o.falsebeard_share,
                "mark_lift": o.mark_lift,
                "integrity": o.attestation_integrity,
                "social": o.social_payoff,
            }
            for o in outcomes
        ]
    )


def main(outdir: Path) -> None:
    tables_dir = outdir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    key: dict[str, object] = {}

    race = build_race_tables(cfg.RACE)
    base = cfg.IDENTITY
    sml = dict(population_size=cfg.POPULATION, beta=cfg.BETA)

    # ---------------------------------------------------------------- layer 0
    _frame(race.payoff, STRATEGIES).to_csv(tables_dir / "race_payoff.csv")
    _frame(race.unsafe_count, STRATEGIES).to_csv(tables_dir / "race_unsafe_count.csv")
    _frame(race.unsafe_frequency, STRATEGIES).to_csv(
        tables_dir / "race_unsafe_frequency.csv"
    )

    key["expected_horizon"] = cfg.RACE.expected_horizon
    key["reciprocity_threshold"] = th.reciprocity_threshold(race)
    key["first_strike_threshold_as"] = th.first_strike_threshold(race, "AS")
    key["first_strike_threshold_cs"] = th.first_strike_threshold(race, "CS")
    key["uncertified_safety_impossible_at_zero"] = th.uncertified_safety_is_impossible(
        race, 0.0
    )
    key["first_strike_assortment_bound"] = th.first_strike_assortment_bound(race)
    # The same statement at the baseline assortment, where it is false: this is
    # the hypothesis Proposition 1 now carries, recorded so it cannot drift.
    key["uncertified_safety_impossible_at_baseline_r"] = (
        th.uncertified_safety_is_impossible(race, 0.0, cfg.IDENTITY.r)
    )
    p0 = th.race_private(race, 0.0)
    idx = {s: i for i, s in enumerate(STRATEGIES)}
    key["payoff_cs_cs"] = float(p0[idx["CS"], idx["CS"]])
    key["payoff_cas_cs"] = float(p0[idx["CAS"], idx["CS"]])
    key["payoff_cas_as"] = float(p0[idx["CAS"], idx["AS"]])
    key["payoff_cas_cas"] = float(p0[idx["CAS"], idx["CAS"]])
    key["payoff_cs_cas"] = float(p0[idx["CS"], idx["CAS"]])
    key["harm_cas_as"] = float(race.unsafe_count[idx["CAS"], idx["AS"]])

    # ------------------------------------------------------- the uncertified world
    rows = []
    for L in cfg.L_GRID:
        for r in (0.0, base.r):
            fun = build_functionals(race, replace(base, r=r), float(L), cfg.SOCIAL_HARM)
            sub = unbadged_subspace(fun)
            eq = th.equilibrium(sub, race, "sml", **sml)
            rows.append(
                {"liability": float(L), "r": r, "unsafe": eq.unsafe_frequency}
            )
    pd.DataFrame(rows).to_csv(tables_dir / "uncertified_by_liability.csv", index=False)
    df = pd.DataFrame(rows)
    key["uncertified_unsafe_L0_r0"] = float(
        df[(df.liability == 0.0) & (df.r == 0.0)].unsafe.iloc[0]
    )
    key["uncertified_unsafe_L0_baseline_r"] = float(
        df[(df.liability == 0.0) & (df.r == base.r)].unsafe.iloc[0]
    )

    # ---------------------------------------------------------------- thresholds
    key["spoof_threshold_cs_club"] = th.spoof_threshold_closed_form(
        race, 0.0, "CS", "CAS", "CAS", base.kappa_g, base.kappa_f, 0.0
    )
    key["spoof_threshold_as_club"] = th.spoof_threshold_closed_form(
        race, 0.0, "AS", "CAS", "CAS", base.kappa_g, base.kappa_f, 0.0
    )
    key["spoof_threshold_ratio"] = (
        key["spoof_threshold_cs_club"] / key["spoof_threshold_as_club"]
    )
    key["mimic_threshold"] = th.mimic_threshold_closed_form(
        race, 0.0, "CS", "CAS", base.kappa_g, base.kappa_f, 0.0
    )
    wm = replace(base, r=0.0)
    key["spoof_threshold_forger_assorted"] = th.spoof_threshold(
        race, base, 0.0, cfg.CLUB, cfg.FALSEBEARD
    )
    key["spoof_threshold_mimic_assorted"] = th.spoof_threshold(
        race, base, 0.0, cfg.CLUB, cfg.MIMIC
    )
    key["nucleation_threshold"], key["nucleation_direction"] = th.nucleation_threshold(
        race, 0.0, "CS", "CAS", "CAS", base.kappa_g
    )
    # The same club against a *safe* unbadged resident: the denominator changes
    # sign, so assortment obstructs the club instead of enabling it.
    key["nucleation_vs_safe_resident"], key["nucleation_vs_safe_direction"] = (
        th.nucleation_threshold(race, 0.0, "CS", "CAS", "CS", base.kappa_g)
    )
    key["badge_futility"] = th.badge_is_futile_against_nonconditioners(race, wm, 0.0)
    key["reentry_forger_r0"] = th.reentry_threshold(
        race, wm, 0.0, cfg.CLUB, cfg.FALSEBEARD
    )
    key["reentry_anarchy_r0"] = th.reentry_threshold(race, wm, 0.0, cfg.CLUB, cfg.ANARCHY)
    key["reentry_anarchy_baseline_r"] = th.reentry_threshold(
        race, base, 0.0, cfg.CLUB, cfg.ANARCHY
    )
    key["dues_gradient"] = th.dues_gradient(race, 0.0, "CS", "CAS", "CAS", 0.0)
    key["invader_ordering_flip"] = th.invader_ordering_flip(
        race, base, cfg.LIABILITY, cfg.CLUB, cfg.FALSEBEARD, cfg.MIMIC
    )
    key["exploiter_immunity_assortment"] = th.exploiter_immunity(
        race, base, cfg.LIABILITY, cfg.CLUB, cfg.FALSEBEARD
    )

    # how the two thresholds move with assortment: the asymmetry behind the flip
    order_rows = []
    for r in np.round(np.linspace(0.0, 0.25, 51), 4):
        p = replace(base, r=float(r))
        order_rows.append(
            {
                "r": float(r),
                "sigma_exploiter": th.spoof_threshold(
                    race, p, cfg.LIABILITY, cfg.CLUB, cfg.FALSEBEARD
                ),
                "sigma_mimic": th.spoof_threshold(
                    race, p, cfg.LIABILITY, cfg.CLUB, cfg.MIMIC
                ),
            }
        )
    pd.DataFrame(order_rows).to_csv(tables_dir / "invader_ordering.csv", index=False)
    key["sigma_exploiter_r0"] = order_rows[0]["sigma_exploiter"]
    key["sigma_mimic_r0"] = order_rows[0]["sigma_mimic"]
    mimic_values = [
        row["sigma_mimic"] for row in order_rows if row["sigma_mimic"] is not None
    ]
    key["mimic_threshold_span_over_assortment"] = max(mimic_values) - min(mimic_values)

    thr_rows = []
    for kg in (0.5, 1.0, 2.0, 4.0, 8.0):
        for rho in (0.0, 5.0, 20.0):
            thr_rows.append(
                {
                    "kappa_g": kg,
                    "rho": rho,
                    "sigma_star_forger": th.spoof_threshold_closed_form(
                        race, 0.0, "CS", "CAS", "CAS", kg, 0.0, rho
                    ),
                    "sigma_star_mimic": th.mimic_threshold_closed_form(
                        race, 0.0, "CS", "CAS", kg, 0.0, rho
                    ),
                    "r_star": th.nucleation_threshold(race, 0.0, "CS", "CAS", "CAS", kg)[0],
                }
            )
    pd.DataFrame(thr_rows).to_csv(tables_dir / "thresholds.csv", index=False)

    # required fines along the sigma grid
    fine_rows = [
        {
            "sigma": float(s),
            "required_fine": th.required_fine(
                race, 0.0, float(s), "CS", "CAS", "CAS", base.kappa_g, base.kappa_f
            ),
        }
        for s in cfg.SIGMA_GRID
        if s < 1.0
    ]
    pd.DataFrame(fine_rows).to_csv(tables_dir / "required_fine.csv", index=False)
    key["required_fine_090"] = th.required_fine(
        race, 0.0, 0.90, "CS", "CAS", "CAS", base.kappa_g, base.kappa_f
    )
    key["required_fine_095"] = th.required_fine(
        race, 0.0, 0.95, "CS", "CAS", "CAS", base.kappa_g, base.kappa_f
    )
    key["required_fine_099"] = th.required_fine(
        race, 0.0, 0.99, "CS", "CAS", "CAS", base.kappa_g, base.kappa_f
    )

    # ---------------------------------------------------------------- sweeps
    ver = iv.verification_sweep(race, base, cfg.SIGMA_GRID, cfg.LIABILITY,
                                cfg.SOCIAL_HARM, "sml", **sml)
    ver_df = _sweep_frame(ver)
    ver_df.to_csv(tables_dir / "verification_sweep.csv", index=False)

    baseline = next(o for o in ver if abs(o.setting - base.sigma) < 1e-9)
    key["baseline_unsafe"] = baseline.unsafe_frequency
    key["baseline_club_share"] = baseline.club_share
    key["baseline_integrity"] = baseline.attestation_integrity
    key["baseline_mark_lift"] = baseline.mark_lift
    key["baseline_social"] = baseline.social_payoff
    perfect = ver[0]
    key["perfect_verification_unsafe"] = perfect.unsafe_frequency
    key["perfect_verification_club_share"] = perfect.club_share
    free = ver[-1]
    key["free_forgery_unsafe"] = free.unsafe_frequency
    key["free_forgery_integrity"] = free.attestation_integrity
    key["free_forgery_club_share"] = free.club_share

    # where the integrity falls through one half, and where the lift inverts
    integ = ver_df.integrity.to_numpy()
    sig = ver_df.setting.to_numpy()
    below = np.flatnonzero(integ < 0.5)
    key["integrity_collapse_sigma"] = float(sig[below[0]]) if below.size else None
    lift = ver_df.mark_lift.to_numpy()
    neg = np.flatnonzero(lift < 0.0)
    key["mark_lift_inversion_sigma"] = float(sig[neg[0]]) if neg.size else None

    aso = iv.assortment_sweep(race, base, cfg.R_GRID, cfg.LIABILITY,
                              cfg.SOCIAL_HARM, "sml", **sml)
    aso_df = _sweep_frame(aso)
    aso_df.to_csv(tables_dir / "assortment_sweep.csv", index=False)
    key["wellmixed_unsafe"] = float(aso_df.unsafe.iloc[0])
    key["wellmixed_club_share"] = float(aso_df.club.iloc[0])

    for sigma_fine in (0.5, 0.9, 0.97):
        fs = iv.fine_sweep(
            race, replace(base, sigma=sigma_fine), cfg.RHO_GRID,
            cfg.LIABILITY, cfg.SOCIAL_HARM, "sml", **sml
        )
        _sweep_frame(fs).to_csv(
            tables_dir / f"fine_sweep_sigma{int(round(100 * sigma_fine)):03d}.csv",
            index=False,
        )

    dues = iv.dues_sweep(race, base, cfg.KAPPA_GRID, cfg.LIABILITY,
                         cfg.SOCIAL_HARM, "sml", **sml)
    _sweep_frame(dues).to_csv(tables_dir / "dues_sweep.csv", index=False)

    forge = iv.forgery_cost_sweep(
        race, replace(base, sigma=0.97), cfg.KAPPA_GRID,
        cfg.LIABILITY, cfg.SOCIAL_HARM, "sml", **sml
    )
    _sweep_frame(forge).to_csv(tables_dir / "forgery_cost_sweep.csv", index=False)

    lia = iv.liability_sweep(race, base, cfg.L_GRID, cfg.SOCIAL_HARM, "sml", **sml)
    _sweep_frame(lia).to_csv(tables_dir / "liability_sweep.csv", index=False)

    # ---------------------------------------------------------------- pools
    fun = build_functionals(race, base, cfg.LIABILITY, cfg.SOCIAL_HARM)
    pools = {
        "full": th.equilibrium(fun, race, "sml", **sml),
        "honest": th.equilibrium(honest_subspace(fun), race, "sml", **sml),
        "uncertified": th.equilibrium(unbadged_subspace(fun), race, "sml", **sml),
        "plain": th.equilibrium(plain_subspace(fun), race, "sml", **sml),
    }
    pd.DataFrame(
        [
            {
                "pool": name,
                "unsafe": eq.unsafe_frequency,
                "club": eq.class_distribution["certified club"],
                "falsebeard": eq.class_distribution["falsebeard"],
                "integrity": eq.attestation_integrity,
                "social": eq.social_payoff,
            }
            for name, eq in pools.items()
        ]
    ).to_csv(tables_dir / "pool_ablation.csv", index=False)
    key["honest_unsafe"] = pools["honest"].unsafe_frequency
    key["uncertified_unsafe"] = pools["uncertified"].unsafe_frequency
    key["plain_unsafe"] = pools["plain"].unsafe_frequency
    key["certification_value_baseline"] = (
        pools["uncertified"].unsafe_frequency - pools["full"].unsafe_frequency
    )
    key["forgery_toll_baseline"] = (
        pools["full"].unsafe_frequency - pools["honest"].unsafe_frequency
    )

    # equilibrium composition at three verification regimes
    comp_rows = []
    for sigma in (0.1, base.sigma, 0.97):
        eq = th.equilibrium(
            build_functionals(race, replace(base, sigma=sigma), cfg.LIABILITY,
                              cfg.SOCIAL_HARM),
            race, "sml", **sml,
        )
        top = np.argsort(eq.frequencies)[::-1][:6]
        labels = design_labels()
        for rank, k in enumerate(top, start=1):
            comp_rows.append(
                {
                    "sigma": sigma,
                    "rank": rank,
                    "design": labels[k],
                    "share": float(eq.frequencies[k]),
                }
            )
    pd.DataFrame(comp_rows).to_csv(tables_dir / "equilibrium_designs.csv", index=False)

    # ---------------------------------------------------------------- decomposition
    dec = iv.detection_decomposition(
        race, replace(base, sigma=0.9), rho=20.0,
        liability=cfg.LIABILITY, social_harm=cfg.SOCIAL_HARM, method="sml", **sml
    )
    pd.DataFrame(
        [
            {
                "cell": name,
                "unsafe": eq.unsafe_frequency,
                "club": eq.class_distribution["certified club"],
                "falsebeard": eq.class_distribution["falsebeard"],
                "integrity": eq.attestation_integrity,
            }
            for name, eq in (
                ("neither", dec.neither),
                ("screening_only", dec.screening_only),
                ("fines_only", dec.fines_only),
                ("both", dec.both),
            )
        ]
    ).to_csv(tables_dir / "detection_decomposition.csv", index=False)
    key["decomposition_screening_effect"] = dec.screening_effect
    key["decomposition_fines_effect"] = dec.fines_effect
    key["decomposition_interaction"] = dec.interaction
    key["decomposition_total"] = dec.total_effect

    # channel effects along the sigma grid (for the figure)
    chan_rows = []
    for sigma in cfg.SIGMA_GRID[::5]:
        d = iv.detection_decomposition(
            race, replace(base, sigma=float(sigma)), rho=20.0,
            liability=cfg.LIABILITY, social_harm=cfg.SOCIAL_HARM, method="sml", **sml
        )
        row: dict[str, float] = {"sigma": float(sigma)}
        for name, eq in (
            ("neither", d.neither),
            ("screening_only", d.screening_only),
            ("fines_only", d.fines_only),
            ("both", d.both),
        ):
            row[f"{name}_unsafe"] = eq.unsafe_frequency
            row[f"{name}_integrity"] = eq.attestation_integrity
            row[f"{name}_club"] = eq.class_distribution["certified club"]
            row[f"{name}_falsebeard"] = eq.class_distribution["falsebeard"]
        chan_rows.append(row)
    pd.DataFrame(chan_rows).to_csv(tables_dir / "detection_channels.csv", index=False)

    # ------------------------------------- bistability and the entry barrier
    # The basin split is the one estimated quantity in the study, so it gets its
    # own and much larger sample; everything else here is a mean over the same
    # draws and converges far sooner.
    rep = dict(n_starts=cfg.BASIN_STARTS, seed=cfg.SEED)
    ends = replicator_attractors(fun.fitness, **rep)
    basins = th.equilibrium(fun, race, "replicator", ends=ends, **rep)
    mono = pools["full"]
    carries = np.array([b != "N" for b in fun.badge], dtype=float)
    is_certified = ends @ carries > 0.5
    rows = []
    for tag, mask in (("certified", is_certified), ("uncertified", ~is_certified)):
        face = ends[mask]
        rows.append(
            {
                "regime": tag,
                "basin_share": float(mask.mean()),
                "unsafe": float(
                    np.mean([aggregate_unsafe_frequency(e, fun) for e in face])
                ),
                "social": float(np.mean([mean_social_payoff(e, fun) for e in face])),
                "n_attractors": int(len(np.unique(np.round(face, 6), axis=0))),
            }
        )
    faces = pd.DataFrame(rows)
    faces.to_csv(tables_dir / "bistability.csv", index=False)
    key["certified_basin_share"] = float(is_certified.mean())

    # A basin share is a sample proportion, not a property of the flow, so it
    # ships with the interval that says how much of its second digit is real.
    lo, hi = _wilson(int(is_certified.sum()), len(ends))
    key["basin_starts"] = int(len(ends))
    key["certified_basin_wilson_lo"] = lo
    key["certified_basin_wilson_hi"] = hi

    # No draw has ever landed between the two faces, but "no badge-mixed
    # attractor" is a claim about the sample and is recorded as one.
    badged_mass = ends @ carries
    key["badge_mixed_end_states"] = int(
        ((badged_mass > 1e-6) & (badged_mass < 1 - 1e-6)).sum()
    )

    # The dominant design of each face is quoted in the text.  Its spread across
    # the face is quoted too: the faces are continua of neutrally stable rest
    # points, so a mean alone reads as if one composition were selected.
    for tag, mask, design in (
        ("certified", is_certified, ("G", "CS", "CAS")),
        ("uncertified", ~is_certified, ("N", "CAS", "CS")),
    ):
        col = fun.designs.index(design)
        share = ends[mask][:, col]
        key[f"{tag}_face_dominant_mean"] = float(share.mean())
        key[f"{tag}_face_dominant_min"] = float(share.min())
        key[f"{tag}_face_dominant_max"] = float(share.max())

    # The residual the data statement quotes as evidence that every end state is
    # a rest point.  max_i (f_i - fbar) is the right statistic; the replicator
    # field itself is automatically tiny wherever x_i is.
    growth = ends @ fun.fitness.T
    growth = growth - (ends * growth).sum(axis=1, keepdims=True)
    key["attractor_max_growth"] = float(growth.max())
    key["basin_unsafe_max"] = float(
        max(aggregate_unsafe_frequency(e, fun) for e in ends)
    )
    key["basin_unsafe_mean"] = basins.unsafe_frequency
    key["certified_face_social"] = float(faces.loc[0, "social"])
    key["uncertified_face_social"] = float(faces.loc[1, "social"])
    key["settled_market_dues_loss"] = (
        key["uncertified_face_social"] - key["certified_face_social"]
    )
    key["n_attractors"] = basins.n_attractors
    # the artefact this replaced, kept so the correction is auditable
    key["unsafe_at_the_mean_state"] = float(
        aggregate_unsafe_frequency(ends.mean(axis=0), fun)
    )

    # A fully specified targeted construction probes a part of the simplex that
    # the uniform sample does not reach.  It is intentionally reported as a
    # numerical reachability and robustness result, not as a basin-volume
    # estimate or proof of a third isolated attractor.
    target_index = fun.index(*cfg.TARGETED_DESIGN)
    target_x0 = focal_mass_start(fun.n, target_index, cfg.TARGETED_MASS)
    target_times, target_traj = integrate_replicator(
        fun.fitness, target_x0, t_end=3000.0, n_points=301
    )
    target_end = target_traj[-1]
    target_fit = fun.fitness @ target_end
    target_mean_fit = float(target_end @ target_fit)
    target_growth = target_fit - target_mean_fit
    target_badged = float(target_end @ carries)
    target_unsafe = aggregate_unsafe_frequency(target_end, fun)
    target_field_inf = float(np.abs(replicator_field(target_end, fun.fitness)).max())

    endpoint_rows = []
    for j, (badge, s_in, s_out) in enumerate(fun.designs):
        endpoint_rows.append(
            {
                "design_index": j,
                "badge": badge,
                "s_in": s_in,
                "s_out": s_out,
                "initial_share": target_x0[j],
                "endpoint_share": target_end[j],
                "invasion_growth": target_growth[j],
            }
        )
    pd.DataFrame(endpoint_rows).to_csv(
        tables_dir / "targeted_mixed_endpoint.csv", index=False
    )

    trajectory = {
        "time": target_times,
        "unsafe": [aggregate_unsafe_frequency(x, fun) for x in target_traj],
        "badged_mass": target_traj @ carries,
        "focal_forger_mass": target_traj[:, target_index],
    }
    for j, (badge, s_in, s_out) in enumerate(fun.designs):
        trajectory[f"share_{j:02d}_{badge}_{s_in}_{s_out}"] = target_traj[:, j]
    pd.DataFrame(trajectory).to_csv(
        tables_dir / "targeted_mixed_trajectory.csv", index=False
    )

    perturbed_starts = full_dimensional_perturbations(
        target_x0,
        cfg.TARGETED_PERTURBATIONS,
        cfg.TARGETED_EPSILON,
        cfg.TARGETED_SEED,
    )
    perturbed_ends = np.vstack(
        [replicator_attractor(fun.fitness, x) for x in perturbed_starts]
    )
    perturbation_rows = []
    for k, (start, end) in enumerate(zip(perturbed_starts, perturbed_ends)):
        fit = fun.fitness @ end
        growth_k = fit - float(end @ fit)
        perturbation_rows.append(
            {
                "probe": k,
                "epsilon": cfg.TARGETED_EPSILON,
                "l1_distance_from_target": float(np.abs(start - target_x0).sum()),
                "unsafe": aggregate_unsafe_frequency(end, fun),
                "badged_mass": float(end @ carries),
                "field_inf": float(np.abs(replicator_field(end, fun.fitness)).max()),
                "external_growth_max": float(growth_k.max()),
            }
        )
    perturbations = pd.DataFrame(perturbation_rows)
    perturbations.to_csv(
        tables_dir / "targeted_mixed_perturbations.csv", index=False
    )

    key["targeted_design"] = "-".join(cfg.TARGETED_DESIGN)
    key["targeted_initial_mass"] = cfg.TARGETED_MASS
    key["targeted_residual_per_design"] = float((1.0 - cfg.TARGETED_MASS) / (fun.n - 1))
    key["targeted_endpoint_unsafe"] = target_unsafe
    key["targeted_endpoint_badged_mass"] = target_badged
    key["targeted_endpoint_field_inf"] = target_field_inf
    key["targeted_endpoint_external_growth_max"] = float(target_growth.max())
    key["targeted_endpoint_support_size"] = int((target_end > 1e-8).sum())
    key["targeted_perturbation_count"] = cfg.TARGETED_PERTURBATIONS
    key["targeted_perturbation_epsilon"] = cfg.TARGETED_EPSILON
    key["targeted_perturbation_unsafe_count"] = int(
        (perturbations.unsafe > 1.0 - 1e-9).sum()
    )
    key["targeted_perturbation_unsafe_min"] = float(perturbations.unsafe.min())
    key["targeted_perturbation_badged_mass_min"] = float(
        perturbations.badged_mass.min()
    )
    key["targeted_perturbation_badged_mass_max"] = float(
        perturbations.badged_mass.max()
    )
    key["targeted_perturbation_external_growth_max"] = float(
        perturbations.external_growth_max.max()
    )

    dec = mono.harm_blocks
    pd.DataFrame(
        [
            {
                "block": b,
                "mass": dec.mass[b],
                "contribution": dec.contribution[b],
                "conditional": dec.conditional[b],
            }
            for b in th.BLOCKS
        ]
    ).to_csv(tables_dir / "harm_blocks_monomorphic.csv", index=False)

    penalty, boundary = th.entry_barrier(race, base, cfg.LIABILITY, "CS", "CAS")
    key["entry_penalty"] = penalty
    key["boundary_unsafe"] = boundary
    gentle_penalty, gentle_boundary = th.entry_barrier(
        race, base, cfg.LIABILITY, "CS", "CS"
    )
    key["gentle_entry_penalty"] = gentle_penalty
    key["gentle_boundary_unsafe"] = gentle_boundary

    scan = th.out_group_policy_scan(
        race, base, "CS", cfg.LIABILITY, cfg.SOCIAL_HARM,
        cfg.POPULATION, cfg.BETA, cfg.REPLICATOR_STARTS, cfg.SEED,
    )
    pd.DataFrame(
        [
            {
                "s_out": p.s_out,
                "spoof_threshold": p.spoof_threshold,
                "mimic_threshold": p.mimic_threshold,
                "nucleation_threshold": p.nucleation_threshold,
                "entry_penalty": p.entry_penalty,
                "boundary_unsafe": p.boundary_unsafe,
                "certified_basin_share": p.certified_basin_share,
                "club_share_monomorphic": p.club_share_monomorphic,
                "unsafe_basin_mean": p.unsafe_basin_mean,
                "unsafe_monomorphic": p.unsafe_monomorphic,
            }
            for p in scan
        ]
    ).to_csv(tables_dir / "out_group_policy.csv", index=False)
    by_out = {p.s_out: p for p in scan}
    key["soft_club_spoof_threshold"] = by_out["CS"].spoof_threshold
    key["soft_club_entry_penalty"] = by_out["CS"].entry_penalty
    key["soft_club_nucleation"] = by_out["CS"].nucleation_threshold
    key["harsh_club_spoof_threshold"] = by_out["CAS"].spoof_threshold
    key["harsh_club_entry_penalty"] = by_out["CAS"].entry_penalty
    key["harsh_club_share"] = by_out["CAS"].club_share_monomorphic

    # can a fine make a gentle club viable?  the barrier, not the purpose
    rescue = th.gentle_club_rescue(
        race, base, (0.0, 5.0, 10.0, 20.0, 40.0, 60.0, 100.0), "CS",
        cfg.LIABILITY, cfg.SOCIAL_HARM, cfg.POPULATION, cfg.BETA,
        cfg.REPLICATOR_STARTS, cfg.SEED,
    )
    pd.DataFrame(
        [
            {
                "rho": row.rho,
                "spoof_threshold": row.spoof_threshold,
                "club_share_mixed": row.club_share_mixed,
                "club_share_monomorphic": row.club_share_monomorphic,
                "unsafe_mixed": row.unsafe_mixed,
                "unsafe_monomorphic": row.unsafe_monomorphic,
                "integrity": row.attestation_integrity,
            }
            for row in rescue
        ]
    ).to_csv(tables_dir / "gentle_club_rescue.csv", index=False)
    by_rho = {row.rho: row for row in rescue}
    key["gentle_spoof_threshold_no_fine"] = by_rho[0.0].spoof_threshold
    key["gentle_spoof_threshold_fine60"] = by_rho[60.0].spoof_threshold
    key["gentle_club_share_max"] = max(row.club_share_monomorphic for row in rescue)
    key["gentle_club_unsafe_mixed"] = by_rho[60.0].unsafe_mixed
    key["gentle_integrity_fine60"] = by_rho[60.0].attestation_integrity
    key["gentle_integrity_no_fine"] = by_rho[0.0].attestation_integrity

    # what the fine buys in the full design space, which is the constructive
    # half of the corollary
    for rho in (0.0, 60.0):
        eq_fine = th.equilibrium(
            build_functionals(race, replace(base, rho=rho), cfg.LIABILITY,
                              cfg.SOCIAL_HARM),
            race, "sml", **sml,
        )
        tag = "no_fine" if rho == 0.0 else "fine60"
        key[f"full_space_social_{tag}"] = eq_fine.social_payoff
        key[f"full_space_unsafe_{tag}"] = eq_fine.unsafe_frequency
        key[f"full_space_integrity_{tag}"] = eq_fine.attestation_integrity

    # ---------------------------------------------------------------- providers
    prov_rows = []
    for k in (1, 2, 3, 4, 6, 8, 12, 20):
        f = th.provider_frontier(race, 0.0, "CS", "CAS", base.kappa_g, k)
        prov_rows.append(
            {
                "providers": k,
                "hhi": f.hhi,
                "unsafe_scoped": f.unsafe_frequency,
                "unsafe_federated": f.federated_unsafe_frequency,
                "member_payoff": f.member_payoff,
                "entrant_payoff": f.entrant_payoff,
                "exclusion_rent": f.exclusion_rent,
            }
        )
    pd.DataFrame(prov_rows).to_csv(tables_dir / "provider_frontier.csv", index=False)
    f2 = th.provider_frontier(race, 0.0, "CS", "CAS", base.kappa_g, 2)
    f8 = th.provider_frontier(race, 0.0, "CS", "CAS", base.kappa_g, 8)
    f1 = th.provider_frontier(race, 0.0, "CS", "CAS", base.kappa_g, 1)
    key["duopoly_unsafe"] = f2.unsafe_frequency
    key["eight_provider_unsafe"] = f8.unsafe_frequency
    key["monopoly_rent"] = f1.exclusion_rent
    key["eight_provider_rent"] = f8.exclusion_rent

    # ---------------------------------------------------------------- process
    key["population_size"] = cfg.POPULATION
    key["beta"] = cfg.BETA
    key["sigma_baseline"] = base.sigma
    key["r_baseline"] = base.r
    key["kappa_g"] = base.kappa_g
    key["kappa_f"] = base.kappa_f
    key["social_harm"] = cfg.SOCIAL_HARM
    key["liability_baseline"] = cfg.LIABILITY
    key["club_dues_share_of_club_payoff"] = base.kappa_g / (
        key["payoff_cs_cs"] - base.kappa_g
    )

    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        raise TypeError(type(o))

    (outdir / "key_numbers.json").write_text(
        json.dumps(key, indent=2, sort_keys=True, default=_default)
    )
    print(f"wrote {outdir / 'key_numbers.json'} with {len(key)} entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=ROOT / "results")
    main(parser.parse_args().outdir)

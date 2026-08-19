"""Verify every scalar quoted in the manuscript against the computed results.

Usage::

    python scripts/check_numbers.py

The manuscript quotes about sixty numbers.  Any one of them can drift when a
parameter changes, and a drifted number is the kind of error a referee finds
and an author does not.  This script pairs each claim in the text with the
entry of ``results/key_numbers.json`` it came from and checks that the two
agree to the precision the text prints.

Exit status is non-zero if any claim fails, so the check can gate a build.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: ``(key in key_numbers.json, printed value, decimals printed)``.  A claim is
#: satisfied when the stored value rounds to the printed one at the stated
#: number of decimals.
CLAIMS: list[tuple[str, float, int]] = [
    # the race and its reciprocity threshold
    ("reciprocity_threshold", 0.551, 3),
    ("first_strike_threshold_as", 42.765, 3),
    ("expected_horizon", 9.0, 1),
    ("payoff_cs_cs", 59.00, 2),
    ("payoff_cas_cs", 61.63, 2),
    ("uncertified_unsafe_L0_r0", 1.000, 3),
    # spoof thresholds
    ("spoof_threshold_cs_club", 0.866, 3),
    ("spoof_threshold_as_club", 0.400, 3),
    ("spoof_threshold_ratio", 2.17, 2),
    ("spoof_threshold_forger_assorted", 0.962, 3),
    # fines and dues
    ("required_fine_090", 11.9, 1),
    ("required_fine_095", 58.2, 1),
    ("required_fine_099", 428.6, 1),
    ("dues_gradient", -0.029, 3),
    # nucleation
    ("nucleation_threshold", 0.063, 3),
    # mimicry and the ordering reversal
    ("mimic_threshold", 0.938, 3),
    ("invader_ordering_flip", 0.077, 3),
    ("exploiter_immunity_assortment", 0.134, 3),
    ("sigma_exploiter_r0", 0.866, 3),
    ("sigma_mimic_r0", 0.938, 3),
    ("mimic_threshold_span_over_assortment", 0.001, 3),
    ("spoof_threshold_mimic_assorted", 0.938, 3),
    ("integrity_collapse_sigma", 0.93, 2),
    ("mark_lift_inversion_sigma", 0.57, 2),
    ("free_forgery_integrity", 0.046, 3),
    # the baseline equilibrium
    ("baseline_unsafe", 0.090, 3),
    ("uncertified_unsafe", 0.159, 3),
    ("honest_unsafe", 0.068, 3),
    ("baseline_social", 39.1, 1),
    # providers
    ("duopoly_unsafe", 0.500, 3),
    ("eight_provider_unsafe", 0.875, 3),
    ("monopoly_rent", 30.4, 1),
    ("eight_provider_rent", 2.5, 1),
    # bistability and the entry barrier
    ("certified_basin_share", 0.44, 2),
    ("certified_face_social", 57.0, 1),
    ("uncertified_face_social", 59.0, 1),
    ("settled_market_dues_loss", 2.0, 1),
    ("entry_penalty", 30.36, 2),
    ("boundary_unsafe", 0.461, 3),
    ("gentle_entry_penalty", -2.00, 2),
    ("gentle_boundary_unsafe", 0.000, 3),
    ("harsh_club_spoof_threshold", 0.962, 3),
    ("harsh_club_entry_penalty", 30.36, 2),
    ("harsh_club_share", 1.000, 3),
    ("soft_club_spoof_threshold", 0.000, 3),
    ("soft_club_entry_penalty", -2.00, 2),
    # the artefact the corrected reading replaced, quoted in the paper
    ("unsafe_at_the_mean_state", 0.379, 3),
    # Corollary 9: the fine restores the barrier, not the purpose
    ("gentle_spoof_threshold_no_fine", 0.000, 3),
    ("gentle_spoof_threshold_fine60", 0.923, 3),
    ("gentle_club_share_max", 0.032, 3),
    ("gentle_integrity_no_fine", 0.16, 2),
    ("gentle_integrity_fine60", 0.98, 2),
    ("full_space_social_no_fine", 39.1, 1),
    ("full_space_social_fine60", 43.6, 1),
]

#: Claims read from the robustness summary instead.
ROBUSTNESS_CLAIMS: list[tuple[str, float, int]] = [
    ("mean_value_below_lr", 0.054, 3),
    ("mean_value_above_lr", 0.002, 3),
    ("replicator_integrity_correlation", 0.965, 3),
    ("process_unsafe_min", 0.044, 3),
    ("process_unsafe_max", 0.330, 3),
    ("value_share_above_001", 0.15, 2),
]

#: Claims that live in a results table rather than in a summary file:
#: ``(csv, row filter, column, printed value, decimals)``.
TABLE_CLAIMS: list[tuple[str, dict, str, float, int]] = [
    ("out_group_policy.csv", {"s_out": "CS"}, "spoof_threshold", 0.000, 3),
    ("out_group_policy.csv", {"s_out": "AS"}, "spoof_threshold", 0.000, 3),
    ("out_group_policy.csv", {"s_out": "CAS"}, "entry_penalty", 30.36, 2),
    ("out_group_policy.csv", {"s_out": "AU"}, "entry_penalty", 40.40, 2),
    ("out_group_policy.csv", {"s_out": "CS"}, "entry_penalty", -2.00, 2),
    ("out_group_policy.csv", {"s_out": "AS"}, "entry_penalty", -2.00, 2),
    ("out_group_policy.csv", {"s_out": "CAS"}, "boundary_unsafe", 0.461, 3),
    ("out_group_policy.csv", {"s_out": "AU"}, "boundary_unsafe", 0.868, 3),
    ("out_group_policy.csv", {"s_out": "CAS"}, "club_share_monomorphic", 1.000, 3),
    ("out_group_policy.csv", {"s_out": "AU"}, "certified_basin_share", 0.395, 3),
    ("out_group_policy.csv", {"s_out": "CAS"}, "certified_basin_share", 0.885, 3),
    ("out_group_policy.csv", {"s_out": "AU"}, "club_share_monomorphic", 1.000, 3),
    ("out_group_policy.csv", {"s_out": "CS"}, "club_share_monomorphic", 0.020, 3),
    ("out_group_policy.csv", {"s_out": "AS"}, "club_share_monomorphic", 0.005, 3),
    ("bistability.csv", {"regime": "certified"}, "basin_share", 0.44, 2),
    ("bistability.csv", {"regime": "certified"}, "social", 57.0, 1),
    ("bistability.csv", {"regime": "uncertified"}, "social", 59.0, 1),
    ("detection_decomposition.csv", {"cell": "neither"}, "integrity", 0.05, 2),
    ("detection_decomposition.csv", {"cell": "screening_only"}, "integrity", 0.68, 2),
    ("detection_decomposition.csv", {"cell": "fines_only"}, "integrity", 0.53, 2),
    ("detection_decomposition.csv", {"cell": "both"}, "integrity", 0.94, 2),
]


def _check(label: str, stored: float, printed: float, decimals: int) -> str | None:
    if stored is None:
        return f"{label}: MISSING from the results"
    if round(float(stored), decimals) != round(printed, decimals):
        return (
            f"{label}: manuscript prints {printed:.{decimals}f}, "
            f"results give {float(stored):.{decimals + 2}f}"
        )
    return None


def main() -> int:
    results = ROOT / "results"
    key = json.loads((results / "key_numbers.json").read_text())
    robustness = json.loads((results / "robustness_summary.json").read_text())

    failures: list[str] = []
    checked = 0

    for name, printed, decimals in CLAIMS:
        failures.append(_check(name, key.get(name), printed, decimals))
        checked += 1
    for name, printed, decimals in ROBUSTNESS_CLAIMS:
        failures.append(_check(name, robustness.get(name), printed, decimals))
        checked += 1

    import pandas as pd

    for csv, where, column, printed, decimals in TABLE_CLAIMS:
        frame = pd.read_csv(results / "tables" / csv)
        for field, value in where.items():
            frame = frame[frame[field] == value]
        label = f"{csv}[{where}].{column}"
        if frame.empty:
            failures.append(f"{label}: no matching row")
        else:
            failures.append(
                _check(label, float(frame[column].iloc[0]), printed, decimals)
            )
        checked += 1

    # the composition quoted at the hollowed-out verification quality
    for column, printed, decimals in (
        ("falsebeard", 0.55, 2),
        ("club", 0.02, 2),
        ("integrity", 0.05, 2),
        ("unsafe", 0.058, 3),
    ):
        failures.append(_sweep_claim(results, 0.97, column, printed, decimals))
        checked += 1

    problems = [f for f in failures if f]
    for problem in problems:
        print(f"FAIL  {problem}")
    print(f"\n{checked - len(problems)}/{checked} quoted numbers verified")
    return 1 if problems else 0


def _sweep_claim(
    results: Path, sigma: float, column: str, printed: float, decimals: int
) -> str | None:
    """One observable at a stated point of the verification sweep."""
    import pandas as pd

    frame = pd.read_csv(results / "tables" / "verification_sweep.csv")
    row = frame.iloc[(frame.setting - sigma).abs().argmin()]
    return _check(
        f"verification_sweep[sigma={sigma}].{column}",
        float(row[column]),
        printed,
        decimals,
    )


if __name__ == "__main__":
    sys.exit(main())

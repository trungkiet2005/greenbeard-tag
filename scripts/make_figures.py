"""Render every figure of the manuscript from the saved results.

Usage::

    python scripts/make_figures.py [--outdir results]

Requires ``run_analysis.py`` and ``run_robustness.py`` to have written
``results/tables`` and ``results/grids.npz``.  Every figure is saved at the
standard width so the whole set is reduced by one factor on the page.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gbtag import config as cfg
from gbtag import theory as th
from gbtag.dynamics import replicator_attractors
from gbtag.functionals import aggregate_unsafe_frequency, build_functionals
from gbtag.identity import CLASSES
from gbtag.plotting import (
    FS,
    PALETTE,
    fitted_legend,
    new_figure,
    panel_title,
    save,
    use_paper_style,
)
from gbtag.race import STRATEGIES, build_race_tables

ROOT = Path(__file__).resolve().parents[1]

CLASS_COLOURS = {c: PALETTE[c] for c in CLASSES}


# --------------------------------------------------------------------------
# fig02: the race rewards the first strike
# --------------------------------------------------------------------------


def fig02(tables, results: Path, figdir: Path) -> None:
    df = pd.read_csv(results / "tables" / "uncertified_by_liability.csv")
    p0 = th.race_private(tables, 0.0)
    lr = th.reciprocity_threshold(tables)

    fig, axes = new_figure(2.65, ncols=2)

    ax = axes[0]
    ax.grid(False)
    ax.imshow(p0, cmap="RdYlBu", vmin=0.0, vmax=105.0)
    ax.set_xticks(range(4), STRATEGIES)
    ax.set_yticks(range(4), STRATEGIES)
    ax.set_xlabel("opponent design")
    ax.set_ylabel("focal design")
    for i in range(4):
        for j in range(4):
            ax.text(
                j,
                i,
                f"{p0[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=FS["tiny"],
                color="white" if 25 < p0[i, j] < 80 else "black",
            )
    # the two cells of the handshake exploit
    for (i, j) in ((2, 1), (1, 1)):
        ax.add_patch(
            plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1, fill=False,
                edgecolor="black", linewidth=1.6,
            )
        )
    panel_title(ax, "A", "race payoff at zero liability")

    ax = axes[1]
    for r, colour, label in (
        (0.0, PALETTE["anarchy"], "well mixed ($r = 0$)"),
        (0.1, PALETTE["club"], "clustered ($r = 0.1$)"),
    ):
        sub = df[df.r == r].sort_values("liability")
        ax.plot(sub.liability, sub.unsafe, color=colour, label=label)
    ax.axvline(lr, color=PALETTE["unsafe"], linestyle="--", linewidth=1.0)
    ax.text(
        lr + 0.12, 0.93, "$L_R = 0.551$", fontsize=FS["annot"],
        color=PALETTE["unsafe"], ha="left", va="top",
    )
    ax.axvspan(0.0, lr, color=PALETTE["unsafe"], alpha=0.06)
    ax.set_xlabel("liability $L$ per unsafe action")
    ax.set_ylabel("long-run unsafe frequency")
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.02, 1.02)
    fitted_legend(ax, loc="upper right")
    panel_title(ax, "B", "the uncertified world by liability")

    save(fig, figdir / "fig02_handshake")


# --------------------------------------------------------------------------
# fig03: the invasion structure of the certified club
# --------------------------------------------------------------------------


def _advantage(tables, params, invader, resident):
    return th.fitness_against_resident(
        tables, params, 0.0, invader, resident
    ) - th.pair_payoff(tables, params, 0.0, resident, resident)


def fig03(tables, results: Path, figdir: Path) -> None:
    sigmas = np.linspace(0.0, 1.0, 201)
    base = cfg.IDENTITY

    fig, axes = new_figure(2.55, ncols=3)

    for ax, r, letter in ((axes[0], 0.0, "A"), (axes[1], base.r, "B")):
        params = replace(base, r=r)
        marks = []
        for invader, colour, label in (
            (cfg.FALSEBEARD, PALETTE["forger"], "exploiter F:CAS/CAS"),
            (cfg.MIMIC, PALETTE["mimic"], "mimic F:CS/CAS"),
        ):
            adv = [
                _advantage(tables, replace(params, sigma=float(s)), invader, cfg.CLUB)
                for s in sigmas
            ]
            ax.plot(sigmas, adv, color=colour, label=label)
            s_star = th.spoof_threshold(tables, params, 0.0, cfg.CLUB, invader)
            if s_star is not None and 0.0 < s_star < 1.0:
                ax.axvline(s_star, color=colour, linestyle=":", linewidth=0.9)
                marks.append((s_star, colour))
        ax.axhline(0.0, color=PALETTE["neutral"], linewidth=0.8)
        # the thresholds differ by 0.024 at the baseline, which is invisible on
        # a unit axis, so the crossing order is stated rather than left to the eye
        # the curves run bottom-left to top-right, so the bottom-right corner
        # is the only region of the panel with no ink in it
        for k, (s_star, colour) in enumerate(sorted(marks)):
            ax.text(
                0.97,
                -22.0 - 6.5 * k,
                rf"$\sigma^{{*}} = {s_star:.3f}$",
                fontsize=FS["tiny"],
                color=colour,
                ha="right",
                va="center",
            )
        ax.set_xlabel(r"spoof success $\sigma$")
        ax.set_ylim(-35, 45)
        title = "well mixed ($r = 0$)" if r == 0.0 else f"clustered ($r = {r}$)"
        panel_title(ax, letter, title)
    axes[0].set_ylabel("invader advantage in the club")
    fitted_legend(axes[0], loc="upper left", fontsize=FS["tiny"])

    # panel C: the two thresholds against assortment, which is where the
    # crossing lives and where a unit sigma axis cannot show it
    ax = axes[2]
    order = pd.read_csv(results / "tables" / "invader_ordering.csv")
    ax.plot(order.r, order.sigma_exploiter, color=PALETTE["forger"],
            label="exploiter")
    ax.plot(order.r, order.sigma_mimic, color=PALETTE["mimic"], label="mimic")
    flip = th.invader_ordering_flip(
        tables, base, cfg.LIABILITY, cfg.CLUB, cfg.FALSEBEARD, cfg.MIMIC
    )
    immune = th.exploiter_immunity(
        tables, base, cfg.LIABILITY, cfg.CLUB, cfg.FALSEBEARD
    )
    if flip is not None:
        ax.axvline(flip, color=PALETTE["neutral"], linestyle="--", linewidth=0.9)
        ax.text(flip - 0.006, 1.015, f"$r = {flip:.3f}$", fontsize=FS["tiny"],
                color=PALETTE["neutral"], ha="right", va="top")
    if immune is not None:
        ax.axvspan(immune, order.r.max(), color=PALETTE["forger"], alpha=0.10)
        ax.text(
            (immune + float(order.r.max())) / 2, 0.885,
            "exploiter\nlocked out", fontsize=FS["tiny"],
            color=PALETTE["forger"], ha="center", va="center",
        )
    ax.set_xlabel("assortment $r$")
    ax.set_ylabel(r"spoof threshold $\sigma^{*}$")
    ax.set_xlim(0, float(order.r.max()))
    ax.set_ylim(0.65, 1.03)
    fitted_legend(ax, loc="lower right", fontsize=FS["tiny"])
    panel_title(ax, "C", "clustering flips it")

    save(fig, figdir / "fig03_invasion")


# --------------------------------------------------------------------------
# fig04: the certification cliff
# --------------------------------------------------------------------------


def fig04(tables, results: Path, figdir: Path) -> None:
    df = pd.read_csv(results / "tables" / "verification_sweep.csv")
    pools = pd.read_csv(results / "tables" / "pool_ablation.csv").set_index("pool")
    base = cfg.IDENTITY
    sml = dict(population_size=cfg.POPULATION, beta=cfg.BETA)

    # class shares along the sweep have to be recomputed briefly: the sweep
    # table stores only club and falsebeard, so rebuild the five-way split
    shares = {c: [] for c in CLASSES}
    for s in df.setting:
        fun = build_functionals(
            tables, replace(base, sigma=float(s)), cfg.LIABILITY, cfg.SOCIAL_HARM
        )
        eq = th.equilibrium(fun, tables, "sml", **sml)
        for c in CLASSES:
            shares[c].append(eq.class_distribution[c])

    fig = plt.figure(figsize=(6.9, 4.7), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.03)
    grid = fig.add_gridspec(2, 2)
    ax_top = fig.add_subplot(grid[0, :])
    ax_u = fig.add_subplot(grid[1, 0])
    ax_m = fig.add_subplot(grid[1, 1])

    x = df.setting.to_numpy()
    stack = np.vstack([shares[c] for c in CLASSES])
    ax_top.stackplot(
        x, stack,
        labels=list(CLASSES),
        colors=[CLASS_COLOURS[c] for c in CLASSES],
        alpha=0.88,
        linewidth=0.0,
    )
    sig_m = th.mimic_threshold_closed_form(
        tables, 0.0, "CS", "CAS", base.kappa_g, base.kappa_f, 0.0
    )
    ax_top.axvline(sig_m, color="black", linestyle="--", linewidth=1.0)
    ax_top.text(
        sig_m - 0.015, 0.5, r"$\sigma_m = 0.938$", fontsize=FS["annot"],
        rotation=90, ha="right", va="center", color="black",
    )
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)
    ax_top.set_xlabel(r"spoof success $\sigma$")
    ax_top.set_ylabel("share of the population")
    fitted_legend(ax_top, loc="center left", ncol=1, fontsize=FS["annot"])
    panel_title(ax_top, "A", "long-run composition at the baseline")

    ax_u.plot(x, df.unsafe, color=PALETTE["unsafe"], label="certified world")
    ax_u.axhline(
        pools.loc["uncertified", "unsafe"], color=PALETTE["anarchy"],
        linestyle="--", linewidth=1.0, label="uncertified world",
    )
    ax_u.axhline(
        pools.loc["honest", "unsafe"], color=PALETTE["safe"],
        linestyle=":", linewidth=1.0, label="unforgeable badges",
    )
    ax_u.set_xlabel(r"spoof success $\sigma$")
    ax_u.set_ylabel("unsafe frequency")
    ax_u.set_xlim(0, 1)
    ax_u.set_ylim(0.0, 0.24)
    fitted_legend(ax_u, loc="upper right", fontsize=FS["annot"])
    panel_title(ax_u, "B", "safety along the sweep")

    ax_m.plot(x, df.integrity, color=PALETTE["club"], label="attestation integrity")
    ax_m.plot(x, df.mark_lift, color=PALETTE["mimic"], label="mark lift")
    ax_m.axvline(sig_m, color="black", linestyle="--", linewidth=1.0)
    ax_m.axhline(0.0, color=PALETTE["neutral"], linewidth=0.8)
    ax_m.set_xlabel(r"spoof success $\sigma$")
    ax_m.set_ylabel("value of a passed check")
    ax_m.set_xlim(0, 1)
    ax_m.set_ylim(-0.15, 1.05)
    fitted_legend(ax_m, loc="center left", fontsize=FS["annot"])
    panel_title(ax_m, "C", "what the mark still means")

    save(fig, figdir / "fig04_cliff")


# --------------------------------------------------------------------------
# fig05: nucleation and hysteresis
# --------------------------------------------------------------------------


def fig05(tables, results: Path, figdir: Path) -> None:
    aso = pd.read_csv(results / "tables" / "assortment_sweep.csv")
    grids = np.load(results / "grids.npz")
    base = cfg.IDENTITY
    r_star = th.nucleation_threshold(tables, 0.0, "CS", "CAS", "CAS", base.kappa_g)

    fig, axes = new_figure(2.55, ncols=3)

    ax = axes[0]
    ax.plot(aso.setting, aso.unsafe, color=PALETTE["unsafe"], label="unsafe frequency")
    ax.plot(aso.setting, aso.club, color=PALETTE["club"], label="club share")
    ax.axvline(r_star, color=PALETTE["neutral"], linestyle="--", linewidth=1.0)
    ax.text(
        r_star + 0.008, 0.86, r"$r^{*} = 0.063$", fontsize=FS["annot"],
        color=PALETTE["neutral"], ha="left",
    )
    ax.set_xlabel("assortment $r$")
    ax.set_ylabel("share / frequency")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-0.02, 1.28)
    fitted_legend(ax, loc="upper right", fontsize=FS["annot"])
    panel_title(ax, "A", "clustering switches phase")

    ax = axes[1]
    ax.grid(False)
    im = ax.pcolormesh(
        grids["plane_sigmas"], grids["plane_rs"], grids["plane_club"],
        cmap="Blues", vmin=0.0, vmax=0.7, shading="nearest",
    )
    ax.axhline(r_star, color=PALETTE["unsafe"], linestyle="--", linewidth=1.0)
    sig_m = th.mimic_threshold_closed_form(
        tables, 0.0, "CS", "CAS", base.kappa_g, base.kappa_f, 0.0
    )
    ax.axvline(sig_m, color=PALETTE["mimic"], linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"spoof success $\sigma$")
    ax.set_ylabel("assortment $r$")
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("club share", fontsize=FS["annot"])
    cbar.ax.tick_params(labelsize=FS["tiny"])
    panel_title(ax, "B", "where the club can live")

    ax = axes[2]
    rs = np.linspace(0.0, 0.3, 121)
    build = [
        _advantage(tables, replace(base, r=float(r)), cfg.CLUB, cfg.ANARCHY)
        for r in rs
    ]
    keep = [
        -_advantage(tables, replace(base, r=float(r)), cfg.FALSEBEARD, cfg.CLUB)
        for r in rs
    ]
    ax.plot(rs, keep, color=PALETTE["club"], label="keeping the club")
    ax.plot(rs, build, color=PALETTE["forger"], label="building the club")
    ax.axhline(0.0, color=PALETTE["neutral"], linewidth=0.8)
    ax.axvline(r_star, color=PALETTE["neutral"], linestyle="--", linewidth=1.0)
    ax.set_xlabel("assortment $r$")
    ax.set_ylabel("selective advantage")
    ax.set_xlim(0, 0.3)
    ax.set_ylim(-3.0, 26.0)
    fitted_legend(ax, loc="upper left", fontsize=FS["annot"])
    panel_title(ax, "C", r"keep vs build ($\sigma = 0.5$)")

    save(fig, figdir / "fig05_nucleation")


# --------------------------------------------------------------------------
# fig06: the instruments
# --------------------------------------------------------------------------


def fig06(tables, results: Path, figdir: Path) -> None:
    req = pd.read_csv(results / "tables" / "required_fine.csv")
    dues = pd.read_csv(results / "tables" / "dues_sweep.csv")
    forge = pd.read_csv(results / "tables" / "forgery_cost_sweep.csv")

    fig, axes = new_figure(4.7, nrows=2, ncols=2)

    ax = axes[0, 0]
    for sigma, colour in ((0.5, PALETTE["safe"]), (0.9, PALETTE["CAS"]), (0.97, PALETTE["unsafe"])):
        fs = pd.read_csv(
            results / "tables" / f"fine_sweep_sigma{int(round(100 * sigma)):03d}.csv"
        )
        ax.plot(fs.setting, fs.falsebeard, color=colour, label=rf"$\sigma = {sigma}$")
    ax.set_xlabel(r"fine $\rho$ on a detected forgery")
    ax.set_ylabel("falsebeard share")
    ax.set_ylim(-0.02, 0.8)
    fitted_legend(ax, loc="upper right", fontsize=FS["annot"])
    panel_title(ax, "A", "fines suppress forgers")

    ax = axes[0, 1]
    positive = req[req.required_fine > 0.0]
    ax.plot(positive.sigma, positive.required_fine, color=PALETTE["unsafe"])
    ax.set_yscale("log")
    for s, v in ((0.90, 11.9), (0.95, 27.7), (0.99, 428.6)):
        row = req.iloc[(req.sigma - s).abs().argmin()]
        ax.plot([row.sigma], [row.required_fine], "o", color=PALETTE["neutral"],
                markersize=3.5)
    # the curve is low on the left and near-vertical on the right, so the
    # upper-left corner is the only part of the panel with no ink in it
    ax.annotate(
        r"$\rho^{*} \to \infty$",
        xy=(0.990, 260),
        xycoords="data",
        xytext=(0.04, 0.93),
        textcoords="axes fraction",
        fontsize=FS["annot"],
        color=PALETTE["unsafe"],
        ha="left",
        va="top",
        arrowprops=dict(arrowstyle="->", color=PALETTE["unsafe"], lw=0.8),
    )
    ax.set_xlabel(r"spoof success $\sigma$")
    ax.set_ylabel(r"smallest protective fine $\rho^{*}$")
    ax.set_xlim(0.85, 1.0)
    panel_title(ax, "B", "no fine substitutes for detection")

    ax = axes[1, 0]
    ax.plot(dues.setting, dues.club, color=PALETTE["club"], label="club share")
    sigmas_of_dues = [
        th.spoof_threshold_closed_form(tables, 0.0, "CS", "CAS", "CAS", float(k), 0.0, 0.0)
        for k in dues.setting
    ]
    ax2 = ax.twinx()
    ax2.plot(dues.setting, sigmas_of_dues, color=PALETTE["forger"], linestyle="--",
             label=r"spoof threshold $\sigma^{*}$")
    ax2.set_ylabel(r"$\sigma^{*}$", fontsize=FS["label"])
    ax2.tick_params(labelsize=FS["tick"])
    ax2.set_ylim(0.55, 1.0)
    ax2.grid(False)
    ax.set_xlabel(r"certification dues $\kappa_g$")
    ax.set_ylabel("club share")
    # both curves fall from the upper left, so the headroom above them is the
    # only region a two-entry legend fits without covering either
    ax.set_ylim(-0.02, 1.30)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right",
              fontsize=FS["annot"])
    panel_title(ax, "C", "dues weaken the club")

    ax = axes[1, 1]
    ax.plot(forge.setting, forge.falsebeard, color=PALETTE["forger"],
            label="falsebeard share")
    ax.plot(forge.setting, forge.club, color=PALETTE["club"], label="club share")
    ax.set_xlabel(r"forgery cost $\kappa_f$")
    ax.set_ylabel("share")
    # both curves stay below 0.56, so the headroom is the free region
    ax.set_ylim(-0.02, 0.92)
    fitted_legend(ax, loc="upper right", fontsize=FS["annot"])
    panel_title(ax, "D", r"forgery cost at $\sigma = 0.97$")

    save(fig, figdir / "fig06_instruments")


# --------------------------------------------------------------------------
# fig07: screening versus auditing
# --------------------------------------------------------------------------


def fig07(tables, results: Path, figdir: Path) -> None:
    dec = pd.read_csv(results / "tables" / "detection_decomposition.csv").set_index("cell")
    chan = pd.read_csv(results / "tables" / "detection_channels.csv")

    fig, axes = new_figure(2.65, ncols=2)

    cells = ["neither", "fines_only", "screening_only", "both"]
    cell_labels = ["neither", "fines\nonly", "screening\nonly", "both"]
    colours = [PALETTE["anarchy"], PALETTE["CAS"], PALETTE["safe"], PALETTE["club"]]

    ax = axes[0]
    values = [dec.loc[c, "integrity"] for c in cells]
    unsafe_vals = [dec.loc[c, "unsafe"] for c in cells]
    bars = ax.bar(cell_labels, values, color=colours, width=0.62)
    for bar, v, u in zip(bars, values, unsafe_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v + 0.02,
            f"{v:.2f}\n$U$={u:.3f}",
            ha="center", va="bottom", fontsize=FS["tiny"], linespacing=1.35,
        )
    ax.set_ylabel("attestation integrity")
    ax.set_ylim(0, 1.24)
    ax.tick_params(axis="x", labelsize=FS["annot"])
    panel_title(ax, "A", r"one detection, two channels ($\sigma = 0.9$)")

    ax = axes[1]
    for cell, colour, label in (
        ("neither", PALETTE["anarchy"], "neither"),
        ("fines_only", PALETTE["CAS"], "fines only (audit)"),
        ("screening_only", PALETTE["safe"], "screening only"),
        ("both", PALETTE["club"], "both"),
    ):
        ax.plot(chan.sigma, chan[f"{cell}_integrity"], color=colour, label=label)
    ax.set_xlabel(r"spoof success $\sigma$")
    ax.set_ylabel("attestation integrity")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.42)
    fitted_legend(ax, loc="upper left", fontsize=FS["annot"], ncol=2)
    panel_title(ax, "B", r"the channels along the sweep ($\rho = 20$)")

    save(fig, figdir / "fig07_channels")


# --------------------------------------------------------------------------
# fig01: the exclusion dividend (the page-one figure)
# --------------------------------------------------------------------------


def fig01(tables, results: Path, figdir: Path) -> None:
    policy = pd.read_csv(results / "tables" / "out_group_policy.csv")
    faces = pd.read_csv(results / "tables" / "bistability.csv").set_index("regime")
    key = json.loads((results / "key_numbers.json").read_text())

    fig, axes = new_figure(2.95, ncols=3)
    ax_a, ax_b, ax_c = axes

    # ---- A: the flow is bistable, and both regimes are safe
    fun = build_functionals(tables, cfg.IDENTITY, cfg.LIABILITY, cfg.SOCIAL_HARM)
    ends = replicator_attractors(fun.fitness, n_starts=cfg.REPLICATOR_STARTS,
                                 seed=cfg.SEED)
    carries = np.array([b != "N" for b in fun.badge], dtype=float)
    badged_mass = ends @ carries
    unsafe_each = np.array([aggregate_unsafe_frequency(e, fun) for e in ends])

    ax_a.scatter(
        badged_mass, np.maximum(unsafe_each, 1e-24),
        s=12, alpha=0.55, edgecolors="none",
        c=[PALETTE["club"] if m > 0.5 else PALETTE["unbadged cooperator"]
           for m in badged_mass],
    )
    ax_a.set_yscale("log")
    ax_a.set_ylim(1e-25, 1e2)
    ax_a.set_xlim(-0.08, 1.08)
    ax_a.axvspan(0.08, 0.92, color=PALETTE["neutral"], alpha=0.07)
    ax_a.text(0.5, 1e-12, "no attractor\nhere", ha="center", va="center",
              fontsize=FS["tiny"], color=PALETTE["neutral"])
    ax_a.text(0.02, 3e0, f"{1 - key['certified_basin_share']:.0%}\nunbadged",
              ha="left", va="top", fontsize=FS["tiny"],
              color=PALETTE["unbadged cooperator"])
    ax_a.text(0.98, 3e0, f"{key['certified_basin_share']:.0%}\nbadged",
              ha="right", va="top", fontsize=FS["tiny"], color=PALETTE["club"])
    ax_a.set_xlabel("badge-carrying share of the attractor")
    ax_a.set_ylabel("unsafe frequency")
    panel_title(ax_a, "A", "two regimes, both safe")

    # ---- B: what each regime is worth
    xs = np.arange(2)
    social = [faces.loc["uncertified", "social"], faces.loc["certified", "social"]]
    ax_b.bar(xs, social, 0.5,
             color=[PALETTE["unbadged cooperator"], PALETTE["club"]])
    for x, v in zip(xs, social):
        ax_b.text(x, 51.0, f"{v:.1f}", ha="center", va="bottom",
                  fontsize=FS["tiny"], color="white")
    ax_b.annotate(
        "", xy=(1, social[1]), xytext=(1, social[0]),
        arrowprops=dict(arrowstyle="<->", color=PALETTE["unsafe"], lw=0.9),
    )
    ax_b.text(1.20, 0.5 * (social[0] + social[1]),
              rf"$\kappa_g = {cfg.IDENTITY.kappa_g:.0f}$",
              fontsize=FS["tiny"], color=PALETTE["unsafe"], va="center",
              ha="left")
    ax_b.set_xticks(xs, ["uncertified\nregime", "certified\nregime"])
    ax_b.tick_params(axis="x", labelsize=FS["tiny"])
    ax_b.set_ylabel("social payoff")
    ax_b.set_ylim(50, 63)
    ax_b.set_xlim(-0.6, 1.7)
    panel_title(ax_b, "B", "the club costs its dues")

    # ---- C: the toll and the defence are the same policy
    pol = policy.set_index("s_out").loc[list(STRATEGIES)]
    xs = np.arange(4)
    ax_c.bar(xs - 0.19, pol.entry_penalty, 0.36, color=PALETTE["unsafe"],
             label="entry toll on an outsider")
    ax2 = ax_c.twinx()
    ax2.bar(xs + 0.19, pol.spoof_threshold, 0.36, color=PALETTE["club"],
            label=r"forgery tolerated, $\sigma^{*}$")
    ax2.set_ylim(0, 2.05)
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.set_ylabel(r"$\sigma^{*}$", fontsize=FS["label"])
    ax2.tick_params(labelsize=FS["tick"])
    ax2.grid(False)
    ax_c.axhline(0.0, color=PALETTE["neutral"], linewidth=0.8)
    ax_c.axvspan(-0.5, 1.5, color=PALETTE["neutral"], alpha=0.07)
    ax_c.text(0.5, 62, "no club\nsurvives", ha="center", va="center",
              fontsize=FS["tiny"], color=PALETTE["neutral"])
    ax_c.set_xticks(xs, list(STRATEGIES))
    ax_c.set_xlabel("outsider conduct")
    ax_c.set_ylabel("entry toll (payoff)")
    ax_c.set_xlim(-0.5, 3.5)
    ax_c.set_ylim(-8, 88)
    handles = ax_c.containers[0:1] + ax2.containers[0:1]
    ax_c.legend(handles, [h.get_label() for h in handles],
                loc="upper left", fontsize=FS["tiny"])
    panel_title(ax_c, "C", "one policy buys both")

    save(fig, figdir / "fig01_exclusion")


# --------------------------------------------------------------------------
# fig08: provider-scoped marks and the concentration frontier
# --------------------------------------------------------------------------


def fig08(tables, results: Path, figdir: Path) -> None:
    prov = pd.read_csv(results / "tables" / "provider_frontier.csv")

    fig, axes = new_figure(2.65, ncols=2)

    ax = axes[0]
    hhi = np.linspace(0.02, 1.0, 200)
    ax.plot(hhi, 1.0 - hhi, color=PALETTE["scoped"], label="provider-scoped marks")
    ax.axhline(0.0, color=PALETTE["federated"], linewidth=1.4,
               label="federated marks")
    for _, row in prov.iterrows():
        if row.providers in (1, 2, 4, 8):
            ax.plot([row.hhi], [row.unsafe_scoped], "o",
                    color=PALETTE["scoped"], markersize=3.5)
            ax.annotate(
                f"$K = {int(row.providers)}$",
                xy=(row.hhi, row.unsafe_scoped),
                xytext=(row.hhi + 0.03, row.unsafe_scoped + 0.055),
                fontsize=FS["tiny"], color=PALETTE["neutral"],
            )
    ax.set_xlabel("market concentration (HHI)")
    ax.set_ylabel("cross-ecosystem unsafe frequency")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.05, 1.05)
    fitted_legend(ax, loc="upper right", fontsize=FS["annot"])
    panel_title(ax, "A", "safety against concentration")

    ax = axes[1]
    ax.plot(prov.providers, prov.member_payoff, "o-", color=PALETTE["club"],
            markersize=3.5, label="club member")
    ax.plot(prov.providers, prov.entrant_payoff, "s-", color=PALETTE["anarchy"],
            markersize=3.5, label="compliant entrant")
    ax.fill_between(
        prov.providers, prov.entrant_payoff, prov.member_payoff,
        color=PALETTE["scoped"], alpha=0.18,
    )
    mid = 2
    ax.text(
        prov.providers.iloc[mid] + 0.4,
        0.5 * (prov.member_payoff.iloc[mid] + prov.entrant_payoff.iloc[mid]),
        "exclusion rent", fontsize=FS["annot"], color=PALETTE["neutral"],
    )
    ax.set_xlabel("number of providers $K$")
    ax.set_ylabel("payoff")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 20], ["1", "2", "4", "8", "20"])
    fitted_legend(ax, loc="upper right", fontsize=FS["annot"])
    panel_title(ax, "B", "the rent fragmentation dissipates")

    save(fig, figdir / "fig08_providers")


# --------------------------------------------------------------------------
# fig09: robustness
# --------------------------------------------------------------------------


def fig09(tables, results: Path, figdir: Path) -> None:
    grids = np.load(results / "grids.npz")
    agree = pd.read_csv(results / "tables" / "robustness_replicator.csv")
    proc = pd.read_csv(results / "tables" / "robustness_process.csv")
    scope = pd.read_csv(results / "tables" / "robustness_setback_scope.csv")

    fig, axes = new_figure(4.7, nrows=2, ncols=2)

    ax = axes[0, 0]
    ax.grid(False)
    vmax = float(np.abs(grids["value"]).max())
    im = ax.pcolormesh(
        grids["value_sigmas"], grids["value_liabilities"], grids["value"],
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest",
    )
    lr = th.reciprocity_threshold(tables)
    ax.axhline(lr, color="black", linestyle="--", linewidth=1.0)
    ax.text(0.03, lr + 0.25, "$L_R$", fontsize=FS["annot"], color="black")
    ax.set_xlabel(r"spoof success $\sigma$")
    ax.set_ylabel("liability $L$")
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("certification value", fontsize=FS["annot"])
    cbar.ax.tick_params(labelsize=FS["tiny"])
    panel_title(ax, "A", "where identity earns its keep")

    ax = axes[0, 1]
    ax.plot(agree.sigma, agree.sml_integrity, "o-", color=PALETTE["club"],
            markersize=3.0, label="integrity, finite population")
    ax.plot(agree.sigma, agree.rep_integrity, "s--", color=PALETTE["mimic"],
            markersize=3.0, label="integrity, replicator")
    ax.plot(agree.sigma, agree.sml_unsafe, "o-", color=PALETTE["unsafe"],
            markersize=3.0, label="unsafe, finite population")
    ax.plot(agree.sigma, agree.rep_unsafe, "s--", color=PALETTE["CAS"],
            markersize=3.0, label="unsafe, replicator")
    ax.set_xlabel(r"spoof success $\sigma$")
    ax.set_ylabel("value")
    ax.set_ylim(-0.05, 1.95)
    fitted_legend(ax, loc="upper left", fontsize=FS["tiny"], ncol=1)
    panel_title(ax, "B", "two dynamics, one collapse")

    ax = axes[1, 0]
    width = 0.24
    zs = sorted(proc.population_size.unique())
    betas = sorted(proc.beta.unique())
    for b_i, beta in enumerate(betas):
        sub = proc[proc.beta == beta].sort_values("population_size")
        offset = (b_i - 1) * width
        ax.bar(
            np.arange(len(zs)) + offset, sub.unsafe, width,
            color=[PALETTE["safe"], PALETTE["CAS"], PALETTE["unsafe"]][b_i],
            label=rf"$\beta = {beta}$",
        )
    ax.set_xticks(range(len(zs)), [f"$Z = {int(z)}$" for z in zs])
    ax.set_ylabel("unsafe frequency")
    fitted_legend(ax, loc="upper right", fontsize=FS["annot"])
    panel_title(ax, "C", "process sensitivity")

    ax = axes[1, 1]
    labels = ["certified", "uncertified"]
    x = np.arange(2)
    for i, (_, row) in enumerate(scope.iterrows()):
        vals = [row.unsafe, row.uncertified_unsafe]
        ax.bar(
            x + (i - 0.5) * 0.32, vals, 0.32,
            color=[PALETTE["club"], PALETTE["anarchy"]][i],
            label=f"setback: {row.scope}",
        )
        for xx, v in zip(x + (i - 0.5) * 0.32, vals):
            ax.text(xx, v + 0.004, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=FS["tiny"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("unsafe frequency")
    ax.set_ylim(0, max(scope.uncertified_unsafe.max(), scope.unsafe.max()) * 1.3)
    fitted_legend(ax, loc="upper left", fontsize=FS["annot"])
    panel_title(ax, "D", "both readings of the setback")

    save(fig, figdir / "fig09_robustness")


# --------------------------------------------------------------------------


def main(outdir: Path) -> None:
    use_paper_style()
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    tables = build_race_tables(cfg.RACE)
    for fn in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09):
        fn(tables, outdir, figdir)
        print(f"rendered {fn.__name__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=ROOT / "results")
    main(parser.parse_args().outdir)

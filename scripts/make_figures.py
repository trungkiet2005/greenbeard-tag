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
    fit_headroom,
    fitted_legend,
    legend_below,
    new_figure,
    panel_title,
    readable_on,
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
    heat = ax.imshow(p0, cmap="RdYlBu", vmin=0.0, vmax=105.0)
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
                # the colour has to come from the cell that was painted, not
                # from the number in it: RdYlBu is lightest in the middle of
                # its range and darkest at both ends, so a rule keyed to the
                # value hides exactly the cells it means to show
                color=readable_on(heat.cmap(heat.norm(p0[i, j]))),
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

    fig, axes = new_figure(4.6, nrows=2, ncols=2)
    axes = axes.ravel()

    for ax, r, letter in ((axes[0], 0.0, "A"), (axes[1], base.r, "B")):
        params = replace(base, r=r)
        marks = []
        for invader, colour, label, symbol in (
            (cfg.FALSEBEARD, PALETTE["forger"], "exploiter F:CAS/CAS",
             r"\sigma^{*}"),
            (cfg.MIMIC, PALETTE["mimic"], "mimic F:CS/CAS",
             r"\sigma_{m}"),
        ):
            adv = [
                _advantage(tables, replace(params, sigma=float(s)), invader, cfg.CLUB)
                for s in sigmas
            ]
            ax.plot(sigmas, adv, color=colour, label=label)
            s_star = th.spoof_threshold(tables, params, 0.0, cfg.CLUB, invader)
            if s_star is not None and 0.0 < s_star < 1.0:
                ax.axvline(s_star, color=colour, linestyle=":", linewidth=0.9)
                marks.append((s_star, colour, symbol))
        ax.axhline(0.0, color=PALETTE["neutral"], linewidth=0.8)
        # the thresholds differ by 0.024 at the baseline, which is invisible on
        # a unit axis, so the crossing order is stated rather than left to the eye
        # the curves run bottom-left to top-right, so the bottom-right corner
        # is the only region of the panel with no ink in it
        # the marks are full-height dotted lines, so the block of labels has
        # to end to the left of the leftmost one rather than at the axis edge
        x_label = min((s for s, _, _ in marks), default=0.99) - 0.02
        for k, (s_star, colour, symbol) in enumerate(sorted(marks)):
            ax.text(
                x_label,
                -22.0 - 6.5 * k,
                rf"${symbol} = {s_star:.3f}$",
                fontsize=FS["tiny"],
                color=colour,
                ha="right",
                va="center",
            )
        ax.set_xlabel(r"spoof success $\sigma$")
        # the curves span [-30.4, +4.6]; the old (-35, 45) left the top four
        # tenths of both panels empty.  The upper-left legend of panel A sets
        # the ceiling, and panel B shares it so the two stay comparable.
        ax.set_ylim(-33, 14)
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
    ax.set_ylabel(r"invasion threshold in $\sigma$")
    ax.set_xlim(0, float(order.r.max()))
    ax.set_ylim(0.65, 1.03)
    fitted_legend(ax, loc="lower right", fontsize=FS["tiny"])
    panel_title(ax, "C", "clustering flips it")

    # panel D: the bifurcation itself.  Every threshold in the paper is a
    # transcritical bifurcation of the edge flow, and the paper asserted that
    # without ever drawing one.  On the {club, exploiter} edge the flow is
    # xdot = x(1-x)(alpha + beta x) with alpha the transversal eigenvalue, so
    # the interior branch sits at -alpha/beta and crosses the vertex exactly
    # where alpha changes sign.
    ax = axes[3]
    for r, colour, label in ((0.0, PALETTE["forger"], "$r = 0$"),
                             (base.r, PALETTE["club"], f"$r = {base.r}$")):
        params = replace(base, r=r)
        branch_s, branch_x, stable = [], [], []
        for s in sigmas:
            p = replace(params, sigma=float(s))
            fun = build_functionals(tables, p)
            i, j = fun.index(*cfg.FALSEBEARD), fun.index(*cfg.CLUB)
            P = fun.pi_P
            alpha = r * P[i, i] + (1 - r) * P[i, j] - P[j, j]
            beta = (1 - r) * (P[i, i] - P[i, j] - P[j, i] + P[j, j])
            if abs(beta) < 1e-12:
                continue
            x = -alpha / beta
            if 0.0 <= x <= 1.0:
                branch_s.append(s)
                branch_x.append(x)
                stable.append(beta < 0.0)
        # the vertex branch x = 0 is the club; solid where it resists invasion
        s_star = th.spoof_threshold(tables, params, 0.0, cfg.CLUB, cfg.FALSEBEARD)
        if s_star is None or not (0.0 < s_star < 1.0):
            s_star = 1.0
        ax.plot([0.0, s_star], [0.0, 0.0], color=colour, linewidth=1.6, label=label)
        ax.plot([s_star, 1.0], [0.0, 0.0], color=colour, linewidth=1.2,
                linestyle=":")
        if branch_s:
            ax.plot(branch_s, branch_x, color=colour, linewidth=1.2,
                    linestyle=(0, (5, 2)))
        ax.plot([s_star], [0.0], marker="o", markersize=3.4, color=colour,
                zorder=5)
    ax.set_xlabel(r"spoof success $\sigma$")
    ax.set_ylabel("forger share on the edge")
    ax.set_xlim(0.75, 1.0)
    ax.set_ylim(-0.055, 0.70)
    ax.text(0.912, 0.42, "solid: club stable\ndotted: club invaded\ndashed: interior branch",
            fontsize=FS["tiny"], color=PALETTE["neutral"], ha="center",
            va="top", linespacing=1.35)
    fitted_legend(ax, loc="upper center", fontsize=FS["tiny"], ncol=2)
    panel_title(ax, "D", "the transcritical bifurcation")

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
    # the top band is flat and uninformative across the whole sweep, so the
    # mark hides nothing there; at mid-height it lay across the moving bands
    ax_top.text(
        sig_m - 0.015, 0.82, r"$\sigma_m = 0.938$", fontsize=FS["annot"],
        rotation=90, ha="right", va="center", color="black",
    )
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)
    ax_top.set_xlabel(r"spoof success $\sigma$")
    ax_top.set_ylabel("share of the population")
    # a stackplot paints the whole panel, so any in-axes legend hides part of
    # the composition, and the fastest-moving part of it at that
    legend_below(ax_top, ncol=5, fontsize=FS["annot"])
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
    r_star, _ = th.nucleation_threshold(tables, 0.0, "CS", "CAS", "CAS", base.kappa_g)

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
    # the nucleation threshold is drawn as a full-height line at r = 0.063,
    # which runs through the upper left; both curves rise, so the free corner
    # is the upper right
    fitted_legend(ax, loc="upper right", fontsize=FS["annot"])
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
    ax.set_ylabel(r"break-even fine $\rho^{*}$")
    ax.set_xlim(0.85, 1.0)
    panel_title(ax, "B", "no finite fine replaces detection")

    ax = axes[1, 0]
    ax.plot(dues.setting, dues.club, color=PALETTE["club"], label="club share")
    sigmas_of_dues = [
        th.spoof_threshold_closed_form(tables, 0.0, "CS", "CAS", "CAS", float(k), 0.0, 0.0)
        for k in dues.setting
    ]
    ax2 = ax.twinx()
    ax2.plot(dues.setting, sigmas_of_dues, color=PALETTE["forger"], linestyle="--",
             label=r"spoof threshold $\sigma^{*}$")
    ax2.set_ylabel(r"$\sigma^{*}$", fontsize=FS["label"], labelpad=6.0)
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
    ax.set_ylim(-0.02, 0.92)
    fitted_legend(ax, loc="upper right", fontsize=FS["annot"])
    # the headroom is whatever seats the legend and nothing more: a band of
    # empty axis under a legend reads as badly as a legend on a curve
    fit_headroom(ax)
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
    colours = [PALETTE["anarchy"], PALETTE["CAS"], PALETTE["safe"], PALETTE["accent"]]

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
    # PALETTE["safe"] and PALETTE["club"] are #2166AC and #0072B2: a contrast
    # ratio of 1.14, which is no contrast at all.  These are the two curves the
    # panel exists to separate, so "both" gets its own hue and a dash as well.
    for cell, colour, style, label in (
        ("neither", PALETTE["anarchy"], "-", "neither"),
        ("fines_only", PALETTE["CAS"], "-", "fines only (audit)"),
        ("screening_only", PALETTE["safe"], "-", "screening only"),
        ("both", PALETTE["accent"], (0, (4, 2)), "both"),
    ):
        ax.plot(
            chan.sigma,
            chan[f"{cell}_integrity"],
            color=colour,
            linestyle=style,
            label=label,
        )
    ax.set_xlabel(r"spoof success $\sigma$")
    ax.set_ylabel("attestation integrity")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.42)
    fitted_legend(ax, loc="upper left", fontsize=FS["annot"], ncol=2)
    fit_headroom(ax)
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

    # The end states carry unsafe frequencies at numerical zero, so the old
    # version of this panel floored them at 1e-24 and plotted them on a log
    # axis spanning twenty-five decades, which invited the reader to read
    # structure into floating-point noise.  What the sample actually says is
    # where the badge-carrying mass lands, so that is what is drawn: two
    # spikes with nothing between them.
    counts, edges = np.histogram(badged_mass, bins=np.linspace(0.0, 1.0, 51))
    centres = 0.5 * (edges[:-1] + edges[1:])
    colours = [PALETTE["club"] if c > 0.5 else PALETTE["unbadged cooperator"]
               for c in centres]
    ax_a.bar(centres, counts / counts.sum(), width=edges[1] - edges[0],
             color=colours)
    ax_a.set_xlim(-0.06, 1.06)
    ax_a.set_ylim(0.0, 0.88)
    unbadged_share = 1.0 - key["certified_basin_share"]
    ax_a.text(0.07, unbadged_share + 0.04,
              f"{unbadged_share:.0%}\nunbadged",
              ha="left", va="bottom", fontsize=FS["tiny"],
              color=PALETTE["unbadged cooperator"])
    ax_a.text(0.93, key["certified_basin_share"] + 0.04,
              f"{key['certified_basin_share']:.0%}\nbadged",
              ha="right", va="bottom", fontsize=FS["tiny"], color=PALETTE["club"])
    ax_a.text(0.5, 0.24, "no end state\nin between", ha="center", va="center",
              fontsize=FS["tiny"], color=PALETTE["neutral"])
    ax_a.set_xlabel("badge-carrying share of the end state")
    ax_a.set_ylabel("share of starts")
    panel_title(ax_a, "A", "the sample splits in two")

    # ---- B: what each regime is worth
    xs = np.arange(2)
    social = [faces.loc["uncertified", "social"], faces.loc["certified", "social"]]
    bar_colours = [PALETTE["unbadged cooperator"], PALETTE["club"]]
    ax_b.bar(xs, social, 0.5, color=bar_colours)
    for x, v, colour in zip(xs, social, bar_colours):
        ax_b.text(x, 51.0, f"{v:.1f}", ha="center", va="bottom",
                  fontsize=FS["tiny"], color=readable_on(colour))
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
    # the label sits in the same column as the tick labels, and at this ylim it
    # lands level with the 1.0 tick, so it needs more than the default pad
    ax2.set_ylabel(r"$\sigma^{*}$", fontsize=FS["label"], labelpad=6.0)
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
    # the band closes as K grows, so the label goes in the widest gap that is
    # not against the axis edge, centred in it rather than hung off a point
    gap = (prov.member_payoff - prov.entrant_payoff).to_numpy()
    mid = int(np.argmax(gap[1:-1])) + 1
    ax.text(
        prov.providers.iloc[mid],
        0.5 * (prov.member_payoff.iloc[mid] + prov.entrant_payoff.iloc[mid]),
        "exclusion rent", fontsize=FS["annot"], color=PALETTE["neutral"],
        ha="center", va="center",
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
    fit_headroom(ax)
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


# --------------------------------------------------------------------------
# fig10: the two faces are one system, and the basin share is a measure
# --------------------------------------------------------------------------


def fig10(tables, results: Path, figdir: Path) -> None:
    """Assortment resolves the global claim the baseline could only assert.

    Panels A and B are the dynamical content of the face-reduction theorem: the
    two faces carry the same conduct flow, so their unsafe frequencies coincide
    at every assortment and their social payoffs stay exactly ``kappa_g`` apart
    wherever the regimes are settled.  Panel C shows that the clean two-face
    split is itself a property of the assortment range, and panel D that the
    basin share is a property of the start measure.
    """
    basins = pd.read_csv(results / "tables" / "assortment_basins.csv")
    measures = pd.read_csv(results / "tables" / "start_measure.csv")
    race = build_race_tables(cfg.RACE)
    r_dagger = th.first_strike_assortment_bound(race)

    fig, axes = new_figure(5.2, nrows=2, ncols=2)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # ---- A: the two faces carry the same unsafe frequency
    ax_a.plot(basins.r, basins.unsafe_certified, color=PALETTE["club"],
              lw=1.8, label="badged face")
    ax_a.plot(basins.r, basins.unsafe_uncertified, color=PALETTE["unbadged cooperator"],
              lw=1.8, ls="--", label="unbadged face")
    ax_a.axvline(r_dagger, color=PALETTE["neutral"], lw=0.9, ls=":")
    ax_a.text(r_dagger + 0.004, 0.60, rf"$r^{{\dagger}}={r_dagger:.3f}$",
              fontsize=FS["tiny"], color=PALETTE["neutral"])
    ax_a.set_xlabel("assortment $r$")
    ax_a.set_ylabel("unsafe frequency")
    ax_a.set_xlim(0.0, float(basins.r.max()))
    ax_a.set_ylim(-0.04, 1.08)
    fitted_legend(ax_a, loc="upper right")
    panel_title(ax_a, "A", "one flow, two labels")

    # ---- B: the payoff gap is exactly the dues wherever the regime is settled
    # Plotting the two levels hides the result: they run from -153 to +59 and
    # the gap the theorem predicts is 2, which is a line's width on that axis.
    # The gap itself is the quantity with something to say.
    gap = basins.social_uncertified - basins.social_certified
    ax_b.plot(basins.r, gap, color=PALETTE["club"], lw=1.8)
    ax_b.axhline(cfg.IDENTITY.kappa_g, color=PALETTE["unsafe"], lw=1.0, ls="--")
    ax_b.axvline(r_dagger, color=PALETTE["neutral"], lw=0.9, ls=":")
    ax_b.text(float(basins.r.max()), cfg.IDENTITY.kappa_g + 0.4,
              rf"$\kappa_g={cfg.IDENTITY.kappa_g:.0f}$", ha="right", va="bottom",
              fontsize=FS["tiny"], color=PALETTE["unsafe"])
    ax_b.set_xlabel("assortment $r$")
    ax_b.set_ylabel("social payoff, unbadged $-$ badged")
    ax_b.set_xlim(0.0, float(basins.r.max()))
    panel_title(ax_b, "B", "the gap is the dues")

    # ---- C: the two-face split exists only above a threshold in r
    n_starts = float(basins.n_starts.iloc[0])
    ax_c.plot(basins.r, basins.mixed_count / n_starts, color=PALETTE["forger"],
              lw=1.8, label="badge-mixed end states")
    ax_c.plot(basins.r, basins.badged_share, color=PALETTE["club"],
              lw=1.8, ls="--", label="badged basin share")
    ax_c.set_xlabel("assortment $r$")
    ax_c.set_ylabel("share of end states")
    ax_c.set_xlim(0.0, float(basins.r.max()))
    ax_c.set_ylim(-0.04, 1.12)
    fitted_legend(ax_c, loc="upper right")
    panel_title(ax_c, "C", "when the split exists")

    # ---- D: the basin share is a property of the start measure
    labels = [str(m) for m in measures.measure]
    xs = np.arange(len(labels))
    ax_d.bar(xs, measures.badged_share, 0.62, color=PALETTE["club"])
    lo, hi = float(measures.badged_share.min()), float(measures.badged_share.max())
    ax_d.axhline(lo, color=PALETTE["unsafe"], lw=0.8, ls=":")
    ax_d.axhline(hi, color=PALETTE["unsafe"], lw=0.8, ls=":")
    # Short tick labels: the concentration is named once in the axis label, so
    # repeating "alpha=" on every tick only makes them collide.
    short = [l.replace("alpha=", "").replace("badge-stratified", "strat.")
             for l in labels]
    ax_d.set_xticks(xs, short)
    ax_d.tick_params(axis="x", labelsize=FS["tick"])
    ax_d.set_xlabel(r"start concentration $\alpha$")
    ax_d.set_ylabel("badged basin share")
    # Exactly the headroom the one annotation needs, and no more.
    ax_d.set_ylim(0.0, hi * 1.18)
    ax_d.text(0.0, hi * 1.15, f"range {lo:.2f} to {hi:.2f}", ha="left",
              va="top", fontsize=FS["tiny"], color=PALETTE["unsafe"])
    panel_title(ax_d, "D", "a measure, not a flow")

    save(fig, figdir / "fig10_assortment")


# --------------------------------------------------------------------------
# fig11: the global regime map, and why conduct monitoring cannot see it
# --------------------------------------------------------------------------


def fig11(tables, results: Path, figdir: Path) -> None:
    """The (sigma, r) plane of the replicator flow, resolved rather than sampled.

    Panel A is the regime map.  Its two boundaries are orthogonal: the safety
    boundary is horizontal and set by assortment alone, the integrity boundary
    vertical and set by verification alone.  Panels B and C are the two fields
    the labels are built from, side by side so the reader can see that the one
    an outcome monitor observes is flat exactly where the other collapses.
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    with np.load(results / "grids.npz", allow_pickle=False) as z:
        sig, rs = z["global_sigmas"], z["global_rs"]
        regime = z["global_regime"]
        names = [str(s) for s in z["global_regime_names"]]
        integrity, unsafe = z["global_integrity"], z["global_unsafe"]

    race = build_race_tables(cfg.RACE)
    r_dagger = th.first_strike_assortment_bound(race)
    sigma_m = th.mimic_threshold_closed_form(
        race, cfg.LIABILITY, "CS", "CAS",
        cfg.IDENTITY.kappa_g, cfg.IDENTITY.kappa_f, cfg.IDENTITY.rho,
    )

    # one colour per regime, ordered as REGIMES is
    regime_colour = {
        "badged-safe": PALETTE["club"],
        "hollow": PALETTE["forger"],
        "mixed-unsafe": PALETTE["unsafe"],
        "unbadged-safe": PALETTE["unbadged cooperator"],
    }
    cmap = ListedColormap([regime_colour[n] for n in names])
    norm = BoundaryNorm(np.arange(len(names) + 1) - 0.5, len(names))
    extent = [sig.min(), sig.max(), rs.min(), rs.max()]

    fig, axes = new_figure(3.25, ncols=3)
    ax_a, ax_b, ax_c = axes

    # ---- A: the regime map
    ax_a.imshow(regime, origin="lower", aspect="auto", extent=extent,
                cmap=cmap, norm=norm, interpolation="nearest")
    ax_a.axhline(r_dagger, color="white", lw=1.3, ls="--")
    ax_a.axvline(sigma_m, color="white", lw=1.3, ls=":")
    ax_a.text(0.03, r_dagger + 0.006, rf"$r^{{\dagger}}={r_dagger:.3f}$",
              fontsize=FS["tiny"], color="white", va="bottom")
    ax_a.text(sigma_m - 0.02, 0.235, rf"$\sigma_m={sigma_m:.3f}$",
              fontsize=FS["tiny"], color="white", ha="right", va="top",
              rotation=90)
    ax_a.set_xlabel(r"spoof success $\sigma$")
    ax_a.set_ylabel("assortment $r$")
    # The key goes outside every axes.  Inside panel A it sits on a filled
    # image, so the audit reads its background as the image colour and the
    # black label text fails the contrast floor however opaque the frame is.
    present = [n for k, n in enumerate(names) if (regime == k).any()]
    fig.legend(
        handles=[Patch(facecolor=regime_colour[n], label=n) for n in present],
        loc="outside lower center", ncol=len(present), fontsize=FS["legend"],
        frameon=False,
    )
    panel_title(ax_a, "A", "orthogonal boundaries")

    # ---- B: attestation integrity, which the vertical boundary governs
    im_b = ax_b.imshow(np.nan_to_num(integrity, nan=0.0), origin="lower",
                       aspect="auto", extent=extent, cmap="Blues",
                       vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_b.axvline(sigma_m, color=PALETTE["neutral"], lw=1.1, ls=":")
    ax_b.set_xlabel(r"spoof success $\sigma$")
    ax_b.set_ylabel("assortment $r$")
    fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.03).ax.tick_params(
        labelsize=FS["tick"])
    panel_title(ax_b, "B", "what a passed badge means")

    # ---- C: unsafe frequency, which the horizontal boundary governs
    im_c = ax_c.imshow(unsafe, origin="lower", aspect="auto", extent=extent,
                       cmap="Reds", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_c.axhline(r_dagger, color=PALETTE["neutral"], lw=1.1, ls="--")
    ax_c.set_xlabel(r"spoof success $\sigma$")
    ax_c.set_ylabel("assortment $r$")
    fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.03).ax.tick_params(
        labelsize=FS["tick"])
    panel_title(ax_c, "C", "what a monitor sees")

    save(fig, figdir / "fig11_phase")


def main(outdir: Path) -> None:
    use_paper_style()
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    tables = build_race_tables(cfg.RACE)
    for fn in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09,
               fig10, fig11):
        fn(tables, outdir, figdir)
        print(f"rendered {fn.__name__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=ROOT / "results")
    main(parser.parse_args().outdir)

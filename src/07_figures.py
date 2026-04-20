"""07_figures.py — Publication figures for the v2 paper.

v2 (post peer review) produces eight figures as PDF + PNG:

  fig1_sg_distributions.pdf   — violin of the 4 SG components (unchanged)
  fig2_sg_decomposition.pdf   — dual panel: SG_total identity (Var shares)
                                next to LMG importance on finish position
                                with 95% CIs.  Makes the v1 "variance
                                decomposition" / importance distinction
                                explicit (addresses C1).
  fig3_effect_sizes.pdf       — Cohen's d with 500-boot 95% CIs and
                                paired Pearson-r continuous effect size,
                                plus the attenuation-corrected within-
                                tournament standardised betas (addresses
                                C2).
  fig4_archetypes.pdf         — 2x2 grid:
                                  (a) K-sweep silhouette vs Gaussian null
                                      95th percentile (absolute scope)
                                  (b) same for residual scope
                                  (c) top-20 concentration: absolute vs
                                      residual (addresses the circular
                                      "ball-striker" claim)
                                  (d) UMAP scatter (absolute scope, for
                                      qualitative context)
                                addresses C3.
  fig5_cv_comparison.pdf      — bar chart of AUC / R² across CV schemes
                                (random, GroupKFold-tournament, GroupKFold-
                                player, out-of-time) for both the
                                same-tournament SANITY check and the
                                genuine trailing-SG classifier, plus the
                                finish-position regressor (addresses
                                C4 & C5).
  fig6_trends.pdf             — winner mean SG_total and within-
                                tournament SG std by season, with
                                linear-trend fit and Chow-break p-values
                                overlaid (addresses C7).
  fig7_corr_matrix.pdf        — Pearson correlation heatmap without
                                sg_t2g (C8).
  fig8_top_radar.pdf          — radar grid of top-20 SG profiles
                                annotated with absolute- and residual-
                                scope archetype labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
})

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "results"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SG = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
LABELS = {"sg_ott": "Off-the-Tee", "sg_app": "Approach",
          "sg_arg": "Around-the-Green", "sg_putt": "Putting"}
COLORS = {"sg_ott": "#1F77B4", "sg_app": "#D62728",
          "sg_arg": "#2CA02C", "sg_putt": "#9467BD"}


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=220)
    plt.close(fig)
    print(f"  saved {stem}.{{pdf,png}}")


# ----------------------------------------------------------------------
# fig1 — distributions (unchanged from v1)
# ----------------------------------------------------------------------
def fig1_distributions():
    df = pd.read_csv(ROOT / "data" / "asa_pga_tourn_level.csv", low_memory=False).replace({"NA": np.nan})
    for c in SG + ["sg_total"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=SG)
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    order = SG
    parts = ax.violinplot([df[c].dropna() for c in order],
                          showmedians=True, widths=0.75)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COLORS[order[i]]); pc.set_alpha(0.55); pc.set_edgecolor("black")
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        parts[key].set_color("black")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([LABELS[c] for c in order])
    ax.set_ylabel("Strokes Gained (per round, vs. field)")
    ax.axhline(0, color="grey", lw=0.7, ls="--")
    ax.set_ylim(-6, 6)
    ax.set_title(f"Distribution of SG components (N = {len(df):,} player–events)")
    save(fig, "fig1_sg_distributions")


# ----------------------------------------------------------------------
# fig2 — identity vs importance (C1)
# ----------------------------------------------------------------------
def fig2_sg_decomposition():
    sg = json.load(open(RES / "sg_correlations.json"))
    ident = sg["sg_total_identity"]["marginal_var_share"]
    lmg_share = sg["lmg_importance_on_finish"]["lmg_share_of_total"]
    lmg_ci = sg["lmg_importance_on_finish"]["lmg_share_95ci"]
    total_r2 = sg["lmg_importance_on_finish"]["total_r2"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    comps = SG
    colors = [COLORS[c] for c in comps]

    # Left: SG_total identity
    vals_i = [ident[c] for c in comps]
    bars = axes[0].bar([LABELS[c] for c in comps], vals_i,
                       color=colors, edgecolor="black")
    for b, v in zip(bars, vals_i):
        axes[0].text(b.get_x() + b.get_width()/2, v + 0.005,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("Share of Var(SG_total)")
    axes[0].set_ylim(0, max(vals_i) * 1.3)
    axes[0].set_title("(a) SG_total identity\n"
                      "$\\mathrm{Var}(\\mathrm{SG\\_total}) = \\sum \\mathrm{Var}(\\mathrm{SG}_c) + \\mathrm{covariances}$")

    # Right: LMG importance on finish position with CIs
    vals_l = [lmg_share[c] for c in comps]
    lo = [vals_l[i] - lmg_ci[c][0] for i, c in enumerate(comps)]
    hi = [lmg_ci[c][1] - vals_l[i] for i, c in enumerate(comps)]
    bars = axes[1].bar([LABELS[c] for c in comps], vals_l,
                       yerr=[lo, hi], capsize=4,
                       color=colors, edgecolor="black",
                       error_kw=dict(ecolor="black", elinewidth=1.2))
    for b, v in zip(bars, vals_l):
        axes[1].text(b.get_x() + b.get_width()/2, v + 0.012,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("Share of explainable variance in finish position")
    axes[1].set_ylim(0, max(vals_l) * 1.35)
    axes[1].set_title(f"(b) LMG dominance analysis on finish position\n"
                      f"(total $R^2$ = {total_r2:.3f}; 200-boot 95% CIs)")
    fig.suptitle("Identity of SG_total  ≠  importance of its components on outcomes", y=1.04, fontsize=11)
    save(fig, "fig2_sg_decomposition")


# ----------------------------------------------------------------------
# fig3 — effect sizes with CIs + attenuation-corrected betas (C2)
# ----------------------------------------------------------------------
def fig3_effect_sizes():
    sg = json.load(open(RES / "sg_correlations.json"))
    tvb = sg["top_vs_bottom_with_ci"]["top_vs_bottom_decile"]
    wt  = sg["within_tournament_standardised"]["components"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    comps = SG
    colors = [COLORS[c] for c in comps]

    # Left: Cohen's d with CIs + paired Pearson r (full sample, continuous)
    ds = [tvb[c]["cohens_d_top_vs_bot"] for c in comps]
    d_lo = [ds[i] - tvb[c]["cohens_d_95ci"][0] for i, c in enumerate(comps)]
    d_hi = [tvb[c]["cohens_d_95ci"][1] - ds[i] for i, c in enumerate(comps)]
    rs = [tvb[c]["full_sample_pearson_r_with_neg_pos"] for c in comps]
    ys = np.arange(len(comps))
    axes[0].barh(ys, ds, xerr=[d_lo, d_hi], capsize=4,
                 color=colors, edgecolor="black",
                 error_kw=dict(ecolor="black", elinewidth=1.2))
    for i, (v, r) in enumerate(zip(ds, rs)):
        axes[0].text(v + 0.05, i, f"d = {v:.2f}; $r_\\mathrm{{cont}}$ = {r:.2f}",
                     va="center", fontsize=9)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([LABELS[c] for c in comps])
    axes[0].set_xlabel("Cohen's d  (top-10% vs bot-10% finishers, 500-boot 95% CI)")
    axes[0].set_xlim(0, max(ds) * 1.45)
    axes[0].set_title("(a) Effect size: extreme-groups d and continuous $r$")

    # Right: raw vs attenuation-corrected within-tournament betas
    raw = [wt[c]["mean_beta_flipped"] for c in comps]
    corr = [wt[c]["attenuation_corrected_beta"] for c in comps]
    x = np.arange(len(comps))
    w = 0.36
    axes[1].bar(x - w/2, raw,  w, label="raw $\\beta$",
                color=colors, edgecolor="black", alpha=0.55)
    axes[1].bar(x + w/2, corr, w, label="attenuation-corrected",
                color=colors, edgecolor="black")
    for i, (r, c_) in enumerate(zip(raw, corr)):
        axes[1].text(i - w/2, r + 0.01, f"{r:.2f}", ha="center", fontsize=8)
        axes[1].text(i + w/2, c_ + 0.01, f"{c_:.2f}", ha="center", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([LABELS[c] for c in comps])
    axes[1].set_ylabel("Standardised $\\beta$ on $-$finish_pos (within-tournament)")
    axes[1].set_ylim(0, max(corr) * 1.35)
    axes[1].legend(fontsize=8, loc="upper left", frameon=False)
    axes[1].set_title("(b) Within-tournament OLS $\\beta$:\nraw vs attenuation-corrected (reliability = $n/(n+1)$)")
    fig.suptitle("How strong is the SG-component → finish association?", y=1.04, fontsize=11)
    save(fig, "fig3_effect_sizes")


# ----------------------------------------------------------------------
# fig4 — archetype sanity: K-sweep null + top-20 concentration + UMAP (C3)
# ----------------------------------------------------------------------
def fig4_archetypes():
    arch = json.load(open(RES / "player_archetypes.json"))
    sweep = pd.read_csv(RES / "archetype_k_sweep.csv")
    players = pd.read_csv(RES / "player_clusters.csv", index_col=0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) K sweep absolute scope
    s_abs = sweep.query("scope == 'absolute'")
    axes[0, 0].plot(s_abs.k, s_abs.observed_silhouette, "-o", color="#B22222",
                    label="observed silhouette", lw=2)
    axes[0, 0].plot(s_abs.k, s_abs.null_95_silhouette, "--", color="#555555",
                    label="Gaussian-null 95th pct (99 perms)")
    axes[0, 0].fill_between(s_abs.k, s_abs.null_95_silhouette,
                            s_abs.observed_silhouette,
                            where=s_abs.observed_silhouette < s_abs.null_95_silhouette,
                            interpolate=True, color="#CCCCCC", alpha=0.4)
    axes[0, 0].axvline(arch["k_selected"], color="grey", ls=":", alpha=0.7)
    axes[0, 0].set_xlabel("k"); axes[0, 0].set_ylabel("Silhouette (Euclidean, standardised SG)")
    axes[0, 0].set_title("(a) Absolute-scope K-sweep vs null")
    axes[0, 0].legend(fontsize=8, frameon=False)

    # (b) K sweep residual scope
    s_res = sweep.query("scope == 'residual'")
    axes[0, 1].plot(s_res.k, s_res.observed_silhouette, "-o", color="#1F77B4",
                    label="observed silhouette", lw=2)
    axes[0, 1].plot(s_res.k, s_res.null_95_silhouette, "--", color="#555555",
                    label="Gaussian-null 95th pct (99 perms)")
    axes[0, 1].axvline(arch["k_selected"], color="grey", ls=":", alpha=0.7)
    axes[0, 1].set_xlabel("k"); axes[0, 1].set_ylabel("Silhouette")
    axes[0, 1].set_title("(b) Residual-scope K-sweep vs null")
    axes[0, 1].legend(fontsize=8, frameon=False)

    # (c) top-20 concentration bar chart
    abs_counts = arch["top20_concentration_abs"]["counts_by_cluster_abs"]
    res_counts = arch["top20_concentration_res"]["counts_by_cluster_res"]
    ids = sorted(set(abs_counts) | set(res_counts), key=lambda x: int(x))
    abs_v = [abs_counts.get(i, 0) for i in ids]
    res_v = [res_counts.get(i, 0) for i in ids]
    x = np.arange(len(ids)); w = 0.4
    axes[1, 0].bar(x - w/2, abs_v, w, color="#B22222", edgecolor="black",
                   label="absolute-scope")
    axes[1, 0].bar(x + w/2, res_v, w, color="#1F77B4", edgecolor="black",
                   label="residual-scope")
    for i, (a, r) in enumerate(zip(abs_v, res_v)):
        axes[1, 0].text(i - w/2, a + 0.15, str(a), ha="center", fontsize=9)
        axes[1, 0].text(i + w/2, r + 0.15, str(r), ha="center", fontsize=9)
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels([f"C{i}" for i in ids])
    axes[1, 0].set_ylabel("Top-20 players (by career SG_total) in cluster")
    axes[1, 0].set_title("(c) Top-20 concentration test\n"
                         f"abs: max share = {arch['top20_concentration_abs']['max_share']:.0%}, "
                         f"res: max share = {arch['top20_concentration_res']['max_share']:.0%}")
    axes[1, 0].legend(fontsize=8, frameon=False)
    axes[1, 0].set_ylim(0, 22)

    # (d) UMAP scatter (absolute scope, context only)
    emb = pd.read_csv(RES / "player_umap.csv", index_col=0)
    emb = emb.join(players[["cluster_abs", "sg_total"]])
    pal = sns.color_palette("tab10", n_colors=int(emb.cluster_abs.nunique()))
    for c in sorted(emb.cluster_abs.dropna().unique()):
        sub = emb[emb.cluster_abs == c]
        axes[1, 1].scatter(sub.umap_x, sub.umap_y,
                           s=12 + sub.sg_total.clip(lower=0) * 18,
                           c=[pal[int(c)]], alpha=0.7,
                           edgecolors="white", linewidths=0.4,
                           label=f"C{int(c)} (n={len(sub)})")
    top10 = emb.sort_values("sg_total", ascending=False).head(10)
    for name, r in top10.iterrows():
        axes[1, 1].annotate(name.split()[-1], (r.umap_x, r.umap_y),
                            fontsize=7, alpha=0.8,
                            xytext=(4, 3), textcoords="offset points")
    axes[1, 1].legend(fontsize=7, frameon=False, loc="best")
    axes[1, 1].set_xlabel("UMAP-1"); axes[1, 1].set_ylabel("UMAP-2")
    axes[1, 1].set_title("(d) UMAP of 4-SG profile, coloured by absolute-scope cluster\n"
                         f"(qualitative context; silhouette@{arch['k_selected']} within Gaussian null)")
    fig.suptitle("Archetypes do not exceed a Gaussian-null baseline, and top-20 concentration is an artefact of the SG_total level dimension", y=1.01, fontsize=11)
    save(fig, "fig4_archetypes")


# ----------------------------------------------------------------------
# fig5 — cross-validation comparison (C4, C5)
# ----------------------------------------------------------------------
def fig5_cv_comparison():
    pm = json.load(open(RES / "predictive_models.json"))
    cv_schemes = ["random_5fold", "group_by_tournament",
                  "group_by_player", "out_of_time_2021_2022"]
    scheme_labels = ["Random\n5-fold", "GroupKFold\n(tournament)",
                     "GroupKFold\n(player)", "Out-of-time\n2021-2022"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    x = np.arange(len(cv_schemes)); w = 0.38

    # (a) Same-tournament cut SANITY check
    san = pm["cut_sanity_check_same_tournament_sg"]
    logi = [san[s]["logistic_auc"] for s in cv_schemes]
    rfa  = [san[s]["rf_auc"] for s in cv_schemes]
    axes[0].bar(x - w/2, logi, w, color="#9467BD", edgecolor="black", label="Logistic")
    axes[0].bar(x + w/2, rfa,  w, color="#2CA02C", edgecolor="black", label="Random Forest")
    for i, (a, b) in enumerate(zip(logi, rfa)):
        axes[0].text(i - w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        axes[0].text(i + w/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(scheme_labels, fontsize=8)
    axes[0].set_ylabel("AUC")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].axhline(0.5, color="grey", ls="--", lw=0.7, alpha=0.5)
    axes[0].set_title("(a) Cut classifier — SANITY CHECK\n(same-tournament SG → made_cut; near-tautology)")
    axes[0].legend(fontsize=8, loc="lower right", frameon=False)

    # (b) Genuine trailing-SG cut classifier
    gen = pm["cut_genuine_trailing_sg"]
    logi = [gen[s]["logistic_auc"] for s in cv_schemes]
    rfa  = [gen[s]["rf_auc"] for s in cv_schemes]
    axes[1].bar(x - w/2, logi, w, color="#9467BD", edgecolor="black", label="Logistic")
    axes[1].bar(x + w/2, rfa,  w, color="#2CA02C", edgecolor="black", label="Random Forest")
    for i, (a, b) in enumerate(zip(logi, rfa)):
        axes[1].text(i - w/2, a + 0.005, f"{a:.2f}", ha="center", fontsize=8)
        axes[1].text(i + w/2, b + 0.005, f"{b:.2f}", ha="center", fontsize=8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(scheme_labels, fontsize=8)
    axes[1].set_ylabel("AUC")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].axhline(0.5, color="grey", ls="--", lw=0.7, alpha=0.5)
    axes[1].set_title("(b) Cut classifier — GENUINE\n(trailing 10-event SG → next cut)")
    axes[1].legend(fontsize=8, loc="lower right", frameon=False)

    # (c) Finish-position regressor R^2
    fp = pm["finish_position"]
    rid = [fp[s]["ridge_r2"] for s in cv_schemes]
    rf  = [fp[s]["rf_r2"] for s in cv_schemes]
    axes[2].bar(x - w/2, rid, w, color="#1F77B4", edgecolor="black", label="Ridge")
    axes[2].bar(x + w/2, rf,  w, color="#D62728", edgecolor="black", label="Random Forest")
    for i, (a, b) in enumerate(zip(rid, rf)):
        axes[2].text(i - w/2, a + 0.005, f"{a:.2f}", ha="center", fontsize=8)
        axes[2].text(i + w/2, b + 0.005, f"{b:.2f}", ha="center", fontsize=8)
    axes[2].set_xticks(x); axes[2].set_xticklabels(scheme_labels, fontsize=8)
    axes[2].set_ylabel("$R^2$ on finish position (cut-makers)")
    axes[2].set_ylim(0, 0.9)
    axes[2].set_title("(c) Finish-position regressor\n(same-tournament SG; identity-dominated)")
    axes[2].legend(fontsize=8, loc="lower right", frameon=False)

    fig.suptitle("Cross-validation grid: random 5-fold vs GroupKFold (tournament, player) vs out-of-time", y=1.03, fontsize=11)
    save(fig, "fig5_cv_comparison")


# ----------------------------------------------------------------------
# fig6 — temporal trends with stationarity tests (C7)
# ----------------------------------------------------------------------
def fig6_trends():
    t = json.load(open(RES / "temporal_trends.json"))
    ann = pd.DataFrame(t["annual_table"]).sort_values("season")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))

    # (a) Winner SG_total with linear trend + Chow p
    seasons = ann["season"].to_numpy()
    wins = ann["sg_total_mean"].to_numpy()
    axes[0].plot(seasons, wins, "-o", color="#B22222", lw=2, label="Winner mean SG_total")
    lt = t["tests"]["linear_trend_winner_sg_total"]
    xfit = np.linspace(seasons.min(), seasons.max(), 50)
    yfit = lt["intercept"] + lt["slope_per_season"] * xfit
    axes[0].plot(xfit, yfit, "--", color="#B22222", alpha=0.5,
                 label=f"linear fit: slope={lt['slope_per_season']:+.3f}/yr, p={lt['p']:.2f}")
    chow = t["tests"]["chow_break_winner_sg_total"]
    axes[0].axvline(chow["boundary"], color="grey", ls=":", alpha=0.6)
    axes[0].set_xlabel("Season"); axes[0].set_ylabel("Winner mean SG_total")
    axes[0].set_title(f"(a) Winners by season\nChow break @{chow['boundary']}: F={chow['F']:.2f}, p={chow['p']:.2f}  ·  "
                      f"Season FE F={t['tests']['season_fixed_effects_winner_sg']['joint_F_seasons_equal_zero']['F']:.2f}, "
                      f"p={t['tests']['season_fixed_effects_winner_sg']['joint_F_seasons_equal_zero']['p']:.2f}")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")

    # (b) Within-tournament SG std by season with linear trend + Chow p
    spread = ann["mean_within_t_sd"].to_numpy()
    axes[1].plot(seasons, spread, "-s", color="#4B0082", lw=2, label="Within-tourn. SG std")
    lt = t["tests"]["linear_trend_within_tourn_sd"]
    yfit = lt["intercept"] + lt["slope_per_season"] * xfit
    axes[1].plot(xfit, yfit, "--", color="#4B0082", alpha=0.5,
                 label=f"linear fit: slope={lt['slope_per_season']:+.4f}/yr, p={lt['p']:.2f}")
    chow = t["tests"]["chow_break_within_t_sd"]
    axes[1].axvline(chow["boundary"], color="grey", ls=":", alpha=0.6)
    axes[1].set_xlabel("Season"); axes[1].set_ylabel("Mean within-tournament SG std")
    axes[1].set_title(f"(b) Field dispersion by season\nChow break @{chow['boundary']}: F={chow['F']:.2f}, p={chow['p']:.2f}  ·  "
                      f"Season FE F={t['tests']['season_fixed_effects_within_t_sd']['joint_F_seasons_equal_zero']['F']:.2f}, "
                      f"p={t['tests']['season_fixed_effects_within_t_sd']['joint_F_seasons_equal_zero']['p']:.2f}")
    axes[1].legend(fontsize=8, frameon=False, loc="lower right")

    fig.suptitle("Stationarity: no linear trend and no Chow break at 2018 (tournament-level FE model)", y=1.04, fontsize=11)
    save(fig, "fig6_trends")


# ----------------------------------------------------------------------
# fig7 — correlation heatmap (C8: no sg_t2g)
# ----------------------------------------------------------------------
def fig7_corr_matrix():
    df = pd.read_csv(ROOT / "data" / "asa_pga_tourn_level.csv", low_memory=False).replace({"NA": np.nan})
    for c in SG + ["sg_total", "pos", "made_cut"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    use = df.dropna(subset=SG + ["sg_total", "pos"])[SG + ["sg_total", "pos", "made_cut"]]
    # Flip finish pos so +ve means better
    use = use.rename(columns={"pos": "finish_pos"})
    use["-finish_pos"] = -use["finish_pos"]
    cols = SG + ["sg_total", "-finish_pos", "made_cut"]
    corr = use[cols].corr()
    pretty = {c: LABELS.get(c, {"sg_total": "SG Total",
                                "-finish_pos": "Finish (rev.)",
                                "made_cut": "Made Cut"}.get(c, c)) for c in cols}
    corr = corr.rename(index=pretty, columns=pretty)

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, cbar_kws={"shrink": 0.75}, ax=ax,
                annot_kws={"fontsize": 8})
    ax.set_title("Pearson correlations: SG components, total, outcomes\n(sg_t2g removed: it is a definitional sum of OTT+APP+ARG)")
    save(fig, "fig7_corr_matrix")


# ----------------------------------------------------------------------
# fig8 — top-20 radar with archetype labels
# ----------------------------------------------------------------------
def fig8_top_radar():
    ep = json.load(open(RES / "elite_profiles.json"))
    pool_mean = ep["career_mean_sg_total_all_eligible"]
    top = pd.DataFrame(ep["top20"]).set_index("player")
    top_vals = top[SG]
    has_abs = "cluster_abs" in top.columns
    has_res = "cluster_res" in top.columns
    fig, axes = plt.subplots(4, 5, figsize=(11, 8.5),
                             subplot_kw=dict(polar=True))
    axes = axes.ravel()
    angles = np.linspace(0, 2 * np.pi, len(SG), endpoint=False).tolist()
    angles += angles[:1]
    lims = (-0.5, 1.1)
    for i, (name, row) in enumerate(top_vals.iterrows()):
        vals = row.tolist() + [row.tolist()[0]]
        axes[i].plot(angles, vals, color="#B22222", lw=1.6)
        axes[i].fill(angles, vals, color="#B22222", alpha=0.25)
        axes[i].set_xticks(angles[:-1])
        axes[i].set_xticklabels([LABELS[c].split("-")[0][:4] for c in SG], fontsize=7)
        axes[i].set_yticks([0, 0.5, 1.0])
        axes[i].set_yticklabels(["0", "0.5", "1"], fontsize=7)
        axes[i].set_ylim(*lims)
        title = name
        tags = []
        if has_abs and pd.notna(top.loc[name, "cluster_abs"]):
            tags.append(f"abs=C{int(top.loc[name, 'cluster_abs'])}")
        if has_res and pd.notna(top.loc[name, "cluster_res"]):
            tags.append(f"res=C{int(top.loc[name, 'cluster_res'])}")
        if tags:
            title = f"{name}\n{' · '.join(tags)}"
        axes[i].set_title(title, fontsize=7.5, pad=6)
        axes[i].grid(alpha=0.4)
    plt.suptitle("SG profiles of the top-20 players with absolute- and residual-scope archetype labels", y=1.01)
    save(fig, "fig8_top_radar")


def main():
    print("Generating v2 figures...")
    fig1_distributions()
    fig2_sg_decomposition()
    fig3_effect_sizes()
    fig4_archetypes()
    fig5_cv_comparison()
    fig6_trends()
    fig7_corr_matrix()
    fig8_top_radar()


if __name__ == "__main__":
    main()

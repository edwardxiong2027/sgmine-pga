"""07_figures.py — Publication figures for the paper.

Produces six figures (both PDF + PNG):
  fig1_sg_distributions.pdf  — violin/KDE of 4 SG components
  fig2_variance_decomp.pdf   — bar chart of unique R^2 for sg_total
  fig3_finish_effect.pdf     — Cohen's d (top-10% vs bot-10%) by component
  fig4_archetypes.pdf        — UMAP scatter coloured by k=4 cluster
  fig5_trends.pdf            — |Pearson r| SG-vs-finish by season, winner SG by season
  fig6_top_players_radar.pdf — radar-like grid of top-20 player SG profiles
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
    ax.set_title("Distribution of Strokes Gained components (N = 29{,}180 player–events)")
    save(fig, "fig1_sg_distributions")


def fig2_variance():
    sg = json.load(open(RES / "sg_correlations.json"))
    u = sg["variance_decomp"]["unique_r2"]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    comps = SG
    vals = [u[c] for c in comps]
    bars = ax.bar([LABELS[c] for c in comps], vals,
                  color=[COLORS[c] for c in comps], edgecolor="black")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Unique $R^2$ of component within sg_total")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("Which SG component contributes most to total?")
    save(fig, "fig2_variance_decomp")


def fig3_effect_size():
    sg = json.load(open(RES / "sg_correlations.json"))
    d = sg["top_vs_bottom"]["top_vs_bottom_decile"]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    comps = SG
    ds = [d[c]["cohens_d"] for c in comps]
    bars = ax.barh([LABELS[c] for c in comps], ds,
                   color=[COLORS[c] for c in comps], edgecolor="black")
    for b, v in zip(bars, ds):
        ax.text(v + 0.05, b.get_y() + b.get_height() / 2,
                f"d = {v:.2f}", va="center", fontsize=9)
    ax.set_xlabel("Cohen's d (top-10% minus bottom-10% finishers)")
    ax.set_xlim(0, max(ds) * 1.25)
    ax.set_title("Effect size separating elite from lagging finishers")
    save(fig, "fig3_finish_effect")


def fig4_archetypes():
    try:
        emb = pd.read_csv(RES / "player_umap.csv", index_col=0)
    except FileNotFoundError:
        print("  (no UMAP embedding found; skipping fig4)"); return
    clust = pd.read_csv(RES / "player_clusters.csv", index_col=0)
    emb = emb.join(clust[["sg_total", "cluster", "wins", "top10s"]])
    centr = pd.read_csv(RES / "cluster_centroids.csv")
    pal = sns.color_palette("tab10", n_colors=emb.cluster.nunique())

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for c in sorted(emb.cluster.unique()):
        sub = emb[emb.cluster == c]
        lbl = centr.loc[c, "label"] if "label" in centr.columns else f"Cluster {c}"
        ax.scatter(sub.umap_x, sub.umap_y, s=14 + sub.sg_total.clip(lower=0) * 18,
                   c=[pal[c]], alpha=0.75, edgecolors="white", linewidths=0.5,
                   label=f"C{c}: {lbl} (n={len(sub)})")
    # Label top-10 players
    top10 = emb.sort_values("sg_total", ascending=False).head(10)
    for name, r in top10.iterrows():
        ax.annotate(name.split()[-1], (r.umap_x, r.umap_y),
                    fontsize=7, alpha=0.8,
                    xytext=(4, 3), textcoords="offset points")
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title("Player archetypes (k=4) in UMAP of 4-component SG profile\n"
                 f"silhouette@4 = {json.load(open(RES / 'player_archetypes.json'))['silhouette_selected']:.3f}; n = {len(emb)} players with ≥10 events")
    save(fig, "fig4_archetypes")


def fig5_trends():
    t = json.load(open(RES / "temporal_trends.json"))
    seasons = sorted(int(s) for s in t["per_season_component_vs_pos"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    for c in SG + ["sg_t2g"]:
        ys = [abs(t["per_season_component_vs_pos"][str(s)][c]["pearson_r_with_pos"]) for s in seasons]
        lbl = LABELS.get(c, "Tee-to-Green") if c != "sg_t2g" else "Tee-to-Green"
        color = COLORS.get(c, "#555555")
        lw = 2.5 if c == "sg_t2g" else 1.5
        ls = "--" if c == "sg_t2g" else "-"
        axes[0].plot(seasons, ys, marker="o", label=lbl, color=color, lw=lw, ls=ls)
    axes[0].set_ylim(0, 0.75)
    axes[0].set_xlabel("Season"); axes[0].set_ylabel("|Pearson r| with finish position")
    axes[0].set_title("Per-season association SG component ↔ finish")
    axes[0].legend(fontsize=7, loc="lower right", frameon=False)

    wins = [t["per_season_winner_sg"][str(s)]["sg_total_mean"] for s in seasons]
    spread = [t["per_season_field_spread"][str(s)]["mean_within_tourn_sg_std"] for s in seasons]
    axes[1].plot(seasons, wins, "-o", color="#B22222", label="Winner mean SG_total")
    ax2 = axes[1].twinx()
    ax2.plot(seasons, spread, "--s", color="#4B0082", label="Field SG std")
    axes[1].set_xlabel("Season"); axes[1].set_ylabel("Winner mean SG_total", color="#B22222")
    ax2.set_ylabel("Within-tournament SG_total std", color="#4B0082")
    axes[1].tick_params(axis="y", labelcolor="#B22222")
    ax2.tick_params(axis="y", labelcolor="#4B0082")
    axes[1].set_title("Winners & field dispersion per season")
    lns = axes[1].get_lines() + ax2.get_lines()
    axes[1].legend(lns, [l.get_label() for l in lns], fontsize=7, loc="lower right", frameon=False)

    save(fig, "fig5_trends")


def fig6_top_radar():
    ep = json.load(open(RES / "elite_profiles.json"))
    pool_mean = ep["career_mean_sg_total_all_eligible"]
    top = pd.DataFrame(ep["top20"]).set_index("player")
    top = top[SG]
    fig, axes = plt.subplots(4, 5, figsize=(11, 8.5),
                             subplot_kw=dict(polar=True))
    axes = axes.ravel()
    angles = np.linspace(0, 2 * np.pi, len(SG), endpoint=False).tolist()
    angles += angles[:1]
    lims = (-0.5, 1.1)
    for i, (name, row) in enumerate(top.iterrows()):
        vals = row.tolist() + [row.tolist()[0]]
        axes[i].plot(angles, vals, color="#B22222", lw=1.6)
        axes[i].fill(angles, vals, color="#B22222", alpha=0.25)
        axes[i].set_xticks(angles[:-1])
        axes[i].set_xticklabels([LABELS[c].split("-")[0][:4] for c in SG], fontsize=7)
        axes[i].set_yticks([0, 0.5, 1.0])
        axes[i].set_yticklabels(["0", "0.5", "1"], fontsize=7)
        axes[i].set_ylim(*lims)
        axes[i].set_title(name, fontsize=8.5, pad=6)
        axes[i].grid(alpha=0.4)
    plt.suptitle("SG profiles of the top-20 players (career mean, ≥30 events, 2015-2022)", y=1.01)
    save(fig, "fig6_top_players_radar")


def fig7_corr_matrix():
    df = pd.read_csv(ROOT / "data" / "asa_pga_tourn_level.csv", low_memory=False).replace({"NA": np.nan})
    for c in SG + ["sg_t2g", "sg_total", "pos", "made_cut"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    use = df.dropna(subset=SG + ["sg_total", "pos"])[SG + ["sg_t2g", "sg_total", "pos", "made_cut"]]
    # Flip finish pos so +ve means better
    use = use.rename(columns={"pos": "finish_pos"})
    use["-finish_pos"] = -use["finish_pos"]
    cols = SG + ["sg_t2g", "sg_total", "-finish_pos", "made_cut"]
    corr = use[cols].corr()
    pretty = {c: LABELS.get(c, {"sg_t2g": "Tee-to-Green",
                                "sg_total": "SG Total",
                                "-finish_pos": "Finish (rev.)",
                                "made_cut": "Made Cut"}.get(c, c)) for c in cols}
    corr = corr.rename(index=pretty, columns=pretty)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, cbar_kws={"shrink": 0.75}, ax=ax,
                annot_kws={"fontsize": 8})
    ax.set_title("Pearson correlations: SG components, aggregates, outcomes")
    save(fig, "fig7_correlation_heatmap")


def main():
    print("Generating figures...")
    fig1_distributions()
    fig2_variance()
    fig3_effect_size()
    fig4_archetypes()
    fig5_trends()
    fig6_top_radar()
    fig7_corr_matrix()


if __name__ == "__main__":
    main()

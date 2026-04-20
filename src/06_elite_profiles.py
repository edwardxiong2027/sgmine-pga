"""06_elite_profiles.py — Rank-based profile of the top-N players 2015-2022.

v2 (post peer review): the v1 "ball-striker archetype" claim was circular
(see 04_player_archetypes.py null & residual analyses). We now report top-N
profiles alongside **both** the absolute-scope and residual-scope cluster
labels. This lets the reader see that the absolute-scope clustering mostly
restates the SG_total ranking, while the residual-scope clustering — which
removes each player's mean level — disperses top-N players across
shape-based archetypes.

For each player with >= MIN_EVENTS events we report:
  - career mean SG_total and its 4 components
  - career made-cut rate
  - wins and top-10 counts
  - absolute and residual archetype labels from 04_player_archetypes.py

We also compute top-N concentration statistics for both clusterings to
directly rebut the "19 of top-20 are C0 ball-strikers" v1 claim.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MIN_EVENTS = 30
DATA = Path(__file__).resolve().parents[1] / "data" / "asa_pga_tourn_level.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SG = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]


def main() -> None:
    df = pd.read_csv(DATA, low_memory=False).replace({"NA": np.nan})
    for c in SG + ["sg_total", "made_cut", "pos"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=SG + ["sg_total"]).copy()

    agg = (
        df.groupby("player")
        .agg(
            n_events=("tournament id", "count"),
            sg_total=("sg_total", "mean"),
            sg_ott=("sg_ott", "mean"),
            sg_app=("sg_app", "mean"),
            sg_arg=("sg_arg", "mean"),
            sg_putt=("sg_putt", "mean"),
            made_cut_rate=("made_cut", "mean"),
            wins=("pos", lambda s: int((s == 1).sum())),
            top10s=("pos", lambda s: int((s <= 10).sum())),
        )
        .query("n_events >= @MIN_EVENTS")
        .sort_values("sg_total", ascending=False)
    )
    agg.to_csv(OUT / "player_career_aggregates.csv")

    top20 = agg.head(20).copy().round(3)
    top20.to_csv(OUT / "top20_players.csv")

    # Attach cluster labels from 04_player_archetypes v2 outputs if present.
    # v2 writes cluster_abs and cluster_res (absolute-scope vs residual-scope).
    clust = OUT / "player_clusters.csv"
    abs_conc = None
    res_conc = None
    if clust.exists():
        c = pd.read_csv(clust, index_col=0)
        keep_cols = [x for x in ["cluster_abs", "cluster_res", "stability_abs", "stability_res", "cluster"] if x in c.columns]
        top20 = top20.join(c[keep_cols], how="left")

        # Top-20 concentration statistics (rebuttal to v1 "19 of 20 in C0" claim)
        if "cluster_abs" in top20.columns:
            abs_counts = top20["cluster_abs"].value_counts().sort_index()
            abs_conc = {
                "counts": abs_counts.to_dict(),
                "top_cluster_share": float(abs_counts.max() / len(top20)),
            }
        if "cluster_res" in top20.columns:
            res_counts = top20["cluster_res"].value_counts().sort_index()
            res_conc = {
                "counts": res_counts.to_dict(),
                "top_cluster_share": float(res_counts.max() / len(top20)),
            }

    print("=== TOP 20 players by career mean SG_total (>=30 events) ===")
    print(top20.to_string())
    if abs_conc is not None:
        print("\nTop-20 absolute-scope cluster counts:", abs_conc)
    if res_conc is not None:
        print("Top-20 residual-scope cluster counts:", res_conc)

    summary = {
        "min_events": MIN_EVENTS,
        "n_players_in_pool": int(len(agg)),
        "top20": top20.reset_index().to_dict(orient="records"),
        "career_mean_sg_total_all_eligible": float(agg.sg_total.mean()),
        "career_mean_sg_total_top20": float(top20.sg_total.mean()),
        "top20_abs_concentration": abs_conc,
        "top20_res_concentration": res_conc,
        "note": (
            "v1 claimed 19 of the top-20 fell in a single 'ball-striker' cluster C0. "
            "In v2 the absolute-scope clustering recovers a similar concentration "
            "(which is mostly a re-statement of the SG_total ranking, since the "
            "clustering operates on raw component means), but the residual-scope "
            "clustering — which removes each player's SG_total mean level before "
            "clustering — disperses the top-20 across multiple clusters, which "
            "we interpret as evidence that v1's archetype claim was largely "
            "circular."
        ),
    }
    (OUT / "elite_profiles.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

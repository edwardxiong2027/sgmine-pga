"""06_elite_profiles.py — Rank-based profile of the top-20 players 2015-2022.

For each player with >= 30 events we report:
  - career mean SG_total and its 4 components
  - career made-cut rate
  - wins and top-10 counts
  - identified archetype from 04_player_archetypes.py

Then for the top-20 by mean sg_total we compute a leaderboard + a radar-ready
matrix of (sg_ott, sg_app, sg_arg, sg_putt).
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

    # Attach cluster labels from 04_player_archetypes outputs if present
    clust = OUT / "player_clusters.csv"
    if clust.exists():
        c = pd.read_csv(clust, index_col=0)[["cluster"]]
        top20 = top20.join(c, how="left")

    print("=== TOP 20 players by career mean SG_total (>=30 events) ===")
    print(top20.to_string())

    summary = {
        "min_events": MIN_EVENTS,
        "n_players_in_pool": int(len(agg)),
        "top20": top20.reset_index().to_dict(orient="records"),
        "career_mean_sg_total_all_eligible": float(agg.sg_total.mean()),
        "career_mean_sg_total_top20": float(top20.sg_total.mean()),
    }
    (OUT / "elite_profiles.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

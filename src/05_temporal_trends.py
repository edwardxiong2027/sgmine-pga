"""05_temporal_trends.py — Has PGA Tour scoring evolved between 2015-2022?

We ask three questions of the data:
  1. Has the within-tournament SG *variance* compressed (competition tighter)?
  2. Has the explanatory mix of SG shifted (e.g., has putting faded in
     importance while approach has grown, consistent with the "driving
     distance inflation" narrative)?
  3. Has the median cut-making SG_total drifted?

We also report winning SG-total by season and per-tournament field size.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA = Path(__file__).resolve().parents[1] / "data" / "asa_pga_tourn_level.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SG = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, low_memory=False).replace({"NA": np.nan})
    for c in SG + ["sg_total", "pos", "made_cut", "purse", "season"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=SG + ["sg_total", "season"]).copy()
    df["season"] = df["season"].astype(int)
    return df


def per_season_spread(df: pd.DataFrame) -> dict:
    """Standard deviation of sg_total within each tournament, averaged by season.
    A shrinking std implies a tighter field (narrower dispersion)."""
    per_t = df.groupby(["season", "tournament id"])["sg_total"].std().reset_index()
    per_s = per_t.groupby("season")["sg_total"].agg(["mean", "std", "count"]).reset_index()
    return {int(r["season"]): {"mean_within_tourn_sg_std": float(r["mean"]),
                               "sd_within_tourn_sg_std": float(r["std"]),
                               "n_tournaments": int(r["count"])}
            for _, r in per_s.iterrows()}


def per_season_winner_sg(df: pd.DataFrame) -> dict:
    winners = df[df["pos"] == 1].copy()
    agg = winners.groupby("season")[SG + ["sg_total"]].agg(["mean", "median"])
    return {int(s): {f"{col}_{stat}": float(agg.loc[s, (col, stat)])
                     for col in SG + ["sg_total"]
                     for stat in ["mean", "median"]}
            for s in sorted(winners["season"].unique())}


def per_season_component_corr(df: pd.DataFrame) -> dict:
    """Pearson correlation of each SG component with finish position, by season."""
    cut = df.dropna(subset=["pos"])
    out = {}
    for s, sub in cut.groupby("season"):
        out[int(s)] = {}
        for c in SG + ["sg_t2g"] if "sg_t2g" in sub.columns else SG:
            if c == "sg_t2g":
                v = sub[c].dropna()
                if len(v) < 100:
                    continue
                r, p = stats.pearsonr(sub.loc[v.index, c], sub.loc[v.index, "pos"])
            else:
                r, p = stats.pearsonr(sub[c], sub["pos"])
            out[int(s)][c] = {"pearson_r_with_pos": float(r), "p": float(p)}
    return out


def per_season_make_cut_sg(df: pd.DataFrame) -> dict:
    """Median SG_total that just makes the cut (defines 'cut-line difficulty')."""
    made = df[df["made_cut"] == 1]
    missed = df[df["made_cut"] == 0]
    out = {}
    for s in sorted(df["season"].unique()):
        out[int(s)] = {
            "made_cut_median_sg_total": float(made.loc[made.season == s, "sg_total"].median()),
            "missed_cut_median_sg_total": float(missed.loc[missed.season == s, "sg_total"].median()),
            "n_made": int((made.season == s).sum()),
            "n_missed": int((missed.season == s).sum()),
        }
    return out


def main() -> None:
    df = load()
    print(f"Rows for trend analysis: {len(df):,}  Seasons: {sorted(df.season.unique())}")

    out = {
        "per_season_field_spread": per_season_spread(df),
        "per_season_winner_sg": per_season_winner_sg(df),
        "per_season_component_vs_pos": per_season_component_corr(df),
        "per_season_cut_profile": per_season_make_cut_sg(df),
    }
    (OUT / "temporal_trends.json").write_text(json.dumps(out, indent=2))
    import pprint; pprint.pprint(out, width=100, sort_dicts=False)


if __name__ == "__main__":
    main()

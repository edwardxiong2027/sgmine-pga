"""01_explore.py — Profile the ASA PGA tournament-level dataset.

Outputs a JSON profile + a short text summary for the paper's Data section.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "asa_pga_tourn_level.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "profile"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise NAs and types on the raw ASA CSV."""
    df = df.replace({"NA": np.nan})
    for col in ["sg_putt", "sg_arg", "sg_app", "sg_ott", "sg_t2g", "sg_total",
                "strokes", "hole_par", "n_rounds", "purse", "made_cut", "pos"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def main() -> None:
    df = _coerce(pd.read_csv(DATA, low_memory=False))

    profile = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "n_unique_players": int(df["player id"].nunique()),
        "n_unique_tournaments": int(df["tournament id"].nunique()),
        "n_unique_courses": int(df["course"].nunique()),
        "seasons": sorted(df["season"].dropna().unique().tolist()),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "missing_pct_by_col": {
            c: round(df[c].isna().mean() * 100, 2) for c in df.columns
        },
        "sg_availability_pct": round(df["sg_total"].notna().mean() * 100, 2),
        "made_cut_rate": round(df["made_cut"].mean() * 100, 2),
        "purse_median_millions": round(df["purse"].median(), 2),
        "purse_mean_millions": round(df["purse"].mean(), 2),
    }

    # --- Strokes Gained distribution ---
    sg_cols = ["sg_putt", "sg_arg", "sg_app", "sg_ott", "sg_t2g", "sg_total"]
    sg_desc = df[sg_cols].describe().round(3).to_dict()
    profile["sg_summary"] = sg_desc

    # --- Player-level aggregates (for later archetype analysis) ---
    player_agg = (
        df.dropna(subset=["sg_total"])
        .groupby("player")
        .agg(
            n_events=("tournament id", "count"),
            sg_total=("sg_total", "mean"),
            sg_ott=("sg_ott", "mean"),
            sg_app=("sg_app", "mean"),
            sg_arg=("sg_arg", "mean"),
            sg_putt=("sg_putt", "mean"),
            made_cut_rate=("made_cut", "mean"),
        )
    )
    profile["n_players_with_any_sg"] = int(len(player_agg))
    profile["n_players_min10_events"] = int((player_agg.n_events >= 10).sum())

    (OUT_DIR / "profile.json").write_text(json.dumps(profile, indent=2, default=str))
    player_agg.to_csv(OUT_DIR / "player_aggregates.csv")

    print(f"Rows: {profile['n_rows']:,}")
    print(f"Players: {profile['n_unique_players']:,}  "
          f"(with >=10 events & SG: {profile['n_players_min10_events']:,})")
    print(f"Tournaments: {profile['n_unique_tournaments']}")
    print(f"Courses:     {profile['n_unique_courses']}")
    print(f"Seasons:     {profile['seasons']}")
    print(f"SG availability: {profile['sg_availability_pct']}%")


if __name__ == "__main__":
    main()

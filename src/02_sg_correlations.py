"""02_sg_correlations.py — Strokes Gained components vs tournament outcomes.

IMPORTANT: A player who misses the 36-hole cut has fewer total strokes than a
player who made the cut and played 72 holes. So we NEVER rank by raw `strokes`.
Outcomes used here:
  - `pos`: numeric finish position among players who made the cut.
  - `made_cut`: binary 0/1.
  - strokes_vs_field within the subset who played the same number of rounds.

Question 1 (variance-decomposition): Which of the four SG components
(OTT, APP, ARG, PUTT) contributes most to sg_total?

Question 2 (rank correlation): Which SG component has the strongest Spearman
rank-correlation with finish position (among made-cut players)?

Question 3 (within-tournament): For made-cut players, fit `pos ~ SG components`
within each tournament and average the standardised coefficients.

Question 4: Effect size (Cohen's d) of SG components between top-10% and
bottom-10% finishers (among made-cut players).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

SEED = 42
DATA = Path(__file__).resolve().parents[1] / "data" / "asa_pga_tourn_level.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SG_COMPONENTS = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, low_memory=False).replace({"NA": np.nan})
    for c in SG_COMPONENTS + ["sg_total", "sg_t2g", "strokes", "hole_par",
                              "n_rounds", "purse", "made_cut", "pos"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=SG_COMPONENTS + ["sg_total"]).copy()
    return df


def variance_decomp(df: pd.DataFrame) -> dict:
    """Regress sg_total on its 4 components; report R^2 and unique R^2."""
    X = df[SG_COMPONENTS].to_numpy()
    y = df["sg_total"].to_numpy()
    model = LinearRegression().fit(X, y)
    yhat = model.predict(X)
    r2 = 1 - np.var(y - yhat) / np.var(y)

    unique = {}
    for i, comp in enumerate(SG_COMPONENTS):
        cols = [j for j in range(X.shape[1]) if j != i]
        m_wo = LinearRegression().fit(X[:, cols], y)
        r2_wo = 1 - np.var(y - m_wo.predict(X[:, cols])) / np.var(y)
        unique[comp] = r2 - r2_wo

    pearson_r = {c: float(np.corrcoef(df[c], df["sg_total"])[0, 1]) for c in SG_COMPONENTS}

    return {
        "r2_full": float(r2),
        "unique_r2": {k: float(v) for k, v in unique.items()},
        "pearson_r_with_total": pearson_r,
        "coefficients": {c: float(b) for c, b in zip(SG_COMPONENTS, model.coef_)},
    }


def rank_correlations(df: pd.DataFrame) -> dict:
    """Spearman rho between each SG component and finish position (lower = better).

    Sign is flipped so a *positive* reported value means 'higher SG predicts
    better finish'.
    """
    cut = df.dropna(subset=["pos"]).copy()
    out = {}
    for comp in SG_COMPONENTS + ["sg_t2g", "sg_total"]:
        rho, p = stats.spearmanr(cut[comp], cut["pos"])
        out[comp] = {"spearman_rho_minus_pos": float(-rho),
                     "p_value": float(p),
                     "n": int(len(cut))}
    return out


def within_tournament_standardised(df: pd.DataFrame) -> dict:
    """For each tournament, fit pos ~ standardised SG components, then average.

    A negative coefficient means 'higher SG => lower (better) finish position'.
    We flip the sign so higher reported value = stronger predictor of a good
    finish.
    """
    cut = df.dropna(subset=["pos"]).copy()
    coefs = {c: [] for c in SG_COMPONENTS}
    r2s = []
    for tid, sub in cut.groupby("tournament id"):
        if len(sub) < 30:
            continue
        X = sub[SG_COMPONENTS].to_numpy()
        # Standardise within tournament to get comparable betas
        Xs = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
        y = sub["pos"].to_numpy()
        ys = (y - y.mean()) / y.std(ddof=0)
        m = LinearRegression().fit(Xs, ys)
        for c, b in zip(SG_COMPONENTS, m.coef_):
            coefs[c].append(-b)
        r2s.append(m.score(Xs, ys))
    return {
        "components": {
            c: {"mean_beta_flipped": float(np.mean(v)),
                "median_beta_flipped": float(np.median(v)),
                "n_tournaments": int(len(v))}
            for c, v in coefs.items()
        },
        "mean_R2_within_tournament": float(np.mean(r2s)),
        "median_R2_within_tournament": float(np.median(r2s)),
    }


def top_vs_bottom(df: pd.DataFrame) -> dict:
    """Mean SG by finish bucket: Top10 / T20 / Cut-makers / Missed-cut.

    Also report Cohen's d between top-10% and bottom-10% finishers among
    cut-makers.
    """
    buckets = {}
    made = df[df["made_cut"] == 1].copy()
    missed = df[df["made_cut"] == 0].copy()

    buckets["made_cut"] = {c: float(made[c].mean()) for c in SG_COMPONENTS + ["sg_total"]}
    buckets["missed_cut"] = {c: float(missed[c].mean()) for c in SG_COMPONENTS + ["sg_total"]}
    buckets["top10_finish"] = {c: float(made.loc[made["pos"] <= 10, c].mean()) for c in SG_COMPONENTS + ["sg_total"]}
    buckets["top3_finish"] = {c: float(made.loc[made["pos"] <= 3, c].mean()) for c in SG_COMPONENTS + ["sg_total"]}
    buckets["win"] = {c: float(made.loc[made["pos"] == 1, c].mean()) for c in SG_COMPONENTS + ["sg_total"]}

    # Cohen's d between top-10% and bottom-10% of cut-makers
    q_lo = made["pos"].quantile(0.10)
    q_hi = made["pos"].quantile(0.90)
    top = made[made["pos"] <= q_lo]
    bot = made[made["pos"] >= q_hi]
    cohens = {}
    for c in SG_COMPONENTS + ["sg_total"]:
        a = top[c].dropna().to_numpy()
        b = bot[c].dropna().to_numpy()
        s = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
        cohens[c] = {
            "top10pct_mean": float(a.mean()),
            "bot10pct_mean": float(b.mean()),
            "cohens_d": float((a.mean() - b.mean()) / s),
            "n_top": int(len(a)),
            "n_bot": int(len(b)),
        }
    return {"bucket_means": buckets, "top_vs_bottom_decile": cohens,
            "top10pct_rank_cutoff": float(q_lo),
            "bot10pct_rank_cutoff": float(q_hi)}


def made_cut_analysis(df: pd.DataFrame) -> dict:
    """Logistic-like summary: mean of each SG component for cut-makers vs missers."""
    grouped = df.groupby("made_cut")[SG_COMPONENTS + ["sg_total"]].mean()
    diffs = {c: float(grouped.loc[1, c] - grouped.loc[0, c]) for c in SG_COMPONENTS + ["sg_total"]}
    return {"means_by_cut": grouped.to_dict(),
            "cut_minus_nocut": diffs}


def main() -> None:
    df = load()
    print(f"Usable rows with all SG: {len(df):,}")
    print(f"... with made_cut=1: {(df['made_cut']==1).sum():,}")

    results = {
        "n_rows": int(len(df)),
        "n_rows_made_cut": int((df['made_cut']==1).sum()),
        "variance_decomp": variance_decomp(df),
        "rank_correlations": rank_correlations(df),
        "within_tournament_standardised": within_tournament_standardised(df),
        "top_vs_bottom": top_vs_bottom(df),
        "made_cut_analysis": made_cut_analysis(df),
    }
    (OUT / "sg_correlations.json").write_text(json.dumps(results, indent=2))
    import pprint; pprint.pprint(results, width=100, sort_dicts=False)


if __name__ == "__main__":
    main()

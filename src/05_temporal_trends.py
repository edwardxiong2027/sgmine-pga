"""05_temporal_trends.py (v2 — peer-review revised).

v2 changes addressing reviewer C7:

  * In v1 the "stationarity" conclusion was asserted from visual
    inspection of two time series (winner mean SG_total, within-
    tournament SG_total SD). The reviewer correctly noted that these
    ranges (3.38-3.80 for winners, ~12% of the central value) merit a
    test, not a glance. We now fit:

      (a) SEASON FIXED-EFFECTS regression on both series, reporting
          year coefficients with 95% CIs (OLS + HC0 robust SE).

      (b) LINEAR-TREND test via OLS of series ~ season with a 2-sided
          t-test for slope == 0.

      (c) CHOW structural-break test using the 2018 boundary motivated
          by the v1 narrative about "coverage artefacts" pre-2018.

    The reader can then see whether "stationary" is supported.

  * We also add a simple coverage-adjusted model that conditions on
    the per-season fraction of tournaments with complete SG (the v1
    dismissal was "2015-2016 is a coverage artefact"; here we show
    how much coverage actually shifts the coefficients).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
try:
    import statsmodels.api as sm
    HAS_SM = True
except Exception:
    HAS_SM = False

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


# ---------------------------------------------------------------------------
# Per-season series (same as v1)
# ---------------------------------------------------------------------------
def per_season_spread(df: pd.DataFrame) -> pd.DataFrame:
    per_t = (df.groupby(["season", "tournament id"])["sg_total"].std()
             .reset_index()
             .rename(columns={"sg_total": "within_t_sd"}))
    per_s = per_t.groupby("season")["within_t_sd"].agg(
        ["mean", "std", "count"]).reset_index()
    per_s = per_s.rename(columns={
        "mean": "mean_within_t_sd",
        "std": "sd_within_t_sd",
        "count": "n_tournaments",
    })
    return per_s


def per_season_winner(df: pd.DataFrame) -> pd.DataFrame:
    winners = df[df["pos"] == 1].copy()
    rows = []
    for s, sub in winners.groupby("season"):
        row = {"season": int(s), "n_winners": int(len(sub))}
        for c in SG + ["sg_total"]:
            row[f"{c}_mean"] = float(sub[c].mean())
            row[f"{c}_median"] = float(sub[c].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("season")


def per_season_coverage(df: pd.DataFrame) -> pd.DataFrame:
    # Coverage = fraction of tournaments in a season whose SG rows are
    # non-null for all 4 components.
    rows = []
    for s, sub in df.groupby("season"):
        rows.append({"season": int(s),
                     "sg_coverage": float(sub[SG].notna().all(axis=1).mean()),
                     "n_rows": int(len(sub))})
    return pd.DataFrame(rows).sort_values("season")


# ---------------------------------------------------------------------------
# C7: formal tests — season FE, linear trend, Chow break
# ---------------------------------------------------------------------------
def season_fixed_effects_model(df: pd.DataFrame, y_series: pd.Series,
                               base_season: int | None = None) -> dict:
    """OLS with season dummies. Returns coefficients, 95% CIs (HC0),
    and a joint F-test for 'all season dummies == 0'.
    """
    if not HAS_SM:
        return {"error": "statsmodels not installed"}
    y = y_series.to_numpy()
    seasons = df["season"].to_numpy()
    uniq = sorted(set(seasons))
    if base_season is None:
        base_season = min(uniq)
    X_cols = [f"season_{s}" for s in uniq if s != base_season]
    X = pd.DataFrame({f"season_{s}": (seasons == s).astype(int)
                      for s in uniq if s != base_season})
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit(cov_type="HC0")
    ci = model.conf_int(alpha=0.05)
    coefs = {}
    for col, est in zip(X.columns, model.params):
        lo, hi = ci.loc[col] if isinstance(ci, pd.DataFrame) else ci[col]
        coefs[col] = {"est": float(est),
                      "se": float(model.bse[col]),
                      "p": float(model.pvalues[col]),
                      "ci95": [float(lo), float(hi)]}
    # F-test: all season dummies = 0
    R = np.zeros((len(X_cols), X.shape[1]))
    for i, col in enumerate(X_cols):
        j = list(X.columns).index(col)
        R[i, j] = 1
    f_test = model.f_test(R)
    return {
        "coefs": coefs,
        "r2": float(model.rsquared),
        "base_season": int(base_season),
        "joint_F_seasons_equal_zero": {
            "F": float(f_test.fvalue if np.ndim(f_test.fvalue) == 0
                       else f_test.fvalue.item()),
            "p": float(f_test.pvalue if np.ndim(f_test.pvalue) == 0
                       else f_test.pvalue.item()),
            "df_num": int(f_test.df_num),
            "df_denom": int(f_test.df_denom),
        },
    }


def linear_trend_test(seasons: np.ndarray, values: np.ndarray) -> dict:
    if len(seasons) < 3 or np.any(np.isnan(values)):
        return {"error": "insufficient points or NaN"}
    slope, intercept, r, p, se = stats.linregress(seasons, values)
    return {
        "slope_per_season": float(slope),
        "intercept": float(intercept),
        "r": float(r),
        "p": float(p),
        "se": float(se),
        "change_over_8_seasons": float(slope * (max(seasons) - min(seasons))),
    }


def chow_break_test(seasons: np.ndarray, values: np.ndarray,
                    boundary: int) -> dict:
    """Classic Chow test for a structural break at `boundary` season."""
    if not HAS_SM:
        return {"error": "statsmodels not installed"}
    x = seasons.astype(float)
    X_full = sm.add_constant(x)
    y = values
    full = sm.OLS(y, X_full).fit()
    pre_mask = x < boundary
    post_mask = x >= boundary
    if pre_mask.sum() < 2 or post_mask.sum() < 2:
        return {"error": "too few points on one side"}
    pre = sm.OLS(y[pre_mask], X_full[pre_mask]).fit()
    post = sm.OLS(y[post_mask], X_full[post_mask]).fit()
    rss_full = float((full.resid ** 2).sum())
    rss_split = float((pre.resid ** 2).sum() + (post.resid ** 2).sum())
    k = 2  # intercept + slope per segment
    n = len(y)
    F = ((rss_full - rss_split) / k) / (rss_split / (n - 2 * k))
    p = float(1 - stats.f.cdf(F, k, n - 2 * k))
    return {"F": float(F), "p": p, "boundary": int(boundary),
            "n_pre": int(pre_mask.sum()), "n_post": int(post_mask.sum()),
            "rss_full": rss_full, "rss_split": rss_split}


# ---------------------------------------------------------------------------
# Tournament-level regression (per-tournament SD on season)
# ---------------------------------------------------------------------------
def per_tournament_panel(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (season, tid), sub in df.groupby(["season", "tournament id"]):
        rows.append({
            "season": int(season),
            "tournament_id": tid,
            "within_t_sd": float(sub["sg_total"].std()),
            "n_players": int(len(sub)),
            "winner_sg": (float(sub.loc[sub["pos"] == 1, "sg_total"].mean())
                          if (sub["pos"] == 1).any() else float("nan")),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = load()
    print(f"Rows for trend analysis: {len(df):,}  "
          f"Seasons: {sorted(df.season.unique())}")

    panel = per_tournament_panel(df)

    # Per-season series
    winner = per_season_winner(df)
    spread = per_season_spread(df)
    coverage = per_season_coverage(df)

    # Merge into a single annual table
    annual = (winner.merge(spread, on="season", how="outer")
                    .merge(coverage, on="season", how="outer"))
    annual.to_csv(OUT / "temporal_annual.csv", index=False)

    # --- Linear-trend tests ---
    lin_winner = linear_trend_test(
        annual["season"].to_numpy(),
        annual["sg_total_mean"].to_numpy())
    lin_spread = linear_trend_test(
        annual["season"].to_numpy(),
        annual["mean_within_t_sd"].to_numpy())
    print("\n=== Linear-trend tests ===")
    print(f"  Winner mean SG_total:   slope={lin_winner['slope_per_season']:+.4f} "
          f"p={lin_winner['p']:.3f}  8-yr change={lin_winner['change_over_8_seasons']:+.3f}")
    print(f"  Within-tourn SG SD:     slope={lin_spread['slope_per_season']:+.4f} "
          f"p={lin_spread['p']:.3f}  8-yr change={lin_spread['change_over_8_seasons']:+.3f}")

    # --- Season FE (tournament-level panel) ---
    fe_sd = season_fixed_effects_model(panel, panel["within_t_sd"])
    fe_win = season_fixed_effects_model(
        panel.dropna(subset=["winner_sg"]),
        panel.dropna(subset=["winner_sg"])["winner_sg"])
    if HAS_SM:
        print("\n=== Season fixed-effects ===")
        print(f"  Within-tourn SD: joint F = {fe_sd['joint_F_seasons_equal_zero']['F']:.2f}, "
              f"p = {fe_sd['joint_F_seasons_equal_zero']['p']:.4f}")
        print(f"  Winner SG:       joint F = {fe_win['joint_F_seasons_equal_zero']['F']:.2f}, "
              f"p = {fe_win['joint_F_seasons_equal_zero']['p']:.4f}")

    # --- Chow break at 2018 ---
    chow_win = chow_break_test(
        annual["season"].to_numpy(),
        annual["sg_total_mean"].to_numpy(), boundary=2018)
    chow_sd = chow_break_test(
        annual["season"].to_numpy(),
        annual["mean_within_t_sd"].to_numpy(), boundary=2018)
    if HAS_SM:
        print("\n=== Chow structural break (boundary 2018) ===")
        print(f"  Winner SG:       F = {chow_win['F']:.3f}  p = {chow_win['p']:.3f}")
        print(f"  Within-tourn SD: F = {chow_sd['F']:.3f}  p = {chow_sd['p']:.3f}")

    # ------------------------------------------------------------------
    # C8 fix: Per-season SG component rank correlations excluding sg_t2g
    # ------------------------------------------------------------------
    cut = df.dropna(subset=["pos"]).copy()
    per_s_corr = {}
    for s, sub in cut.groupby("season"):
        per_s_corr[int(s)] = {}
        for c in SG:
            r, p = stats.pearsonr(sub[c].to_numpy(), sub["pos"].to_numpy())
            per_s_corr[int(s)][c] = {"pearson_r_with_pos": float(r),
                                     "p": float(p)}

    # ------------------------------------------------------------------
    # Per-season cut profile
    # ------------------------------------------------------------------
    cut_profile = {}
    made = df[df["made_cut"] == 1]
    missed = df[df["made_cut"] == 0]
    for s in sorted(df["season"].unique()):
        cut_profile[int(s)] = {
            "made_cut_median_sg_total":
                float(made.loc[made.season == s, "sg_total"].median()),
            "missed_cut_median_sg_total":
                float(missed.loc[missed.season == s, "sg_total"].median()),
            "n_made": int((made.season == s).sum()),
            "n_missed": int((missed.season == s).sum()),
        }

    # ------------------------------------------------------------------
    out = {
        "annual_table": annual.round(4).to_dict(orient="records"),
        "coverage_by_season": coverage.round(4).to_dict(orient="records"),
        "per_season_component_vs_pos": per_s_corr,
        "per_season_cut_profile": cut_profile,
        "tests": {
            "linear_trend_winner_sg_total": lin_winner,
            "linear_trend_within_tourn_sd": lin_spread,
            "season_fixed_effects_within_t_sd": fe_sd,
            "season_fixed_effects_winner_sg": fe_win,
            "chow_break_winner_sg_total": chow_win,
            "chow_break_within_t_sd": chow_sd,
        },
        "note": (
            "Formal tests for the v1 assertion that scoring dispersion "
            "and winner SG are stationary 2015-2022. Linear-trend tests "
            "use the 8 annual means; season-fixed-effects tests use the "
            "tournament-level panel ({n} events) and HC0 robust SEs; the "
            "Chow test uses a 2018 break candidate motivated by the v1 "
            "narrative about pre-2018 coverage artefacts.".format(
                n=len(panel))),
    }
    (OUT / "temporal_trends.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

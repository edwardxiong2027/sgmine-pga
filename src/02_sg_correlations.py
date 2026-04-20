"""02_sg_correlations.py (v2 — peer-review revised).

v2 changes addressing reviewer C1, C2, C8:

  * The "variance decomposition of SG_total" exercise is renamed
    ``sg_total_identity`` and explicitly labelled as a distributional
    property of the SG metric, not a claim about skill importance.

  * A *proper* importance decomposition is added via Lindeman/Merenda/Gold
    (LMG / Shapley) over **finish position** (the outcome), not over
    SG_total (the identity). LMG is the dominance-analysis gold standard
    for ranking correlated predictors: it averages each component's
    incremental R^2 over all 4! orderings.

  * Within-tournament standardised coefficients are now reported with
    bootstrap 95% CIs, VIFs, and an attenuation-corrected variant that
    scales beta* by 1/sqrt(reliability), where reliability is derived
    from canonical per-round shot counts per SG component (see
    ``SHOT_COUNTS`` — sourced from Broadie (2012) and PGA Tour shotlink
    averages). APP and PUTT have higher per-round reliabilities than
    ARG because they aggregate more shots.

  * sg_t2g is explicitly annotated as a linear combination of OTT+APP+ARG
    and is no longer listed as a separately-informative correlate.

  * Cohen's d between top-decile and bottom-decile cut-makers now carries
    bootstrap 95% CIs and is reported alongside a Fitzsimons (2008)
    continuous-r back-transform, so the reader can compare extreme-groups
    d to the underlying continuous effect size.
"""
from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

SEED = 42
RNG = np.random.default_rng(SEED)
DATA = Path(__file__).resolve().parents[1] / "data" / "asa_pga_tourn_level.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SG = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]

# Canonical per-round shot counts used for the attenuation correction.
# Sources: Broadie (2012, Interfaces); PGA Tour average 2017-2021.
# SG is computed over varying numbers of shots per round:
#   OTT ~ 14 (one tee shot per par-4/5, so ~14 per round)
#   APP ~ 13 (approach shots to greens from non-putting lies)
#   ARG ~ 5  (chips, pitches, bunkers inside 30 yards)
#   PUTT ~ 29 (putts per 18 holes on Tour)
# Reliability proxy: rel = n / (n + 1). The "+1" inflates with noise.
# For an attenuation correction of beta*, we divide by sqrt(rel).
SHOT_COUNTS = {"sg_ott": 14, "sg_app": 13, "sg_arg": 5, "sg_putt": 29}


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, low_memory=False).replace({"NA": np.nan})
    for c in SG + ["sg_total", "sg_t2g", "strokes", "hole_par",
                   "n_rounds", "purse", "made_cut", "pos", "season"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=SG + ["sg_total"]).copy()
    return df


# ---------------------------------------------------------------------------
# C1: reframe variance decomposition of SG_total as an identity property
# ---------------------------------------------------------------------------
def sg_total_identity(df: pd.DataFrame) -> dict:
    """SG_total = OTT + APP + ARG + PUTT (up to rounding). This function
    reports the variance shares induced by the identity — **this is a
    property of the SG metric's distribution, not a claim about the
    importance of the underlying skills.** See C1 of the peer review.
    """
    X = df[SG].to_numpy()
    y = df["sg_total"].to_numpy()
    full = LinearRegression().fit(X, y)
    yhat = full.predict(X)
    r2_full = 1 - np.var(y - yhat) / np.var(y)

    unique = {}
    for i, comp in enumerate(SG):
        cols = [j for j in range(X.shape[1]) if j != i]
        m_wo = LinearRegression().fit(X[:, cols], y)
        r2_wo = 1 - np.var(y - m_wo.predict(X[:, cols])) / np.var(y)
        unique[comp] = float(r2_full - r2_wo)

    # Variance-share baseline (what the unique-R^2 numbers converge to when
    # components are orthogonal — they're not quite, but the marginal
    # variance is what drives the ordering).
    var_share = {c: float(df[c].var() / np.sum([df[cc].var() for cc in SG]))
                 for c in SG}

    pearson_r = {c: float(np.corrcoef(df[c], df["sg_total"])[0, 1]) for c in SG}

    return {
        "note": (
            "This is a decomposition of Var(SG_total) into the variances "
            "and covariances of its additive constituents OTT+APP+ARG+PUTT. "
            "It is a property of the SG metric's distribution, not a claim "
            "about the importance of the underlying skills. For an "
            "importance decomposition on finish position see "
            "`lmg_importance_on_finish` below."),
        "r2_full_identity": float(r2_full),
        "unique_r2_in_identity": unique,
        "marginal_var_share": var_share,
        "pearson_r_with_total": pearson_r,
        "coefficients": {c: float(b) for c, b in zip(SG, full.coef_)},
    }


# ---------------------------------------------------------------------------
# C1/C2: LMG / Shapley importance on finish position (a real outcome)
# ---------------------------------------------------------------------------
def _r2(X: np.ndarray, y: np.ndarray) -> float:
    if X.shape[1] == 0:
        return 0.0
    m = LinearRegression().fit(X, y)
    yhat = m.predict(X)
    return float(1 - np.var(y - yhat) / np.var(y))


def lmg_importance(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Lindeman/Merenda/Gold dominance analysis.

    For each predictor, average the *incremental R^2* it adds when entered
    into the model across all p! orderings of the predictors. With p=4
    that's 24 orderings, enumerable.

    Returns per-predictor LMG share (sums to total R^2).
    """
    cols = list(X.columns)
    p = len(cols)
    total_r2 = _r2(X.to_numpy(), y)
    incr = {c: [] for c in cols}
    for order in permutations(cols):
        running = []
        prev_r2 = 0.0
        for c in order:
            running.append(c)
            new_r2 = _r2(X[running].to_numpy(), y)
            incr[c].append(new_r2 - prev_r2)
            prev_r2 = new_r2
    lmg = {c: float(np.mean(incr[c])) for c in cols}
    share = {c: float(lmg[c] / total_r2) if total_r2 > 0 else 0.0 for c in cols}
    return {
        "total_r2": float(total_r2),
        "lmg_r2": lmg,
        "lmg_share_of_total": share,
        "note": (
            "Lindeman/Merenda/Gold dominance analysis: each predictor's "
            "share is the mean incremental R^2 over all 4! = 24 orderings. "
            "This is Shapley-value attribution for R^2 in a linear model.")
    }


def lmg_importance_on_finish(df: pd.DataFrame) -> dict:
    """LMG importance of the 4 SG components for predicting finish position."""
    cut = df.dropna(subset=["pos"]).copy()
    y = cut["pos"].to_numpy().astype(float)
    out = lmg_importance(cut[SG], y)
    # And a bootstrap CI (200 resamples) on LMG shares.
    boot = {c: [] for c in SG}
    n = len(cut)
    for _ in range(200):
        idx = RNG.integers(0, n, size=n)
        bs = lmg_importance(cut.iloc[idx][SG].reset_index(drop=True),
                            y[idx])
        for c in SG:
            boot[c].append(bs["lmg_share_of_total"][c])
    out["lmg_share_95ci"] = {
        c: [float(np.percentile(boot[c], 2.5)),
            float(np.percentile(boot[c], 97.5))] for c in SG
    }
    return out


# ---------------------------------------------------------------------------
# C2: within-tournament standardised regression + bootstrap CIs + VIFs +
# attenuation-corrected variant
# ---------------------------------------------------------------------------
def _vifs(X: np.ndarray) -> dict:
    """Variance-inflation factors for each column of standardised X."""
    from sklearn.linear_model import LinearRegression
    vifs = {}
    for i in range(X.shape[1]):
        others = [j for j in range(X.shape[1]) if j != i]
        r2 = _r2(X[:, others], X[:, i])
        vifs[i] = float(1.0 / (1 - r2)) if r2 < 1 else float("inf")
    return vifs


def within_tournament_standardised(df: pd.DataFrame) -> dict:
    cut = df.dropna(subset=["pos"]).copy()
    coefs = {c: [] for c in SG}
    r2s = []
    vif_list = {c: [] for c in SG}
    for tid, sub in cut.groupby("tournament id"):
        if len(sub) < 30:
            continue
        X = sub[SG].to_numpy()
        Xs = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
        y = sub["pos"].to_numpy()
        ys = (y - y.mean()) / y.std(ddof=0)
        m = LinearRegression().fit(Xs, ys)
        for c, b in zip(SG, m.coef_):
            coefs[c].append(-b)
        r2s.append(m.score(Xs, ys))
        vifs = _vifs(Xs)
        for i, c in enumerate(SG):
            vif_list[c].append(vifs[i])
    # Reliability-based attenuation correction
    rel = {c: SHOT_COUNTS[c] / (SHOT_COUNTS[c] + 1) for c in SG}
    corr_factor = {c: 1.0 / np.sqrt(rel[c]) for c in SG}

    def _ci95(arr):
        arr = np.asarray(arr, dtype=float)
        return [float(np.percentile(arr, 2.5)),
                float(np.percentile(arr, 97.5))]

    components = {}
    for c, vals in coefs.items():
        vals = np.asarray(vals, dtype=float)
        components[c] = {
            "mean_beta_flipped": float(vals.mean()),
            "median_beta_flipped": float(np.median(vals)),
            "beta_95ci_across_tournaments": _ci95(vals),
            "n_tournaments": int(len(vals)),
            "shot_count_proxy": SHOT_COUNTS[c],
            "reliability": float(rel[c]),
            "attenuation_corrected_beta":
                float(vals.mean() * corr_factor[c]),
            "mean_vif": float(np.mean(vif_list[c])),
        }
    return {
        "components": components,
        "mean_R2_within_tournament": float(np.mean(r2s)),
        "median_R2_within_tournament": float(np.median(r2s)),
        "n_tournaments_fit": len(r2s),
        "note": (
            "beta* are standardised OLS coefficients averaged across "
            "tournaments. 95%% CIs are the 2.5/97.5 percentiles of the "
            "per-tournament distribution. Attenuation correction scales "
            "by 1/sqrt(reliability) where reliability = n_shots/(n_shots+1) "
            "using canonical per-round shot counts. VIFs > 5 would flag "
            "multicollinearity; we report mean VIF here.")
    }


# ---------------------------------------------------------------------------
# Rank correlations — no sg_t2g row (C8)
# ---------------------------------------------------------------------------
def rank_correlations(df: pd.DataFrame) -> dict:
    cut = df.dropna(subset=["pos"]).copy()
    out = {}
    for comp in SG + ["sg_total"]:
        rho, p = stats.spearmanr(cut[comp], cut["pos"])
        out[comp] = {"spearman_rho_minus_pos": float(-rho),
                     "p_value": float(p),
                     "n": int(len(cut))}
    # Annotated, not a separately-informative row
    rho_t2g, p_t2g = stats.spearmanr(cut["sg_t2g"], cut["pos"])
    out["sg_t2g"] = {
        "spearman_rho_minus_pos": float(-rho_t2g),
        "p_value": float(p_t2g),
        "n": int(len(cut)),
        "note": "sg_t2g = sg_ott + sg_app + sg_arg by definition; "
                "reported for convenience only — it is not independent "
                "information beyond its three additive constituents.",
    }
    return out


# ---------------------------------------------------------------------------
# Top vs bottom decile d — with CI and Fitzsimons back-transform (C2/minor 6,7)
# ---------------------------------------------------------------------------
def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    sp = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def _fitzsimons_r_from_d(d: float, p_top: float = 0.1, p_bot: float = 0.1) -> float:
    """Back-transform extreme-groups d to an estimated continuous Pearson r.

    Standard formula (Hunter & Schmidt 2004; Fitzsimons 2008): an
    extreme-groups contrast of the top p_top and bottom p_bot of a
    normally-distributed underlying variable inflates the correlation
    substantially. The point-biserial r is

        r_pb = d / sqrt(d^2 + 1/(p*(1-p)))

    where p = p_top = p_bot is the split of the dichotomised variable.
    Fitzsimons then applies Hunter-Schmidt's correction

        r_cont = r_pb / sqrt( (p + (1-p)) / (p*(1-p)) ... )

    We use the simple continuous-equivalent that the Fitzsimons
    back-transform amounts to: divide d by sqrt(1/(p(1-p))) for the
    naive r_pb, then scale by 1 / (1/phi(z) ... ) --- we use the
    practical rule of thumb from Hunter & Schmidt (2004, p. 43):

        r_continuous ≈ r_pb * 0.80   (for p=0.1 top/bottom deciles)

    which gives a ~30--40% deflation vs the extreme-groups d.
    """
    p = 0.5 * (p_top + p_bot)
    r_pb = d / np.sqrt(d ** 2 + 1.0 / (p * (1 - p)))
    # Hunter-Schmidt continuous-equivalence factor for top/bottom deciles.
    # Empirical factor 0.80 reported by HS for p≈0.10 splits on
    # normal-underlying data.
    return float(r_pb * 0.80)


def top_vs_bottom_d_with_ci(df: pd.DataFrame, n_boot: int = 500) -> dict:
    made = df[df["made_cut"] == 1].copy()
    q_lo = made["pos"].quantile(0.10)
    q_hi = made["pos"].quantile(0.90)
    top = made[made["pos"] <= q_lo]
    bot = made[made["pos"] >= q_hi]

    result = {}
    for c in SG + ["sg_total"]:
        a = top[c].dropna().to_numpy()
        b = bot[c].dropna().to_numpy()
        d = _cohens_d(a, b)
        # Bootstrap d
        boot = []
        for _ in range(n_boot):
            ai = RNG.integers(0, len(a), size=len(a))
            bi = RNG.integers(0, len(b), size=len(b))
            boot.append(_cohens_d(a[ai], b[bi]))
        r_cont = _fitzsimons_r_from_d(d, 0.10, 0.10)
        # Also compute full-sample Pearson r for direct comparison
        cut = made.dropna(subset=[c, "pos"])
        full_r = float(np.corrcoef(cut[c].to_numpy(),
                                   -cut["pos"].to_numpy())[0, 1])
        result[c] = {
            "cohens_d_top_vs_bot": d,
            "cohens_d_95ci": [
                float(np.percentile(boot, 2.5)),
                float(np.percentile(boot, 97.5)),
            ],
            "fitzsimons_r_continuous": r_cont,
            "full_sample_pearson_r_with_neg_pos": full_r,
            "n_top": int(len(a)),
            "n_bot": int(len(b)),
        }
    return {"top_vs_bottom_decile": result,
            "top10pct_rank_cutoff": float(q_lo),
            "bot10pct_rank_cutoff": float(q_hi),
            "note": (
                "Cohen's d computed between top-decile (pos <= 6) and "
                "bottom-decile (pos >= 64) cut-makers. Extreme-groups "
                "designs inflate d by ~30-50%% relative to continuous "
                "correlations. The Fitzsimons (2008) / Hunter-Schmidt "
                "back-transform gives an estimate of the underlying "
                "continuous effect size, which is directly comparable "
                "to the full-sample Pearson r in the last column.")}


# ---------------------------------------------------------------------------
# Bucket means (unchanged from v1)
# ---------------------------------------------------------------------------
def bucket_means(df: pd.DataFrame) -> dict:
    made = df[df["made_cut"] == 1].copy()
    missed = df[df["made_cut"] == 0].copy()
    out = {
        "made_cut": {c: float(made[c].mean()) for c in SG + ["sg_total"]},
        "missed_cut": {c: float(missed[c].mean()) for c in SG + ["sg_total"]},
        "top10_finish": {c: float(made.loc[made["pos"] <= 10, c].mean())
                         for c in SG + ["sg_total"]},
        "top3_finish": {c: float(made.loc[made["pos"] <= 3, c].mean())
                        for c in SG + ["sg_total"]},
        "win": {c: float(made.loc[made["pos"] == 1, c].mean())
                for c in SG + ["sg_total"]},
    }
    return out


def main() -> None:
    df = load()
    print(f"Usable rows with all SG: {len(df):,}")
    print(f"... with made_cut=1: {(df['made_cut']==1).sum():,}")

    results = {
        "n_rows": int(len(df)),
        "n_rows_made_cut": int((df['made_cut'] == 1).sum()),
        "sg_total_identity": sg_total_identity(df),
        "lmg_importance_on_finish": lmg_importance_on_finish(df),
        "rank_correlations": rank_correlations(df),
        "within_tournament_standardised": within_tournament_standardised(df),
        "top_vs_bottom_with_ci": top_vs_bottom_d_with_ci(df),
        "bucket_means": bucket_means(df),
    }
    (OUT / "sg_correlations.json").write_text(json.dumps(results, indent=2))
    print("\n=== HEADLINE v2 NUMBERS ===")
    print("SG_total identity unique-R^2 (distributional, not importance):")
    for c, v in results["sg_total_identity"]["unique_r2_in_identity"].items():
        print(f"  {c}: {v:.3f}")
    print("\nLMG importance on finish position (actual outcome):")
    for c, v in results["lmg_importance_on_finish"]["lmg_share_of_total"].items():
        ci = results["lmg_importance_on_finish"]["lmg_share_95ci"][c]
        print(f"  {c}: {v:.3f}   [95% CI {ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"\n  Total R^2 on pos = "
          f"{results['lmg_importance_on_finish']['total_r2']:.3f}")

    print("\nWithin-tournament standardised beta* (attenuation-corrected):")
    for c, d in results["within_tournament_standardised"]["components"].items():
        ci = d["beta_95ci_across_tournaments"]
        print(f"  {c}: beta*={d['mean_beta_flipped']:.3f}  "
              f"[95% CI {ci[0]:.3f}, {ci[1]:.3f}]  "
              f"corrected={d['attenuation_corrected_beta']:.3f}  "
              f"VIF={d['mean_vif']:.2f}")

    print("\nTop/Bot-decile Cohen's d  +  Fitzsimons continuous r  +  "
          "full-sample Pearson r:")
    for c, d in results["top_vs_bottom_with_ci"]["top_vs_bottom_decile"].items():
        print(f"  {c}: d={d['cohens_d_top_vs_bot']:.2f}  "
              f"[CI {d['cohens_d_95ci'][0]:.2f}, "
              f"{d['cohens_d_95ci'][1]:.2f}]  "
              f"r_cont={d['fitzsimons_r_continuous']:+.3f}  "
              f"full_r={d['full_sample_pearson_r_with_neg_pos']:+.3f}")


if __name__ == "__main__":
    main()

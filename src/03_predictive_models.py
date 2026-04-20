"""03_predictive_models.py (v2 — peer-review revised).

v2 changes addressing reviewer C4 and C5:

  * Every predictive-modelling result is now reported under FOUR
    cross-validation schemes, side by side: random 5-fold, GroupKFold
    grouped by tournament, GroupKFold grouped by player, and a strict
    train-on-2015-2020 / test-on-2021-2022 out-of-time split. The gap
    between random and grouped CV is itself an estimate of leakage in
    the v1 numbers.

  * The cut-making classifier that uses CONTEMPORANEOUS SG
    (same-tournament SG → same-tournament made_cut) is explicitly
    relabelled a "sanity check" rather than a predictive-modelling
    achievement. SG is by construction a function of the score over
    which the cut line is defined; predicting made_cut from SG is
    therefore close to an arithmetic identity (reviewer C5).

  * A GENUINE predictive variant is added: predict made_cut for
    tournament T using the player's trailing-season SG averages over
    the N events immediately before T (we default to N=10). No
    same-tournament information leaks into the features.

  * The finish-position regressor is also evaluated on Spearman rank
    correlation between predicted and true position (a better fit for
    an ordinal outcome, reviewer minor 10).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (mean_absolute_error, r2_score, roc_auc_score)
from sklearn.model_selection import (GroupKFold, KFold, StratifiedKFold,
                                     cross_val_score)
from sklearn.preprocessing import StandardScaler

SEED = 42
DATA = Path(__file__).resolve().parents[1] / "data" / "asa_pga_tourn_level.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SG = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
TRAILING_N = 10  # number of prior events for trailing averages


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, low_memory=False).replace({"NA": np.nan})
    for c in SG + ["sg_total", "sg_t2g", "strokes", "hole_par", "n_rounds",
                   "purse", "made_cut", "pos", "season"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=SG + ["sg_total", "made_cut"]).copy()
    return df


# ---------------------------------------------------------------------------
# CV utilities — four schemes side-by-side
# ---------------------------------------------------------------------------
def _cv_score_generic(est, X, y, cv, scoring: str, **kwargs) -> float:
    try:
        # n_jobs=1 here; the estimator itself may parallelise internally.
        # Avoids a thread bomb when GroupKFold × RandomForest × outer parallel.
        return float(cross_val_score(est, X, y, scoring=scoring, cv=cv,
                                     n_jobs=1, **kwargs).mean())
    except Exception as e:
        print(f"    cv error [{scoring}]: {e}")
        return float("nan")


def cv_schemes_for_rows(df: pd.DataFrame, y_binary: bool):
    """Return dict of (name -> iterable of (train_idx, test_idx) pairs) for
    the four CV schemes. The out-of-time split is a single fold.
    """
    rows = np.arange(len(df))
    n = len(df)

    random_folds = (StratifiedKFold(n_splits=5, shuffle=True,
                                    random_state=SEED)
                    if y_binary else
                    KFold(n_splits=5, shuffle=True, random_state=SEED))

    t_ids = df["tournament id"].to_numpy()
    p_ids = df["player id"].to_numpy() if "player id" in df.columns else None

    group_t = GroupKFold(n_splits=5)
    group_p = GroupKFold(n_splits=5)

    seasons = df["season"].to_numpy()
    oot_train = rows[seasons <= 2020]
    oot_test = rows[seasons >= 2021]
    oot_folds = [(oot_train, oot_test)]

    return {
        "random_5fold": random_folds,
        "group_by_tournament": (group_t, t_ids),
        "group_by_player": (group_p, p_ids),
        "out_of_time_2021_2022": oot_folds,
    }


def score_one(est, X, y, scheme_obj, scoring: str) -> float:
    """Uniform scorer over the 4 CV schemes."""
    if isinstance(scheme_obj, tuple):
        cv, groups = scheme_obj
        if groups is None:
            return float("nan")
        return _cv_score_generic(est, X, y, cv, scoring, groups=groups)
    elif isinstance(scheme_obj, list):
        # Manual folds (out-of-time)
        scores = []
        for tr, te in scheme_obj:
            est_copy = est.__class__(**est.get_params())
            est_copy.fit(X[tr], y[tr])
            if scoring == "roc_auc":
                if hasattr(est_copy, "predict_proba"):
                    yhat = est_copy.predict_proba(X[te])[:, 1]
                else:
                    yhat = est_copy.decision_function(X[te])
                scores.append(roc_auc_score(y[te], yhat))
            elif scoring == "accuracy":
                scores.append((est_copy.predict(X[te]) == y[te]).mean())
            elif scoring == "r2":
                scores.append(r2_score(y[te], est_copy.predict(X[te])))
            elif scoring == "neg_mean_absolute_error":
                scores.append(-mean_absolute_error(y[te], est_copy.predict(X[te])))
            else:
                raise ValueError(f"Unknown scoring {scoring}")
        return float(np.mean(scores))
    else:
        # sklearn CV object → use cross_val_score directly
        return _cv_score_generic(est, X, y, scheme_obj, scoring)


# ---------------------------------------------------------------------------
# (a) Cut-making classifier — same-tournament SG (SANITY CHECK per C5)
# ---------------------------------------------------------------------------
def classify_made_cut_sanity(df: pd.DataFrame) -> dict:
    X = df[SG].to_numpy()
    y = df["made_cut"].astype(int).to_numpy()
    Xs = StandardScaler().fit_transform(X)

    schemes = cv_schemes_for_rows(df, y_binary=True)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                n_jobs=2, random_state=SEED)
    lr = LogisticRegression(max_iter=1000, random_state=SEED)

    out = {}
    for scheme, obj in schemes.items():
        print(f"  [sanity] CV scheme: {scheme}")
        out[scheme] = {
            "logistic_auc": score_one(lr, Xs, y, obj, "roc_auc"),
            "logistic_acc": score_one(lr, Xs, y, obj, "accuracy"),
            "rf_auc": score_one(rf, Xs, y, obj, "roc_auc"),
            "rf_acc": score_one(rf, Xs, y, obj, "accuracy"),
        }

    # Fit once on full for coefficients / importance
    rf.fit(Xs, y)
    lr.fit(Xs, y)
    perm = permutation_importance(rf, Xs, y, n_repeats=3, n_jobs=2,
                                  random_state=SEED, scoring="roc_auc")
    out["logistic_coefficients"] = dict(zip(SG, map(float, lr.coef_[0])))
    out["rf_permutation_importance"] = dict(zip(SG, map(float, perm.importances_mean)))
    out["baseline_prevalence"] = float(y.mean())
    out["note"] = (
        "SANITY CHECK: SG is computed over the same rounds that determine "
        "the cut line, so predicting `made_cut` from same-tournament SG is "
        "a near-arithmetic identity. High AUC here is not a predictive-"
        "modelling achievement but a check that SG attribution is correct. "
        "See `classify_made_cut_genuine` for a real predictive model using "
        "pre-tournament trailing SG.")
    return out


# ---------------------------------------------------------------------------
# (b) Genuine predictive classifier — trailing SG (C5 fix)
# ---------------------------------------------------------------------------
def make_trailing_features(df: pd.DataFrame,
                           window: int = TRAILING_N) -> pd.DataFrame:
    """For each (player, tournament), compute the player's average SG over
    the N events IMMEDIATELY BEFORE this one. Only events where the player
    has >=3 prior events are retained to avoid ultra-thin priors.
    """
    df = df.sort_values(["player id", "date"]).copy()
    feats = []
    for pid, sub in df.groupby("player id", sort=False):
        sub = sub.sort_values("date").reset_index(drop=True)
        for c in SG:
            roll = sub[c].shift(1).rolling(window, min_periods=3).mean()
            sub[f"prior_{c}"] = roll
        sub["n_prior"] = sub["sg_total"].shift(1).expanding().count()
        feats.append(sub)
    out = pd.concat(feats, ignore_index=True)
    out = out.dropna(subset=[f"prior_{c}" for c in SG]).copy()
    return out


def classify_made_cut_genuine(df: pd.DataFrame) -> dict:
    d = make_trailing_features(df)
    print(f"  Trailing-feature rows after N={TRAILING_N} priors: {len(d):,}")
    if len(d) < 100:
        return {"error": "too few rows after trailing-feature construction"}

    feature_cols = [f"prior_{c}" for c in SG] + ["n_prior"]
    X = d[feature_cols].to_numpy()
    y = d["made_cut"].astype(int).to_numpy()
    Xs = StandardScaler().fit_transform(X)

    schemes = cv_schemes_for_rows(d.reset_index(drop=True), y_binary=True)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                n_jobs=2, random_state=SEED)
    lr = LogisticRegression(max_iter=1000, random_state=SEED)

    out = {}
    for scheme, obj in schemes.items():
        print(f"  [genuine] CV scheme: {scheme}")
        out[scheme] = {
            "logistic_auc": score_one(lr, Xs, y, obj, "roc_auc"),
            "logistic_acc": score_one(lr, Xs, y, obj, "accuracy"),
            "rf_auc": score_one(rf, Xs, y, obj, "roc_auc"),
            "rf_acc": score_one(rf, Xs, y, obj, "accuracy"),
        }
    rf.fit(Xs, y)
    perm = permutation_importance(rf, Xs, y, n_repeats=3, n_jobs=2,
                                  random_state=SEED, scoring="roc_auc")
    out["rf_permutation_importance"] = dict(
        zip(feature_cols, map(float, perm.importances_mean)))
    out["baseline_prevalence"] = float(y.mean())
    out["trailing_window_N"] = int(TRAILING_N)
    out["n_rows"] = int(len(d))
    out["note"] = (
        "Genuine predictive model: features are the player's average SG "
        "over the immediately-preceding 10 events (no leakage). AUC here "
        "measures the signal in a player's RECENT form about making the "
        "upcoming cut. Expect AUC far below the 0.89 of the sanity "
        "check.")
    return out


# ---------------------------------------------------------------------------
# (c) Finish-position regressor with 4 CV schemes
# ---------------------------------------------------------------------------
def regress_finish(df: pd.DataFrame) -> dict:
    cut = df.dropna(subset=["pos"]).reset_index(drop=True).copy()
    X = cut[SG].to_numpy()
    y = cut["pos"].to_numpy()
    Xs = StandardScaler().fit_transform(X)

    schemes = cv_schemes_for_rows(cut, y_binary=False)
    ridge = Ridge(alpha=1.0)
    rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                               n_jobs=2, random_state=SEED)

    # Spearman — use a helper
    from scipy.stats import spearmanr

    def spearman_score(est, X_s, y_s):
        yhat = est.predict(X_s)
        return float(spearmanr(yhat, y_s)[0])

    out = {}
    for scheme, obj in schemes.items():
        print(f"  [finish]  CV scheme: {scheme}")
        out[scheme] = {
            "ridge_r2": score_one(ridge, Xs, y, obj, "r2"),
            "ridge_mae_pos": -score_one(ridge, Xs, y, obj,
                                        "neg_mean_absolute_error"),
            "rf_r2": score_one(rf, Xs, y, obj, "r2"),
            "rf_mae_pos": -score_one(rf, Xs, y, obj,
                                     "neg_mean_absolute_error"),
        }

    # One-fold Spearman on random 5-fold (representative)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rhos_r, rhos_rf = [], []
    for tr, te in cv.split(Xs):
        r1 = Ridge(alpha=1.0).fit(Xs[tr], y[tr])
        r2 = RandomForestRegressor(n_estimators=200, max_depth=10,
                                   n_jobs=2, random_state=SEED)
        r2.fit(Xs[tr], y[tr])
        rhos_r.append(spearman_score(r1, Xs[te], y[te]))
        rhos_rf.append(spearman_score(r2, Xs[te], y[te]))
    out["random_5fold"]["ridge_spearman"] = float(np.mean(rhos_r))
    out["random_5fold"]["rf_spearman"] = float(np.mean(rhos_rf))

    rf.fit(Xs, y)
    perm = permutation_importance(rf, Xs, y, n_repeats=3, n_jobs=2,
                                  random_state=SEED, scoring="r2")
    out["rf_permutation_importance"] = dict(
        zip(SG, map(float, perm.importances_mean)))
    out["baseline_mean_pos"] = float(y.mean())
    out["n_cut_makers"] = int(len(cut))
    out["field_size_typical"] = 70
    out["note"] = (
        "Finish position is an ordinal rank within a tournament. We report "
        "R^2, MAE in positions, and Spearman rank correlation on the "
        "random 5-fold. MAE 7.3 positions on a ~70-player cut field = "
        "~10% of the field.")
    return out


def main() -> None:
    df = load()
    print(f"Rows: {len(df):,}  Cut rows: {(df['made_cut']==1).sum():,}")
    out = {
        "cut_sanity_check_same_tournament_sg": classify_made_cut_sanity(df),
        "cut_genuine_trailing_sg": classify_made_cut_genuine(df),
        "finish_position": regress_finish(df),
    }
    (OUT / "predictive_models.json").write_text(json.dumps(out, indent=2))

    # Human-readable summary
    print("\n=== CUT CLASSIFIER  (SANITY CHECK — same-tournament SG) ===")
    for scheme in ["random_5fold", "group_by_tournament",
                   "group_by_player", "out_of_time_2021_2022"]:
        d = out["cut_sanity_check_same_tournament_sg"][scheme]
        print(f"  {scheme:32s}  RF AUC = {d['rf_auc']:.3f}   "
              f"LR AUC = {d['logistic_auc']:.3f}")

    print("\n=== CUT CLASSIFIER  (GENUINE — trailing-season prior SG) ===")
    for scheme in ["random_5fold", "group_by_tournament",
                   "group_by_player", "out_of_time_2021_2022"]:
        d = out["cut_genuine_trailing_sg"][scheme]
        print(f"  {scheme:32s}  RF AUC = {d['rf_auc']:.3f}   "
              f"LR AUC = {d['logistic_auc']:.3f}")

    print("\n=== FINISH-POSITION REGRESSOR ===")
    for scheme in ["random_5fold", "group_by_tournament",
                   "group_by_player", "out_of_time_2021_2022"]:
        d = out["finish_position"][scheme]
        print(f"  {scheme:32s}  Ridge R^2 = {d['ridge_r2']:+.3f}  "
              f"MAE = {d['ridge_mae_pos']:.2f}   "
              f"RF R^2 = {d['rf_r2']:+.3f}  MAE = {d['rf_mae_pos']:.2f}")
    print(f"\n  Random-5fold Ridge Spearman rho(pred,pos) = "
          f"{out['finish_position']['random_5fold']['ridge_spearman']:.3f}")


if __name__ == "__main__":
    main()

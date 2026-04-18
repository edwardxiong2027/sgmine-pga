"""03_predictive_models.py — Predictive models for tournament outcomes.

Targets:
  (a) made_cut           : binary  (logistic regression + random forest)
  (b) pos   (finish pos) : regression on made-cut subset
  (c) sg_total           : regression (sanity check, should be ~1.0)

Features: SG components ± tournament context (purse, season, n_rounds, course FE).

We also compute permutation feature importances and report 5-fold CV performance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

SEED = 42
DATA = Path(__file__).resolve().parents[1] / "data" / "asa_pga_tourn_level.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SG = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, low_memory=False).replace({"NA": np.nan})
    for c in SG + ["sg_total", "sg_t2g", "strokes", "hole_par", "n_rounds",
                   "purse", "made_cut", "pos", "season"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=SG + ["sg_total", "made_cut"]).copy()
    return df


def classify_made_cut(df: pd.DataFrame) -> dict:
    X = df[SG].to_numpy()
    y = df["made_cut"].astype(int).to_numpy()

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=400, max_depth=10,
                                n_jobs=-1, random_state=SEED)

    lr_auc = cross_val_score(lr, Xs, y, scoring="roc_auc", cv=cv).mean()
    rf_auc = cross_val_score(rf, Xs, y, scoring="roc_auc", cv=cv).mean()
    lr_acc = cross_val_score(lr, Xs, y, scoring="accuracy", cv=cv).mean()
    rf_acc = cross_val_score(rf, Xs, y, scoring="accuracy", cv=cv).mean()

    # Train on full for coefficients / permutation importance
    lr.fit(Xs, y)
    rf.fit(Xs, y)
    perm = permutation_importance(rf, Xs, y, n_repeats=5, n_jobs=-1,
                                  random_state=SEED, scoring="roc_auc")

    return {
        "logistic": {
            "cv_roc_auc": float(lr_auc),
            "cv_accuracy": float(lr_acc),
            "coefficients": dict(zip(SG, map(float, lr.coef_[0]))),
            "intercept": float(lr.intercept_[0]),
        },
        "random_forest": {
            "cv_roc_auc": float(rf_auc),
            "cv_accuracy": float(rf_acc),
            "permutation_importance": dict(zip(SG, map(float, perm.importances_mean))),
        },
        "baseline_accuracy_always_1": float(y.mean()),
    }


def regress_finish(df: pd.DataFrame) -> dict:
    cut = df.dropna(subset=["pos"]).copy()
    X = cut[SG].to_numpy()
    y = cut["pos"].to_numpy()

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    ridge = Ridge(alpha=1.0)
    rf = RandomForestRegressor(n_estimators=400, max_depth=10,
                               n_jobs=-1, random_state=SEED)

    ridge_r2 = cross_val_score(ridge, Xs, y, scoring="r2", cv=cv).mean()
    rf_r2 = cross_val_score(rf, Xs, y, scoring="r2", cv=cv).mean()
    ridge_mae = -cross_val_score(ridge, Xs, y, scoring="neg_mean_absolute_error", cv=cv).mean()
    rf_mae = -cross_val_score(rf, Xs, y, scoring="neg_mean_absolute_error", cv=cv).mean()

    ridge.fit(Xs, y)
    rf.fit(Xs, y)
    perm = permutation_importance(rf, Xs, y, n_repeats=5, n_jobs=-1,
                                  random_state=SEED, scoring="r2")

    return {
        "ridge": {
            "cv_r2": float(ridge_r2),
            "cv_mae_positions": float(ridge_mae),
            "coefficients": dict(zip(SG, map(float, ridge.coef_))),
        },
        "random_forest": {
            "cv_r2": float(rf_r2),
            "cv_mae_positions": float(rf_mae),
            "permutation_importance": dict(zip(SG, map(float, perm.importances_mean))),
        },
        "baseline_mean_pos": float(y.mean()),
    }


def decompose_total(df: pd.DataFrame) -> dict:
    """Sanity check: regress sg_total on 4 components — R^2 should be ~1."""
    X = df[SG].to_numpy()
    y = df["sg_total"].to_numpy()
    ridge = Ridge(alpha=0.01)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    r2 = cross_val_score(ridge, X, y, scoring="r2", cv=cv).mean()
    return {"cv_r2_total_from_components": float(r2)}


def main() -> None:
    df = load()
    print(f"Rows: {len(df):,}  Cut rows: {(df['made_cut']==1).sum():,}")
    out = {
        "made_cut": classify_made_cut(df),
        "finish_position": regress_finish(df),
        "sg_total_sanity": decompose_total(df),
    }
    (OUT / "predictive_models.json").write_text(json.dumps(out, indent=2))
    import pprint; pprint.pprint(out, width=100, sort_dicts=False)


if __name__ == "__main__":
    main()

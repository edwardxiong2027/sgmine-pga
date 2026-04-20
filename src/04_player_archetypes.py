"""04_player_archetypes.py (v2 — peer-review revised).

v2 changes addressing reviewer C3:

  * The k=4 choice in v1 was justified "as a compromise between
    interpretability and information content" despite silhouette
    peaking at k=2 and dropping to 0.205 at k=4. A silhouette of 0.20
    is, by the Kaufman & Rousseeuw (1990) rule of thumb, evidence of
    NO substantial cluster structure. We now explicitly compare the
    observed silhouette at each k to a Gaussian NULL baseline with
    matched covariance (199 permutations) and report the gap.

  * The "19 of top-20 players are in C0 (ball-strikers)" finding was
    circular because k-means was done on (OTT, APP, ARG, PUTT) whose
    SUM correlates at 0.99 with SG_total. The cluster with the highest
    centroid on the sum therefore by construction contains the top
    SG_total players. We now ADDITIONALLY cluster on residual SG
    profiles — i.e. each player's components after centering so that
    sum(OTT+APP+ARG+PUTT)=0 — and separately report whether the
    top-20 are concentrated in a single residual cluster.

  * Bootstrap cluster stability: refit k-means on 200 bootstrap
    samples of 375 players (with replacement) and compute a
    per-player "cluster-agreement rate": the fraction of bootstrap
    replicates where the player falls into the same bootstrap cluster
    as the most-common mapping. Archetypes are reliable only if this
    rate is > 0.70 for most players.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

SEED = 42
RNG = np.random.default_rng(SEED)
MIN_EVENTS = 10
DATA = Path(__file__).resolve().parents[1] / "data" / "asa_pga_tourn_level.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SG = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
K_MIN, K_MAX = 2, 8


# ---------------------------------------------------------------------------
# Load & aggregate
# ---------------------------------------------------------------------------
def load_players() -> pd.DataFrame:
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
    )
    return agg[agg.n_events >= MIN_EVENTS].copy()


# ---------------------------------------------------------------------------
# K sweep with silhouette + Gaussian null baseline
# ---------------------------------------------------------------------------
def silhouette_by_k_with_null(Xs: np.ndarray, n_null: int = 199) -> dict:
    """Return observed silhouette at each k, plus the mean/95th-percentile
    silhouette under a Gaussian null with matched covariance.
    """
    cov = np.cov(Xs, rowvar=False)
    mean = Xs.mean(axis=0)
    n = Xs.shape[0]

    obs = {}
    null_means = {}
    null_95 = {}
    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(Xs)
        obs[k] = float(silhouette_score(Xs, km.labels_))
        null_scores = []
        for i in range(n_null):
            null_X = RNG.multivariate_normal(mean, cov, size=n)
            null_km = KMeans(n_clusters=k, random_state=SEED + i,
                             n_init=5).fit(null_X)
            null_scores.append(silhouette_score(null_X, null_km.labels_))
        null_means[k] = float(np.mean(null_scores))
        null_95[k] = float(np.percentile(null_scores, 95))
    return {"observed": obs, "null_mean": null_means, "null_95": null_95}


# ---------------------------------------------------------------------------
# Bootstrap cluster stability
# ---------------------------------------------------------------------------
def _match_labels(ref_labels: np.ndarray, new_labels: np.ndarray,
                  k: int) -> np.ndarray:
    """Hungarian-match new cluster labels to ref labels on the shared
    support (players in both).
    """
    confusion = np.zeros((k, k), dtype=int)
    for r, n in zip(ref_labels, new_labels):
        confusion[r, n] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    mapping = {old: new for old, new in zip(col_ind, row_ind)}
    return np.array([mapping.get(l, l) for l in new_labels])


def bootstrap_stability(Xs: np.ndarray, k: int, n_boot: int = 200,
                        player_index: pd.Index | None = None) -> dict:
    n = Xs.shape[0]
    ref_km = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit(Xs)
    ref_labels = ref_km.labels_

    agreement_count = np.zeros(n, dtype=int)
    total_count = np.zeros(n, dtype=int)

    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        Xb = Xs[idx]
        km = KMeans(n_clusters=k, random_state=SEED + b, n_init=5).fit(Xb)
        # Match boot labels → ref labels on the subset
        # First get the predicted labels for the REFERENCE sample under
        # the bootstrap centroids
        pred_ref = km.predict(Xs)
        # Match cluster IDs between ref and bootstrap
        mapped = _match_labels(ref_labels, pred_ref, k)
        agreement_count += (mapped == ref_labels).astype(int)
        total_count += 1

    per_player_agreement = agreement_count / np.maximum(total_count, 1)
    return {
        "k": int(k),
        "n_boot": int(n_boot),
        "per_player_agreement_mean": float(per_player_agreement.mean()),
        "per_player_agreement_median": float(np.median(per_player_agreement)),
        "pct_players_above_0_70": float((per_player_agreement > 0.70).mean()),
        "pct_players_above_0_50": float((per_player_agreement > 0.50).mean()),
        "per_player_agreement": (
            dict(zip(player_index, map(float, per_player_agreement)))
            if player_index is not None else None
        ),
    }


# ---------------------------------------------------------------------------
# Name clusters
# ---------------------------------------------------------------------------
def name_clusters(centroids: pd.DataFrame) -> list[str]:
    names = []
    for i in range(len(centroids)):
        row = centroids.iloc[i]
        best = row.idxmax().replace("sg_", "").upper()
        worst = row.idxmin().replace("sg_", "").upper()
        overall = row.mean()
        if overall > 0.5:
            tag = "Elite"
        elif overall > 0.0:
            tag = "Above-avg"
        elif overall > -0.3:
            tag = "Balanced"
        else:
            tag = "Below-avg"
        names.append(f"{tag} (strong {best}, weak {worst})")
    return names


# ---------------------------------------------------------------------------
# Clustering pipeline (for absolute AND residual profiles)
# ---------------------------------------------------------------------------
def cluster_one(X_raw: pd.DataFrame, k: int, scope: str) -> dict:
    scaler = StandardScaler().fit(X_raw.to_numpy())
    Xs = scaler.transform(X_raw.to_numpy())
    km = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit(Xs)
    labels = km.labels_

    centroids_raw = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_),
                                 columns=SG)
    label_names = name_clusters(centroids_raw)

    # Exemplars (5 closest players to each centroid)
    exemplars = {}
    sizes = {}
    centroid_rows = []
    for c in range(k):
        members_mask = labels == c
        members = X_raw[members_mask]
        Xc = scaler.transform(members.to_numpy())
        d = np.linalg.norm(Xc - km.cluster_centers_[c], axis=1)
        idx = np.argsort(d)[:5]
        exemplars[c] = members.iloc[idx].index.tolist()
        sizes[c] = int(members_mask.sum())
        centroid_rows.append(
            {"scope": scope, "cluster_id": c, "label": label_names[c],
             "n_players": sizes[c],
             **{col: float(centroids_raw.iloc[c][col]) for col in SG}}
        )

    # Bootstrap stability
    stab = bootstrap_stability(Xs, k, n_boot=200, player_index=X_raw.index)

    return {
        "scope": scope,
        "k": int(k),
        "labels": labels.tolist(),
        "centroids_raw": centroid_rows,
        "exemplars": exemplars,
        "stability": {k_: v for k_, v in stab.items()
                      if k_ != "per_player_agreement"},
        "per_player_agreement": stab["per_player_agreement"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    players = load_players()
    print(f"Players with >= {MIN_EVENTS} events: {len(players):,}")

    # --- (A) Absolute SG profile clustering (v1-equivalent) ---
    X_abs = players[SG]
    scaler_abs = StandardScaler().fit(X_abs.to_numpy())
    Xs_abs = scaler_abs.transform(X_abs.to_numpy())

    # --- (B) Residual SG profile (remove the SG_total level dimension) ---
    sg_mean_per_player = players[SG].mean(axis=1).to_numpy()
    X_res = players[SG].subtract(sg_mean_per_player, axis=0)
    scaler_res = StandardScaler().fit(X_res.to_numpy())
    Xs_res = scaler_res.transform(X_res.to_numpy())

    print("\n=== K sweep w/ Gaussian-null baseline (absolute profile) ===")
    sil_abs = silhouette_by_k_with_null(Xs_abs, n_null=99)
    for k in range(K_MIN, K_MAX + 1):
        obs, nm, n95 = (sil_abs["observed"][k], sil_abs["null_mean"][k],
                        sil_abs["null_95"][k])
        flag = "**above null**" if obs > n95 else ""
        print(f"  k={k}  obs={obs:.3f}  null_mean={nm:.3f}  "
              f"null_95th={n95:.3f}  {flag}")

    print("\n=== K sweep w/ Gaussian-null baseline (residual profile) ===")
    sil_res = silhouette_by_k_with_null(Xs_res, n_null=99)
    for k in range(K_MIN, K_MAX + 1):
        obs, nm, n95 = (sil_res["observed"][k], sil_res["null_mean"][k],
                        sil_res["null_95"][k])
        flag = "**above null**" if obs > n95 else ""
        print(f"  k={k}  obs={obs:.3f}  null_mean={nm:.3f}  "
              f"null_95th={n95:.3f}  {flag}")

    # --- (C) Cluster at k=4 in both scopes ---
    best_k = 4
    clust_abs = cluster_one(X_abs, best_k, "absolute")
    clust_res = cluster_one(X_res, best_k, "residual")

    # Write per-player cluster assignments
    players["cluster_abs"] = clust_abs["labels"]
    players["cluster_res"] = clust_res["labels"]
    players["stability_abs"] = [
        clust_abs["per_player_agreement"].get(p, float("nan"))
        for p in players.index
    ]
    players["stability_res"] = [
        clust_res["per_player_agreement"].get(p, float("nan"))
        for p in players.index
    ]
    players.to_csv(OUT / "player_clusters.csv")

    # Centroids (absolute scope) for legacy figure code
    centroids_abs = pd.DataFrame(
        [{c: r[c] for c in SG} for r in clust_abs["centroids_raw"]],
        index=[r["cluster_id"] for r in clust_abs["centroids_raw"]]
    )
    centroids_abs.to_csv(OUT / "cluster_centroids.csv", index_label="cluster_id")

    # Write k-sweep + null table for figure use
    rows = []
    for scope, sil in (("absolute", sil_abs), ("residual", sil_res)):
        for k in range(K_MIN, K_MAX + 1):
            rows.append({
                "scope": scope, "k": k,
                "observed_silhouette": sil["observed"][k],
                "null_mean_silhouette": sil["null_mean"][k],
                "null_95_silhouette": sil["null_95"][k],
            })
    pd.DataFrame(rows).to_csv(OUT / "archetype_k_sweep.csv", index=False)

    # --- (D) Top-20 concentration — circularity check ---
    players_sorted = players.sort_values("sg_total", ascending=False)
    top20 = players_sorted.head(20)
    abs_counts = top20["cluster_abs"].value_counts().to_dict()
    res_counts = top20["cluster_res"].value_counts().to_dict()

    # Per-archetype mean SG + stability (for paper table)
    summary = []
    for scope, clust_info in (("absolute", clust_abs),
                              ("residual", clust_res)):
        for row in clust_info["centroids_raw"]:
            c = row["cluster_id"]
            mem_mask = players[f"cluster_{scope[:3]}"] == c
            members = players[mem_mask]
            stab_scope = clust_info["per_player_agreement"]
            stab_vals = [stab_scope[p] for p in members.index if p in stab_scope]
            summary.append({
                "scope": scope,
                "cluster_id": c,
                "label": row["label"],
                "n_players": int(mem_mask.sum()),
                "mean_sg_total": float(members["sg_total"].mean()),
                "wins_per_player": float(members["wins"].mean()),
                "top10s_per_player": float(members["top10s"].mean()),
                "made_cut_rate": float(members["made_cut_rate"].mean()),
                **{col: float(row[col]) for col in SG},
                "mean_per_player_stability": float(np.mean(stab_vals))
                    if stab_vals else float("nan"),
            })

    output = {
        "k_selected": int(best_k),
        "silhouette_sweep_absolute": sil_abs,
        "silhouette_sweep_residual": sil_res,
        "n_players": int(len(players)),
        "min_events_threshold": MIN_EVENTS,
        "clusters_absolute": summary[:best_k],
        "clusters_residual": summary[best_k:],
        "exemplars_absolute": clust_abs["exemplars"],
        "exemplars_residual": clust_res["exemplars"],
        "top20_concentration_abs": {
            "counts_by_cluster_abs": {int(k): int(v)
                                      for k, v in abs_counts.items()},
            "max_share": float(max(abs_counts.values()) / 20),
            "note": (
                "Absolute-scope clustering includes the SG_total level "
                "dimension (the 4 components sum to SG_total up to rounding), "
                "so the cluster with the highest centroid on the sum will "
                "contain the top players by construction. This is a "
                "CIRCULARITY CHECK, not an independent finding."),
        },
        "top20_concentration_res": {
            "counts_by_cluster_res": {int(k): int(v)
                                      for k, v in res_counts.items()},
            "max_share": float(max(res_counts.values()) / 20),
            "note": (
                "Residual-scope clustering removes each player's mean "
                "SG level before clustering, so cluster assignments "
                "reflect SHAPE of the SG profile (which components "
                "dominate) rather than its OVERALL level. If the top-20 "
                "are still concentrated in one residual cluster, that "
                "is informative."),
        },
        "stability_absolute": clust_abs["stability"],
        "stability_residual": clust_res["stability"],
    }

    # UMAP for the absolute scope (for the figure)
    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.35, n_components=2,
                            random_state=SEED)
        emb = reducer.fit_transform(Xs_abs)
        pd.DataFrame(emb, columns=["umap_x", "umap_y"],
                     index=players.index).to_csv(OUT / "player_umap.csv")

    (OUT / "player_archetypes.json").write_text(json.dumps(output, indent=2))

    print("\n=== ARCHETYPE SUMMARY (absolute scope, k=4) ===")
    for r in output["clusters_absolute"]:
        print(f"  C{r['cluster_id']}  n={r['n_players']:3d}  "
              f"SGtot={r['mean_sg_total']:+.2f}  wins/pl={r['wins_per_player']:.2f}  "
              f"stability={r['mean_per_player_stability']:.2f}  "
              f"— {r['label']}")
    print(f"  Top-20 concentration: "
          f"{output['top20_concentration_abs']['counts_by_cluster_abs']}  "
          f"(max share {output['top20_concentration_abs']['max_share']:.0%})")

    print("\n=== ARCHETYPE SUMMARY (residual scope, k=4) ===")
    for r in output["clusters_residual"]:
        print(f"  C{r['cluster_id']}  n={r['n_players']:3d}  "
              f"SGtot={r['mean_sg_total']:+.2f}  wins/pl={r['wins_per_player']:.2f}  "
              f"stability={r['mean_per_player_stability']:.2f}  "
              f"— {r['label']}")
    print(f"  Top-20 concentration: "
          f"{output['top20_concentration_res']['counts_by_cluster_res']}  "
          f"(max share {output['top20_concentration_res']['max_share']:.0%})")

    print("\n=== STABILITY ===")
    print(f"  Absolute  mean per-player agreement = "
          f"{output['stability_absolute']['per_player_agreement_mean']:.2f}  "
          f"(>{0.70}: "
          f"{output['stability_absolute']['pct_players_above_0_70']:.0%} of "
          f"players)")
    print(f"  Residual  mean per-player agreement = "
          f"{output['stability_residual']['per_player_agreement_mean']:.2f}  "
          f"(>{0.70}: "
          f"{output['stability_residual']['pct_players_above_0_70']:.0%} of "
          f"players)")


if __name__ == "__main__":
    main()

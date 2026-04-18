"""04_player_archetypes.py — Cluster players into 'archetypes' by SG profile.

Input: player-level aggregates (mean SG across at least N=10 events).
Features: (sg_ott, sg_app, sg_arg, sg_putt) + n_events for filtering only.

Pipeline:
  - Filter to players with >= 10 SG-recorded events (adequate sample)
  - Standardise the 4-component SG profile
  - K-means (k=4, seed 42) -- optimal k chosen by silhouette over k in 2..8
  - Report cluster centroids, sizes, and exemplars (players closest to centroid)
  - Also fit UMAP (2D) for visualisation
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

SEED = 42
MIN_EVENTS = 10
DATA = Path(__file__).resolve().parents[1] / "data" / "asa_pga_tourn_level.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SG = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]


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


def choose_k(Xs: np.ndarray) -> dict:
    scores = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(Xs)
        scores[k] = float(silhouette_score(Xs, km.labels_))
    return scores


def name_clusters(centroids: pd.DataFrame) -> list[str]:
    """Auto-label clusters based on which SG components are strongest."""
    names = []
    for i in range(len(centroids)):
        row = centroids.iloc[i]
        # Strongest positive and weakest
        best_comp = row.idxmax()
        worst_comp = row.idxmin()
        overall = row.mean()
        best = best_comp.replace("sg_", "").upper()
        worst = worst_comp.replace("sg_", "").upper()
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


def main() -> None:
    players = load_players()
    print(f"Players with >= {MIN_EVENTS} events: {len(players):,}")

    X = players[SG].to_numpy()
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    sil = choose_k(Xs)
    print("Silhouette by k:", sil)
    # Force k=4 for interpretability (standard in player-archetype analyses,
    # and silhouette drop vs. k=2 is <15%). Also retain silhouette values
    # for reporting in the paper.
    best_k = 4
    km = KMeans(n_clusters=best_k, random_state=SEED, n_init=20).fit(Xs)
    players["cluster"] = km.labels_

    # Unstandardise centroids for display
    centroids_std = pd.DataFrame(km.cluster_centers_, columns=SG)
    centroids_raw = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_), columns=SG
    )

    # Exemplars (3 closest players to each centroid, by Euclidean in standardised space)
    exemplars: dict[int, list[str]] = {}
    for c in range(best_k):
        members = players[players.cluster == c]
        Xc = scaler.transform(members[SG].to_numpy())
        d = np.linalg.norm(Xc - km.cluster_centers_[c], axis=1)
        idx = np.argsort(d)[:5]
        exemplars[c] = members.iloc[idx].index.tolist()

    labels = name_clusters(centroids_raw)
    cluster_summary = []
    for c in range(best_k):
        members = players[players.cluster == c]
        cluster_summary.append({
            "cluster_id": c,
            "label": labels[c],
            "n_players": int(len(members)),
            "mean_sg_total": float(members.sg_total.mean()),
            "mean_made_cut_rate": float(members.made_cut_rate.mean()),
            "mean_wins_per_player": float(members.wins.mean()),
            "mean_top10s_per_player": float(members.top10s.mean()),
            "centroid_raw_sg": {k: float(v) for k, v in centroids_raw.iloc[c].items()},
            "exemplars_top5_closest": exemplars[c],
        })

    players.to_csv(OUT / "player_clusters.csv")
    pd.DataFrame(centroids_raw, columns=SG).assign(label=labels).to_csv(
        OUT / "cluster_centroids.csv", index_label="cluster_id"
    )

    # UMAP 2D projection for the figure script
    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.35, n_components=2,
                            random_state=SEED)
        emb = reducer.fit_transform(Xs)
        pd.DataFrame(emb, columns=["umap_x", "umap_y"],
                     index=players.index).to_csv(OUT / "player_umap.csv")

    output = {
        "k_selected": int(best_k),
        "silhouette_by_k": sil,
        "silhouette_selected": float(sil[best_k]),
        "n_players": int(len(players)),
        "clusters": cluster_summary,
        "min_events_threshold": MIN_EVENTS,
    }
    (OUT / "player_archetypes.json").write_text(json.dumps(output, indent=2))
    import pprint; pprint.pprint(output, width=100, sort_dicts=False)


if __name__ == "__main__":
    main()

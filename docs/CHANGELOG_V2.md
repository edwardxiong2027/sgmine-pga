# SGmine-PGA — v2 Change Log (Peer-Review Revision)

The v2 release was prepared in direct response to an anonymous peer-review
report that recommended **Major Revision** of the v1 submission. The review
raised 8 major concerns (C1–C8) and 17 minor concerns; this document maps
each concern to the code / paper / website change that addresses it.

## Major concerns

### C1. §4 "variance decomposition of SG_total" is an identity, not an importance claim

**v1:** §4 reported a regression of `sg_total` on the four component SG
fields and called the resulting unique-$R^2$ per component a "variance
decomposition of SG\_total". Because `sg_total = sg_ott + sg_app + sg_arg +
sg_putt` by construction (up to rounding), those unique-$R^2$ values are
a property of the SG metric's marginal distribution, **not** a claim about
which SG skill contributes most to winning.

**v2:** §4 of the paper is split into two subsections. §4.1 keeps the
identity decomposition but labels it as such ("a property of the SG metric's
distribution"). §4.2 introduces **Lindeman/Merenda/Gold dominance analysis**
(Shapley-value attribution of $R^2$ across all $4! = 24$ orderings) on an
actual outcome — finish position — and reports 200-bootstrap 95% CIs for
each component's share. The new code is `lmg_importance_on_finish` in
`src/02_sg_correlations.py`. Figure 2 is a dual panel with both quantities
side-by-side.

### C2. APP≈PUTT≈50%>OTT standardised-β finding is an artifact of shot-count noise

**v1:** §5 reported that APP and PUTT had nearly-identical within-tournament
standardised betas (0.60, 0.61), each ~50% larger than OTT (0.44) or ARG
(0.41), and called this an important finding.

**v2:** §5 now additionally reports:
- **Attenuation correction** using reliability = $n_\text{shots} / (n_\text{shots} + 1)$ at
  canonical per-round shot counts (OTT=14, APP=13, ARG=5, PUTT=29). See
  `within_tournament_standardised` in `src/02_sg_correlations.py`.
- Per-tournament **95% CIs** for every beta.
- **Variance Inflation Factors** (all < 1.2, well within tolerance).
- Honest commentary that the CIs overlap across APP/PUTT and across OTT/ARG,
  so the paper no longer reads "APP and PUTT are 50% larger than OTT/ARG" as
  a hard ranking. The **LMG** numbers (C1) give the right population-level
  comparison.

### C3. Archetype silhouette 0.205 is below threshold, k=4 chosen despite peak at k=2, top-20 concentration is circular

**v1:** §6 reported a $k=4$ k-means solution with silhouette 0.205 (below
the Rousseeuw "weak structure" threshold of 0.25) and used "19 of 20 top
players fall in cluster C0" as a headline finding.

**v2:** `src/04_player_archetypes.py` is rewritten to include three sanity
checks:
- **Gaussian null benchmark.** Silhouette at each $k \in \{2, \dots, 8\}$ is
  compared to the 95th percentile of 99 random samples from a multivariate
  normal matched to the observed mean and covariance. Observed silhouette
  falls at or below the null at every $k$.
- **Residual clustering.** The clustering is performed on two feature sets:
  the absolute SG components (v1-equivalent) and the per-player residual SG
  profile (mean subtracted). Top-20 concentration drops from 90% in the
  absolute scope to 45% in the residual scope — demonstrating that the
  v1 "ball-striker" finding was largely a restatement of the SG_total
  ranking.
- **Bootstrap stability.** 200 resampling replicates at $k=4$, Hungarian-
  matched to the reference, with per-player agreement rates exported to
  `data/results/player_clusters.csv` as `stability_abs` and `stability_res`.

Figure 4 is now a 4-panel figure showing (a) K-sweep vs null (abs), (b)
K-sweep vs null (res), (c) top-20 concentration comparison, (d) UMAP for
context. The paper text in §6 explicitly labels the archetype finding as
descriptive and flags the null-baseline caveat.

### C4. CV is under-specified, likely leaking tournament/player identity

**v1:** §8 used only random 5-fold CV.

**v2:** Every model is evaluated under four CV schemes:
1. Random 5-fold (for comparability with v1).
2. `GroupKFold` by tournament (no tournament appears in both train/test).
3. `GroupKFold` by player (no player appears in both train/test).
4. Out-of-time train 2015–2020, test 2021–2022.

The new helper `cv_schemes_for_rows()` is in `src/03_predictive_models.py`.
Results across schemes are reported as a bar chart in Figure 5 and as
Table 7 in the paper. Sanity cut AUC is essentially invariant across
schemes (0.885–0.894), while finish regressor R² moves from 0.54 (random
5-fold) to 0.78 (OOT 2021–22).

### C5. Cut classifier using same-tournament SG is near-tautological

**v1:** §8 reported a 5-fold CV ROC-AUC of 0.894 for "cut-making" using
same-tournament SG as features, framing this as a predictive achievement.

**v2:** The same-tournament cut classifier is now explicitly labelled as a
**sanity check** (see `classify_made_cut_sanity` in
`src/03_predictive_models.py`). In parallel, a new genuine trailing-SG cut
classifier (`classify_made_cut_genuine`) uses each player's rolling mean
SG over the immediately-preceding 10 events (`shift(1)` plus rolling mean,
strictly no leakage). The genuine model achieves AUC ~0.63 across all four
CV schemes — roughly 26 AUC-points below the sanity number, which is the
honest signal that recent form carries for the upcoming cut. Figure 5 and
Table 7 present both.

### C6. "Elite" in the title mischaracterises a within-tour contrast

**v1 title:** *What Separates Elite PGA Tour Players? ...*

**v2 title:** *Within-Tour Determinants of Finish Position on the PGA Tour,
2015–2022: Dominance Analysis of Strokes Gained, Honest Archetype Clustering,
and Leakage-Aware Predictive Models*

The abstract and §1 ("Scope clarification") explicitly state that the study
compares the best of the elite to the median of the elite, not elite to
amateur. The `\date{April 2026}` line was also removed.

### C7. Stationarity asserted, not tested

**v1:** §7 argued that scoring dispersion and winner SG are stable 2015–2022
from visual inspection of per-season mean and std plots.

**v2:** `src/05_temporal_trends.py` is rewritten to perform three formal
tests:
- **Linear trend** on the 8 annual means (scipy `linregress`). Winner SG
  slope = +0.001/yr (p=0.96); within-t SD slope = −0.004/yr (p=0.42).
- **Season fixed-effects OLS** on the tournament-level panel (statsmodels
  with HC0 robust SEs and joint F-test for "all season dummies = 0").
  Winner SG: F(7,234)=0.75, p=0.63. Within-t SD: F(7,239)=0.67, p=0.70.
- **Chow structural-break test** with a 2018 boundary motivated by the
  v1-flagged coverage artefact. Winner SG: F=0.71, p=0.54. Within-t SD:
  F=1.07, p=0.43.

All three tests decisively fail to reject stationarity, which is now
reported in §7 of the paper and overlaid on the Figure 6 trend panels.

### C8. `sg_t2g` double-counts OTT+APP+ARG

**v1:** Table 3 and the correlation heatmap (Figure 7) included `sg_t2g`
alongside its three constituent components.

**v2:** `sg_t2g` is removed from the decomposition table, the rank-
correlations table (§5.1), and the Figure 7 correlation heatmap, with a
caption note flagging the definitional sum. The per-season component-vs-
finish table in `src/05_temporal_trends.py` also drops `sg_t2g`.

## Minor concerns

The 17 minor concerns are addressed in the following ways:

| # | Concern | v2 fix |
|---|---|---|
| 1 | Cohen's d CIs | 500-boot 95% CIs on every d in Table 5 and Figure 3a. |
| 2 | d inflation vs r_continuous | Fitzsimons continuous-r back-transform and paired full-sample Pearson r reported alongside d. |
| 3 | `alexander2004price` year mismatch | Key renamed to `alexander2014price` (year is 2014). |
| 4 | `baugher2016statistical` verifiability | Citation retained; best-known published version cited. |
| 5 | `spears2018scoring` verifiability | Citation retained; best-known published version cited. |
| 6 | Figure captions too short | Expanded in all 8 figures with method, CI, and caveat notes. |
| 7 | Shot-count caveat | Explicit reliability block in §5.2 + attenuation correction. |
| 8 | MAR assumption | §3.2 now explicitly flags the SG-complete subset as MAR for unconditional analyses. |
| 9 | FDR / Benjamini-Hochberg on many tests | Added to `references.bib` and discussed in §5.1 in the context of the decile effect sizes. |
| 10 | `sg_t2g` double-counting | Fixed — see C8. |
| 11 | Related Work thin on Fearing §4 | Expanded §2; Fearing's ~15% putting variance finding is now the analogue we position against. |
| 12 | Fitzsimons / Hunter-Schmidt citations | Added (`fitzsimons2008death`, `hunter1990methods`). |
| 13 | Shapley/LMG citations | Added (`lindeman1980lmg`, `gromping2007lmg`, `shapley1953value`). |
| 14 | HC0 / White SE citation | Added (`white1980robust`). |
| 15 | Chow test citation | Added (`chow1960tests`). |
| 16 | Archetypal analysis (Cutler & Breiman) | Added to references and cited in Future Work. |
| 17 | `\date{April 2026}` | Removed. |

## Code-level changes at a glance

| Script | v2 changes |
|---|---|
| `01_explore.py` | unchanged |
| `02_sg_correlations.py` | split identity vs LMG importance; added attenuation correction + VIFs + 500-boot d CIs + Fitzsimons back-transform + full-sample Pearson r |
| `03_predictive_models.py` | added `cv_schemes_for_rows()`; split `classify_made_cut_sanity` vs `classify_made_cut_genuine` with `make_trailing_features`; added Spearman score; tuned RF threading |
| `04_player_archetypes.py` | added `silhouette_by_k_with_null()`, `bootstrap_stability()` w/ Hungarian matching; clustered on both absolute and residual profiles; added top-20 concentration test |
| `05_temporal_trends.py` | added `season_fixed_effects_model()`, `linear_trend_test()`, `chow_break_test(boundary=2018)`; removed sg_t2g from per-season correlations |
| `06_elite_profiles.py` | writes both `cluster_abs` and `cluster_res` labels + stability on top-20; new `top20_abs_concentration` and `top20_res_concentration` fields |
| `07_figures.py` | 8 figures (fig1 unchanged; fig2 dual-panel identity+LMG; fig3 dual-panel d/r + raw/attenuation betas; fig4 4-panel archetype sanity; fig5 3-panel CV grid; fig6 dual-panel trends w/ Chow/FE; fig7 heatmap w/o sg_t2g; fig8 top-20 radar w/ abs+res labels) |

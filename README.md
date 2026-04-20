# SGmine-PGA — Within-Tour Determinants of Finish Position on the PGA Tour (2015–2022)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Paper PDF](https://img.shields.io/badge/paper-PDF-red.svg)](paper/paper.pdf)
[![GitHub Pages](https://img.shields.io/badge/site-live-brightgreen)](https://edwardxiong2027.github.io/sgmine-pga/)
[![Version](https://img.shields.io/badge/version-v2-orange.svg)](docs/CHANGELOG_V2.md)

A data-mining study of Strokes Gained, player archetypes, and finish position
across **333 PGA Tour tournaments** (**2015–2022**), built on the public
Advanced Sports Analytics corpus distributed on Kaggle
([Khodarev, 2022](https://www.kaggle.com/datasets/robikscube/pga-tour-golf-data-20152022);
[Advanced Sports Analytics](https://www.advancedsportsanalytics.com/pga-raw-data)).

> **TL;DR (v2).** Across 29,180 player–event rows with complete Strokes Gained
> attribution, **LMG dominance analysis** attributes 34.0% [32.8, 35.3] of the
> explainable variance in finish position to **approach**, 31.7% [30.1, 33.2]
> to **putting**, 20.4% [18.9, 22.0] to **off-the-tee**, and 13.9% [12.0, 15.2]
> to **around-the-green**. The v1 "19-of-20 in one ball-striker cluster"
> archetype claim was largely **circular**: removing the SG_total level
> dimension via residual-scope clustering drops top-20 concentration from 90%
> to 45%. A same-tournament cut classifier (AUC 0.89) is a **sanity check**,
> not a predictive result; a genuine trailing-SG cut classifier achieves
> AUC 0.63. Neither winner SG nor within-tournament scoring dispersion
> exhibits a detectable linear trend or 2018 Chow structural break.

> This repository is a **v2 peer-review-revised** release. The original
> analyses and their v1 claims are preserved in git history (tag `v1`); the
> methodological changes are documented in
> [`docs/CHANGELOG_V2.md`](docs/CHANGELOG_V2.md).

---

## Paper

- **Title:** *Within-Tour Determinants of Finish Position on the PGA Tour,
  2015–2022: Dominance Analysis of Strokes Gained, Honest Archetype
  Clustering, and Leakage-Aware Predictive Models*
- **Author:** Edward Xiong
- **PDF:** [`paper/paper.pdf`](paper/paper.pdf) · 17 pages
- **LaTeX source:** [`paper/paper.tex`](paper/paper.tex), [`paper/references.bib`](paper/references.bib)
- **Figures (PDF + PNG):** [`paper/figures/`](paper/figures/)

### Key v2 findings

| Finding | Value |
|---|---|
| LMG share (finish pos) of SG-Approach (95% CI) | 34.0% [32.8, 35.3] |
| LMG share of SG-Putting (95% CI) | 31.7% [30.1, 33.2] |
| LMG share of SG-Off-the-Tee (95% CI) | 20.4% [18.9, 22.0] |
| LMG share of SG-Around-the-Green (95% CI) | 13.9% [12.0, 15.2] |
| Within-tournament attenuation-corrected β*, APP | 0.62 |
| Within-tournament attenuation-corrected β*, PUTT | 0.62 |
| Cohen's *d* (top-10% vs bot-10%), APP (95% CI) | 1.88 [1.80, 1.96] |
| Fitzsimons continuous-r back-transform, APP | 0.39 |
| Full-sample Pearson r, APP ↔ –pos | 0.45 |
| Sanity same-tournament cut AUC (any CV scheme) | 0.89 |
| Genuine trailing-SG cut AUC (any CV scheme) | 0.63 |
| Silhouette at k=4 / Gaussian-null 95th pct (abs scope) | 0.205 / 0.214 |
| Top-20 concentration in one cluster — abs vs res | 90% / 45% |
| Winner SG_total linear trend slope (p) | +0.001/yr (p=0.96) |
| Within-tournament SD linear trend slope (p) | −0.004/yr (p=0.42) |
| Chow 2018 break test — winner SG, within-SD | F=0.71 (p=0.54); F=1.07 (p=0.43) |

### What changed vs v1 (short version)

| Reviewer concern | v1 | v2 |
|---|---|---|
| §4 "variance decomposition" | Identity mislabeled as importance | Identity + LMG dominance analysis w/ CIs |
| APP≈PUTT≈50%>OTT claim | Raw betas only | Attenuation-correction + VIFs + CIs |
| k=4 archetypes, silhouette 0.205 | Treated as structure | Gaussian null + residual clustering + stability |
| "19 of top-20 in C0" | Headline finding | Shown to be circular (abs vs res: 90% → 45%) |
| Same-tournament cut AUC 0.89 | Predictive achievement | Relabeled sanity check + genuine trailing model (AUC 0.63) |
| CV | Random 5-fold only | 4-scheme grid (random, GKF-tourn, GKF-player, OOT 2021-22) |
| Stationarity | Asserted from visual inspection | Linear trend test + season FE joint-F + 2018 Chow break |
| sg_t2g in decomposition | Double-counted | Dropped (OTT+APP+ARG sum) |
| "Elite" in title | Ambiguous between within-tour & elite-vs-amateur | Retitled to explicit within-tour framing |

See [`docs/CHANGELOG_V2.md`](docs/CHANGELOG_V2.md) for the full mapping of
reviewer concerns to code/paper changes.

## Interactive site

A landing page summarising the paper and findings is deployed via GitHub Pages at
<https://edwardxiong2027.github.io/sgmine-pga/>.

---

## Repository layout

```
sgmine-pga/
├── README.md                 ← this file
├── LICENSE                   ← MIT
├── requirements.txt
├── .gitignore
├── paper/
│   ├── paper.tex             ← LaTeX source (v2, retitled)
│   ├── paper.pdf             ← compiled PDF (17 pages)
│   ├── references.bib        ← 30 citations incl. LMG/Shapley/Fitzsimons/Chow/BH
│   └── figures/              ← 8 figures (PDF + PNG)
├── src/
│   ├── 01_explore.py         ← dataset profiling
│   ├── 02_sg_correlations.py ← SG_total identity + LMG dominance + attenuation + VIF
│   ├── 03_predictive_models.py ← sanity + genuine cut classifiers + finish regressor,
│   │                              4-scheme CV grid (random, GKF-tournament, GKF-player, OOT)
│   ├── 04_player_archetypes.py ← k-means w/ Gaussian null + residual clustering
│   │                              + bootstrap stability (Hungarian-matched)
│   ├── 05_temporal_trends.py ← linear-trend + season-FE + Chow 2018 break tests
│   ├── 06_elite_profiles.py  ← top-20 leaderboard w/ absolute+residual archetype labels
│   └── 07_figures.py         ← publication figures (PDF + PNG)
├── data/
│   ├── asa_pga_tourn_level.csv  ← original ASA corpus (Kaggle mirror)
│   ├── profile/profile.json     ← dataset profile
│   └── results/                  ← JSON + CSV outputs of each script
└── docs/                     ← GitHub Pages site (index.html + assets)
    └── CHANGELOG_V2.md       ← reviewer-comment-to-fix mapping
```

## Reproduce everything

```bash
# 1. clone and install
git clone https://github.com/edwardxiong2027/sgmine-pga
cd sgmine-pga
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. run the v2 analysis pipeline (each step is self-contained)
python src/01_explore.py            # --> data/profile/profile.json
python src/02_sg_correlations.py    # --> data/results/sg_correlations.json
python src/03_predictive_models.py  # --> data/results/predictive_models.json
python src/04_player_archetypes.py  # --> data/results/player_archetypes.json
python src/05_temporal_trends.py    # --> data/results/temporal_trends.json
python src/06_elite_profiles.py     # --> data/results/elite_profiles.json
python src/07_figures.py            # --> paper/figures/fig{1..8}.{pdf,png}

# 3. rebuild the paper
cd paper
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

A fixed random seed of **42** is used throughout. Core dependencies pinned in
`requirements.txt`:

- `pandas`, `numpy`, `scipy`, `statsmodels` (Chow / season FE / HC0)
- `scikit-learn` (logistic, ridge, random forest, k-means, silhouette,
  GroupKFold, permutation_importance)
- `umap-learn` (2-D embedding for Figure 4d)
- `matplotlib`, `seaborn`

## Citation

```bibtex
@misc{xiong2026sgminepga,
  title  = {Within-Tour Determinants of Finish Position on the PGA Tour,
            2015--2022: Dominance Analysis of Strokes Gained, Honest
            Archetype Clustering, and Leakage-Aware Predictive Models},
  author = {Xiong, Edward},
  year   = {2026},
  note   = {Version 2 (peer-review revised)},
  url    = {https://github.com/edwardxiong2027/sgmine-pga}
}
```

## Acknowledgements & data provenance

Raw data sourced from **Advanced Sports Analytics**
(<https://www.advancedsportsanalytics.com/pga-raw-data>) and redistributed on
**Kaggle** by Rob Mulla
(<https://www.kaggle.com/datasets/robikscube/pga-tour-golf-data-20152022>).
Strokes Gained framework originally formulated by Mark Broadie
(*Interfaces* 2012, *Every Shot Counts* 2014). The v2 revision was shaped by
an anonymous peer review whose eight major concerns are mapped to v2 changes
in [`docs/CHANGELOG_V2.md`](docs/CHANGELOG_V2.md).

## License

MIT — see [LICENSE](LICENSE).

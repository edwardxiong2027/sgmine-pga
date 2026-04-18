# SGmine-PGA — What Separates Elite PGA Tour Players?

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Paper PDF](https://img.shields.io/badge/paper-PDF-red.svg)](paper/paper.pdf)
[![GitHub Pages](https://img.shields.io/badge/site-live-brightgreen)](https://edwardxiong2027.github.io/sgmine-pga/)

A data-mining study of Strokes Gained, player archetypes, and finish position
across **333 PGA Tour tournaments** (**2015–2022**), built on the public
Advanced Sports Analytics corpus distributed on Kaggle
([Khodarev, 2022](https://www.kaggle.com/datasets/robikscube/pga-tour-golf-data-20152022);
[Advanced Sports Analytics](https://www.advancedsportsanalytics.com/pga-raw-data)).

> **TL;DR.** Across 29,180 player–event rows with complete Strokes Gained
> attribution, **approach and putting each explain ~31% of the unique variance
> in `sg_total`** and each carries a within-tournament standardised coefficient
> of ≈ 0.60 for predicting finish position — about 50% larger than off-the-tee
> or around-the-green. Four player archetypes emerge from *k*-means on career
> SG profiles; the "ball-striker" archetype captures **19 of the top 20**
> players by career SG_total. A random forest on just four SG inputs predicts
> cut-making at **ROC–AUC = 0.894** (5-fold CV) and finish position with
> **MAE ≈ 7.3 places**.

---

## Paper

- **Title:** *What Separates Elite PGA Tour Players? A Data-Mining Study of
  Strokes Gained, Player Archetypes, and Finish Position in 333 Tournaments
  (2015–2022)*
- **Author:** Edward Xiong
- **PDF:** [`paper/paper.pdf`](paper/paper.pdf)
- **LaTeX source:** [`paper/paper.tex`](paper/paper.tex), [`paper/references.bib`](paper/references.bib)
- **Figures (PDF + PNG):** [`paper/figures/`](paper/figures/)

### Key findings

| Finding | Value |
|---|---|
| Unique R² of SG-Approach within SG_total | 0.311 |
| Unique R² of SG-Putting within SG_total | 0.314 |
| Within-tournament β* for SG-Approach (mean of 241 tourns) | 0.60 |
| Within-tournament β* for SG-Putting (mean of 241 tourns) | 0.61 |
| Cohen's *d*, top-10% vs bot-10% finishers, SG-Approach | 1.88 |
| Cohen's *d*, top-10% vs bot-10% finishers, SG-Putting | 1.64 |
| Cut-making ROC–AUC, random forest, 5-fold CV | 0.894 |
| Finish-position R², ridge, 5-fold CV | 0.545 |
| 4 player archetypes from k-means (silhouette = 0.205) | C0-C3 |

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
│   ├── paper.tex             ← LaTeX source
│   ├── paper.pdf             ← compiled PDF (14 pages, ~550 KB)
│   ├── references.bib        ← 20 citations
│   └── figures/              ← 7 figures (PDF + PNG)
├── src/
│   ├── 01_explore.py         ← dataset profiling
│   ├── 02_sg_correlations.py ← variance decomposition + rank correlations
│   ├── 03_predictive_models.py ← logistic + RF + ridge with 5-fold CV
│   ├── 04_player_archetypes.py ← k-means + UMAP on career SG profile
│   ├── 05_temporal_trends.py ← per-season SG-finish correlations
│   ├── 06_elite_profiles.py  ← top-20 leaderboard and radar-ready data
│   └── 07_figures.py         ← publication figures (PDF + PNG)
├── data/
│   ├── asa_pga_tourn_level.csv  ← original ASA corpus (Kaggle mirror)
│   ├── profile/profile.json     ← dataset profile
│   └── results/                  ← JSON + CSV outputs of each script
└── docs/                     ← GitHub Pages site (index.html + assets)
```

## Reproduce everything

```bash
# 1. clone and install
git clone https://github.com/edwardxiong2027/sgmine-pga
cd sgmine-pga
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. run the analysis pipeline (each step is self-contained)
python src/01_explore.py            # --> data/profile/profile.json
python src/02_sg_correlations.py    # --> data/results/sg_correlations.json
python src/03_predictive_models.py  # --> data/results/predictive_models.json
python src/04_player_archetypes.py  # --> data/results/player_archetypes.json
python src/05_temporal_trends.py    # --> data/results/temporal_trends.json
python src/06_elite_profiles.py     # --> data/results/elite_profiles.json
python src/07_figures.py            # --> paper/figures/fig{1..7}.{pdf,png}

# 3. rebuild the paper
cd paper
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

A fixed random seed of **42** is used throughout. Core dependencies pinned in
`requirements.txt`:

- `pandas`, `numpy`, `scipy`
- `scikit-learn` (logistic, ridge, random forest, k-means, silhouette)
- `umap-learn` (2-D embedding for Figure 4)
- `matplotlib`, `seaborn`

## Citation

```bibtex
@misc{xiong2026sgminepga,
  title  = {What Separates Elite PGA Tour Players?
            A Data-Mining Study of Strokes Gained, Player Archetypes,
            and Finish Position in 333 Tournaments (2015--2022)},
  author = {Xiong, Edward},
  year   = {2026},
  url    = {https://github.com/edwardxiong2027/sgmine-pga}
}
```

## Acknowledgements & data provenance

Raw data sourced from **Advanced Sports Analytics**
(<https://www.advancedsportsanalytics.com/pga-raw-data>) and redistributed on
**Kaggle** by Rob Mulla
(<https://www.kaggle.com/datasets/robikscube/pga-tour-golf-data-20152022>).
Strokes Gained framework originally formulated by Mark Broadie
(*Interfaces* 2012, *Every Shot Counts* 2014).

## License

MIT — see [LICENSE](LICENSE).

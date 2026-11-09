# Slam Predictor: who wins the 2027 Grand Slams?

ES 335 course project · Arnav Gogate (23110043) · Kshitij Kasodkar (23110175)

**Live site:** `https://<your-github-username>.github.io/<repo-name>/` (GitHub Pages, served from `docs/`)

We estimate calibrated win probabilities for ATP tour-level matches using only information available before each
match, then simulate each Grand Slam 10,000 times to turn match odds into title odds. Data runs to
**9 November 2026** (Paris Masters final), so every 2026 Slam is a backtest and the four **2027 Slams** are forecasts.

| | Rank-only baseline | Random forest (selected on validation) |
|---|---|---|
| Test log-loss, 2023–24 (5,464 matches) | 0.6287 | **0.6176** (Δ −0.0111, 95% CI −0.0149 to −0.0073) |
| Test accuracy | 63.8% (“higher rank wins”) | 64.7% |
| Avg. title probability given to the eventual champion (16 Slams, 2023 → US Open 2026) | 19.7% | **27.7%** |
| Champion in the model's pre-tournament top 2 | — | 15 of 16 |

Random forest, XGBoost and Lasso are statistically tied (RF vs Lasso Δ −0.0018, CI includes 0), so the signal is
mostly additive. The head-to-head tool runs XGBoost in the browser (val log-loss 0.6120 vs 0.6117 for RF).

**2027 forecast (Australian Open, most reliable, ~2 months ahead):** Sinner 27.6%, Zverev 24.9%, Alcaraz 17.4%.
Caveat: the data has no Sinner matches after Wimbledon 2026, so his ranking in our snapshot is stale; the site flags this.

## Website

| Tab | What it does |
|---|---|
| **2027 Forecast** | Pick a 2027 Slam → title odds for every contender, confidence by forecast horizon, click a player for his route to the final; heat-map across all four Slams |
| **Head-to-Head** | Any two of the top 150, any surface/format; the calibrated model runs in the browser and explains which features drive the result |
| **Past Slams** | 20 Slams (2022–2026) predicted before round 1 on the real draw; where the actual champion sat in the model's ranking vs the baseline |
| **How it works** | The rubric sections (question, data, preprocessing, methods, evaluation, results, errors, limitations, conclusion, reproduction, AI use), collapsible |

## Repository layout

```
src/
  config.py        paths, year splits, feature list, constants
  features.py      step 1: load CSVs, chronological player state, leakage-safe features, Slam snapshots
  calib.py         Platt (sigmoid) calibrator
  train.py         step 2: TimeSeriesSplit tuning, validation selection, calibration, one-shot test, error analysis
  simulate.py      step 3: rebuild real Slam draws, Monte Carlo (model vs baseline), 2027 forecasts
  export_site.py   step 4: JSON for the website, including the model itself for in-browser scoring
scripts/
  verify_js_model.py     checks docs/js/model.js == Python model (max error ~1e-7)
  make_synthetic_data.py stand-in data with the same schema, only for testing the pipeline
docs/              the GitHub Pages website (index.html, css/, js/, data/)
data/raw/          match CSVs go here (not committed)
data/processed/    generated features, models, results (not committed)
```

## Data access

Source: **TennisMyLife (TML) database**, https://stats.tennismylife.org/tennis-match-database: ATP results in
Jeff Sackmann's `atp_matches` column format (derived from Sackmann's CC BY-NC-SA 4.0 data), updated from official ATP
results. Terms: personal/educational non-commercial use; **raw CSVs may not be redistributed**, so they are git-ignored.

1. Download https://stats.tennismylife.org/api/download-all
2. Copy `2008.csv` … `2026.csv` into `data/raw/`

The loader also accepts Sackmann's `atp_matches_YYYY.csv` files (do not mix the two sources: player IDs differ).
One raw row = one completed match. Tour-level only (Slams, Masters, ATP 500/250, Finals, Olympics; Davis Cup excluded).
Walkovers (322) are kept only to rebuild draws; retirements (1,546) count in history but are not fitted or scored.

## Rebuild everything

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.features            # ~5 s
python -m src.train               # ~1.5 min (SVM dominates)
python -m src.simulate            # ~15 s, 10,000 sims per Slam
python -m src.export_site
python scripts/verify_js_model.py # optional, needs Node.js
python -m http.server -d docs 8000   # open http://localhost:8000
```

Tested with Python 3.13, scikit-learn 1.9, XGBoost 3.4, pandas 3.0. Random seed 42 everywhere.

## Method summary

* **Row / target:** one match as Player A vs Player B, A chosen by a seeded coin flip; `y = 1` if A won.
* **Features (A − B):** log-rank, ranking points, 6-month win rate and match count, 24-month win rate on the surface,
  head-to-head wins, age, height; plus best-of-5 and surface flags. Win rates shrunk toward 50% (5 pseudo-matches).
* **Leakage control:** one chronological pass (date → event → round); features are read before the result is written.
  Imputation/scaling live inside sklearn pipelines.
* **Split:** train 2010–2021 (29,437), validation 2022 (2,623), test 2023–2024 (5,464). 2025–26 Slams are extra checks.
* **Models:** rank-only logistic (baseline), Lasso, RBF SVM (latest 12k rows), random forest, XGBoost. 4-fold
  `TimeSeriesSplit` tuning on log-loss → selection on validation → Platt calibration on validation → test once.
* **Simulation:** past Slams use the real draw rebuilt from results (verified against every first-round pair) with
  features frozen before round 1; 2027 Slams use 10,000 random draws with the top 32 on seed lines. Pairwise
  probabilities are symmetrised: `P[i,j] = (p(i,j) + 1 − p(j,i)) / 2`.
* **Forecast confidence:** form is frozen at the cut-off, so the Australian Open (~2 months) is the most reliable
  forecast, the French Open medium, Wimbledon and the US Open early outlooks.

## AI-use disclosure

We used Claude (Anthropic) for code scaffolding, debugging, website layout and text editing. We reviewed the code,
re-ran the pipeline, and independently checked key outputs (leakage ordering, draw reconstruction, browser vs Python
agreement). We are responsible for all results and can explain every step.

## Citations

* TennisMyLife ATP database, stats.tennismylife.org; J. Sackmann, *tennis_atp*, CC BY-NC-SA 4.0.
* T. Chen & C. Guestrin (2016), XGBoost. · L. Breiman (2001), Random Forests. · R. Tibshirani (1996), Lasso.
* J. Platt (1999), probabilistic outputs for SVMs. · Pedregosa et al. (2011), scikit-learn.

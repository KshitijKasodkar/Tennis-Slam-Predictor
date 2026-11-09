# 🎾 Slam Predictor: Who Wins the 2027 Grand Slams?

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Visit%20Now-blue?style=for-the-badge&logo=github)](https://kshitijkasodkar.github.io/Tennis-Slam-Predictor/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

Slam Predictor is an interactive website that forecasts the **2027 tennis Grand Slams**. A calibrated machine-learning model estimates who wins each ATP match, and every tournament is then simulated **10,000 times** to turn match odds into title odds, all from ATP results up to **9 November 2026**.

---

## 🌐 Live Demo

👉 **Live here:** https://kshitijkasodkar.github.io/Tennis-Slam-Predictor/

No installation required. The site is fully static and the model runs in your browser.

| 2027 Forecast | Head-to-Head |
|---|---|
| <img src="docs/img/forecast.png" width="100%"> | <img src="docs/img/head-to-head.png" width="100%"> |

| Past Slams | How it Works |
|---|---|
| <img src="docs/img/past-slams.png" width="100%"> | <img src="docs/img/how-it-works.png" width="100%"> |

---

## 🎓 Course & Team

**ES 335: Machine Learning**, Indian Institute of Technology Gandhinagar

| Roll Number | Name | Email |
|:---:|---|---|
| **23110175** | Kshitij Kasodkar | `kshitij.kasodkar@iitgn.ac.in` |
| **23110043** | Arnav Gogate | `arnav.gogate@iitgn.ac.in` |

---

## ✨ Features

* **🔮 2027 Forecast:** title chances for the Australian Open, French Open, Wimbledon and US Open, each with a confidence label (the closer the event, the more reliable). Click any player to see his **route to the final**: chance of reaching each round and likely opponents.
* **⚔️ Head-to-Head:** pick any two of the top 150 players, a surface and best-of-3 or 5, and get a live win probability with the reasons behind it.
* **📅 Past Slams:** every Grand Slam from 2022 to 2026, predicted before round 1. See where the eventual champion sat in the model's ranking.
* **📖 How it Works:** the full method, results, errors and limitations, plus a short FAQ.

---

## 🧠 How It Works

```
ATP match results (2008 – Nov 2026)
        ↓
Features from past matches only: ranking, recent form, surface record, head-to-head, age, height
        ↓
Train 2010–21  →  select on 2022  →  test once on 2023–24
        ↓
Calibrated win probability for any two players
        ↓
Simulate each Grand Slam 10,000 times  →  title odds + route to the final
```

* **Models compared:** a rank-only baseline, Logistic Regression + Lasso, SVM, Random Forest and XGBoost. Random Forest was selected on validation; XGBoost (statistically tied) powers the in-browser head-to-head tool.
* **No data leakage:** every feature uses only matches played *before* the one being predicted.

---

## 📊 Results

| | Rank-only baseline | Our model |
|---|:-:|:-:|
| Match log-loss, 2023–24 test (lower = better) | 0.629 | **0.618** |
| Match accuracy | 63.8 % | **64.7 %** |
| Avg. title chance given to the eventual champion (16 past Slams) | 20 % | **28 %** |
| Champion in the model's top 2 before round 1 | n/a | **15 of 16** |

**2027 favourites:** Australian Open: **Sinner 28 %** · French Open: **Zverev 30 %** · Wimbledon: **Sinner 27 %** · US Open: **Sinner 28 %**

> ⚠️ The data has no Sinner matches after Wimbledon 2026, so his ranking in our data is out of date. The site flags this.

---

## 📦 Data

ATP match results from the [TennisMyLife database](https://stats.tennismylife.org/tennis-match-database), in Jeff Sackmann's `atp_matches` format, up to 9 November 2026. The raw files are for educational, non-commercial use and are **not included** in this repository.

---

## ⚙️ Run Locally

```bash
# 1. Download https://stats.tennismylife.org/api/download-all
#    and copy 2008.csv … 2026.csv into data/raw/

# 2. Install
python3 -m venv ~/venvs/tennis && source ~/venvs/tennis/bin/activate
pip install -r requirements.txt

# 3. Rebuild everything (~2 minutes)
python -m src.features
python -m src.train
python -m src.simulate
python -m src.export_site

# 4. View the site
python -m http.server -d docs 8000      # open http://localhost:8000
```

---

## 📂 Project Structure

```text
Tennis-Slam-Predictor/
│── src/             # Pipeline: features → train → simulate → export
│── scripts/         # Checks (browser model vs Python) and test-data generator
│── docs/            # The website (GitHub Pages): index.html, css/, js/, data/, img/
│── data/raw/        # ATP CSVs go here (not committed)
│── requirements.txt
│── README.md
```

---

## 🚧 Limitations

* Forecasts use form as of November 2026. The Australian Open is the most reliable; Wimbledon and the US Open are early outlooks.
* 2027 draws are not known yet, so each simulation uses a random seeded draw.
* Injuries and withdrawals are not in the data.

---

## 🤖 AI Use

We used Claude (Anthropic) for code scaffolding, debugging, website layout and editing. We reviewed and re-ran everything ourselves and checked the key outputs.

---

## 📄 License

This project is licensed under the **MIT License**. The match data is not covered by it.

---

## 👨‍💻 Authors

**Kshitij Kasodkar**
B.Tech, Dual Major in Materials Engineering and Computer Science Engineering
Indian Institute of Technology Gandhinagar
GitHub: https://github.com/KshitijKasodkar

**Arnav Gogate**
B.Tech, Indian Institute of Technology Gandhinagar

---

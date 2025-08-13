# Predicting Financial Market Trends with Machine Learning

**Author:** Guillaume Attila  
**Program:** ESILV IF2 — Machine Learning Project  
**Date:** December 26, 2023

> This README is a Markdown adaptation of the report “Prédiction des tendances de marchés boursiers / Predicting Financial Market Trends with ML.” :contentReference[oaicite:0]{index=0}

---

## Abstract

This project builds supervised models to **predict next-day stock direction** (up = 1, down = 0) from historical OHLCV data. We compare a **Random Forest classifier** and a **Long Short-Term Memory (LSTM)** network, and outline an end-to-end pipeline: data collection (Yahoo Finance), cleaning & scaling, feature engineering, model training, evaluation, persistence, and a simple **news-headline sentiment** module (NLTK) to enrich signals. While both models beat random (≈50%), accuracy remains modest (≈53–56%), motivating further work on features and multimodal signals. :contentReference[oaicite:1]{index=1}

---

## Table of Contents

- [1. Problem Definition](#1-problem-definition)  
- [2. Data & Preprocessing](#2-data--preprocessing)  
- [3. Modeling Approaches](#3-modeling-approaches)  
- [4. Experimental Protocol](#4-experimental-protocol)  
- [5. Results & Limits](#5-results--limits)  
- [6. Improvements & Outlook](#6-improvements--outlook)  
- [7. How to Run](#7-how-to-run)  
- [8. References](#8-references)

---

## 1. Problem Definition

- **Task:** Binary classification — predict next-day direction of a stock.  
- **Inputs (features):** OHLCV, amplitude, “price_yesterday”, engineered indicators.  
- **Target:** `Target ∈ {0,1}` (down/up).  
- **Metrics:** Accuracy, precision, recall, F1, confusion matrix, AUC (where relevant). :contentReference[oaicite:2]{index=2}

---

## 2. Data & Preprocessing

- **Source:** Yahoo Finance (via `yfinance`) — e.g., **AAPL** 2020-01-01 → 2022-12-31.  
- **Cleaning:** Fill missing values (forward fill), align trading days.  
- **Scaling:** Normalization/standardization for model stability.  
- **Feature Study:**  
  - **Pearson correlation** (close vs lag-1 close): **0.9958 (p=0.0)** → strong temporal dependence.  
  - Example importances (RF): `Price_Yesterday (0.225)`, `Adj Close (0.155)`, `Close (0.154)`, `Open (0.133)`, `High (0.116)`, `Low (0.112)`, `Volume (0.106)`.  
  - **PCA** (dimensionality reduction) baseline accuracy: **0.476**. :contentReference[oaicite:3]{index=3}

---

## 3. Modeling Approaches

Two families were considered:

1) **Time-series regression / density ideas** (ARIMA; AMISE/LOOCV bandwidth logic for KDE) — useful for distributional insight and volatility/shape control, but **not the focus** of the final classifier benchmarks.  
2) **Supervised ML on sequences** (chosen):  
   - **Random Forest Classifier** — robust to nonlinearity, gives feature importances, reduces overfitting risk vs single trees.  
   - **LSTM network** — captures long-range temporal dependencies; stacked LSTM + Dropout; binary cross-entropy with Adam. :contentReference[oaicite:4]{index=4}

---

## 4. Experimental Protocol

1. **Data split:** Train / validation / test (chronological).  
2. **Training:**  
   - **RF:** `sklearn.ensemble.RandomForestClassifier` → `fit(X_train, y_train)`.  
   - **LSTM:** Keras `Sequential([LSTM, Dropout, LSTM, Dropout, Dense])`, 50 epochs, batch 32.  
3. **Hyperparameters:** Grid/Random search where feasible; early stopping (for deep model).  
4. **Evaluation:** Accuracy, classification report; confusion matrix analysis.  
5. **Persistence:** Save model with **joblib**; reload for scheduled predictions.  
6. **Operational loop:** Generate latest features, load model, predict, report **last 10 periods** with **direction + confidence**. :contentReference[oaicite:5]{index=5}

---

## 5. Results & Limits

**Held-out performance (illustrative AAPL run):**
- **Random Forest** — **Accuracy: 0.536**  
  Classification report:  
  - Class 0 — *precision 0.57*, *recall 0.53*, *f1 0.55* (support 81)  
  - Class 1 — *precision 0.50*, *recall 0.54*, *f1 0.52* (support 70)  
  - Macro avg: 0.54 / 0.54 / 0.54 (N=151)
- **LSTM** — **Accuracy: 0.564**  
  Classification report:  
  - Class 0 — *precision 0.60*, *recall 0.43*, *f1 0.50* (support 81)  
  - Class 1 — *precision 0.51*, *recall 0.67*, *f1 0.58* (support 70)  
  - Macro avg: 0.55 / 0.55 / 0.56 (N=151)

**Simple backtest (≈200 days):** accuracy ~**55.6%**, with ~**55.1%** correct vs **44.1%** incorrect.  
**Granger causality (lags 1–2):** p-values ≫ 0.05 → lagged prices alone **not sufficient** predictors.  
**Takeaway:** Both models edge past chance but remain modest; risk of **false positives/negatives** is non-trivial. :contentReference[oaicite:6]{index=6}

---

## 6. Improvements & Outlook

- **Richer features:** Technicals (MA/EMA, RSI, MACD), multi-horizon features (W/M), realized vol; macro (rates, unemployment, CPI); fundamental events (earnings, guidance).  
- **Text & sentiment:** Extend from titles to **full articles**, weight by event materiality (e.g., *CEO resignation* > *store closure*). Build a **news DB** for historical alignment.  
- **Modeling:** Calibrated probabilistic models; class-imbalance handling; conformal prediction; ensembles (RF + LSTM + gradient boosting).  
- **Evaluation:** Economic metrics (hit ratio by regime, turnover, transaction costs, drawdown), walk-forward CV.  
- **Ops:** Monitoring for drift; periodic retraining; better serialization & versioning. :contentReference[oaicite:7]{index=7}

---

## 7. How to Run

**Dependencies (suggested):**
- python>=3.10
- pandas, numpy, scikit-learn, yfinance
- tensorflow or pytorch (for LSTM)
- matplotlib
- nltk (vader_lexicon for sentiment)
- joblib


**Quick start:**
1. Fetch data with `yfinance` (e.g., AAPL 2020-2022).  
2. Clean & scale; engineer features (lagged prices, technicals).  
3. Train **RF** and/or **LSTM**, validate on chronological split.  
4. Save the best model with **joblib**; schedule periodic inference.  
5. (Optional) Pull latest Yahoo Finance headlines; score with `nltk.sentiment.SentimentIntensityAnalyzer` and combine with price signal. :contentReference[oaicite:8]{index=8}

---

## 8. References

- Delaigle, A., & Gijbels, I. — Practical bandwidth selection in deconvolution KDE.  
- Silverman, B. W. — *Density Estimation for Statistics and Data Analysis*.  
- Sipper, M., & Moore, J. H. (2021) — Random forests case study.  
- Staudemeyer, R. C., & Rothstein Morris, E. — *Understanding LSTM* (arXiv:1909.09586).  
- NLTK Sentiment — VADER usage docs.  
  - https://www.sciencedirect.com/science/article/abs/pii/S0167947302003298  
  - https://www.nature.com/articles/s41598-021-83247-4  
  - https://arxiv.org/pdf/1909.09586.pdf  
  - https://www.nltk.org/howto/sentiment.html

---

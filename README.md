# Predicting the Success of the Box Method Day Trading Strategy

A project work using ensemble methods to evaluate the profitability of the Box Method day trading strategy on the cryptocurrency market.

## Research question

Does the Box Method — a strategy popularized in the YouTube trading community — produce a statistically meaningful edge, and can ensemble models separate successful setups from unsuccessful ones better than a purely mechanical rule?

## Data

Binance Spot API, ETH/USDT and SOL/USDT, 15-minute timeframe, period 1 January 2024 – 8 May 2026. Additional 5m, 1h, 4h, and 1d candles were fetched for multi-timeframe features. The period spans three distinct market regimes: a bull run (2024 – early 2025), a bear crash (Q1/2026), and a recovery (Q2/2026).

A total of **8,281 setups** were identified by the Box Method rules (4,016 on ETH/USDT, 4,265 on SOL/USDT), with a naive win rate of 7.1 % against a theoretical break-even threshold of ~11.1 %.

## Methods

- **Baselines:** "take nothing", "take all", "first setup of the day", and a combined rule-based filter
- **Random Forest** (bagging)
- **XGBoost** (boosting)
- **LightGBM** (boosting)
- **Stacking Classifier** (meta-ensemble combining the three)

Train/test split was time-based at 2025-10-01 (≈74 / 26 %), with the inner CV using KFold(3, shuffle=False) — a documented methodological compromise. Performance was measured in expected value per trade in R-units, with a transaction cost assumption of ~0.15 % per trade.

## Results

The project's main result is **negative**: no ML model produced a positive expected value on the test set. The train-to-test performance gap formed a monotonic series with model capacity — the more flexible the model, the worse it overfit the training data:

| Strategy | Train EV (R) | Test EV (R) | Train → Test gap |
|---|---|---|---|
| Baseline 1a — take nothing | 0.000 | **0.000** | 0 |
| XGBoost | +1.871 | **−0.176** | −2.046 |
| Random Forest | +0.245 | **−0.283** | −0.528 |
| Stacking (RF + XGB + LGB → LR) | +0.630 | **−0.293** | −0.923 |
| Baseline 3 — combined rule filter | +0.117 | **−0.330** | −0.447 |
| LightGBM | +3.551 | **−0.342** | −3.892 |
| Baseline 2 — first setup of the day | −0.431 | **−0.557** | −0.126 |
| Baseline 1b — take all | −0.678 | **−0.660** | +0.018 |

No ML model beat the "take nothing" baseline on the test set. More strikingly, the train-to-test gap scaled **monotonically with model capacity**: simpler models overfit less, gradient boosting overfit dramatically. LightGBM produced a train EV of +3.55 R per trade — a money-printing illusion that collapsed to −0.342 R in test.

The strongest individual predictive feature was `minutes_in_opposite_half` (speed of the reversal): fast-reaction setups won 11.4 % vs. 2.1 % for slow ones — but this was not enough to flip the strategy into positive territory under realistic costs.

## Scientific contribution

While the strategy itself proved unprofitable, the process produced several generalizable methodological findings about ML applied to weak-signal financial data, including:

- A clean demonstration of how model capacity correlates monotonically with overfitting on weak-signal datasets
- An instance of Simpson's paradox in the RSI feature (opposite monotonic trends within long/short subgroups)
- A spurious-causation unwinding (`setup_index_in_day` → `minutes_in_opposite_half`)
- Honest reporting of CV compromises and threshold-tuning risks

A well-executed negative result is more valuable than a poorly-executed positive one — and this project documents a process that can be reused to evaluate other trading methods.

## Status

✅ **Complete** — final report (`docs/loppuraportti.docx`) delivered May 2026.

## Repository structure

```
boxmethod/
├── src/
│   ├── data_fetcher.py        # Phase 1 — Binance API
│   ├── box_method.py          # Phase 2 — setup detector
│   └── features.py            # Phase 3 — feature groups A–E
├── notebooks/
│   ├── 01_data_fetch.ipynb
│   ├── 02_box_setups.ipynb
│   ├── 03_features.ipynb
│   └── 04_modeling.ipynb
├── data/                       # raw + processed parquet (gitignored)
├── models/                     # trained joblib models + comparison tables
├── reports/figures/            # key plots
├── docs/                       # phase findings and final report
└── requirements.txt
```

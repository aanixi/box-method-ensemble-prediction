# Ensemble Methods for Predicting Trading Setup Quality

*An empirical evaluation of the Box Method using machine learning*

**Final report — summary of phases 1–4**
**May 2026**

---

## Abstract

This project empirically evaluates whether machine learning can turn the Box Method — a technical-analysis trading strategy popularized on YouTube Shorts — into a profitable strategy. The work was carried out in four phases: (1) fetching historical price data from the Binance Spot API, (2) translating the Box Method rules into an algorithm and identifying all qualifying setups, (3) designing and constructing predictive features, and (4) training and evaluating ensemble models.

The resulting dataset contained 8,281 setups on the ETHUSDT and SOLUSDT pairs between 1 January 2024 and 8 May 2026. The naive win rate (taking every setup) was 7.1 %, while the strategy's theoretical break-even threshold is approximately 11.1 %. Phase 3 identified several strong individual predictive features, the most important being `minutes_in_opposite_half` — the speed of the reversal: fast-reaction setups won 11.4 % vs. 2.1 % for slow ones. Phase 4 trained Random Forest, XGBoost, LightGBM, and a stacking ensemble combining them.

**Main result: no ML model produced a positive expected value on the test set.** The train-to-test performance gap formed a monotonic series with model capacity: rules −0.45 R → RF −0.53 R → Stacking −0.92 R → XGBoost −2.05 R → LightGBM −3.89 R. The more flexible the model, the worse it overfit the training data.

The project's scientific contribution is strongly methodological: although the strategy itself proved unprofitable, the process produced several generalizable observations about machine-learning pitfalls in trading research, including the unwinding of spurious causation (`setup_index_in_day` → `minutes_in_opposite_half`), Simpson's paradox in the RSI feature, and a concrete demonstration of how model capacity correlates with overfitting.

---

## 1. Introduction and research question

The Box Method is a technical-analysis trading method presented in YouTube Shorts videos. It uses the high and low of the previous UTC day's price range — the "box" edges — as support and resistance levels. Entries are taken after a reversal candle, take-profit is placed at the opposite box edge, and stop-loss behind the reversal candle. The strategy promises a high reward-to-risk ratio, but its empirical profitability has never been publicly documented.

The research question was two-part:

- Is the Box Method profitable on its own without any filtering?
- Can machine learning identify the subset of setups with positive expected value, thereby turning the strategy profitable?

The study design was set up so that either question could be answered positively or negatively — both outcomes have scientific value. Particular attention was paid to methodological honesty: time-aware train/test split, systematic avoidance of look-ahead bias, EV-based evaluation, and a priori regularization of model parameters.

---

## 2. Phase 1: Data acquisition

### 2.1 Objective and implementation

The goal of Phase 1 was to fetch historical price data from the Binance Spot exchange for ML model training. Data was stored locally in Parquet format so that subsequent phases could reload it quickly without API calls.

### 2.2 Chosen parameters

- **Pairs:** ETHUSDT and SOLUSDT (liquid, well-known, allow assessing cross-pair generalization)
- **Primary timeframe:** 15 minutes (a compromise between data volume and noise; additionally 1h and 4h for multi-timeframe features)
- **Period:** 1 January 2024 – 8 May 2026, covering three distinct market regimes:

| Period | Market regime | Significance for modeling |
|---|---|---|
| 2024 – early 2025 | Bull market (uptrend) | Long trends, momentum common |
| Spring 2026 | Bear crash | Volatility spikes, false breakouts |
| Late spring 2026 | Sideways (range-bound) | Range strategies useful, trend-following weak |

### 2.3 Reproducibility

Fixing the end date to 8 May 2026 is a deliberate decision: re-running the data fetch later produces exactly the same dataset. This is essential for scientific reproducibility. Reproducibility was further ensured by fixed fetch parameters and a `requirements.txt` file.

### 2.4 Phase outputs

Phase 1 produced a working data-fetching module, a documented notebook, and 6 Parquet files (ETHUSDT and SOLUSDT × 15m, 1h, 4h). Data integrity (row counts, time intervals, no NaN, reasonable values) was verified with sanity checks.

---

## 3. Phase 2: Identifying Box setups

### 3.1 Objective

The goal of Phase 2 was to translate the Box Method's trading rules into an algorithm that scans the raw candle data and identifies all situations meeting the rules. Each setup was also labeled with its outcome (label = 1 win, 0 loss, NaN expired). The setup definition and labeling approach are among the project's most methodologically critical choices — a careless definition undermines everything built on top of it.

### 3.2 Box Method rules and interpretation decisions

The box was defined as the previous UTC day's high–low range. Entry happens in three stages: price visits the opposite half of the box, a reversal candle forms (for longs, a green candle after a chain of reds), and a later candle closes above the reversal candle's high. Take-profit is at the opposite box edge; stop-loss is the reversal candle's low minus a 0.15 % buffer.

The video left several details ambiguous. The most significant interpretations and their justifications were documented: close-based half-of-box definition, reversal candle selection (first green after a chain of reds), stop-loss placement (reversal candle low − 0.15 % buffer, chosen from three alternatives based on empirical comparison), symmetric addition of the short side, and a conservative "SL hit first" assumption in simultaneity cases.

### 3.3 Results

**Number of setups**

| Pair | Long | Short | Total | Per day |
|---|---|---|---|---|
| ETHUSDT | 1,941 | 2,075 | 4,016 | 4.7 |
| SOLUSDT | 2,150 | 2,115 | 4,265 | 5.0 |
| **Total** | **4,091** | **4,190** | **8,281** | **4.8** |

Naive win rate (all labeled setups): 7.1 %. The median risk/reward ratio is 7.8 (long) and 8.7 (short) — a typical trade aims to win about 8× what it risks. The strategy's theoretical break-even win rate is approximately 11.1 %, so the naive 7.1 % falls below it.

### 3.4 Key finding: setup index in day

Exploratory analysis showed that the win rate decreases monotonically with `setup_index_in_day`. The day's first setup wins 11.0 %, the tenth essentially 0 %. This was Phase 2's most important finding and served as the starting point for Phase 3's feature design. Practical implication: simply taking only the first setup of the day nearly exceeds the break-even threshold without any ML modeling. As it later turned out in Phase 3, this finding was partly misleading.

**Phase 2 summary**

- 8,281 setups (4,016 ETHUSDT, 4,265 SOLUSDT) identified by the Box Method rules
- Naive win rate 7.1 %, break-even ~11.1 % → the strategy is not profitable as-is
- `setup_index_in_day` monotonic win rate 11.0 % → 0 % — strongest individual signal
- Losing setups concentrate on trending days → the ML model should be able to recognize this

---

## 4. Phase 3: Feature engineering

### 4.1 Design principles

Phase 3 built the predictive features that form the input space for the ML models. Feature selection followed three principles:

- Less is more — 8,281 setups do not support a 30+ feature space without overfitting risk
- Correlated features are pruned (threshold |r| > 0.8)
- Every feature has a documented justification — no ad-hoc ideas

Look-ahead bias was prevented systematically: all features are computed using only closed candles preceding `entry_time`. For multi-timeframe features, `pd.merge_asof` ensures only closed higher-timeframe candles are available.

The features were split into five groups, each implemented and analyzed separately. Two features were pruned during analysis on grounds of high correlation. The final feature space consists of 19 features, of which 17 were used in modeling.

### 4.2 Group A: Setup features

Group A consists of the setup's own geometric and temporal properties: `setup_index_in_day`, `direction`, `risk_reward_ratio`, `entry_position_in_box`, `entry_outside_box` (flag), `hour_sin` / `hour_cos` (cyclical time encoding).

**Finding 1: Box breaking is common and informative**

Box breaking at entry time turned out to be surprisingly common (45.9 % of all setups) and a strong predictive feature:

| `entry_outside_box` | n | Win % |
|---|---|---|
| 0 (box intact) | 3,911 | 9.8 % |
| 1 (box broken) | 2,730 | 3.1 % |

Setups where the box is intact win over three times more often than setups where the box is broken. Interpretation: a broken box means the previous day's price level no longer describes the present — the support-level assumption fails.

**Finding 2: Risk/Reward is an inverse predictor**

This is Group A's strongest and most counter-intuitive finding. Conventional trading wisdom favors high-R/R setups, but in the Box Method, high R/R systematically signals a less probable setup:

| Quintile | R/R mean | n | Win % | EV (R/trade) |
|---|---|---|---|---|
| Q1 (lowest) | 3.82 | 1,329 | 18.0 % | −0.13 |
| Q2 | 6.13 | 1,328 | 8.5 % | −0.39 |
| Q3 | 8.15 | 1,328 | 4.9 % | −0.55 |
| Q4 | 10.78 | 1,328 | 2.6 % | −0.70 |
| Q5 (highest) | 16.84 | 1,328 | 1.4 % | −0.76 |

Win rate falls monotonically from 18.0 % to 1.4 %. The mechanism is geometric: high R/R arises either when the entry is close to the box edge (a long distance to TP) or when the reversal candle is small (a tight SL). In both cases, SL is more likely to be hit first.

**Finding 3: EV is negative in every R/R quintile**

The combination "first setup + box intact" yields a win rate of 11.3 % (close to break-even), but its EV is still −0.29 R per trade. The asymmetric R/R distribution (winners' R/R median 5.10, losers' 8.39) explains the gap. EV is negative across all five R/R quintiles — no simple R/R-based filter turns the strategy profitable. This defines the bar that ML models need to clear.

### 4.3 Group B: Box context

Group B links the setup to broader market context with two features: `prev_day_range_pct` and `box_size_vs_atr14d`.

**Finding 4: Previous day's range is a weak predictor**

Win rate varies between 5.7–8.2 % across quintiles, with a mildly upward but non-monotonic relationship. The feature is retained in the ML input space in case of interactions.

**Finding 5: `box_size_vs_atr14d` is very strong**

| Quintile | Ratio mean | n | Win % |
|---|---|---|---|
| Q1 (smallest) | 0.46 | 1,317 | 12.53 % |
| Q2 | 0.67 | 1,312 | 8.69 % |
| Q3 | 0.86 | 1,311 | 6.41 % |
| Q4 | 1.08 | 1,313 | 4.95 % |
| Q5 (largest) | 1.64 | 1,313 | 2.67 % |

Win rate falls monotonically from 12.5 % to 2.7 %. Q1's win rate exceeds the break-even threshold — small boxes relative to ATR form an almost profitable subgroup on their own. Interpretation: a small box means a quiet market day, with TP and SL levels closer together, making TP relatively easy to reach.

Correlation analysis revealed that Group A's original `box_size_pct` feature correlated r = 0.811 with `box_size_vs_atr14d`. Stronger monotonicity and a clearer theoretical justification (14-day market context) resolved the comparison, and `box_size_pct` was pruned.

### 4.4 Group C: Multi-timeframe trend

Group C extends the feature set by one feature (`trend_4h` = (EMA50−EMA200)/EMA200) and a derived flag `trend_aligned`.

**Finding 6: Counter-trend setups beat trend-aligned setups slightly**

| `trend_aligned` | n | Win % |
|---|---|---|
| 0 (against trend) | 3,322 | 7.38 % |
| 1 (with trend) | 3,319 | 6.75 % |

The difference is small (0.63 pp), but its direction is opposite to the "the trend is your friend" wisdom. The interpretation supports the evolving working hypothesis: the Box Method is structurally a mean-reversion strategy, expecting price to return inside the box. Strong trends push price consistently in one direction — headwind for a mean-reversion strategy.

**Finding 7: Shorts in a downtrend are the weakest subgroup**

A cross-tabulation of `direction` × `trend_aligned` revealed asymmetry: "short + downtrend" (6.28 %) is the weakest subgroup. This is the opposite of the traditional "short in a falling market" mantra and reinforces the mean-reversion hypothesis.

### 4.5 Group D: Volatility and momentum

Group D examines market conditions at entry time: ATR, RSI, volume ratio. Three of the original seven features were pruned on correlation grounds; the final space retains `atr_15m_vs_atr_daily`, `atr_1h_pct`, `rsi_1h`, and `volume_vs_ma20_15m`.

**Findings 9–10: ATR features are strong and monotonic**

`atr_1h_pct` is Group D's strongest individual feature: win rate falls from 11.6 % to 4.5 % from Q1 to Q5. Consistent with Group B's finding — the Box Method works best in calm markets.

**Finding 11: Simpson's paradox in RSI**

`rsi_1h` looked superficially weak (Q5−Q1 ≈ 0), but a direction-split analysis revealed a classic Simpson's paradox: the combined inverted-U shape is composed of two opposite monotonic trends.

| Quintile (long) | RSI 1h range | n | Win % |
|---|---|---|---|
| Q1 | [6.8 – 29.7] | 661 | 1.4 % |
| Q2 | [29.7 – 35.0] | 658 | 5.3 % |
| Q3 | [35.0 – 39.6] | 659 | 5.2 % |
| Q4 | [39.6 – 44.4] | 659 | 9.7 % |
| Q5 | [44.4 – 63.0] | 659 | 15.8 % |

Long setups' win rate grows monotonically with RSI (1.4 % → 15.8 %), while short setups' win rate falls monotonically (12.3 % → 3.0 %). In both subgroups, the best results come when the setup goes least against short-term momentum — a pullback structure where mild with-momentum predicts a better outcome.

> **Updated working hypothesis after Phase 3 (interim)**
>
> The Box Method is a mean-reversion strategy that works best in a calm market when the setup does not go radically against short-term momentum.

### 4.6 Group E: Reversal candle structure

Group E examines the quality of the reversal moment and the intraday dynamics preceding it. Final features: `reversal_candle_size_pct` and `minutes_in_opposite_half`.

**Finding 16: `minutes_in_opposite_half` is Group E's star feature**

| Quintile | Time range (min) | n | Win % |
|---|---|---|---|
| Q1 (fastest) | 10–55 | 1,371 | 11.43 % |
| Q2 | 55–170 | 1,296 | 11.21 % |
| Q3 | 175–415 | 1,326 | 6.85 % |
| Q4 | 420–745 | 1,321 | 3.54 % |
| Q5 (slowest) | 745–1415 | 1,327 | 2.12 % |

The Q5−Q1 difference of −9.31 pp is stronger than `atr_1h_pct` (−7.1 pp) and on par with `box_size_vs_atr14d` (−9.8 pp). Q1+Q2 together (minutes < 175) yield about 11.3 % win rate on 40.5 % of labeled setups — exceeding the break-even threshold without any ML.

**Finding 18: `setup_index_in_day` was a proxy for fast reaction**

A correlation of r = +0.75 between `minutes_in_opposite_half` and `setup_index_in_day` motivated a 2×2 disambiguation analysis:

|  | Slow reaction (≥175 min) | Fast reaction (<175 min) |
|---|---|---|
| Early setup (≤2) | 5.41 % (n=629) | 11.41 % (n=2,384) |
| Later setup (≥3) | 3.98 % (n=3,366) | 11.07 % (n=262) |

The table reveals that when reaction speed is controlled for, the effect of `setup_index_in_day` essentially disappears (11.41 % vs. 11.07 %, a difference of 0.34 pp). In contrast, the effect of `minutes_in_opposite_half` persists in both setup_index groups (+6.00 and +7.09 pp). Phase 2's key finding was therefore not wrong, but its causal interpretation was partly misleading — the actual underlying mechanism is the speed of the reversal moment.

> **Final working hypothesis after Phase 3**
>
> The Box Method is a mean-reversion strategy that works best in a calm market, when (a) the opposing move preceding the entry is fast and sharp rather than prolonged, and (b) the setup goes with short-term momentum but against the medium-term structural trend.

---

## 5. Phase 4: Model training and evaluation

### 5.1 Metric and split

The primary metric is Expected Value (EV) per trade in R-units, where 1 R equals the amount risked in the trade. Classic classification metrics (accuracy, F1, ROC-AUC) do not answer the question "does the strategy make money?". A transaction cost of 0.15 % per trade (taker fee 0.1 % + spread 0.05 %) was subtracted from returns.

The train/test split is time-aware: all setups created before 2025-10-01 went to training (4,817) and all after that to test (1,749). The test period contains three distinct market regimes — a strong test of generalization.

### 5.2 Baseline strategies: the ML bar

Four baselines were defined to provide an interpretive framework for the ML results:

| Baseline | Logic | Test EV (R/trade) |
|---|---|---|
| 1a: Take nothing | No trades | 0.000 (the ML bar) |
| 1b: Take all | Every setup is taken | −0.660 |
| 2: First setup of the day only | Group A finding | −0.557 |
| 3: Combined rule filter | 5 rules from Phase 3 findings | −0.330 |

Baseline 3's collapse from train to test (train EV +0.117 R → test EV −0.330 R, gap −0.447 R) was the first strong indication of overfitting. This formed **Finding #23** and posed a critical question for the ML models.

### 5.3 Modeling protocol

All models (RF, XGBoost, LightGBM, stacking) were trained with a uniform protocol: 17 features (auxiliary features `ema50_4h` and `ema200_4h` removed), `class_weight='balanced'` for class imbalance (ratio 13.17:1), TimeSeriesSplit expanding-window CV (5 folds), threshold tuning by EV with a minimum of 5 trades per fold, and a priori regularization of parameters (RF `max_depth=5`, `min_samples_leaf=30`; gradient boosting `max_depth=4`, `learning_rate=0.05`).

### 5.4 Results: models

| Model | Train EV | Test EV | Train→test gap | % of setups |
|---|---|---|---|---|
| Random Forest | +0.245 | −0.283 | −0.528 | 7.3 % |
| XGBoost | +1.871 | −0.176 | −2.046 | 10.1 % |
| LightGBM | +3.551 | −0.342 | −3.892 | 5.0 % |
| Stacking | +0.630 | −0.293 | −0.923 | 13.3 % |

Test EV was negative across all four models. XGBoost's best test result (−0.176 R) is quantitatively the least bad — but qualitatively, when judged on the train-to-test gap and therefore on reliability, it is exceptionally poor.

### 5.5 Finding #25: capacity monotonic with overfit

Phase 4's central methodological finding is that model capacity and the train-to-test gap form a clear monotonic series:

| Strategy / model | Train→test gap | Capacity |
|---|---|---|
| Rule filter (Baseline 3) | −0.447 R | 5 rules |
| Random Forest | −0.528 R | 300 trees, max_depth 5 |
| Stacking ensemble | −0.923 R | 3 base + meta |
| XGBoost | −2.046 R | 300 trees, gradient boost |
| LightGBM | −3.892 R | 300 trees, leaf-wise |

The more complex the model, the worse the overfit. ML did not provide overfit protection over simple rules — on the contrary, more complex models overfit more. This is a generalizable observation that applies to ML applied to weak-signal datasets more broadly.

### 5.6 Finding #27: the stacking meta-learner negates LightGBM

The stacking meta-learner (logistic regression) coefficients for each base model's prediction were: RF +3.73, XGBoost +1.82, LightGBM −0.32. The negative coefficient on LightGBM is striking — it means the meta-learner inferred from the out-of-fold predictions that a high LightGBM probability predicts a loss, not a win. When LightGBM "shouts take the trade", the meta tries to do the opposite. This is empirical confirmation of Finding #25 from another angle: LightGBM did not just overfit relative to the test period, but already inside the training data at the level of out-of-fold predictions.

---

## 6. Conclusions

### 6.1 Answer to the research question

The project began with a two-part research question:

- **Is the Box Method profitable on its own?** No. The naive win rate of 7.1 % falls clearly below the break-even threshold of ~11.1 %, and the "take all" strategy's EV is −0.66 R/trade in the test period.
- **Can machine learning turn the strategy profitable?** Not within the bounds of this dataset and this model lineup. None of the four tested models produced positive expected value in the test period, and none cleared Baseline 1a's bar (take nothing).

### 6.2 Why

The individual Phase 3 features are genuine qualitative signals — `minutes_in_opposite_half`, for example, produced a Q1–Q5 difference of 11.4 % → 2.1 %. But when these signals are combined into ML models and evaluated by EV in the test period, the profitability disappears. Three factors explain this:

- The signals are weak in absolute terms. Even the strongest individual quintile yields a win rate of about 11–14 %, close to the break-even threshold. A small change in market conditions is enough to drop below the threshold.
- The signals are regime-specific. The training data was primarily a bull market, while the test period included a bear crash and a recovery. The signals do not generalize across market regimes strongly enough to overcome transaction costs.
- ML model capacity is too large relative to signal strength. The models find stronger (but spurious) patterns in the training data than the genuine weak signals — Finding #25 demonstrates this quantitatively.

### 6.3 Scientific contribution

Although the strategy itself proved unprofitable, the process produced several generalizable observations that can be applied to evaluating other trading methods, or more broadly to weak-signal datasets:

- **Unwinding spurious causation (Finding #18).** The effect of `setup_index_in_day` turned out to be mediated through `minutes_in_opposite_half`. Phase 2's "obvious" finding was partly misleading — the true mechanism was revealed only by disambiguating two variables.
- **Simpson's paradox (Finding #11).** RSI looked nearly useless when aggregated, but a direction split showed long and short setups behaving oppositely and monotonically. A classic reminder that an aggregated view can hide subgroup-level signals.
- **Capacity–overfit monotonicity (Finding #25).** Across five strategies (rule filter, RF, stacking, XGBoost, LightGBM), the train-to-test gap follows model capacity monotonically. A generalizable observation that applies to ML on weak-signal data more broadly.
- **The stacking meta reveals overfitting (Finding #27).** A negative coefficient on a base model is a healthy diagnostic signal — the meta-learner detected from out-of-fold predictions that the "most confident" predictor was systematically wrong. A practical diagnostic for evaluating ensemble models.
- **Baselines are essential.** The ML bar is not "take all" but "take nothing" (EV = 0). This matters methodologically: the alternative of "no strategy" is always available, and a loss-making model has no value.
- **EV-based evaluation rather than classification metrics.** High accuracy does not guarantee profitability, and low accuracy does not preclude it — especially with asymmetric R/R ratios. The choice of metric determines what gets optimized.

### 6.4 Status of the hypothesis

The working hypothesis formulated at the end of Phase 3 — that the Box Method is a mean-reversion strategy — receives empirical support at the qualitative level (low R/R, small box, fast reaction, counter-trend direction systematically yield a higher win rate). The Phase 4 results do not refute the hypothesis, but they show that these qualitative regularities are not strong enough to translate quantitatively into a profitable strategy within the bounds of this dataset and the market-regime variation it contains.

### 6.5 Directions for further research

The negative outcome opens several research directions:

- A longer and more varied data span covering multiple market cycles — generalization could be assessed across several cycles
- Alternative labeling strategies (e.g. different SL levels, different TP levels, inclusion of expired setups under various labelings)
- Regime-conditional models that detect the market regime (trend/range/volatility) and apply a different model in each regime
- Closer study of individual strong features as standalone strategies — e.g. "take only setups where `minutes < 175` and `box_size_vs_atr14d < 0.5`"
- Alternative transaction-cost assumptions — the project's 0.15 % per-trade assumption is conservative, and a market-maker fee of 0.02 % would meaningfully shift the break-even threshold

### 6.6 Closing remarks

The project's outcome is negative: the Box Method is not a profitable strategy, and machine learning did not turn it into one within this study design. This is, however, a scientifically valuable result — it clearly answers the question posed and documents the process by which the answer was obtained. The internet is full of unsubstantiated profitability claims about trading strategies; this project shows that careful empirical research can confirm or refute such claims.

The project's scientific contribution lies in the process and the methodological observations it produced. Findings #11, #18, #25, and #27 are worth remembering in any future machine-learning project working with weak-signal data. A well-executed negative result is more valuable than a poorly-executed positive one.

# PHASE0_FREE_PROXY — QQQ
**FREE PROXY EXPERIMENT — NOT FUTURES VALIDATION.** QQQ is an ETF proxy. Nothing here validates any futures strategy.

## Conclusion: **MIXED**
- final-test expectancy +0.296 R over 6 trades; failed checks: ['beats random-direction control (p<0.10)', 'neighbours form a plateau in the final test', 'final-test t-stat >= 2']
- Rule (stated before looking at the test): NO PRELIMINARY EVIDENCE if final-test expectancy <= 0; PROMISING only if validation > 0, survives $0.03/share, beats random-direction control at p<0.10, plateau in final test and t>=2; else MIXED

## Data used
- Source: Yahoo Finance via yfinance 1.7.0 (free, no key, no account)
- Interval: **5m** (chosen by measured availability: 1m: 20 sessions from 2026-08-28, 2m: 31 sessions from 2026-08-13, 5m: 42 sessions from 2026-07-29, 15m: 42 sessions from 2026-07-29)
- Returned: 2026-07-29 09:30:00-04:00 to 2026-09-25 15:55:00-04:00; 3,276 bars; source timezone America/New_York; unadjusted (auto_adjust=False); regular trading hours only (prepost=False)
- Downloaded 2026-09-26T07:26:48.858911+00:00; processed file SHA-256 `91386b2b22a4c337…`

## Data quality
- Phase-0 DQ verdict: **REVIEW**
- Sessions 42 (2026-07-29 … 2026-09-25); UTC offsets seen ['-1 days +20:00:00'] (DST transition in sample: False)
- Sessions whose first bar is not 09:30 ET: 0; incomplete sessions: 0; missing XNYS sessions: none; early closes: none
- Modal peak-volume minute: 570 (570 = 09:30 ET)
- Engine DQ issues: ['[warning] off_tick_prices n=2202']
- 1-minute cross-check: {"bars_compared": 1560, "max_abs_diff": {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0}, "share_ohlc_identical_within_half_cent": 1.0, "volume_ratio_median": 1.0}
- note: warning [off_tick_prices]: 67.2% of bars have prices off the 0.01 tick grid (ratio back-adjusted continuous contract?). Tick-based order levels become approximate. (n=2202)

## Sample design (chronological, no shuffling)
- TRAIN: 2026-07-29 … 2026-09-01 (25 sessions)
- VALIDATION: 2026-09-02 … 2026-09-14 (8 sessions)
- FINAL TEST: 2026-09-15 … 2026-09-25 (9 sessions); not in memory during selection. The selection was frozen to `selection_frozen.json` (SHA-256 `642d22eab1b71dc1…`) before the test was evaluated.
- Configurations evaluated: 236,640 (raw 344,250; invalid 11,475; canonical duplicates 73,950; limit-entry 50 %-width stops identical to midpoint stops removed 22,185)
- Friction for selection and headline: $0.02/share charged on EVERY fill (entry and exit, market, stop and limit alike), i.e. a fixed round-trip cost per share; limit fills must also trade through by $0.01. $0.00 is a reference only.
- Multiple testing (train Sharpe across configs): best 9.69 vs 14.07 expected from selection alone under the null.

## Best in-sample configuration (reported, not emphasised)
- Best training score: `07642b8bf78a` (range 10m, entry TF 10, limit, confirm close+0t+0W, stop or_opposite, 0.6R, cutoff 13:00, short)
- Highest train+validation net P&L: `d4689dafa80a` (range 5m, entry TF 5, market, confirm close+0t+0W, stop or_mid, 1.5R, cutoff 10:00, both)

In-sample winners on every phase (degradation check):

| configuration                     | phase   |   trades |   exp_r |   net_pnl |
|:----------------------------------|:--------|---------:|--------:|----------:|
| best_in_sample_train_score        | train   |       12 |  +0.336 | +1202.450 |
| best_in_sample_train_score        | val     |        3 |  +0.240 |  +197.500 |
| best_in_sample_train_score        | test    |        2 |  -0.223 |  -208.000 |
| best_in_sample_train_plus_val_pnl | train   |       20 |  +0.354 | +3235.640 |
| best_in_sample_train_plus_val_pnl | val     |        8 |  -0.092 |  -252.130 |
| best_in_sample_train_plus_val_pnl | test    |        9 |  +0.080 |   +69.870 |

## Validation-selected configuration
`66a6f5109b68` (range 5m, entry TF 5, limit, confirm close+1t+0W, stop or_pct:0.75, 0.6R, cutoff 13:00, both)
- Why: highest selection score = ½·validation score + ½·(½·training score + ½·median training score of its immediate parameter neighbours) among the 25 finalists ranked by that neighbourhood-robust training score. Training neighbourhood class: **mixed**.
- Training: 17 trades, +0.297 R; Validation: 7 trades, +0.115 R.

Top finalists:
|   range_minutes |   entry_tf | entry_method   | confirmation   | stop        |   target_r | cutoff   | direction   |   train_trades |   train_exp_r |   neighbour_median_train_score |   val_trades |   val_exp_r |   selection_score |
|----------------:|-----------:|:---------------|:---------------|:------------|-----------:|:---------|:------------|---------------:|--------------:|-------------------------------:|-------------:|------------:|------------------:|
|               5 |          5 | limit          | 1tick          | or_pct:0.75 |      0.600 | 13:00    | both        |         17.000 |         0.297 |                          0.759 |        7.000 |       0.115 |             0.648 |
|               5 |          5 | limit          | 1tick          | or_pct:0.75 |      0.600 | 11:30    | both        |         16.000 |         0.279 |                          0.770 |        7.000 |       0.115 |             0.647 |
|               5 |          5 | limit          | 1tick          | or_pct:0.75 |      0.600 | 12:00    | both        |         16.000 |         0.279 |                          0.770 |        7.000 |       0.115 |             0.647 |
|               5 |          5 | limit          | 2tick          | or_pct:0.75 |      0.600 | 13:00    | both        |         17.000 |         0.297 |                          0.747 |        7.000 |       0.115 |             0.645 |
|               5 |          5 | limit          | 1tick          | or_pct:0.75 |      0.600 | 11:00    | both        |         16.000 |         0.279 |                          0.770 |        6.000 |       0.037 |             0.463 |
|               5 |          5 | limit          | 1tick          | or_pct:0.67 |      0.700 | 13:00    | both        |         17.000 |         0.350 |                          0.760 |        7.000 |      -0.061 |             0.442 |
|               5 |          5 | limit          | 1tick          | or_pct:0.67 |      0.700 | 11:30    | both        |         16.000 |         0.330 |                          0.770 |        7.000 |      -0.061 |             0.441 |
|               5 |          5 | limit          | 1tick          | or_pct:0.67 |      0.700 | 12:00    | both        |         16.000 |         0.330 |                          0.770 |        7.000 |      -0.061 |             0.441 |

## Unseen final-test result (the result that matters)
| metric                     | TRAIN     | VALIDATION                              | FINAL TEST (unseen)                     |
|:---------------------------|:----------|:----------------------------------------|:----------------------------------------|
| Trades                     | 17        | 7                                       | 6                                       |
| Win rate                   | 82.4%     | 71.4%                                   | 83.3%                                   |
| Expectancy (R)             | +0.297    | +0.115                                  | +0.296                                  |
| t-stat of mean R           | +1.94     | not meaningful (n<10 or ~zero variance) | not meaningful (n<10 or ~zero variance) |
| Profit factor              | 3.20      | 1.63                                    | 1.56                                    |
| Net P&L (100 sh)           | $1,081.50 | $160.26                                 | $99.00                                  |
| Gross P&L (100 sh)         | $1,149.50 | $188.26                                 | $123.00                                 |
| Friction paid              | $68.00    | $28.00                                  | $24.00                                  |
| Sharpe (daily, annualised) | +7.16     | +3.25                                   | +2.27                                   |
| Max drawdown (R)           | -1.03     | -1.50                                   | -1.02                                   |
| Max drawdown ($, 100 sh)   | $-179.00  | $-151.00                                | $-178.00                                |
| Average winner             | $112.39   | $83.25                                  | $55.40                                  |
| Average loser              | $-164.00  | $-128.00                                | $-178.00                                |
| Largest winner             | $167.00   | $120.00                                 | $88.00                                  |
| Largest loser              | $-179.00  | $-151.00                                | $-178.00                                |
| Long trades                | 8         | 5                                       | 4                                       |
| Long expectancy (R)        | +0.177    | +0.252                                  | +0.562                                  |
| Long net P&L               | $224.50   | $145.26                                 | $237.00                                 |
| Short trades               | 9         | 2                                       | 2                                       |
| Short expectancy (R)       | +0.403    | -0.229                                  | -0.238                                  |
| Short net P&L              | $857.00   | $15.00                                  | $-138.00                                |

## Controls and baselines (same dates)
| phase   |   strategy mean R |   random-direction median |   random 5% |   random 95% |   p(random >= strategy) |   buy&hold % |   buy&hold $ (100 sh) |   intraday-long $ (100 sh) |   sim check |dR| |
|:--------|------------------:|--------------------------:|------------:|-------------:|------------------------:|-------------:|----------------------:|---------------------------:|-----------------:|
| train   |             0.297 |                     0.202 |      -0.080 |        0.391 |                   0.215 |        4.766 |              3219.500 |                    761.500 |            0.000 |
| val     |             0.115 |                     0.114 |      -0.344 |        0.345 |                   0.472 |        0.290 |               205.000 |                   1302.500 |            0.000 |
| test    |             0.296 |                     0.028 |      -0.506 |        0.296 |                   0.158 |        5.034 |              3568.000 |                   1637.470 |            0.000 |

## Robustness: neighbourhood of the selected configuration (expectancy R per phase)
| config_key   |   range_minutes |   entry_tf | entry_method   | confirmation   | stop        |   target_r | cutoff   | direction   |   train_n_trades |   train_exp_r |   val_n_trades |   val_exp_r |   test_n_trades |   test_exp_r | role      |
|:-------------|----------------:|-----------:|:---------------|:---------------|:------------|-----------:|:---------|:------------|-----------------:|--------------:|---------------:|------------:|----------------:|-------------:|:----------|
| 66a6f5109b68 |               5 |          5 | limit          | 1tick          | or_pct:0.75 |      0.600 | 13:00    | both        |           17.000 |         0.297 |          7.000 |       0.115 |           6.000 |        0.296 | SELECTED  |
| eacec601eaea |               5 |          5 | limit          | close          | or_pct:0.75 |      0.600 | 13:00    | both        |           17.000 |         0.297 |          7.000 |       0.115 |           6.000 |        0.296 | neighbour |
| 80a3424b430b |               5 |          5 | limit          | 1tick          | or_pct:0.67 |      0.600 | 13:00    | both        |           17.000 |         0.294 |          7.000 |      -0.118 |           6.000 |        0.292 | neighbour |
| 8428a66d1f81 |               5 |          5 | limit          | 1tick          | or_pct:0.75 |      0.500 | 13:00    | both        |           17.000 |         0.214 |          7.000 |       0.043 |           6.000 |        0.211 | neighbour |
| a9099430dc36 |               5 |          5 | limit          | 1tick          | or_pct:0.75 |      0.600 | 12:00    | both        |           16.000 |         0.279 |          7.000 |       0.115 |           6.000 |        0.296 | neighbour |
| 603cab7ff6e7 |               5 |          5 | limit          | 1tick          | or_pct:0.75 |      0.700 | 13:00    | both        |           17.000 |         0.151 |          7.000 |       0.187 |           6.000 |        0.381 | neighbour |
| 78c56a19b7bd |               5 |          5 | limit          | 2tick          | or_pct:0.75 |      0.600 | 13:00    | both        |           17.000 |         0.297 |          7.000 |       0.115 |           6.000 |        0.296 | neighbour |
| 650702c6055f |               5 |         10 | limit          | 1tick          | or_pct:0.75 |      0.600 | 13:00    | both        |           16.000 |         0.080 |          7.000 |       0.115 |           5.000 |       -0.076 | neighbour |
| a6409bd30f15 |              10 |          5 | limit          | 1tick          | or_pct:0.75 |      0.600 | 13:00    | both        |           19.000 |        -0.032 |          7.000 |       0.121 |           6.000 |        0.041 | neighbour |

Selected configuration neighbourhood class by phase: {'train': 'mixed', 'val': 'mixed', 'test': 'mixed'}; neighbour median expectancy: train +0.247, val +0.115, test +0.294
Rank correlation of all configurations' expectancy: train→test +0.029, validation→test -0.210.
Heatmaps: `heatmaps/*.png` (median expectancy across hidden parameters, per phase).

## Execution sensitivity (selected configuration)
| split   |   friction_usd_per_share |   trades |   exp_r |   t_stat_r |   profit_factor |   net_pnl |   friction_cost |
|:--------|-------------------------:|---------:|--------:|-----------:|----------------:|----------:|----------------:|
| train   |                    0.000 |       17 |   0.319 |      2.091 |           3.395 |  1149.500 |           0.000 |
| val     |                    0.000 |        7 |   0.145 |      0.490 |           1.759 |   188.260 |           0.000 |
| test    |                    0.000 |        6 |   0.336 |      1.258 |           1.707 |   123.000 |           0.000 |
| train   |                    0.010 |       17 |   0.308 |      2.015 |           3.295 |  1115.500 |          34.000 |
| val     |                    0.010 |        7 |   0.130 |      0.438 |           1.692 |   174.260 |          14.000 |
| test    |                    0.010 |        6 |   0.316 |      1.190 |           1.631 |   111.000 |          12.000 |
| train   |                    0.020 |       17 |   0.297 |      1.939 |           3.198 |  1081.500 |          68.000 |
| val     |                    0.020 |        7 |   0.115 |      0.387 |           1.626 |   160.260 |          28.000 |
| test    |                    0.020 |        6 |   0.296 |      1.121 |           1.556 |    99.000 |          24.000 |
| train   |                    0.030 |       17 |   0.286 |      1.864 |           3.103 |  1047.500 |         102.000 |
| val     |                    0.030 |        7 |   0.100 |      0.335 |           1.563 |   146.260 |          42.000 |
| test    |                    0.030 |        6 |   0.275 |      1.051 |           1.483 |    87.000 |          36.000 |
| train   |                    0.050 |       17 |   0.263 |      1.713 |           2.921 |   979.500 |         170.000 |
| val     |                    0.050 |        7 |   0.069 |      0.233 |           1.441 |   118.260 |          70.000 |
| test    |                    0.050 |        6 |   0.235 |      0.907 |           1.342 |    63.000 |          60.000 |

## Concentration (sum of R after removing the best trades)
|                       |   train |     val |    test |
|:----------------------|--------:|--------:|--------:|
| sum_r                 |   +5.05 |   +0.80 |   +1.77 |
| sum_r_without_best_1  |   +4.46 |   +0.22 |   +1.19 |
| sum_r_without_best_3  |   +3.29 |   -0.93 |   +0.07 |
| sum_r_without_best_5  |   +2.12 |   -2.07 |   -1.02 |
| sum_r_without_best_10 |   -0.78 | +nan    | +nan    |
(n/a = fewer trades than removed)

## Grid-wide descriptive evidence (median expectancy R across all configurations sharing the value that traded in that phase)
Configurations with at least one trade: {'train': 231897, 'val': 224094, 'test': 228973}. Descriptive only; never used for selection.
### Opening-range duration
|    |   train |    val |   test |
|---:|--------:|-------:|-------:|
|  5 |  -0.19  | -0.28  |  0.145 |
| 10 |  -0.121 | -0.462 |  0.269 |
| 15 |  -0.058 | -0.483 |  0.207 |
| 20 |  -0.018 | -0.486 |  0.223 |
| 30 |  -0.174 | -0.22  |  0.127 |
### Targets below 2R vs 2R and above
|      |   train |    val |   test |
|:-----|--------:|-------:|-------:|
| R<2  |  -0.096 | -0.369 |  0.164 |
| R>=2 |  -0.164 | -0.527 |  0.368 |
### Direction
|       |   train |    val |   test |
|:------|--------:|-------:|-------:|
| both  |  -0.098 | -0.38  |  0.234 |
| long  |  -0.067 | -0.225 |  0.082 |
| short |  -0.148 | -0.622 |  0.301 |
### Entry method
|        |   train |    val |   test |
|:-------|--------:|-------:|-------:|
| limit  |  -0.13  | -0.419 |  0.096 |
| market |  -0.047 | -0.371 |  0.268 |
| stop   |  -0.249 | -0.409 |  0.163 |
### Entry cutoff
|       |   train |    val |   test |
|:------|--------:|-------:|-------:|
| 10:00 |  -0.058 | -0.58  |  0.311 |
| 10:30 |  -0.108 | -0.488 |  0.229 |
| 11:00 |  -0.062 | -0.421 |  0.163 |
| 11:30 |  -0.1   | -0.343 |  0.163 |
| 12:00 |  -0.132 | -0.351 |  0.176 |
| 13:00 |  -0.134 | -0.335 |  0.173 |
### Share of configurations with positive expectancy: train 31.8%, val 14.1%, test 70.6%

## Resolution cross-check (frozen candidate on 1-minute bars, overlapping final-test dates)
{"period": ["2026-09-15", "2026-09-25"], "trades_1m": 6, "trades_5m": 6, "exp_r_1m": 0.29562604405850107, "exp_r_5m": 0.29562604405850107, "same_direction_days": 6}

## Trade verification
Complete trade log: `trade_log_selected.csv`. Charts (20): `trade_charts/`.
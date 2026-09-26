# PHASE0_FREE_PROXY — SPY
**FREE PROXY EXPERIMENT — NOT FUTURES VALIDATION.** SPY is an ETF proxy. Nothing here validates any futures strategy.

## Conclusion: **MIXED**
- final-test expectancy +0.511 R over 3 trades; failed checks: ['beats random-direction control (p<0.10)', 'neighbours form a plateau in the final test']
- Rule (stated before looking at the test): NO PRELIMINARY EVIDENCE if final-test expectancy <= 0; PROMISING only if validation > 0, survives $0.03/share, beats random-direction control at p<0.10, plateau in final test and t>=2; else MIXED

## Data used
- Source: Yahoo Finance via yfinance 1.7.0 (free, no key, no account)
- Interval: **5m** (chosen by measured availability: 1m: 20 sessions from 2026-08-28, 2m: 31 sessions from 2026-08-13, 5m: 42 sessions from 2026-07-29, 15m: 42 sessions from 2026-07-29)
- Returned: 2026-07-29 09:30:00-04:00 to 2026-09-25 15:55:00-04:00; 3,276 bars; source timezone America/New_York; unadjusted (auto_adjust=False); regular trading hours only (prepost=False)
- Downloaded 2026-09-26T07:17:35.803150+00:00; processed file SHA-256 `37c64ceee49a6bf9…`

## Data quality
- Phase-0 DQ verdict: **REVIEW**
- Sessions 42 (2026-07-29 … 2026-09-25); UTC offsets seen ['-1 days +20:00:00'] (DST transition in sample: False)
- Sessions whose first bar is not 09:30 ET: 0; incomplete sessions: 0; missing XNYS sessions: none; early closes: none
- Modal peak-volume minute: 570 (570 = 09:30 ET)
- Engine DQ issues: ['[warning] off_tick_prices n=3276']
- 1-minute cross-check: {"bars_compared": 1560, "max_abs_diff": {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0}, "share_ohlc_identical_within_half_cent": 1.0, "volume_ratio_median": 1.0}
- note: warning [off_tick_prices]: 100.0% of bars have prices off the 0.01 tick grid (ratio back-adjusted continuous contract?). Tick-based order levels become approximate. (n=3276)

## Sample design (chronological, no shuffling)
- TRAIN: 2026-07-29 … 2026-09-01 (25 sessions)
- VALIDATION: 2026-09-02 … 2026-09-14 (8 sessions)
- FINAL TEST: 2026-09-15 … 2026-09-25 (9 sessions); not in memory during selection. The selection was frozen to `selection_frozen.json` (SHA-256 `2c312c5c7e2c1711…`) before the test was evaluated.
- Configurations evaluated: 236,640 (raw 344,250; invalid 11,475; canonical duplicates 73,950; limit-entry 50 %-width stops identical to midpoint stops removed 22,185)
- Friction for selection and headline: $0.02/share per market or stop fill; limit fills (limit entries, targets) must trade through by $0.01.
- Multiple testing (train Sharpe across configs): best 14.52 vs 13.28 expected from selection alone under the null.

## Best in-sample configuration (reported, not emphasised)
- Best training score: `fed00179f271` (range 5m, entry TF 15, market, confirm close+0t+0.1W, stop or_pct:0.5, 0.5R, cutoff 11:00, long)
- Highest train+validation net P&L: `7c265d2ad5c4` (range 5m, entry TF intrabar, stop, confirm close+0t+0.05W, stop or_opposite, 3.0R, cutoff 10:00, both)

In-sample winners on every phase (degradation check):

| configuration                     | phase   |   trades |   exp_r |   net_pnl |
|:----------------------------------|:--------|---------:|--------:|----------:|
| best_in_sample_train_score        | train   |       12 |  +0.509 |  +389.637 |
| best_in_sample_train_score        | val     |        4 |  +0.123 |   +22.995 |
| best_in_sample_train_score        | test    |        3 |  +0.511 |   +60.989 |
| best_in_sample_train_plus_val_pnl | train   |       22 |  +0.605 | +2632.002 |
| best_in_sample_train_plus_val_pnl | val     |        6 |  +0.642 |  +311.000 |
| best_in_sample_train_plus_val_pnl | test    |        8 |  +0.489 |   -55.000 |

## Validation-selected configuration
`fed00179f271` (range 5m, entry TF 15, market, confirm close+0t+0.1W, stop or_pct:0.5, 0.5R, cutoff 11:00, long)
- Why: highest selection score = ½·validation score + ½·(½·training score + ½·median training score of its immediate parameter neighbours) among the 25 finalists ranked by that neighbourhood-robust training score. Training neighbourhood class: **mixed**.
- Training: 12 trades, +0.509 R; Validation: 4 trades, +0.123 R.

Top finalists:
|   range_minutes |   entry_tf | entry_method   | confirmation   | stop       |   target_r | cutoff   | direction   |   train_trades |   train_exp_r |   neighbour_median_train_score |   val_trades |   val_exp_r |   selection_score |
|----------------:|-----------:|:---------------|:---------------|:-----------|-----------:|:---------|:------------|---------------:|--------------:|-------------------------------:|-------------:|------------:|------------------:|
|               5 |         15 | market         | 10pct          | or_pct:0.5 |      0.500 | 11:00    | long        |         12.000 |         0.509 |                          0.846 |        4.000 |       0.123 |             0.663 |
|              10 |         10 | market         | close          | or_pct:0.5 |      0.700 | 11:30    | long        |         12.000 |         0.564 |                          0.908 |        5.000 |       0.016 |             0.355 |
|              10 |         10 | market         | 1tick          | or_pct:0.5 |      0.700 | 11:30    | long        |         12.000 |         0.564 |                          0.908 |        5.000 |       0.016 |             0.355 |
|              10 |         10 | market         | 2tick          | or_pct:0.5 |      0.700 | 11:30    | long        |         12.000 |         0.564 |                          0.893 |        5.000 |       0.016 |             0.351 |
|              10 |         10 | market         | close          | or_pct:0.5 |      0.500 | 12:00    | long        |         13.000 |         0.508 |                          0.882 |        5.000 |      -0.101 |             0.215 |
|              10 |         10 | market         | 1tick          | or_pct:0.5 |      0.500 | 12:00    | long        |         13.000 |         0.508 |                          0.882 |        5.000 |      -0.101 |             0.215 |
|              10 |         10 | market         | 5pct           | or_pct:0.5 |      0.500 | 12:00    | long        |         13.000 |         0.508 |                          0.879 |        5.000 |      -0.101 |             0.215 |
|              10 |         10 | market         | close          | or_pct:0.5 |      0.500 | 11:30    | long        |         12.000 |         0.508 |                          0.859 |        5.000 |      -0.101 |             0.209 |

## Unseen final-test result (the result that matters)
| metric                     | TRAIN   | VALIDATION   | FINAL TEST (unseen)   |
|:---------------------------|:--------|:-------------|:----------------------|
| Trades                     | 12      | 4            | 3                     |
| Win rate                   | 100.0%  | 75.0%        | 100.0%                |
| Expectancy (R)             | +0.509  | +0.123       | +0.511                |
| t-stat of mean R           | +272.48 | +0.31        | +89.71                |
| Profit factor              | inf     | 1.55         | inf                   |
| Net P&L (100 sh)           | $389.64 | $23.00       | $60.99                |
| Gross P&L (100 sh)         | $413.64 | $33.00       | $66.99                |
| Friction paid              | $24.00  | $10.00       | $6.00                 |
| Sharpe (daily, annualised) | +13.61  | +2.15        | +9.62                 |
| Max drawdown (R)           | +0.00   | -1.05        | +0.00                 |
| Max drawdown ($, 100 sh)   | $0.00   | $-42.00      | $0.00                 |
| Average winner             | $32.47  | $21.67       | $20.33                |
| Average loser              | $0.00   | $-42.00      | $0.00                 |
| Largest winner             | $56.00  | $28.00       | $31.00                |
| Largest loser              | $16.00  | $-42.00      | $14.00                |
| Long trades                | 12      | 4            | 3                     |
| Long expectancy (R)        | +0.509  | +0.123       | +0.511                |
| Long net P&L               | $389.64 | $23.00       | $60.99                |
| Short trades               | 0       | 0            | 0                     |
| Short expectancy (R)       | +0.000  | +0.000       | +0.000                |
| Short net P&L              | $0.00   | $0.00        | $0.00                 |

## Controls and baselines (same dates)
| phase   |   strategy mean R |   random-direction median |   random 5% |   random 95% |   p(random >= strategy) |   buy&hold % |   buy&hold $ (100 sh) |   intraday-long $ (100 sh) |   sim check |dR| |
|:--------|------------------:|--------------------------:|------------:|-------------:|------------------------:|-------------:|----------------------:|---------------------------:|-----------------:|
| train   |             0.509 |                    -0.135 |      -0.518 |        0.250 |                   0.001 |        2.937 |              2173.004 |                    -83.997 |            0.000 |
| val     |             0.123 |                    -0.265 |      -0.665 |        0.513 |                   0.305 |       -0.220 |              -167.999 |                    300.989 |            0.000 |
| test    |             0.511 |                    -0.015 |      -0.545 |        0.517 |                   0.249 |        1.471 |              1118.298 |                     24.695 |            0.000 |

## Robustness: neighbourhood of the selected configuration (expectancy R per phase)
| config_key   |   range_minutes |   entry_tf | entry_method   | confirmation   | stop        |   target_r | cutoff   | direction   |   train_n_trades |   train_exp_r |   val_n_trades |   val_exp_r |   test_n_trades |   test_exp_r | role      |
|:-------------|----------------:|-----------:|:---------------|:---------------|:------------|-----------:|:---------|:------------|-----------------:|--------------:|---------------:|------------:|----------------:|-------------:|:----------|
| fed00179f271 |               5 |         15 | market         | 10pct          | or_pct:0.5  |      0.500 | 11:00    | long        |           12.000 |         0.509 |          4.000 |       0.123 |           3.000 |        0.511 | SELECTED  |
| 290fc92bf66b |               5 |         10 | market         | 10pct          | or_pct:0.5  |      0.500 | 11:00    | long        |           13.000 |        -0.206 |          4.000 |      -0.659 |           4.000 |       -0.274 | neighbour |
| 9d661b8437fd |               5 |         15 | market         | 5pct           | or_pct:0.5  |      0.500 | 11:00    | long        |           12.000 |         0.248 |          4.000 |       0.123 |           3.000 |        0.513 | neighbour |
| d5d75dba91da |               5 |         15 | market         | 10pct          | or_mid      |      0.500 | 11:00    | long        |           12.000 |         0.377 |          4.000 |      -0.261 |           3.000 |        0.004 | neighbour |
| fe93ae653b81 |               5 |         15 | market         | 10pct          | or_pct:0.5  |      0.500 | 10:30    | long        |           10.000 |         0.509 |          4.000 |       0.123 |           3.000 |        0.511 | neighbour |
| 7e05c600b193 |               5 |         15 | market         | 10pct          | or_pct:0.5  |      0.500 | 11:30    | long        |           13.000 |         0.390 |          4.000 |       0.123 |           3.000 |        0.511 | neighbour |
| 673845be2e58 |               5 |         15 | market         | 10pct          | or_pct:0.5  |      0.600 | 11:00    | long        |           12.000 |         0.473 |          4.000 |       0.198 |           3.000 |        0.618 | neighbour |
| 830386d2ca83 |               5 |         15 | market         | 10pct          | or_pct:0.67 |      0.500 | 11:00    | long        |           12.000 |         0.378 |          4.000 |       0.123 |           3.000 |        0.519 | neighbour |
| 574c14a28508 |              10 |         15 | market         | 10pct          | or_pct:0.5  |      0.500 | 11:00    | long        |           12.000 |         0.124 |          3.000 |      -0.531 |           4.000 |       -0.644 | neighbour |

Selected configuration neighbourhood class by phase: {'train': 'mixed', 'val': 'mixed', 'test': 'mixed'}; neighbour median expectancy: train +0.378, val +0.123, test +0.511
Rank correlation of all configurations' expectancy: train→test +0.282, validation→test +0.163.
Heatmaps: `heatmaps/*.png` (median expectancy across hidden parameters, per phase).

## Execution sensitivity (selected configuration)
| split   |   friction_usd_per_share |   trades |   exp_r |   t_stat_r |   profit_factor |   net_pnl |   friction_cost |
|:--------|-------------------------:|---------:|--------:|-----------:|----------------:|----------:|----------------:|
| train   |                    0.000 |       12 |   0.509 |    272.476 |         inf     |   389.637 |           0.000 |
| val     |                    0.000 |        4 |   0.135 |      0.357 |           1.625 |    24.995 |           0.000 |
| test    |                    0.000 |        3 |   0.511 |     89.714 |         inf     |    60.989 |           0.000 |
| train   |                    0.010 |       12 |   0.509 |    272.476 |         inf     |   389.637 |          12.000 |
| val     |                    0.010 |        4 |   0.129 |      0.335 |           1.585 |    23.995 |           5.000 |
| test    |                    0.010 |        3 |   0.511 |     89.714 |         inf     |    60.989 |           3.000 |
| train   |                    0.020 |       12 |   0.509 |    272.476 |         inf     |   389.637 |          24.000 |
| val     |                    0.020 |        4 |   0.123 |      0.314 |           1.547 |    22.995 |          10.000 |
| test    |                    0.020 |        3 |   0.511 |     89.714 |         inf     |    60.989 |           6.000 |
| train   |                    0.030 |       12 |   0.378 |      2.904 |           6.235 |   303.637 |          39.000 |
| val     |                    0.030 |        4 |   0.116 |      0.293 |           1.511 |    21.995 |          15.000 |
| test    |                    0.030 |        3 |   0.511 |     89.714 |         inf     |    60.989 |           9.000 |
| train   |                    0.050 |       12 |   0.375 |      2.815 |           6.027 |   301.637 |          65.000 |
| val     |                    0.050 |        4 |   0.104 |      0.253 |           1.444 |    19.995 |          25.000 |
| test    |                    0.050 |        3 |   0.511 |     89.714 |         inf     |    60.989 |          15.000 |

## Concentration (sum of R after removing the best trades)
|                       |   train |     val |    test |
|:----------------------|--------:|--------:|--------:|
| sum_r                 |   +6.10 |   +0.49 |   +1.53 |
| sum_r_without_best_1  |   +5.58 |   -0.04 |   +1.02 |
| sum_r_without_best_3  |   +4.55 |   -1.05 | +nan    |
| sum_r_without_best_5  |   +3.53 | +nan    | +nan    |
| sum_r_without_best_10 |   +1.00 | +nan    | +nan    |
(n/a = fewer trades than removed)

## Grid-wide descriptive evidence (median expectancy R across all configurations sharing the value)
### Opening-range duration
|    |   train |    val |   test |
|---:|--------:|-------:|-------:|
|  5 |  -0.031 | -0.472 | -0.011 |
| 10 |  -0.106 | -0.344 | -0.135 |
| 15 |  -0.037 | -0.294 | -0.111 |
| 20 |  -0.054 | -0.332 | -0.005 |
| 30 |  -0.052 | -0.325 | -0.149 |
### Targets below 2R vs 2R and above
|      |   train |    val |   test |
|:-----|--------:|-------:|-------:|
| R<2  |  -0.044 | -0.372 | -0.078 |
| R>=2 |  -0.138 | -0.276 | -0.018 |
### Direction
|       |   train |    val |   test |
|:------|--------:|-------:|-------:|
| both  |  -0.033 | -0.324 | -0.007 |
| long  |  -0.087 | -0.312 | -0.172 |
| short |  -0.049 | -0.415 | -0.046 |
### Entry method
|        |   train |    val |   test |
|:-------|--------:|-------:|-------:|
| limit  |  -0.226 | -0.527 | -0.366 |
| market |   0.058 | -0.21  |  0.021 |
| stop   |  -0.13  | -0.511 | -0.034 |
### Entry cutoff
|       |   train |    val |   test |
|:------|--------:|-------:|-------:|
| 10:00 |   0     | -0.047 |  0     |
| 10:30 |  -0.104 | -0.327 | -0.04  |
| 11:00 |  -0.059 | -0.412 | -0.071 |
| 11:30 |  -0.036 | -0.384 | -0.131 |
| 12:00 |  -0.024 | -0.384 | -0.136 |
| 13:00 |  -0.087 | -0.386 | -0.119 |
### Share of configurations with positive expectancy: train 39.3%, val 14.8%, test 37.2%

## Resolution cross-check (frozen candidate on 1-minute bars, overlapping final-test dates)
{"period": ["2026-09-15", "2026-09-25"], "trades_1m": 3, "trades_5m": 3, "exp_r_1m": 0.511346424736965, "exp_r_5m": 0.511346424736965, "same_direction_days": 3}

## Trade verification
Complete trade log: `trade_log_selected.csv`. Charts (19): `trade_charts/`.
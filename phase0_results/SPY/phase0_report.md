# PHASE0_FREE_PROXY — SPY
**FREE PROXY EXPERIMENT — NOT FUTURES VALIDATION.** SPY is an ETF proxy. Nothing here validates any futures strategy.

## Conclusion: **MIXED**
- final-test expectancy +0.398 R over 3 trades; failed checks: ['beats random-direction control (p<0.10)', 'neighbours form a plateau in the final test']
- Rule (stated before looking at the test): NO PRELIMINARY EVIDENCE if final-test expectancy <= 0; PROMISING only if validation > 0, survives $0.03/share, beats random-direction control at p<0.10, plateau in final test and t>=2; else MIXED

## Data used
- Source: Yahoo Finance via yfinance 1.7.0 (free, no key, no account)
- Interval: **5m** (chosen by measured availability: 1m: 20 sessions from 2026-08-28, 2m: 31 sessions from 2026-08-13, 5m: 42 sessions from 2026-07-29, 15m: 42 sessions from 2026-07-29)
- Returned: 2026-07-29 09:30:00-04:00 to 2026-09-25 15:55:00-04:00; 3,276 bars; source timezone America/New_York; unadjusted (auto_adjust=False); regular trading hours only (prepost=False)
- Downloaded 2026-09-26T07:21:31.936089+00:00; processed file SHA-256 `47c4788bf43e9916…`

## Data quality
- Phase-0 DQ verdict: **REVIEW**
- Sessions 42 (2026-07-29 … 2026-09-25); UTC offsets seen ['-1 days +20:00:00'] (DST transition in sample: False)
- Sessions whose first bar is not 09:30 ET: 0; incomplete sessions: 0; missing XNYS sessions: none; early closes: none
- Modal peak-volume minute: 570 (570 = 09:30 ET)
- Engine DQ issues: ['[warning] off_tick_prices n=1730']
- 1-minute cross-check: {"bars_compared": 1560, "max_abs_diff": {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0}, "share_ohlc_identical_within_half_cent": 1.0, "volume_ratio_median": 1.0}
- note: warning [off_tick_prices]: 52.8% of bars have prices off the 0.01 tick grid (ratio back-adjusted continuous contract?). Tick-based order levels become approximate. (n=1730)

## Sample design (chronological, no shuffling)
- TRAIN: 2026-07-29 … 2026-09-01 (25 sessions)
- VALIDATION: 2026-09-02 … 2026-09-14 (8 sessions)
- FINAL TEST: 2026-09-15 … 2026-09-25 (9 sessions); not in memory during selection. The selection was frozen to `selection_frozen.json` (SHA-256 `a78c355bcbdcff15…`) before the test was evaluated.
- Configurations evaluated: 236,640 (raw 344,250; invalid 11,475; canonical duplicates 73,950; limit-entry 50 %-width stops identical to midpoint stops removed 22,185)
- Friction for selection and headline: $0.02/share charged on EVERY fill (entry and exit, market, stop and limit alike), i.e. a fixed round-trip cost per share; limit fills must also trade through by $0.01. $0.00 is a reference only.
- Multiple testing (train Sharpe across configs): best 14.20 vs 13.37 expected from selection alone under the null.

## Best in-sample configuration (reported, not emphasised)
- Best training score: `fed00179f271` (range 5m, entry TF 15, market, confirm close+0t+0.1W, stop or_pct:0.5, 0.5R, cutoff 11:00, long)
- Highest train+validation net P&L: `7c265d2ad5c4` (range 5m, entry TF intrabar, stop, confirm close+0t+0.05W, stop or_opposite, 3.0R, cutoff 10:00, both)

In-sample winners on every phase (degradation check):

| configuration                     | phase   |   trades |   exp_r |   net_pnl |
|:----------------------------------|:--------|---------:|--------:|----------:|
| best_in_sample_train_score        | train   |       12 |  +0.435 |  +336.630 |
| best_in_sample_train_score        | val     |        4 |  +0.030 |    +8.000 |
| best_in_sample_train_score        | test    |        3 |  +0.398 |   +48.990 |
| best_in_sample_train_plus_val_pnl | train   |       22 |  +0.593 | +2570.000 |
| best_in_sample_train_plus_val_pnl | val     |        6 |  +0.642 |  +299.000 |
| best_in_sample_train_plus_val_pnl | test    |        8 |  +0.457 |   -87.000 |

## Validation-selected configuration
`fed00179f271` (range 5m, entry TF 15, market, confirm close+0t+0.1W, stop or_pct:0.5, 0.5R, cutoff 11:00, long)
- Why: highest selection score = ½·validation score + ½·(½·training score + ½·median training score of its immediate parameter neighbours) among the 25 finalists ranked by that neighbourhood-robust training score. Training neighbourhood class: **mixed**.
- Training: 12 trades, +0.435 R; Validation: 4 trades, +0.030 R.

Top finalists:
|   range_minutes |   entry_tf | entry_method   | confirmation   | stop       |   target_r | cutoff   | direction   |   train_trades |   train_exp_r |   neighbour_median_train_score |   val_trades |   val_exp_r |   selection_score |
|----------------:|-----------:|:---------------|:---------------|:-----------|-----------:|:---------|:------------|---------------:|--------------:|-------------------------------:|-------------:|------------:|------------------:|
|               5 |         15 | market         | 10pct          | or_pct:0.5 |      0.500 | 11:00    | long        |         12.000 |         0.435 |                          0.834 |        4.000 |       0.030 |             0.532 |
|              10 |         10 | market         | 2tick          | or_pct:0.5 |      0.700 | 11:30    | long        |         12.000 |         0.507 |                          0.901 |        5.000 |      -0.052 |             0.277 |
|              10 |         10 | market         | close          | or_pct:0.5 |      0.700 | 11:30    | long        |         12.000 |         0.507 |                          0.901 |        5.000 |      -0.052 |             0.277 |
|              10 |         10 | market         | 1tick          | or_pct:0.5 |      0.700 | 11:30    | long        |         12.000 |         0.507 |                          0.901 |        5.000 |      -0.052 |             0.277 |
|              10 |         10 | market         | 1tick          | or_pct:0.5 |      0.700 | 12:00    | long        |         13.000 |         0.518 |                          0.858 |        5.000 |      -0.052 |             0.273 |
|              10 |         10 | market         | close          | or_pct:0.5 |      0.700 | 12:00    | long        |         13.000 |         0.518 |                          0.858 |        5.000 |      -0.052 |             0.273 |
|              10 |         10 | market         | 1tick          | or_pct:0.5 |      0.900 | 12:00    | long        |         13.000 |         0.702 |                          0.820 |        5.000 |      -0.315 |             0.161 |
|              10 |         10 | market         | close          | or_pct:0.5 |      0.900 | 12:00    | long        |         13.000 |         0.702 |                          0.820 |        5.000 |      -0.315 |             0.161 |

## Unseen final-test result (the result that matters)
| metric                     | TRAIN                                   | VALIDATION                              | FINAL TEST (unseen)                     |
|:---------------------------|:----------------------------------------|:----------------------------------------|:----------------------------------------|
| Trades                     | 12                                      | 4                                       | 3                                       |
| Win rate                   | 100.0%                                  | 75.0%                                   | 100.0%                                  |
| Expectancy (R)             | +0.435                                  | +0.030                                  | +0.398                                  |
| t-stat of mean R           | not meaningful (n<10 or ~zero variance) | not meaningful (n<10 or ~zero variance) | not meaningful (n<10 or ~zero variance) |
| Profit factor              | inf (no losing trades)                  | 1.18                                    | inf (no losing trades)                  |
| Net P&L (100 sh)           | $336.63                                 | $8.00                                   | $48.99                                  |
| Gross P&L (100 sh)         | $384.63                                 | $24.00                                  | $60.99                                  |
| Friction paid              | $48.00                                  | $16.00                                  | $12.00                                  |
| Sharpe (daily, annualised) | +13.20                                  | +0.78                                   | +9.20                                   |
| Max drawdown (R)           | +0.00                                   | -1.10                                   | +0.00                                   |
| Max drawdown ($, 100 sh)   | $0.00                                   | $-44.00                                 | $0.00                                   |
| Average winner             | $28.05                                  | $17.33                                  | $16.33                                  |
| Average loser              | $0.00                                   | $-44.00                                 | $0.00                                   |
| Largest winner             | $52.00                                  | $24.00                                  | $27.00                                  |
| Largest loser              | $12.00                                  | $-44.00                                 | $10.00                                  |
| Long trades                | 12                                      | 4                                       | 3                                       |
| Long expectancy (R)        | +0.435                                  | +0.030                                  | +0.398                                  |
| Long net P&L               | $336.63                                 | $8.00                                   | $48.99                                  |
| Short trades               | 0                                       | 0                                       | 0                                       |
| Short expectancy (R)       | +0.000                                  | +0.000                                  | +0.000                                  |
| Short net P&L              | $0.00                                   | $0.00                                   | $0.00                                   |

## Controls and baselines (same dates)
| phase   |   strategy mean R |   random-direction median |   random 5% |   random 95% |   p(random >= strategy) |   buy&hold % |   buy&hold $ (100 sh) |   intraday-long $ (100 sh) |   sim check |dR| |
|:--------|------------------:|--------------------------:|------------:|-------------:|------------------------:|-------------:|----------------------:|---------------------------:|-----------------:|
| train   |             0.435 |                    -0.192 |      -0.567 |        0.183 |                   0.001 |        2.937 |              2173.000 |                    -84.000 |            0.000 |
| val     |             0.030 |                     0.027 |      -0.348 |        0.405 |                   0.493 |       -0.220 |              -168.000 |                    301.000 |            0.000 |
| test    |             0.398 |                    -0.109 |      -0.614 |        0.398 |                   0.249 |        1.471 |              1118.300 |                     24.700 |            0.000 |

## Robustness: neighbourhood of the selected configuration (expectancy R per phase)
| config_key   |   range_minutes |   entry_tf | entry_method   | confirmation   | stop        |   target_r | cutoff   | direction   |   train_n_trades |   train_exp_r |   val_n_trades |   val_exp_r |   test_n_trades |   test_exp_r | role      |
|:-------------|----------------:|-----------:|:---------------|:---------------|:------------|-----------:|:---------|:------------|-----------------:|--------------:|---------------:|------------:|----------------:|-------------:|:----------|
| fed00179f271 |               5 |         15 | market         | 10pct          | or_pct:0.5  |      0.500 | 11:00    | long        |           12.000 |         0.435 |          4.000 |       0.030 |           3.000 |        0.398 | SELECTED  |
| 290fc92bf66b |               5 |         10 | market         | 10pct          | or_pct:0.5  |      0.500 | 11:00    | long        |           13.000 |        -0.260 |          4.000 |      -0.725 |           4.000 |        0.036 | neighbour |
| 9d661b8437fd |               5 |         15 | market         | 5pct           | or_pct:0.5  |      0.500 | 11:00    | long        |           12.000 |         0.184 |          4.000 |       0.030 |           3.000 |        0.401 | neighbour |
| d5d75dba91da |               5 |         15 | market         | 10pct          | or_mid      |      0.500 | 11:00    | long        |           12.000 |         0.341 |          4.000 |      -0.288 |           3.000 |       -0.047 | neighbour |
| fe93ae653b81 |               5 |         15 | market         | 10pct          | or_pct:0.5  |      0.500 | 10:30    | long        |           10.000 |         0.442 |          4.000 |       0.030 |           3.000 |        0.398 | neighbour |
| 7e05c600b193 |               5 |         15 | market         | 10pct          | or_pct:0.5  |      0.500 | 11:30    | long        |           13.000 |         0.320 |          4.000 |       0.030 |           3.000 |        0.398 | neighbour |
| 673845be2e58 |               5 |         15 | market         | 10pct          | or_pct:0.5  |      0.600 | 11:00    | long        |           12.000 |         0.402 |          4.000 |       0.110 |           3.000 |        0.504 | neighbour |
| 830386d2ca83 |               5 |         15 | market         | 10pct          | or_pct:0.67 |      0.500 | 11:00    | long        |           12.000 |         0.326 |          4.000 |       0.057 |           3.000 |        0.425 | neighbour |
| 574c14a28508 |              10 |         15 | market         | 10pct          | or_pct:0.5  |      0.500 | 11:00    | long        |           12.000 |         0.072 |          3.000 |      -0.595 |           4.000 |       -0.695 | neighbour |

Selected configuration neighbourhood class by phase: {'train': 'mixed', 'val': 'mixed', 'test': 'mixed'}; neighbour median expectancy: train +0.323, val +0.030, test +0.398
Rank correlation of all configurations' expectancy: train→test +0.313, validation→test +0.191.
Heatmaps: `heatmaps/*.png` (median expectancy across hidden parameters, per phase).

## Execution sensitivity (selected configuration)
| split   |   friction_usd_per_share |   trades |   exp_r |   t_stat_r |   profit_factor |   net_pnl |   friction_cost |
|:--------|-------------------------:|---------:|--------:|-----------:|----------------:|----------:|----------------:|
| train   |                    0.000 |       12 |   0.506 |    291.852 |         inf     |   384.630 |           0.000 |
| val     |                    0.000 |        4 |   0.130 |      0.346 |           1.600 |    24.000 |           0.000 |
| test    |                    0.000 |        3 |   0.511 |     88.884 |         inf     |    60.990 |           0.000 |
| train   |                    0.010 |       12 |   0.470 |    148.292 |         inf     |   360.630 |          24.000 |
| val     |                    0.010 |        4 |   0.080 |      0.213 |           1.381 |    16.000 |           8.000 |
| test    |                    0.010 |        3 |   0.454 |     65.626 |         inf     |    54.990 |           6.000 |
| train   |                    0.020 |       12 |   0.435 |     64.736 |         inf     |   336.630 |          48.000 |
| val     |                    0.020 |        4 |   0.030 |      0.081 |           1.182 |     8.000 |          16.000 |
| test    |                    0.020 |        3 |   0.398 |     20.318 |         inf     |    48.990 |          12.000 |
| train   |                    0.030 |       12 |   0.400 |     38.406 |         inf     |   312.630 |          72.000 |
| val     |                    0.030 |        4 |  -0.020 |     -0.052 |           1.000 |     0.000 |          24.000 |
| test    |                    0.030 |        3 |   0.341 |     10.573 |         inf     |    42.990 |          18.000 |
| train   |                    0.050 |       12 |   0.329 |     18.427 |         inf     |   264.630 |         120.000 |
| val     |                    0.050 |        4 |  -0.120 |     -0.317 |           0.680 |   -16.000 |          40.000 |
| test    |                    0.050 |        3 |   0.227 |      3.942 |         inf     |    30.990 |          30.000 |

## Concentration (sum of R after removing the best trades)
|                       |   train |     val |    test |
|:----------------------|--------:|--------:|--------:|
| sum_r                 |   +5.22 |   +0.12 |   +1.19 |
| sum_r_without_best_1  |   +4.75 |   -0.31 |   +0.76 |
| sum_r_without_best_3  |   +3.83 |   -1.10 | +nan    |
| sum_r_without_best_5  |   +2.94 | +nan    | +nan    |
| sum_r_without_best_10 |   +0.80 | +nan    | +nan    |
(n/a = fewer trades than removed)

## Grid-wide descriptive evidence (median expectancy R across all configurations sharing the value that traded in that phase)
Configurations with at least one trade: {'train': 231761, 'val': 225029, 'test': 227409}. Descriptive only; never used for selection.
### Opening-range duration
|    |   train |    val |   test |
|---:|--------:|-------:|-------:|
|  5 |  -0.043 | -0.497 | -0.044 |
| 10 |  -0.12  | -0.374 | -0.179 |
| 15 |  -0.06  | -0.339 | -0.165 |
| 20 |  -0.087 | -0.39  | -0.041 |
| 30 |  -0.066 | -0.343 | -0.168 |
### Targets below 2R vs 2R and above
|      |   train |    val |   test |
|:-----|--------:|-------:|-------:|
| R<2  |  -0.062 | -0.403 | -0.121 |
| R>=2 |  -0.154 | -0.329 | -0.064 |
### Direction
|       |   train |    val |   test |
|:------|--------:|-------:|-------:|
| both  |  -0.05  | -0.342 | -0.04  |
| long  |  -0.105 | -0.363 | -0.218 |
| short |  -0.068 | -0.445 | -0.087 |
### Entry method
|        |   train |    val |   test |
|:-------|--------:|-------:|-------:|
| limit  |  -0.266 | -0.612 | -0.453 |
| market |   0.055 | -0.221 |  0.016 |
| stop   |  -0.139 | -0.516 | -0.06  |
### Entry cutoff
|       |   train |    val |   test |
|:------|--------:|-------:|-------:|
| 10:00 |  -0.046 | -0.356 |  0.028 |
| 10:30 |  -0.12  | -0.346 | -0.074 |
| 11:00 |  -0.072 | -0.431 | -0.092 |
| 11:30 |  -0.047 | -0.395 | -0.151 |
| 12:00 |  -0.037 | -0.395 | -0.158 |
| 13:00 |  -0.099 | -0.397 | -0.141 |
### Share of configurations with positive expectancy: train 38.8%, val 15.8%, test 38.0%

## Resolution cross-check (frozen candidate on 1-minute bars, overlapping final-test dates)
{"period": ["2026-09-15", "2026-09-25"], "trades_1m": 3, "trades_5m": 3, "exp_r_1m": 0.3975012364173076, "exp_r_5m": 0.3975012364173076, "same_direction_days": 3}

## Run history (disclosure)
- `run1_fill_relative_slippage_model`: selected `fed00179f271` (range 5m, entry TF 15, market, confirm close+0t+0.1W, stop or_pct:0.5, 0.5R, cutoff 11:00, long); final test 3 trades, +0.511 R; verdict MIXED. Superseded because the engine's path-slippage model moves stop and target with the fill, so slippage never reduced the P&L of trades that still reached their target (net P&L was identical at $0.00, $0.01 and $0.02). This did not match the specified per-share adverse-execution cost. The cost model was changed to a fixed per-share charge on every fill. The change was made after run 1's report (including final-test rows) had been produced; this is disclosed here, and both runs are in `phase0_results/phase0_log.jsonl`.

## Trade verification
Complete trade log: `trade_log_selected.csv`. Charts (28): `trade_charts/`.
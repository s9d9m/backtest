# PHASE0_FREE_PROXY — Final report (SPY, QQQ)

**FREE PROXY EXPERIMENT — NOT FUTURES VALIDATION.** SPY and QQQ are ETF proxies for the equity-index
leg of the research. Nothing here says anything validated about ES, NQ, 6E or GC. The futures registry
and lockbox were not touched.

Run: 2026-09-26. Per-symbol detail: `SPY/phase0_report.md`, `QQQ/phase0_report.md`. Trade logs:
`*/trade_log_selected.csv`. Charts: `*/trade_charts/`. Heatmaps: `*/heatmaps/`. Append-only experiment log:
`phase0_log.jsonl`.

## Bottom line

| | SPY | QQQ |
|---|---|---|
| Category (rule fixed before the test was run) | **MIXED** | **MIXED** |
| Final-test trades | 3 | 6 |
| Final-test expectancy (net, $0.02/share) | +0.40 R | +0.30 R |
| Beats random-direction control? | no (p = 0.25) | no (p = 0.16) |
| Median expectancy of all 236,640 configurations (train / val / test) | −0.07 / −0.39 / −0.11 R | −0.11 / −0.39 / +0.18 R |

**Overall reading: MIXED, leaning to NO PRELIMINARY EVIDENCE.**
- Both validation-selected candidates were positive in their unseen final weeks. But they did so on
  3 and 6 trades, which is indistinguishable from choosing the trade direction at random.
- Neither sits on a parameter plateau.
- Across the whole grid, the 09:30 ORB lost money in most configurations on SPY in every phase. On QQQ
  it lost in training and validation and was profitable only in a final test window where QQQ itself
  rose 5 % in 9 sessions.
- This sample cannot show an edge. It also cannot rule one out: it is simply too small (see Limitations).

What Phase 0 *did* establish:
1. The research pipeline runs end to end on real market prices.
2. The engine enters and exits correctly on real bars. This was checked by hand and on 48 charts
   covering market, limit and stop entries.
3. The final-test isolation holds. It is tested, including a deliberately injected leak.

## Data used (measured, not assumed)

Yahoo Finance via `yfinance` 1.7.0: free, no key, no account. Regular trading hours, unadjusted prices,
timestamps in America/New_York.

| Interval | Earliest bar Yahoo returned | Sessions to 2026-09-25 | Limit reported by Yahoo |
|---|---|---|---|
| 1m | 2026-08-28 | 20 | "must be within the last 30 days" |
| 2m | 2026-08-13 | 31 | accepts 60 days, but no data before 08-13 |
| **5m (used)** | **2026-07-29** | **42** | "must be within the last 60 days" |
| 15m | 2026-07-29 | 42 | "must be within the last 60 days" |

Same for SPY and QQQ. 5-minute bars were chosen: 1-minute gives only 20 sessions, and 5-minute is the
finest interval with the maximum history. On 5-minute bars:
- every requested opening range (5/10/15/20/30 min) is constructible exactly;
- entry timeframes are 5/10/15 min (1- and 3-minute entry bars are impossible);
- 3,276 bars per symbol, 2026-07-29 09:30 → 2026-09-25 15:55 ET.

The 1-minute data (20 sessions) was also downloaded, but only for cross-checks. Raw downloads are cached
per request chunk in `data/raw/yahoo/`; processed files and provenance (hashes, requested/returned
periods) are in `data/phase0/` (git-ignored).

## Data quality (Phase-0 verdict: REVIEW for both; no blocking issue)

- 42 of 42 XNYS sessions present. Every session starts at 09:30 ET and has all 78 bars. No
  duplicates, no invalid OHLC, no early closes in the sample. The highest-volume minute is 09:30 ET.
- UTC offset is −4 h throughout, so **no DST transition in the sample**; DST handling could not be
  exercised on this data. It is covered by the unit tests.
- The 5-minute bars equal the aggregated 1-minute bars **exactly** on all 1,560 overlapping bars
  (open/high/low/close identical; volume ratio 1.0).
- REVIEW reason: sub-penny prices (SPY ~17 % of OHLC values, QQQ more). These are real sub-penny
  prints in Yahoo's data. Float noise was removed (4-dp rounding), but the prints were not forced onto
  the cent grid. The engine handles fractional ticks; order levels are therefore approximate to under
  half a cent.

## Sample design (chronological 60 / 20 / 20, no shuffling)

| | Dates | Sessions |
|---|---|---|
| TRAIN | 2026-07-29 … 2026-09-01 | 25 |
| VALIDATION | 2026-09-02 … 2026-09-14 | 8 |
| FINAL TEST | 2026-09-15 … 2026-09-25 | 9 |

- **Selection ran on a copy of the data that physically ended on 2026-09-14.** The test bars were not
  in memory.
- The selection was written to `selection_frozen.json` (SHA-256 in the log) before the test was
  evaluated.
- A rolling walk-forward was not possible with 42 sessions.

**Search.** 236,640 unique configurations per symbol:
- raw grid 344,250; 11,475 invalid (30-minute range with a 10:00 cutoff); 73,950 canonical
  duplicates;
- 22,185 limit-entry "50 %-width" stops removed, because for a limit fill at the boundary they are
  mathematically identical to the midpoint stop.

Dimensions: 09:30 start; ranges 5/10/15/20/30; entry TF 5/10/15; market / limit / stop entries;
5 confirmations; 5 stops; 17 targets 0.5–3R; 6 cutoffs; both/long/short; 1 trade/day.

**Selection rule.** Same as the futures walk-forward:
1. Rank by the neighbourhood-robust training score (½ own score + ½ median of immediate parameter
   neighbours).
2. Take 25 finalists and score them on validation.
3. Pick the best ½ validation + ½ robust-training score.

**Friction.** $0.02/share on every fill (entry and exit, any order type) for selection and headline
results. The grid covers $0 (reference only), $0.01, $0.02, $0.03 and $0.05. Limit fills must also
trade through the price by $0.01.

## Best in-sample configuration (reported, not emphasised)

| | SPY | QQQ |
|---|---|---|
| Best training score | 5m OR, market, 15m bar close > OR high + 10 % width, stop 50 % width, **0.5R**, cutoff 11:00, **long only** | 10m OR, limit retest, stop at opposite boundary, **0.6R**, cutoff 13:00, **short only** |
| → train / val / test expectancy | +0.44 / +0.03 / +0.40 R | +0.34 / +0.24 / **−0.22 R** |
| Highest train+validation net P&L | 5m OR, stop entry, opposite-boundary stop, 3R, cutoff 10:00, both | 5m OR, market, midpoint stop, 1.5R, cutoff 10:00, both |
| → train / val / test | +0.59 / +0.64 / +0.46 R (test net **−$87**: the losers were larger-risk trades) | +0.35 / **−0.09** / +0.08 R |

The in-sample best is also a multiple-testing artefact. SPY's best training Sharpe (14.2) is only just
above the 13.4 expected from picking the best of 236,640 no-edge configurations. QQQ's best (9.7) is
below the 14.1 expected.

## Validation-selected configurations

- **SPY**: identical to the best-training-score configuration above: 5m OR, market entry,
  15-minute close confirmation (+10 % OR width), stop 50 % of OR width, **0.5R**, cutoff 11:00,
  **long only**.
  - Why it was chosen: highest combined score. Train 12 trades +0.44 R (12 of 12 winners); validation
    4 trades +0.03 R.
  - Its training neighbourhood is **mixed**, not a plateau.
- **QQQ**: 5m OR, **limit retest** at the broken boundary after a 5-minute close one tick outside,
  stop 75 % of OR width, **0.6R**, cutoff 13:00, both directions.
  - Train 17 trades +0.30 R; validation 7 trades +0.12 R. Neighbourhood **mixed**.

Both selections sit at the edge of the grid: the shortest range (5 min) and the smallest targets
(0.5–0.6R).

## Unseen final-test result ($0.02/share friction, 100 shares)

| | SPY | QQQ |
|---|---|---|
| Trades | 3 | 6 |
| Win rate | 100 % | 83 % |
| Expectancy | +0.40 R | +0.30 R (t ≈ 1.1) |
| Profit factor | ∞ (no losers) | 1.56 |
| Net P&L | +$49 | +$99 |
| Sharpe (daily, annualised) | 9.2 (3 trades; meaningless) | 2.3 |
| Max drawdown | 0 R / $0 | −1.0 R / −$178 |
| Average winner / loser | +$16 / none | +$55 / −$178 |
| Largest winner / loser | +$27 / (smallest winner +$10) | +$88 / −$178 |
| Long | 3 trades, +0.40 R | 4 trades, +0.56 R |
| Short | none (long-only rule) | 2 trades, −0.24 R |

Controls on the same dates:
- **Random direction.** Same entries, risk and targets, direction chosen by coin flip, 10,000 draws.
  90 % of draws fall in −0.61…+0.40 R (SPY) and −0.51…+0.30 R (QQQ). The strategies are at the top edge
  of that range, not beyond it (p = 0.25 and 0.16).
- **Buy-and-hold.** SPY +1.47 %, QQQ +5.03 % over the test. That's +$1,118 and +$3,568 per 100 shares,
  more than either ORB candidate in dollars. The comparison is not like-for-like (overnight exposure
  vs ~1-hour intraday risk), so it is context, not a verdict.

## Robustness (neighbours of the selected configuration)

- **SPY.** Neighbours that change the range to 10 minutes or the entry TF to 10 minutes lose in the
  final test (−0.64 R and −0.27 R). Neighbours that change only the cutoff, confirmation or R
  stay positive, largely because they produce the same few trades.
- **QQQ.** Changing the entry TF to 10 minutes gives −0.08 R; changing the range to 10 minutes gives
  +0.04 R (−0.03 R in training).
- Both are classified **mixed** (neither plateau nor clean spike) in train, validation and test.
- **Grid-wide, rankings do not persist.** Rank correlation of expectancy, train→test: +0.31 (SPY),
  +0.03 (QQQ). Validation→test: +0.19 (SPY), −0.21 (QQQ).

## Execution sensitivity (final test)

| Friction/share | $0.00 | $0.01 | $0.02 | $0.03 | $0.05 |
|---|---|---|---|---|---|
| SPY expectancy | +0.51 R | +0.45 R | +0.40 R | +0.34 R | +0.23 R |
| QQQ expectancy | +0.34 R | +0.32 R | +0.30 R | +0.28 R | +0.23 R |

Neither candidate is execution-sensitive up to $0.05/share. The risk per trade (~$0.65–1.80/share) is
large compared with cent-level costs.

**Caveat:** 5-minute bars hide the true fill path. At this bar size, the same-bar stop/target ambiguity
decided 1 of SPY's 19 trades (resolved conservatively as a loss).

## Concentration (sum of R)

| | Total | −best 1 | −best 3 | −best 5 | −best 10 |
|---|---|---|---|---|---|
| SPY train | +5.22 | +4.75 | +3.84 | +2.94 | +0.80 |
| SPY validation | +0.12 | −0.32 | −1.10 | n/a | n/a |
| SPY test | +1.19 | +0.76 | n/a | n/a | n/a |
| QQQ train | +5.05 | +4.46 | +3.29 | +2.12 | **−0.78** |
| QQQ validation | +0.80 | +0.22 | **−0.93** | −2.07 | n/a |
| QQQ test | +1.77 | +1.19 | +0.07 | **−1.02** | n/a |

Removing the few best trades erases QQQ's result in every phase and SPY's validation result.

## Answers to the ten questions (from the real Yahoo data)

1. **Does the 09:30 ORB show positive expectancy at all?**
   - The selected candidates did (+0.40 R and +0.30 R on 3 and 6 unseen trades).
   - The ORB family as a whole did **not**. Median expectancy across all configurations was negative
     for SPY in train, validation and test (−0.07 / −0.39 / −0.11 R). Only 39 % / 16 % / 38 % of SPY
     configurations were profitable.
   - For QQQ it was negative in train and validation (32 % and 14 % profitable). It was positive only
     in the test window (71 %), which coincided with a strong QQQ rally.
2. **Does it survive execution friction?** The selected candidates stay positive at $0.05/share. The
   question barely matters here, because the gross effect itself is not distinguishable from random
   direction.
3. **Is performance stable across neighbouring parameters?** No. Both selections are "mixed". Changing
   the range or entry timeframe one step turns results negative. Configuration rankings do not persist
   from training to the test.
4. **Which OR duration is strongest?** None consistently.
   - SPY: 5 and 20 minutes are "least bad" in train and test (−0.04 R).
   - QQQ: 20 minutes is best in training (−0.02 R), 5 minutes in validation, 10 minutes in the test
     (+0.27 R).
   - The ordering flips between phases.
5. **Do longer opening ranges beat 5 minutes (the video's claim)?** Not visible. SPY's 5-minute
   range is at least as good as the longer ones in every phase. QQQ is inconsistent. It is neither
   supported nor refuted at this sample size.
6. **Do sub-2R targets beat large targets?** Inconsistent.
   - SPY: R < 2 is better in training (−0.06 vs −0.15 R) but worse in validation and test.
   - QQQ: R < 2 is better in training and validation but worse in the test (+0.16 vs +0.37 R).
   - Both selected candidates use 0.5–0.6R, which is exactly the "high win rate, small target" profile
     that looks best in-sample.
7. **Long vs short?**
   - SPY's selection is long-only in a rising sample.
   - QQQ's selection earned its training profit mostly from shorts (+0.40 R), and its shorts then lost
     in both validation and test (−0.23 / −0.24 R).
   - Grid-wide direction medians change sign and order between phases. No stable directional effect.
8. **Does most performance come from a few trades?** Yes. Removing QQQ's best 3 test trades leaves
   +0.07 R, and removing its best 5 gives −1.02 R. Removing QQQ's best 10 training trades turns
   training negative.
9. **Does performance persist in the unseen period?** The two frozen candidates were positive, but
   within the random-direction range. QQQ's best-training-score configuration lost in the test
   (−0.22 R).
10. **Do SPY and QQQ agree?** Only superficially. Both picked a 5-minute opening range with a target
    below 1R, but through different mechanisms:
    - SPY: market entry, long only, 15-minute confirmation;
    - QQQ: limit retest, both directions, wider stop.

    The grid-wide pictures also differ (QQQ's test window was broadly profitable, SPY's was not).
    Conclusions depend heavily on the instrument and the specific weeks.

Everything observed from the OOS/test analysis is an **exploratory hypothesis** only (e.g. "5-minute
range with sub-1R targets"). It is recorded as such in `phase0_log.jsonl`, not in the futures registry,
and it is not validated by anything here.

## Limitations (why this cannot answer the real question)

- **Sample size.** 42 sessions in total; 9 test sessions; 3 and 6 test trades.
  - With a per-trade standard deviation of ~0.6–1.0 R, the standard error of mean R over 6 trades is
    ~0.3–0.4 R. Even a true +0.2 R edge would need roughly 100+ trades to show up at t ≈ 2.
  - Selecting among 236,640 configurations on 25 training sessions guarantees an impressive-looking
    in-sample winner (e.g. SPY's 12-for-12).
- **One regime.** July–September 2026, mostly rising markets. No DST change, no early close, no
  crisis.
- **Resolution.** 5-minute bars: no 1- or 3-minute entry timeframes, and more same-bar ambiguity
  (resolved conservatively).
  - A cross-check on 1-minute bars for the overlapping test dates reproduced every final-test trade
    exactly (same prices; entry and exit timestamps at 1-minute precision).
- **Proxy ≠ futures.** Different instrument, costs and hours (no overnight session). No roll issue,
  but also no information about 6E or GC, the markets the original claim is really about.
- **Disclosures.**
  1. The first SPY run used the engine's path-slippage model. That model moves stop and target with the
     fill, so slippage never reduced a trade that still reached its target (identical net P&L at
     $0 / $0.01 / $0.02). This did not match the specified per-share cost. It was replaced by a fixed
     per-share charge on every fill. The first run's report, including its final-test rows, had already
     been produced. The **same configuration was selected under both cost models**; run 1 is archived
     in `SPY/run1_fill_relative_slippage_model/`.
  2. Grid-wide medians initially counted configurations with zero trades as 0 R. This was corrected
     to exclude them. It affects descriptive tables only, never selection.

## Engine verification on real bars

- 48 charts in total:
  - SPY: 19 selected trades plus 9 verification trades from stop, limit and market configurations
    chosen on development data;
  - QQQ: 16 selected trades plus 4 verification trades.

  Each shows the 09:30 open, the opening-range window with its high, low and midpoint, the signal
  bar, the entry, the stop and target lines, and the exit.
- Hand-checked to the cent:
  - SPY 2026-07-30 (market entry);
  - SPY 2026-09-10 (a same-bar stop/target, correctly flagged ambiguous and booked as the stop);
  - QQQ 2026-09-17 (limit retest filled on a trade-through, stopped at 75 % width);
  - SPY 2026-08-21 (stop entry with a 5 %-width trigger computed from a sub-penny OR low).
- The random-direction simulator reproduces every engine trade's R exactly (max |ΔR| = 0).

## Recommendation

Phase 0 neither justifies nor rules out the futures study. The free sample is roughly 1/60 of the
planned 10-year history and has no statistical power. What it does show:
- the pipeline, engine and leakage controls work on real prices;
- a naive reading of an in-sample winner (100 % win rate, Sharpe 13) would have been badly
  misleading.

**The decision to buy futures data should rest on:**
1. The original claim to be tested, which concerns 6E and GC, not the equity proxies.
2. The low cost of finding out: a Databento quote is free, and the downloader enforces a hard budget
   cap.

It should **not** rest on these proxy results in either direction.

# Research notes

This file records every assumption that materially affects backtest validity, and why it was made.
Where a choice was uncertain, the more conservative option was taken. Codes (A-xx) are referenced from
the source code.

The platform exists to find out **whether** ORB works, not to prove that it does. A null result
("no variation keeps a meaningful edge after costs and walk-forward testing") is a valid, expected
outcome and must be reported as such.

---

## Time and sessions

**A-01. Internal time is UTC, marking the bar start.** Loaders convert everything to tz-aware UTC.
Files that label bars by their close use `timestamp_convention="end"`, and the loader shifts them back
by one bar. Mislabelled conventions shift every signal by one bar, so this is always explicit.

**A-02. Naive timestamps are never guessed.** A naive file must come with a declared source timezone.
Localising uses `ambiguous="infer"`. Timestamps that are invalid in the source zone (DST gaps, or an
ambiguous fall-back hour that cannot be inferred) are dropped, and the loader records this in its notes.

**A-03. Strategy clocks are America/New_York wall-clock.** "09:30" means 09:30 EST in winter and
09:30 EDT in summer. Tests check that identical ET paths trade identically across DST.

**A-04. Trading sessions, not calendar dates.** A bar belongs to the CME Globex session
`[D-1 18:00 ET, D 17:00 ET)`. Bars in the 17:00–18:00 ET maintenance window belong to no session.
`tod` is ET minutes after midnight of the session date, and is negative for the previous evening.
The ET wall-clock is ambiguous during the fall-back hour (01:00–02:00 ET). That hour lies outside the
intraday analysis window (≥ 08:00 ET), so it never affects signals.

**A-05. Calendar = NYSE (XNYS) by default, for every instrument.** The research anchor is the U.S.
cash-equity open. Sessions that are not XNYS trading days (e.g. MLK Day, when Globex trades an
abbreviated session) are excluded. XNYS early closes (13:00 ET) are kept, but the forced exit moves
to 5 minutes before the close. They can be excluded with `exclude_early_close`. Each instrument can
set its own calendar (e.g. `CMES`) in `instruments.yaml`, or use `WEEKDAYS` for synthetic data.

**A-06. Sessions containing more than one contract are excluded.** If the data has a `contract`
column and it changes inside a session, the roll happened intraday. Prices on either side of that
change are not comparable, so the whole session is dropped and reported.

## Opening range and signals

**A-07. The opening range window is `[start, start + length)` on bar start times.** OR high/low come
only from base bars that start inside the window. The range is known at `start + length`. A session
needs at least `min_or_coverage` (default 80 %) of the expected base bars in the window, otherwise it
is skipped as `invalid_or_insufficient_range_data`. A session with zero OR width is skipped too.

**A-08. Entry-timeframe bars are aligned to the range end.** Buckets are
`[range_end + k·tf, range_end + (k+1)·tf)`, so a 15-minute range with 10-minute entry bars gives
09:45–09:55, 09:55–10:05, and so on. Clock-aligned buckets would straddle the range end. A bucket's
close is known only at the bucket's end, and the kernel evaluates it only once a *later* base bar
exists.

**A-09. Breakout confirmation is strict.** A long needs `close > boundary + offset`, where
`offset = confirm_ticks·tick + confirm_or_frac·width`. The spec's "close > boundary + 1 tick" is
implemented literally, so it needs the close to be ≥ 2 ticks beyond the boundary. "Intrabar"
confirmation is the `stop` entry method: a stop order at the first tick strictly beyond
`boundary + offset`, plus `entry_buffer_ticks`. A stop at exactly the boundary could fill without
any breakout.

## Execution (all in tick units inside the kernel)

**A-10. Stop orders fill at the stop price plus slippage, or at the open plus slippage on a gap.**
This assumes every tick trades as price moves through the level. That is reasonable for liquid
futures, but **optimistic when price jumps**. Synthetic validation made this concrete: when the price
process moved several ticks per step, stop entries showed a small, systematic positive frictionless
edge on a pure random walk (+0.045 R, 11 of 12 seeds positive). The edge disappeared once the process
moved about one tick per step (mean t = 0.27 over 8 seeds). Consequences:
- stop-entry and stop-exit results in fast markets are optimistic before slippage;
- judge stop-entry variants at ≥ 1 tick slippage and run the slippage sensitivity (0–3 ticks);
- stop entries that beat market entries only by fractions of a tick are not trustworthy.

**A-11. Market entries fill at the open of the first base bar at or after the signal time, plus
slippage.** If that bar starts more than `max_fill_delay_minutes` (default 5) after the signal, the
signal is dropped as `signal_not_filled`. The model never fills at the signal bar's close.

**A-12. Limit orders (limit entries and profit targets) never get slippage.** Under the default
`conservative` fill model they fill only when price trades *through* the limit by one tick. The
`standard` model accepts a touch. Results must be reported under both models. A limit entry that is
not filled by the cutoff is cancelled.

**A-13. Same-bar ambiguity.** When stop and target both lie inside one bar and the open does not
settle the order:
- `optimistic`: the target was hit first.
- `pessimistic`: the stop was hit first.
- `conservative` (default, used for primary results): stop first. In addition, a position filled
  *intrabar* (stop or limit entry) cannot reach its target on the fill bar, while its stop can be
  hit there.

Ambiguity is resolved on **1-minute bars** regardless of the entry timeframe: signals come from
entry-timeframe bars, but fills and exits are simulated at the finest resolution available. The
number and share of ambiguous exits is reported for every backtest.

**A-14. Stop-entry ambiguity.** If a single base bar triggers both the long and the short stop order
and its open does not decide which came first, the session is skipped
(`ambiguous_both_sides_entry`) rather than guessed.

**A-15. Stops and targets are rounded away from the entry.** Stops are floored below a long and
ceiled above a short; targets are ceiled or floored further away. Risk is therefore never understated
and targets are never easier than the nominal R. Risk is measured from the *actual* fill, including
slippage, and must be at least 1 tick or the signal is skipped (`invalid_stop`).

**A-16. Break-even moves apply from the next bar.** The trigger is evaluated on a completed bar's
extreme. It is never applied on an intrabar-fill entry bar, because that bar's extreme may have
happened before the fill.

**A-17. Costs.** Round trip = `2 × (commission + exchange fee)` per contract. Default slippage is
1 tick per market or stop fill. The defaults in `instruments.yaml` are illustrative retail figures:
$0.85 commission plus $1.40–1.60 exchange/NFA fees per side. Replace them with your own schedule.
Bid/ask columns are loaded but not yet used; slippage stands in for the spread. **Headline results
always include costs.** The frictionless run is a labelled comparison only.

**A-18. Time and session exits.** A position still open at the time exit (default: instrument
`session_exit` = 15:55 ET, or 5 minutes before an early close) exits at that bar's open, minus
slippage. If data ends before then, it exits at the last available close (`data_end`, reported).

**A-19. Cutoff semantics.** A close-confirmed signal is valid if its bucket **ends** at or before the
cutoff. The resulting market order may fill at the cutoff minute. Resting limit and stop orders are
cancelled at the cutoff. The cutoff is capped at the time exit.

## Sizing and metrics

**A-20. Sizing is applied after simulation.** With one instrument and non-overlapping trades, size
never changes *which* trades happen. Signals whose stop is too wide to size even `min_qty` contracts
are skipped and counted (`sized_out`), never traded fractionally. Grid search always uses 1 fixed
contract, so compounding cannot distort comparisons between configurations.

**A-21. Daily statistics use every eligible session**, including days with no trade. Returns on a
fixed-contract run are measured against running equity. Sharpe/Sortino are annualised with √252.
CAGR uses the calendar span. Max drawdown is measured on session-close equity, so intraday drawdown
within a trade is not included.

**A-22. Holding time** is exit-bar start minus entry-bar start, in minutes. MFE/MAE come from bar
extremes and are approximate; they are diagnostics only.

## Look-ahead controls

**A-23. Indicators use only prior sessions.** Daily ATR (Wilder) for session *d* is computed from
full-session bars of sessions *< d*. Percentile ranks (`trailing_percentile_rank`) use strictly prior
values. Regime thresholds (Milestone 6) will use the same trailing machinery. Nothing is computed
from the whole dataset.

**A-24. Leak tests** (`tests/test_lookahead.py`) actively try to detect leakage:
- truncating a session at a random time must not change any earlier decision;
- scrambling everything after a fill-bar's open must not change entries up to that open;
- future sessions must not change past trades;
- ATR and percentile ranks must be causal.

The suite was mutation-tested: a deliberately injected one-bar peek (using the fill bar's close) is
caught by both the targeted and the generic tests.

## Optimization

**A-25. Grid results are in-sample.** The top row is labelled "highest historical result" or "best
composite in-sample score", never "optimal". The composite score rewards Sharpe, the t-stat of mean
R, profit factor, Calmar and year-consistency. It penalises fewer than 100 trades, a top-5 trade
share above 50 %, a single-year share above 50 %, and drawdowns deeper than 30 %. Weights live in
`config/strategy.yaml`. Rankings under several alternative objectives are always reported, along
with how much their top-10s overlap.

**A-26. Multiple testing.** Every grid reports:
- the number of configurations tested;
- an estimate of effective independent trials (participation ratio of the P&L correlation matrix
  for a seeded sample of configurations);
- the best Sharpe that selection alone would produce under the null (Bailey & López de Prado). If
  the observed best does not beat it, a warning is shown.

The full Deflated Sharpe Ratio and PBO/CSCV are Milestone 5.

**A-27. Heatmaps aggregate hidden dimensions by median by default.** "max" is labelled optimistic
because it re-optimises the hidden parameters in every cell.

**A-28. Gradients in heatmaps are not evidence of an edge.** On pure random-walk data the reduced
09:30 grid showed smooth gradients: wider ranges and larger R looked "less bad". This happens because
larger risk per trade shrinks fixed costs when measured in R. Any real-data gradient must be checked
against the frictionless view and the walk-forward results before it is interpreted.

## Synthetic validation (Milestone 1)

Command: `python -m orb_lab.cli validate-synthetic`. Data: ES-like, 2019–2022, seed 123, weekdays
calendar. Costs: 1 tick slippage, $4.50 round trip.

| trend (daily σ) | entry | frictionless mean R (t) | net mean R (t) |
|---|---|---|---|
| 0 (null) | market 15m / or_opposite / 1R | +0.022 (0.73) | −0.003 (−0.10) |
| 0 (null) | market 30m / or_mid / 1.5R | +0.051 (1.29) | +0.016 (0.41) |
| 0 (null) | stop 10m / 1R | +0.052 (1.71) | +0.021 (0.69) |
| 0 (null) | limit 15m / 1R | +0.032 (0.91) | +0.019 (0.54) |
| 0.5 | market 15m | +0.174 (5.81) | +0.155 (5.14) |
| 1.0 | market 30m / or_mid / 1.5R | +0.762 (22.2) | +0.731 (21.1) |

On the null model no variant shows a significant edge (|t| < 2), and costs always reduce the result.
The planted edge is detected by every entry type. The 5-year null backtest of the reference strategy
(`data/synth_ES_null.parquet`, seed 7) lost $27k net against roughly zero gross (t = −0.26). The loss
is entirely transaction costs. That is exactly what an honest engine should show.

---

## Real-market phase (v0.3.0): governance and walk-forward

**Status: no real market data has been analysed yet.** Nothing above or below is evidence about ORB
profitability. See `DATA_SOURCES.md` for the source decision and acquisition steps.

**A-29. Data source: Databento GLBX.MDP3 `ohlcv-1m`, individual contracts.** The front month is built
causally: the contract with the highest volume in the *previous* session, forward-only rolls, and the
first session dropped. Raw prices, **no back-adjustment**. Each session contains exactly one contract.
Spreads are removed with an outright-only symbol filter.

**A-30. Daily true range never spans a roll.** On the first session of a new contract the previous
close is set to NaN, so TR = H − L. Roll gaps therefore cannot inflate ATR stops or the OR/ATR filter.

**A-31. Real-data DQ adds month-by-month 09:30 alignment.** A timezone or DST error confined to part
of the history shows up as a ~60-minute displacement of the volume peak, or as the 09:30 volume
step-up moving to 08:30 or 10:30. Any such month gives the verdict **STOP** and research on that
instrument halts until it is explained. The DQ report also checks every DST transition week, whether
early-close sessions really stop trading, and year-by-year completeness. (Test: a synthetic one-hour
error in summer 2021 is flagged in exactly those months.)

**A-32. Research gates.** Optimisation on real data (`cli wfo`) refuses to run unless a DQ report
exists for the exact file hash with verdict PASS or REVIEW. The dataset is always the
lockbox-truncated development view. Every CLI path and the dashboard withhold a sealed lockbox, so no
strategy result for lockbox sessions can be produced during development.

**A-33. Lockbox.** The most recent 12 months, fixed when the DQ stage first runs (immutable; newer
data does not move it). It is used only if ≥ 5 years of development data remain. Unlocking requires
a frozen candidate file whose SHA-256 is recorded. The lockbox test can run exactly once; later calls
return the stored result.

**A-34. Registry.** `research/registry.jsonl` is append-only. Six hypotheses from the prior ORB
research, and the full WFO design (structures, selection rule, search-space fingerprint
`3d2f58d3a329379d` with 488,070 configurations, primary OOS variant), were registered on 2026-09-26
**before any real data was loaded**. Anything found later from OOS analysis is registered as
EXPLORATORY and cannot be called confirmed without new untouched data (the lockbox).

**A-35. Walk-forward mechanics.**
- *Monthly statistics cube.* Every configuration is simulated once over the development history, and
  additive monthly statistics are stored: trades, wins, ΣP&L, Σdaily-P&L², ΣR, ΣR², gross win/loss,
  long-side stats, gross P&L, ambiguous exits. Any window is an exact sum of months. This is valid
  because strategies are intraday, hold no state across sessions, and are evaluated with 1 fixed
  contract. A test checks that cube metrics equal trade-by-trade backtests.
- *Windows.* Calendar months, with partial first and last months dropped. The primary structure is
  12 / 3 / 3, rolled 3 months. Sensitivity structures: 24/3/3, 24/6/6, 36/6/6 (and 48/12/12). Roll
  equals the OOS length, so OOS segments tile without overlap.
- *Selection.* Per window, as defined in `optimization/walk_forward.py`:
  1. Training score = 0.35·Sharpe/3 + 0.35·t/5 + 0.30·log PF, each clipped. Positive scores are shrunk
     by trades / min_trades (min_trades = 40 for training).
  2. Robust training score = ½ own score + ½ median score of the immediate parameter neighbours. With
     no neighbours the second half is 0.
  3. The top 25 by robust training score become finalists.
  4. Validation score is computed with the same formula (min 10 trades).
  5. Selection score = ½ validation + ½ robust training. The winner is frozen.
- *OOS variants.* Primary is **always_trade**. Secondary is **stand_aside**: flat when the selected
  configuration's validation expectancy ≤ 0. Both use only pre-OOS information.
- *Structural isolation.* Selection receives a `CubeView` limited to `[train_start, val_end)` and
  raises on any other month. Two tests prove that scrambling OOS and later data leaves every earlier
  selection and every finalist score unchanged. A deliberately injected leak (validation extended into
  OOS) is caught by both.
- *Headline OOS result.* Stitched blind-OOS trades, 1 contract, net of costs. A 1 %-risk equity curve
  is also produced. Slippage sensitivity (0.5 / 1 / 1.5 / 2 / 3 ticks) re-runs the **same frozen
  selections**. A result is flagged **execution-sensitive** if OOS expectancy turns non-positive within
  +2 ticks of the base assumption.
- *Overfit flags.* Raised when:
  - selected configurations are profitable in training but not OOS (median);
  - the best raw training configuration collapses OOS;
  - fewer than half of the OOS windows are profitable;
  - the selection changes in more than 75 % of windows.

**A-36. Phase heatmaps are descriptive.** Training = mean over windows. Validation and OOS = pooled
over all validation or OOS months with each configuration held fixed. They show where plateaus lie;
they are **not** walk-forward results and are never used to choose parameters.

**A-37. Start-time experiment uses cutoffs relative to the ORB start** (+60, +120, +210 minutes) so
different start times get comparable entry windows.

**A-38. Machinery check on the synthetic null file** (not evidence about ORB): primary 12/3/3
walk-forward over 540 configurations. Stitched OOS −0.055 R/trade (t = −1.35); only 40 % of windows
profitable; median training expectancy +0.058 R versus OOS −0.096 R. All four overfit flags fired.
This is the intended behaviour: the in-sample "winners" of a no-edge market fail out of sample.

**Pending before any real result is quoted:** verify transaction costs against a real broker/CME fee
schedule (research plan Task 5). The `instruments.yaml` numbers are still unverified placeholders.

---

## PHASE0_FREE_PROXY (v0.4.0): free Yahoo SPY/QQQ experiment, NOT futures validation

**A-39. Measured Yahoo limits (2026-09-26).**
- 1m: last 30 days (20 sessions).
- 2m: the 60-day window is accepted, but data starts 2026-08-13 (31 sessions).
- 5m and 15m: last 60 days (42 sessions).

5m was used: the finest interval with the maximum history, and every requested OR length is a
multiple of 5. Entry timeframes are therefore 5/10/15 (1- and 3-minute entries are impossible).

**A-40. Yahoo prices.**
- Unadjusted and RTH-only; bar-start stamps in America/New_York.
- Float noise is removed by 4-dp rounding. Genuine sub-penny prints (~17 % of SPY OHLC values) are
  kept, so tick-based levels are approximate to under half a cent.
- 5m bars equal aggregated 1m bars exactly on all 1,560 overlapping bars.

**A-41. ETF friction model.** $X/share is charged on every fill (entry and exit, market, stop and
limit), via the engine's per-side cost field (1 contract = 1 share). Stop and target distances are
unaffected. The engine's default path-slippage model moves brackets with the fill and only charges
market and stop fills, so it never reduced the P&L of a trade that still reached its target. It was
used in the first SPY run and replaced; this is disclosed in the reports. The same configuration was
selected under both models.

**A-42. Split and isolation.**
- Chronological 60/20/20 by session.
- Selection runs on `subset_before(test_start)`: the test bars are not in memory.
- The frozen selection file (SHA-256 logged) is written before the full dataset is evaluated.
- Test: replacing every final-test bar with a different market leaves the selection and all finalist
  scores unchanged. A mutation that keeps the test in memory is caught.

**A-43. Duplicates.** A limit entry fills exactly at the broken boundary (buffer 0, conservative fill),
so its 50 %-of-width stop equals the midpoint stop. Those configurations are removed.

**A-44. Random-direction control.** Same entry bar, entry price, risk distance and target multiple;
direction by coin flip; 10,000 draws; the same exit rules and costs. It reproduces every engine trade's
R exactly when given the actual direction.

**A-45. Conclusion rule, stated in code before the test was run.**
- NO PRELIMINARY EVIDENCE if final-test expectancy ≤ 0.
- PROMISING only if validation > 0, the result survives $0.03/share, it beats the random-direction
  control at p < 0.10, the final-test neighbourhood is a plateau, and t ≥ 2.
- MIXED otherwise.

**A-46. Descriptive medians exclude configurations with zero trades in that phase.** This was
corrected after the first report. It never affected selection.

## Pre-data platform finalisation (v0.5.0)

**A-47. Cost categories and units.** Every $ cost is **per unit (contract or share), per side (fill)**:
commission, exchange/regulatory fees and modelled bid/ask friction (`friction_per_side`). Slippage is in
ticks per side and only hits market and stop fills (limit fills and targets instead need a one-tick
trade-through). A round trip pays each $ item twice; all scale with quantity, never per order. The
trade log shows commission, fees, friction, slippage_cost and total_cost separately. The ETF $0.02/share
friction moved from the commission field to `friction_per_side`; totals are identical and the committed
Phase-0 numbers reproduce exactly. A per-order (quantity-independent) commission is not modelled.

**A-48. Contracts vs shares.** `Instrument.asset_class` is `future` (sized in whole contracts) or `etf`
(sized in shares); the unit is shown in every cost and sizing label. Micro contracts (MES, MNQ, MGC,
M6E) are configured so small accounts can size risk-based positions on the same price series.

**A-49. Risk-based sizing.** Quantity = floor(risk budget / risk per unit), where risk per unit =
entry-to-stop distance x $ per point, floored to the instrument's quantity step. Budget = % of current
equity (compounding) or % of starting equity (fixed $). A trade that cannot be sized to one unit is
skipped and flagged. An optional notional cap (position value <= equity x leverage; ETFs default to 4x)
prevents impossible share counts when a stop is very tight; capped trades are flagged.

**A-50. Session-block walk-forward.** For samples too short for monthly windows (the free ~60-day data),
walk-forward windows can be measured in trading sessions. The cube is built over blocks of
gcd(train, validation, OOS) sessions; everything else (neighbourhood-robust selection on a view limited
to train + validation, frozen OOS, stitching) is identical. Monthly presets 12/3/3/3 (primary) and
24/3/3, 24/6/6, 36/6/6 are unchanged and remain the design for real futures data. Every window records
the SHA-256 of its selected parameters, computed before the OOS segment is simulated. A mutation test
(replace all data after an OOS start with a different market) shows earlier selections, finalist scores
and hashes do not change.

**A-51. Dashboard blind holdout.** The PIPELINE tab splits sessions chronologically into train /
validation / blind holdout (default 60/20/20) before optimisation. Development tabs then receive only
sessions before the holdout. Candidates are frozen to read-only files named by the SHA-256 of their
parameters, execution and sizing. Only a frozen, hash-verified candidate can be run on the holdout; the
first run is recorded as BLIND, later runs as NOT BLIND. The futures lockbox (A-26) is applied before any
of this.

**A-52. Monte Carlo is not evidence.** It resamples historical R outcomes (reshuffle, bootstrap, block
bootstrap, missed trades, extra cost per trade). It describes path risk (drawdowns, streaks, probability
of loss) given those outcomes; it cannot create or confirm an edge. Seeds make runs reproducible.

**A-53. Execution stress.** Full re-simulations with +0.5/1/2/3 ticks per side, fixed costs x1.5/2/3,
pessimistic ambiguity and conservative fills, plus a labelled approximation of a delayed entry (entry k
ticks worse, same exits). FRAGILE = positive at baseline but not positive under any moderate stress
(+1 tick, 2x costs, pessimistic ambiguity, 1 tick adverse entry).

**A-54. Robustness sweeps.** One parameter at a time around the candidate (others fixed), on the chosen
development period. plateau = every immediate neighbour positive and their median >= 50 % of the
candidate's expectancy; spike = neighbour median <= 25 % or not positive. In-sample description only.

**A-55. Report verdicts (fixed in code before any real data).** Evidence = first blind holdout test and
stitched walk-forward OOS only. INSUFFICIENT EVIDENCE without either or with < 30 OOS trades;
NO PRELIMINARY EVIDENCE if every OOS expectancy <= 0; PROMISING only with positive OOS expectancy,
t >= 2 in every OOS source, >= 100 OOS trades, and no fragility flag (spike, FRAGILE stress, >= 2
walk-forward overfitting flags, disagreement between OOS sources); MIXED otherwise.

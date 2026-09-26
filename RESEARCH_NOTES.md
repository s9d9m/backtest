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

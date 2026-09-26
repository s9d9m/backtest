# ORB Lab

A local research platform for testing whether **Opening Range Breakout** strategies on futures (ES,
NQ, GC, 6E and anything you add) have a persistent, tradable edge after realistic costs and
out-of-sample testing.

It is built to *find out whether* ORB works, not to prove that it does. Grid-search winners are
labelled "highest historical result", never "optimal". The intended primary evidence is the stitched
out-of-sample walk-forward curve (Milestone 3).

## Status

| Milestone | Scope | Status |
|---|---|---|
| 1 | Loader, sessions/DST, basic 09:30 ORB, entries, stops, R targets, cutoff, costs, trade log, metrics, UI, tests, synthetic validation | **done** |
| 2 | Parameter grid search (parallel, checkpointed), heatmaps, objectives, multiple-testing warning | **done** |
| 3 | Walk-forward optimisation (train / validation / OOS, stitched OOS curve), lockbox, research registry | **done** |
| 4 | Robustness: neighbour/plateau analysis, parameter-stability scores | planned |
| 5 | Monte Carlo, Deflated Sharpe, PBO/CSCV | planned |
| 6 | Retest, FVG, candle-confirmation entries; trailing stops; regime/percentile filters | planned |

## Phase 0: free proxy experiment (NOT futures validation)

```bash
pip install yfinance            # optional; free, no key, no account
python -m orb_lab.cli free-test --symbol SPY,QQQ
```

It measures Yahoo's actual intraday limits, downloads and caches, runs the DQ gate, splits
chronologically (60/20/20), selects without the final test in memory, freezes the selection, then runs
the final test once. Outputs: reports, trade logs, heatmaps and trade charts in `phase0_results/`.
Results from 2026-09-26: `phase0_results/PHASE0_REPORT.md`.

## Real-market research workflow

No real data is included (licensing), and none has been analysed yet. See
[DATA_SOURCES.md](DATA_SOURCES.md).

```bash
python -m orb_lab.cli fetch --instrument 6E --start 2015-06-01 --end 2026-09-01 --estimate-only
python -m orb_lab.cli fetch --instrument 6E --start 2015-06-01 --end 2026-09-01
python -m orb_lab.cli dq-report --data data/6E_databento_front_1m.parquet --instrument 6E   # seals the lockbox
python -m orb_lab.cli wfo --data data/6E_databento_front_1m.parquet --instrument 6E --space real_930_primary
python -m orb_lab.cli hypothesis list
```

## Quick start

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest                                  # 126 tests
streamlit run app.py                    # dashboard

# command line
python -m orb_lab.cli synth --instrument ES --start 2019-01-01 --end 2023-12-29 --out data/synth_ES.parquet
python -m orb_lab.cli dq --data data/ES_1min.csv --instrument ES --tz America/Chicago
python -m orb_lab.cli backtest --data data/synth_ES.parquet --instrument ES --set target_r=1.2 --save
python -m orb_lab.cli grid --data data/synth_ES.parquet --instrument ES --space primary_930 --workers 8
python -m orb_lab.cli rerun --manifest runs/<experiment-id>
python -m orb_lab.cli validate-synthetic
```

Your data: 1-minute OHLCV in CSV or Parquet with `timestamp, open, high, low, close, volume`
(optional `bid, ask, contract, symbol`). Timestamps must carry an offset, or you must declare the
source timezone. Say whether timestamps mark the bar start or the bar end.

## Architecture

```
app.py                         Streamlit entry point
config/
  instruments.yaml             tick size/value, multiplier, costs, calendar, session times
  strategy.yaml                reference strategy, execution, sizing, objective weights
  search_spaces.yaml           grid presets (primary_930, start_time_experiment, ...)
orb_lab/
  engine/
    data_loader.py             adapters (CSV/Parquet/registry) -> canonical UTC bar-start frame
    data_quality.py            DQ report + explicit cleaning policy (errors block backtests)
    sessions.py                ET/DST clocks, Globex session assignment, exchange calendars
    session_data.py            PreparedData: per-session flattened arrays (tick units), OR/ATR caches
    orb.py                     opening ranges (numba), causal ATR, causal percentile ranks
    execution.py               fill rules, cost resolution, kernel codes
    params.py                  StrategyParams / ExecutionParams / SizingParams
    backtest.py                single-configuration orchestration -> trade log, equity, metrics
    portfolio.py               position sizing, daily equity
    metrics.py                 full metric set + fast grid summary
    synthetic.py               null and planted-edge synthetic data
    pipeline.py                load -> check -> prepare
  strategies/
    basic_breakout.py          numba simulation kernel (market / limit / stop entries)
  data_sources/
    databento_source.py        Databento GLBX.MDP3 ohlcv-1m download, cost estimate, provenance
    futures_roll.py            outright filter + causal previous-session-volume front-month roll
  research/
    registry.py                append-only record of hypotheses, experiments, WFO design
    lockbox.py                 12-month final lockbox: seal, withhold, one-shot unlock
    gates.py                   DQ-before-optimisation and lockbox enforcement
  optimization/
    parameter_space.py         YAML spaces -> canonical, de-duplicated configurations
    grid_search.py             multiprocessing + checkpoint/resume
    objectives.py              composite and alternative ranking objectives
    heatmaps.py                parameter-pair tables
    multiple_testing.py        expected max Sharpe under the null, effective trials
    stats_cube.py              monthly sufficient-statistics cube (every config simulated once)
    walk_forward.py            windows, neighbourhood-robust selection, frozen OOS, stitching
    robustness.py              parameter neighbourhoods, plateau vs spike
    wfo_runner.py              all structures x entry families, saved + registered
  reports/
    plots.py                   Plotly figures
    experiment.py              manifests (IDs, hashes, versions, costs) and exact rerun
    dq_report.py               real-data DQ: month-by-month 09:30 ET alignment, DST, early closes
  ui/app_main.py               dashboard tabs
  cli.py
tests/                         unit, exact-outcome, DST, look-ahead, statistical, grid, UI
```

**Data flow.** File → `data_loader` (UTC, bar start) → `data_quality` (report, block on errors) →
`session_data.prepare_data` (sessions, eligibility, ET `tod`, tick units, daily bars) → kernel per
configuration → per-contract trade arrays → `portfolio` sizing → `metrics`.

**No-lookahead design.** The opening range uses only bars that start inside its window. Entry
buckets are evaluated only once a later bar exists, and fills happen on that later bar. Signals use
entry-timeframe bars; fills, stops and targets are simulated on 1-minute bars. ATR and percentiles
use prior sessions only. Leak tests delete or scramble future data and require every earlier decision
to stay identical.

## Assumptions that most affect validity

All of these are documented in [RESEARCH_NOTES.md](RESEARCH_NOTES.md), with the reasoning for each:

- **Same-bar stop/target ambiguity.** Primary results use the conservative model (stop first; no
  target on an intrabar-fill bar). Ambiguous exits are counted and reported.
- **Limit fills.** The conservative model requires trade-through by one tick; results are also
  available under a touch model.
- **Stop fills** at the stop price plus slippage are optimistic when price gaps. Synthetic tests
  confirmed this, so stop-entry variants need slippage-sensitivity checks.
- **Costs.** Commission, exchange fees and slippage are always included in headline numbers. The
  frictionless run is a comparison only.
- **Calendar.** XNYS eligibility, early-close handling, and exclusion of intraday contract rolls.
- **Timestamps.** Naive timestamps are never guessed. The bar-start/bar-end convention is explicit.
  A DST volume-profile check catches data recorded in a fixed UTC offset.

## Reproducibility

Every saved backtest or grid run gets `runs/<experiment-id>/manifest.json` containing:
- parameters, instrument spec, resolved costs and sizing;
- data file path and SHA-256, and the cleaning policy;
- the date range, objective weights and seed;
- package versions and the configuration count.

`cli rerun` refuses to run if the data hash differs from the manifest.

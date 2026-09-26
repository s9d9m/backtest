# Changelog

## 0.4.0: PHASE0_FREE_PROXY (free Yahoo ETF proxy experiment; NOT futures validation)
- `data_sources/yahoo.py`: free intraday provider (yfinance). It measures availability (per-chunk
  refusal messages recorded), refines boundaries day by day, keeps a cached raw chunk store and a
  processed canonical file with provenance (hashes, requested/returned periods, timezone).
- `phase0/`: `free-test` pipeline:
  - measured interval choice, DQ gate, chronological 60/20/20 split;
  - selection on a copy of the data truncated before the final test, frozen with SHA-256 before the
    test is evaluated;
  - per-share friction on every fill ($0 reference, $0.01–$0.05);
  - neighbourhood robustness, phase heatmaps, random-direction control, buy-and-hold baselines,
    concentration analysis;
  - complete trade logs; 10 winner + 10 loser verification charts (supplemented from dev-selected
    configurations when the candidate has too few); 1-minute resolution cross-check.
- `optimization/stats_cube.grouped_stats`: cube over arbitrary session groups.
- `config/phase0.yaml`, kept separate from the futures configuration. CLI `free-test --symbol SPY,QQQ`.
- Executed on real data: see `phase0_results/PHASE0_REPORT.md` (SPY and QQQ both MIXED; no
  statistically meaningful evidence at 42 sessions).
- 126 tests (+9: Yahoo timestamps across DST, refusal reporting, interval choice, ETF tick and costs,
  5-minute OR/entry/exit, missing bars and misaligned opens, chronological split, final-test leakage
  (mutation-checked), trade log and control consistency).

## 0.3.0: Real-data infrastructure and Milestone 3 (walk-forward)
- `DATA_SOURCES.md`: evaluation of Databento, FirstRate, Kibot, Portara/CQG, broker feeds and free
  sources. Recommendation: Databento GLBX.MDP3 individual contracts. Exact acquisition steps.
- `data_sources/databento_source.py`: cost estimate, resumable per-year raw cache, provenance JSON.
  `data_sources/futures_roll.py`: outright filter, causal previous-session-volume forward-only roll,
  roll report.
- ATR/true range never spans a contract roll.
- `reports/dq_report.py`: month-by-month 09:30 ET alignment, DST-transition checks, early-close
  verification, year-by-year completeness, PASS/REVIEW/STOP verdict and a markdown report.
- `research/`: append-only registry (hypotheses with statuses, experiments, WFO design), lockbox
  (seal → development view → one-shot unlock with a frozen candidate), research gates (DQ before
  optimisation, lockbox always withheld).
- Milestone 3: monthly statistics cube, WFO structures (12/3/3 primary; 24/3/3, 24/6/6, 36/6/6
  sensitivity), neighbourhood-robust selection, frozen blind OOS, stitched OOS equity (1 contract and
  1 % risk), stand-aside variant, slippage sensitivity, execution-sensitivity and overfit flags,
  per-window finalists, parameter stability, per-phase heatmap data, entry-family restriction.
- Parameter spaces: `real_930_primary` (488,070 configurations) and `real_start_time` (relative
  cutoffs).
- CLI: `fetch`, `dq-report`, `wfo`, `hypothesis`, `lockbox-status`. Dashboard WALK-FORWARD tab.
- Six hypotheses and the WFO design pre-registered before any real data.
- 116 tests. New: roll construction, mocked Databento, DQ alignment, lockbox/registry/gates, WFO
  segmentation, cube-vs-backtest equality, and two mutation-checked WFO leakage tests.

## 0.2.0 — Milestone 2: grid search and heatmaps
- `optimization/parameter_space.py`: YAML parameter spaces (`config/search_spaces.yaml`), with
  compound `stop` and `confirmation` keys, validation, and canonicalisation that removes duplicates
  differing only in irrelevant parameters.
- `optimization/grid_search.py`: grid search across processes (forkserver/spawn) with atomic
  chunk-level checkpoints, safe resume and a fingerprint check against mixing runs. Results are
  deterministic, and parallel output equals serial output. Configurations are evaluated with fixed
  1-contract sizing. Runs at roughly 120 configs/s on 4 cores over 5 years of 1-minute data.
- `optimization/objectives.py`: configurable composite in-sample score with penalties, plus rankings
  under alternative objectives.
- `optimization/multiple_testing.py`: expected maximum Sharpe under the null and an estimate of
  effective trials; warns when the best result does not beat selection bias.
- `optimization/heatmaps.py`: two-parameter views with median, mean, p25, max and share-positive
  aggregation, and the ability to fix other parameters.
- Presets, including `primary_930` (spec §33, 107,640 valid configurations) and
  `start_time_experiment` (§34).
- Dashboard OPTIMIZATION tab: parameter selection, live progress with ETA, sortable table, CSV
  export, objective-overlap diagnostic and heatmaps. CLI `grid` command.

## 0.1.0 — Milestone 1: core engine
- Project architecture, instrument and strategy configuration (ES, NQ, GC, 6E).
- CSV/Parquet loader with an adapter registry, column aliases, epoch/offset/naive timestamp handling
  (an explicit timezone is required for naive timestamps), and bar-start/bar-end conventions.
- Data-quality report: ordering, exact and conflicting duplicates, invalid OHLC, zero volume, grid
  misalignment, off-tick prices, weekend and maintenance-window bars, a DST volume-profile timezone
  check, price spikes, missing sessions and RTH bars, large gaps and intraday contract changes. Errors
  block backtests unless an explicit, recorded policy resolves them.
- Session handling: Globex session assignment, America/New_York DST-aware clocks, XNYS holidays and
  early closes.
- Numba simulation kernel for the basic breakout: market, limit and stop entries; confirmation
  offsets; order buffers; OR-opposite, OR-mid, %-of-width, ATR and fixed-tick stops; fixed-R targets;
  break-even; cutoff; time and session exits; direction filter; max trades/day with re-entry
  policies; OR/ATR filter; standard vs conservative limit fills; optimistic, pessimistic and
  conservative same-bar ambiguity; slippage and commissions.
- Position sizing (fixed contracts, fixed risk, % of equity), daily equity, the full metric set
  (gross and net, long and short separately), and profit-concentration diagnostics.
- Reproducible experiment manifests with exact rerun and data-hash verification.
- Streamlit dashboard (DATA, BACKTEST, TRADE LOG, REPORT), Plotly charts, slippage sensitivity and a
  frictionless comparison benchmark.
- Synthetic data generator (null and planted-edge) and a validation study.
- 90 automated tests, including exact-outcome kernel tests, DST tests, mutation-checked lookahead
  tests, statistical null/edge tests and a headless dashboard test.

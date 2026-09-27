# Changelog

## 0.6.1: Free 6E (Euro FX futures) data
- New free data source: Yahoo **6E=F** 5-minute bars (last ~60 days, full Globex hours), loaded with the real 6E
  futures specification (contracts, 0.00005 tick = $6.25, futures commission and fees). It is the first free market
  on the Data page and is preferred when the app opens; the Overview offers **Load free 6E data**.
- Clearly labelled as free continuous front-month data (short, spliced at the roll, not the Databento research
  dataset) on Overview, Data, Validate and Results.
- `yahoo.download` keeps 6 decimals for 6E (the ETF default of 4 would have corrupted 0.00005 prices) and records the
  Yahoo ticker in the provenance.
- Opening-range start times 08:00, 08:20 (CME FX pit open) and 08:30 ET added to the strategy editor.
- Prices are shown with the decimals the instrument's tick needs (1.15405 for 6E, not 1.15).
- Verified on real data: 40 sessions (2026-07-31 to 2026-09-25), data-quality gate passes (32 zero-volume bars noted).
- 180 tests (+3): 6E download precision, 6E dashboard load with the futures spec, price formatting.

## 0.6.0: UI/UX redesign (presentation only; no research calculations changed)
- **New information architecture**, research first, data and administration last: Overview · Strategy · Backtest ·
  Optimize · Validate · Stress test · Results · Trade explorer · Data · Settings & research. Sidebar menu; only the
  selected page is computed.
- **Overview** home screen: current strategy, market, account and risk, execution, headline historical result
  (balance, return, P&L, trades, win rate, expectancy, profit factor, drawdown), research-stage badges (in-sample,
  validation, blind OOS, walk-forward, robustness, execution, Monte Carlo) and the research verdict. A backtest alone is
  never shown as validated.
- **Current strategy** is one object (parameters + execution + sizing) used by every page, shown in a header on each
  research page with an *Edit strategy* button. Results computed for a strategy are keyed to it and the data, so editing
  anything marks them "not run" instead of showing stale numbers.
- **Strategy** page with grouped controls (opening range, entry, stop & target, trade management & direction, position
  sizing, execution) and an Advanced section; saved/frozen strategies can be loaded.
- **Backtest**: headline cards first, then equity, drawdown, cumulative R and trade-P&L charts, then details; raw tables
  under Advanced; one consolidated caution box.
- **Optimize**: *Best historical configuration* vs *Robust candidate* cards, compact ranked table (expectancy, PF,
  trades, Sharpe, 1-unit drawdown, t-stat, top-5 share, stability), one-click *Use this strategy / Backtest / Validate /
  Stress test*; multiple-testing warning kept; heatmaps and the full table behind expanders.
- **Validate**: TRAIN → VALIDATION → FREEZE → BLIND OOS → ROLL FORWARD diagram; blind-holdout steps; walk-forward
  launcher and results led by stitched blind-OOS cards, cumulative OOS R, fold-by-fold bars and a degradation chart
  with in-sample phases greyed out.
- **Stress test**: parameter robustness (BROAD PLATEAU / MODERATE / SPIKE-FRAGILE per parameter, charts with your
  setting highlighted, heatmap with the candidate marked), execution sensitivity (scenario chart and verdict) and a
  redesigned Monte Carlo (headline balance/drawdown/loss-probability cards, four distribution charts, path fan;
  percentile tables under Advanced).
- **Results** page: verdict banner and stage cards; downloadable Markdown report.
- **Trade explorer**: filters, per-trade detail cards and a candlestick chart with range, entry, stop, target and exit.
- **Data** page: loading, data quality, sessions, provenance, and a prepared-but-inactive Databento panel (ES, NQ, GC,
  6E · GLBX.MDP3 · OHLCV-1m).
- **Settings & research**: market specifications, frozen strategies and hashes, holdout ledger, research registry,
  diagnostics, Phase-0 archive, test runner.
- Consistent visual system: number formats ($43,532 · 8.8% · +0.13R · 1.64), cards, status colours with icons,
  help tooltips for every key statistic, one chart style.
- Usability fixes found while testing: the header refreshes immediately after saving a strategy; skipped signals
  (stop too wide for the risk budget) are explained with a micro-contract hint; stitched walk-forward results are shown
  in R rather than at 1 share.
- On first open a locally cached free SPY copy is loaded automatically; otherwise the Overview offers one-click loading.
- 177 tests (+9 net): 5 page-based dashboard tests (navigation order, empty state, free-data autoload with ETF units,
  strategy editor, full workflow including a walk-forward job) and 7 helper tests (formatting, robust-candidate choice,
  fold consistency, degradation, Monte Carlo headline, trade filters, strategy descriptions).

## 0.5.0: Pre-data platform finalisation (browser-only research workflow)
- **Walk-forward from the browser**: configure structure (monthly presets 12/3/3/3 primary, 24/3/3, 24/6/6,
  36/6/6, custom; or trading-session windows for short samples), parameter space, entry families, costs,
  selection rules, sizing of the stitched curve and date range; launch as a background job
  (`orb_lab/jobs.py`) with live progress, Stop and Resume (monthly cube checkpoints are reused). Results
  browser for dashboard and command-line runs; "only stitched blind OOS is evidence" shown throughout;
  each window records the SHA-256 of its parameters, frozen before the OOS segment is simulated.
- **ROBUSTNESS tab**: one-at-a-time sweeps around a candidate (OR length, entry TF, target R, stop,
  cutoff, entry type, confirmation, buffer, direction, OR start), plateau / spike / mixed per parameter,
  overall verdict, degradation vs the candidate, two-parameter heatmaps.
- **MONTE CARLO tab**: reshuffle, bootstrap, block bootstrap, missed trades, extra cost per trade; fixed-$
  or compounding risk; seeded; distributions of ending equity, return, max drawdown, losing streaks,
  probabilities; historical path vs simulated paths. Sources: candidate, last backtest, walk-forward OOS
  trades, blind holdout trades.
- **Risk-based sizing**: starting equity (default $40,000) with 0.25 / 0.5 / 1 / 2 % or custom risk per
  trade, compounding or fixed $; quantity from entry-to-stop distance, floored to whole contracts/shares;
  optional notional (buying-power) cap, ETF default 4x; trade log shows quantity, $ at risk, notional.
- **Cost model and units**: separate commission, exchange/regulatory fees, modelled friction (all $ per
  contract or share, per side) and slippage (ticks per side); every label shows its unit; trade log and
  metrics split each category plus total and per-trade cost. ETF friction moved to its own field (Phase-0
  numbers reproduce exactly). `Instrument.asset_class`/`unit` (future→contract, etf→share). Micro
  contracts MES, MNQ, MGC, M6E added.
- **Execution stress test** (BACKTEST and PIPELINE): +0.5…3 ticks, 1.5-3x fixed costs, pessimistic
  ambiguity, conservative fills, approximate delayed entry; verdict ROBUST / FRAGILE / NOT POSITIVE.
- **Optimizer UX**: reduced vs comprehensive presets (confirmation required above 5,000 configurations),
  configurations searched, effective trials, expected best Sharpe under no edge, per-row stability
  (plateau/spike, neighbour median), optimizer defaults to the train period, "Use as candidate".
- **Metric explanations**: tooltips and a "What do these numbers mean?" table for every headline metric,
  with automatic warnings for small samples, annualised figures from < 1 year, |t| < 2, outlier
  dependence, one-sided profits and ambiguous exits. Equity & risk view (equity, drawdown, cumulative R,
  trade P&L, position size and $ at risk).
- **PIPELINE tab**: DATA → SPLIT → BACKTEST/OPTIMIZE → VALIDATE → FREEZE → WALK-FORWARD → ROBUSTNESS +
  STRESS → MONTE CARLO → BLIND HOLDOUT → REPORT with status; chronological train/validation/blind-holdout
  split saved per dataset and withheld from development tabs; frozen candidates as read-only files named by
  SHA-256; one-shot blind test ledger (first test BLIND, later NOT BLIND).
- **REPORT tab**: candidate report across all stages with conservative verdicts (PROMISING / MIXED /
  NO PRELIMINARY EVIDENCE / INSUFFICIENT EVIDENCE), Markdown download.
- **DIAGNOSTICS tab**: null control, planted edge, lookahead truncation, random-direction control, cost-unit
  check, full test suite; clearly separated from real results.
- Databento workflow documented in the dashboard as prepared but inactive (no key, no purchase).
- START_HERE.md rewritten for browser-only use.
- 168 tests (+41): cost units and per-unit/per-side scaling, contracts vs shares, risk % sizing and compounding,
  notional cap, Monte Carlo reproducibility and resampling, stress monotonicity and verdicts, sweeps,
  session-block walk-forward leakage (mutation-checked), split/freeze/blind ledger, background jobs
  (run, stop, resume, failure), report verdict language, metric cautions, diagnostics, malformed data,
  and a headless end-to-end dashboard test of the whole pipeline.

## 0.4.1: Dashboard usable with free data
- Dashboard:
  - new default data source **Free Yahoo (SPY/QQQ)**. It uses the correct ETF spec (tick $0.01,
    $0.02/share friction, 100 shares by default); previously SPY files could only be loaded under a
    futures spec;
  - it reuses the Phase-0 experiment's saved copy, or downloads the last ~60 days into
    `data/dashboard_yahoo/` (the experiment's own file is never overwritten);
  - SPY/QQQ added to the market list;
  - "FREE PROXY — NOT FUTURES VALIDATION" banner whenever ETF data is loaded;
  - range and entry-timeframe choices limited to multiples of the loaded bar size.
- New **PHASE-0 RESULTS** tab (read-only, works on a fresh clone): conclusions, trade logs with CSV
  download, trade-chart gallery, heatmaps, cost table, neighbourhood.
- Easy launch: `.devcontainer/` (GitHub Codespaces opens the dashboard in the browser
  automatically), double-click launchers `start_dashboard.bat` / `.command` / `.sh`, and
  `START_HERE.md` for non-technical users. Streamlit usage statistics disabled.
- Fixed: sidebar cost caption rendered as a LaTeX formula (unescaped `$`).
- 127 tests (+1 headless dashboard test with the free-data source, offline).

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

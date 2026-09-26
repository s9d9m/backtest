# Changelog

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

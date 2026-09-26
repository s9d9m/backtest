"""Command-line interface.

Examples::

    python -m orb_lab.cli synth --instrument ES --start 2018-01-01 --end 2023-12-31 --out data/synth_ES.parquet
    python -m orb_lab.cli dq --data data/ES_1min.csv --instrument ES --tz America/Chicago
    python -m orb_lab.cli backtest --data data/synth_ES.parquet --instrument ES --set range_minutes=15 --set target_r=1.2 --save
    python -m orb_lab.cli grid --data data/synth_ES.parquet --instrument ES --space primary_930_reduced --workers 4
    python -m orb_lab.cli rerun --manifest runs/backtest-20260926-120000-abc123
    python -m orb_lab.cli validate-synthetic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from .engine.backtest import run_backtest
from .engine.data_quality import CleaningPolicy
from .engine.instruments import get_instrument
from .engine.params import ExecutionParams, StrategyParams, load_strategy_config
from .engine.pipeline import DataQualityError, build_dataset
from .engine.synthetic import generate_bars
from .optimization.grid_search import results_table, run_grid
from .optimization.heatmaps import STANDARD_PAIRS, heatmap_table
from .optimization.objectives import ObjectiveConfig
from .optimization.parameter_space import load_search_spaces
from .reports.experiment import metrics_table, rerun_backtest, save_backtest


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.lower() in ("none", "null"):
        return None
    return value


def _dataset(args, instrument):
    policy = CleaningPolicy(duplicate_policy=args.duplicates, invalid_policy=args.invalid)
    cfg = load_strategy_config()
    try:
        ds = build_dataset(
            args.data,
            instrument,
            source_tz=args.tz,
            timestamp_convention=args.convention,
            policy=policy,
            allow_errors=args.allow_errors,
            min_or_coverage=cfg["data"].min_or_coverage,
            exclude_early_close=cfg["data"].exclude_early_close,
        )
    except DataQualityError as exc:
        print(exc.report.to_frame().to_string(index=False))
        print("\nRefusing to backtest data with unresolved errors. Fix the data, choose an explicit cleaning "
              "policy (--duplicates/--invalid) or pass --allow-errors (recorded in the manifest).")
        sys.exit(2)
    return _withhold_lockbox(ds, instrument)


def _withhold_lockbox(ds, instrument):
    """Strategy results are never produced for a sealed lockbox, whichever command is used."""
    from .research.lockbox import apply_lockbox

    prep, lock = apply_lockbox(instrument.symbol, ds.prep)
    if lock and lock.get("lockbox_start"):
        print(f"[lockbox] {instrument.symbol}: sessions from {lock['lockbox_start']} are withheld")
        ds.prep = prep
    return ds


def _add_data_args(p):
    p.add_argument("--data", required=True)
    p.add_argument("--instrument", required=True)
    p.add_argument("--tz", default=None, help="source timezone for naive timestamps")
    p.add_argument("--convention", default="start", choices=["start", "end"], help="timestamp marks bar start or end")
    p.add_argument("--duplicates", default="error", choices=["error", "keep_first", "keep_last"])
    p.add_argument("--invalid", default="error", choices=["error", "drop"])
    p.add_argument("--allow-errors", action="store_true")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)


def cmd_synth(args):
    inst = get_instrument(args.instrument)
    bars = generate_bars(inst, args.start, args.end, seed=args.seed, trend_strength=args.trend)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    bars.reset_index().rename(columns={"ts": "timestamp"}).to_parquet(args.out, index=False)
    print(f"wrote {len(bars):,} bars to {args.out} (seed={args.seed}, trend_strength={args.trend})")


def cmd_dq(args):
    inst = get_instrument(args.instrument)
    args.allow_errors = True
    ds = _dataset(args, inst)
    print(json.dumps(ds.report.summary(), indent=2, default=str))
    print(ds.report.to_frame().to_string(index=False))
    print(f"\ntradable sessions: {ds.prep.n_days}; excluded: {ds.prep.excluded['reason'].value_counts().to_dict()}")


def cmd_backtest(args):
    inst = get_instrument(args.instrument)
    cfg = load_strategy_config(args.config)
    params = cfg["strategy"]
    overrides = dict(kv.split("=", 1) for kv in args.set)
    if overrides:
        params = replace(params, **{k: _coerce(v) for k, v in overrides.items()})
    ds = _dataset(args, inst)
    res = run_backtest(ds.prep, params, cfg["execution"], cfg["sizing"], start=args.start, end=args.end)
    bench = run_backtest(ds.prep, params, replace(cfg["execution"], frictionless=True), cfg["sizing"], start=args.start, end=args.end)
    print(f"{inst.symbol} {params}")
    table = metrics_table(res.metrics, res.metrics_gross)
    table["Frictionless benchmark"] = metrics_table(bench.metrics)["Net"]
    print(table.to_string(index=False))
    print("\nlong :", {k: round(res.metrics_long[k], 3) for k in ("n_trades", "net_pnl", "avg_r", "win_rate")})
    print("short:", {k: round(res.metrics_short[k], 3) for k in ("n_trades", "net_pnl", "avg_r", "win_rate")})
    print("diagnostics:", json.dumps(res.diagnostics, default=str))
    if args.save:
        print("saved:", save_backtest(res, ds, inst))


def cmd_grid(args):
    inst = get_instrument(args.instrument)
    cfg = load_strategy_config()
    spaces = load_search_spaces(base=cfg["strategy"])
    space = spaces[args.space]
    ds = _dataset(args, inst)
    exp = space.expand(ds.prep.base_minutes)
    print(f"space {space.name}: raw {exp.n_raw:,}, invalid {exp.n_invalid:,}, duplicates {exp.n_duplicates:,}, evaluating {len(exp.configs):,}")

    def progress(done, total, elapsed):
        rate = done / elapsed if elapsed > 0 else 0
        print(f"\r  {done:,}/{total:,} configs  {elapsed:,.0f}s  ({rate:,.0f}/s)", end="", flush=True)

    result = run_grid(ds.prep, space, cfg["execution"], start=args.start, end=args.end, n_workers=args.workers,
                      chunk_size=args.chunk, run_dir=args.run_dir, progress=progress, objective=ObjectiveConfig.from_dict(cfg["objective"]),
                      data_description=ds.describe())
    print(f"\nrun dir: {result.run_dir}")
    res = result.results
    table = results_table(res)
    pd.set_option("display.width", 250)
    print("\nTop 15 by composite IN-SAMPLE score (not evidence of an out-of-sample edge):")
    print(table.head(15).to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    b = result.selection_bias
    print(f"\nMultiple testing: {b.get('n_configs', 0):,} configurations (effective ~{b.get('n_effective', 0):,.0f}). "
          f"Best Sharpe {b.get('best_sharpe', 0):.2f} vs {b.get('expected_max_sharpe_under_null', 0):.2f} expected from selection alone "
          f"under the null. Share with positive Sharpe: {b.get('share_positive_sharpe', 0):.1%}.")
    if not b.get("best_exceeds_null_max", False):
        print("WARNING: the best in-sample result is NOT better than what selecting the maximum of this many "
              "no-edge configurations would typically produce.")
    for x, y, title in STANDARD_PAIRS:
        if x in res and y in res and res[x].nunique() > 1 and res[y].nunique() > 1:
            print(f"\n{title}: median Sharpe across other parameters")
            print(heatmap_table(res, y, x, "sharpe", "median").round(2).to_string())
    res.to_csv(result.run_dir / "results.csv", index=False)


def cmd_rerun(args):
    res = rerun_backtest(args.manifest, args.data)
    print(metrics_table(res.metrics, res.metrics_gross).to_string(index=False))


def cmd_validate_synthetic(args):
    from .engine.sessions import TradingCalendar

    inst = get_instrument(args.instrument)
    params = [
        StrategyParams(range_minutes=15, entry_tf=5, target_r=1.0, cutoff="11:00"),
        StrategyParams(range_minutes=30, entry_tf=10, target_r=1.5, cutoff="12:00", stop_method="or_mid"),
        StrategyParams(range_minutes=10, entry_method="stop", target_r=1.0, cutoff="11:30"),
        StrategyParams(range_minutes=15, entry_tf=5, entry_method="limit", target_r=1.0, cutoff="11:00"),
    ]
    rows = []
    for trend in (0.0, 0.5, 1.0):
        bars = generate_bars(inst, args.start, args.end, seed=args.seed, trend_strength=trend, calendar="WEEKDAYS")
        ds = build_dataset(bars, inst, calendar=TradingCalendar("WEEKDAYS"))
        for p in params:
            for fr in (True, False):
                r = run_backtest(ds.prep, p, ExecutionParams(frictionless=fr)).trades["r_multiple"].to_numpy()
                t = r.mean() / r.std(ddof=1) * np.sqrt(len(r)) if len(r) > 1 else 0.0
                rows.append({"trend": trend, "entry": p.entry_method, "range": p.range_minutes, "stop": p.stop_method, "R": p.target_r,
                             "costs": "frictionless" if fr else "net", "trades": len(r), "mean_R": round(r.mean(), 4), "t_stat": round(t, 2)})
    print(pd.DataFrame(rows).to_string(index=False))


def cmd_fetch(args):
    from .data_sources import databento_source as dbs

    if args.estimate_only:
        rows, total = [], 0.0
        for sym in args.instrument.split(","):
            q = dbs.uncached_cost(sym, args.start, args.end, args.mode)
            rows.append((sym, q["total_usd"]))
            total += q["total_usd"]
        spent = dbs.spend_ledger()["spent_estimate_usd"]
        for sym, cost in rows:
            print(f"{sym:4s} ${cost:10.2f}  (not yet downloaded part)")
        print(f"total ${total:.2f}; already spent ~${spent:.2f}; budget ${args.budget:.2f} -> "
              + ("FITS" if spent + total <= args.budget else "DOES NOT FIT: nothing will be downloaded"))
        return
    try:
        res = dbs.fetch(args.instrument, args.start, args.end, mode=args.mode, budget_usd=args.budget)
    except dbs.BudgetExceeded as exc:
        print(exc)
        sys.exit(4)
    print(json.dumps({k: res.provenance[k] for k in ("output_file", "output_rows", "rolls", "construction")}, indent=2, default=str))


def cmd_dq_report(args):
    from .research.gates import run_dq_stage

    inst = get_instrument(args.instrument)
    prov = None
    prov_file = os.path.splitext(args.data)[0].replace("_front_1m", "_provenance") + ".json"
    if os.path.exists(prov_file):
        prov = json.load(open(prov_file))
    ds, rdq, report, lock = run_dq_stage(args.data, inst, source_tz=args.tz, convention=args.convention,
                                          lockbox_months=args.lockbox_months, provenance=prov)
    print(f"verdict: {rdq.verdict}")
    for r in rdq.reasons:
        print("  -", r)
    print(f"report: {report}")
    print(f"lockbox: {lock}")
    if rdq.verdict == "STOP":
        print("Research on this instrument is STOPPED until the data problem is explained.")
        sys.exit(3)


def cmd_wfo(args):
    from .engine.params import ExecutionParams as EP
    from .optimization.walk_forward import SelectionConfig
    from .optimization.wfo_runner import run_experiment
    from .research.gates import research_dataset

    inst = get_instrument(args.instrument)
    cfg = load_strategy_config()
    space = load_search_spaces(base=cfg["strategy"])[args.space]
    if args.research:
        ds, lock = research_dataset(args.data, inst, source_tz=args.tz, convention=args.convention)
    else:
        print("WARNING: non-research run (no DQ gate, no lockbox, not registered). Not evidence of anything.")
        ds, lock = _dataset(args, inst), None
    print(f"{inst.symbol}: development sessions {ds.prep.dates[0]} .. {ds.prep.dates[-1]}"
          + (f" (lockbox from {lock['lockbox_start']} withheld)" if lock else ""))
    execution = cfg["execution"] if args.slippage is None else EP(**{**cfg["execution"].to_dict(), "slippage_ticks": args.slippage})

    def progress(stage, done, total, elapsed):
        print(f"\r  {stage}: {done:,}/{total:,} {elapsed:,.0f}s", end="", flush=True)

    run_dir, results = run_experiment(ds.prep, space, execution, structures=args.structures.split(","), families=args.families.split(","),
                                      n_workers=args.workers, selection=SelectionConfig(), research=args.research,
                                      data_description=ds.describe(), lockbox=lock, progress=progress)
    print(f"\nrun dir: {run_dir}")
    print(pd.read_csv(run_dir / "wfo_overview.csv").to_string(index=False, float_format=lambda v: f"{v:,.3f}"))


def cmd_hypothesis(args):
    from .research import registry

    if args.action == "add":
        e = registry.add_hypothesis(args.statement, status=args.status, source=args.source, data_seen=args.data_seen, test_plan=args.test_plan)
        print(e["hypothesis_id"])
    elif args.action == "update":
        registry.update_hypothesis(args.id, args.status, args.evidence, args.test_data)
    else:
        for h in registry.hypotheses().values():
            print(f"{h['hypothesis_id']} [{h['status']}] {h['statement']}  (introduced {h['ts_utc'][:19]}, source: {h['source']})")


def cmd_lockbox_status(args):
    from .research.lockbox import load_ledger

    print(json.dumps(load_ledger(), indent=2))


def cmd_free_test(args):
    from .data_sources.yahoo import PHASE0_LABEL
    from .phase0.pipeline import RESULTS_DIR, run_free_test
    from .phase0.report import write_combined

    print(PHASE0_LABEL)
    summaries = {}
    for sym in args.symbol.split(","):
        s = run_free_test(sym, n_workers=args.workers, refresh=args.refresh, interval=args.interval)
        summaries[sym] = s
        t = s["headline"]["test"]
        print(f"{sym}: interval {s['interval']} | split {s['split']['sessions']} | selected {s['frozen']['selected']['config_key']} | "
              f"FINAL TEST {t['trades']} trades, {t['exp_r']:+.3f} R, net ${t['net_pnl']:,.2f} | verdict {s['verdict']['category']}")
        print(f"   report: {RESULTS_DIR / sym / 'phase0_report.md'}")
    if len(summaries) > 1:
        print("comparison:", write_combined(summaries, RESULTS_DIR))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="orb_lab")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("synth")
    p.add_argument("--instrument", default="ES")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2023-12-31")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--trend", type=float, default=0.0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_synth)
    p = sub.add_parser("dq")
    _add_data_args(p)
    p.set_defaults(func=cmd_dq)
    p = sub.add_parser("backtest")
    _add_data_args(p)
    p.add_argument("--config", default=None)
    p.add_argument("--set", action="append", default=[], help="override strategy parameter, e.g. --set target_r=1.2")
    p.add_argument("--save", action="store_true")
    p.set_defaults(func=cmd_backtest)
    p = sub.add_parser("grid")
    _add_data_args(p)
    p.add_argument("--space", default="primary_930_reduced")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--chunk", type=int, default=200)
    p.add_argument("--run-dir", default=None, help="existing run directory to resume")
    p.set_defaults(func=cmd_grid)
    p = sub.add_parser("rerun")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data", default=None)
    p.set_defaults(func=cmd_rerun)
    p = sub.add_parser("validate-synthetic")
    p.add_argument("--instrument", default="ES")
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2022-12-30")
    p.add_argument("--seed", type=int, default=123)
    p.set_defaults(func=cmd_validate_synthetic)
    p = sub.add_parser("fetch", help="download real 1-minute futures bars (Databento)")
    p.add_argument("--instrument", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--mode", default="parent", choices=["parent", "continuous"])
    p.add_argument("--estimate-only", action="store_true", help="free quote; comma-separate instruments to quote several")
    p.add_argument("--budget", type=float, default=125.0, help="hard spending cap in USD (default: the $125 free credit)")
    p.set_defaults(func=cmd_fetch)
    p = sub.add_parser("dq-report", help="real-data quality report; seals the lockbox")
    p.add_argument("--data", required=True)
    p.add_argument("--instrument", required=True)
    p.add_argument("--tz", default=None)
    p.add_argument("--convention", default="start", choices=["start", "end"])
    p.add_argument("--lockbox-months", type=int, default=12)
    p.set_defaults(func=cmd_dq_report)
    p = sub.add_parser("wfo", help="walk-forward optimisation")
    _add_data_args(p)
    p.add_argument("--space", default="real_930_primary")
    p.add_argument("--structures", default="primary_12_3_3,sens_24_3_3,sens_24_6_6,sens_36_6_6")
    p.add_argument("--families", default="all,market,limit,stop")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    p.add_argument("--slippage", type=float, default=None)
    p.add_argument("--no-research", dest="research", action="store_false", help="bypass DQ gate/lockbox (synthetic only)")
    p.set_defaults(func=cmd_wfo)
    p = sub.add_parser("hypothesis")
    p.add_argument("action", choices=["add", "update", "list"])
    p.add_argument("--statement", default="")
    p.add_argument("--status", default="EXPLORATORY")
    p.add_argument("--source", default="")
    p.add_argument("--data-seen", default="")
    p.add_argument("--test-plan", default="")
    p.add_argument("--id", default="")
    p.add_argument("--evidence", default="")
    p.add_argument("--test-data", default="")
    p.set_defaults(func=cmd_hypothesis)
    p = sub.add_parser("free-test", help="PHASE0_FREE_PROXY: free Yahoo ETF proxy experiment (NOT futures validation)")
    p.add_argument("--symbol", default="SPY", help="e.g. SPY, QQQ or SPY,QQQ")
    p.add_argument("--interval", default="auto", choices=["auto", "1m", "2m", "5m", "15m"])
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    p.add_argument("--refresh", action="store_true", help="re-download instead of using the cache")
    p.set_defaults(func=cmd_free_test)
    p = sub.add_parser("lockbox-status")
    p.set_defaults(func=cmd_lockbox_status)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

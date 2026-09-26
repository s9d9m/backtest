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
        return build_dataset(
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
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

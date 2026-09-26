"""Exhaustive grid search with multiprocessing and chunk-level checkpointing.

Design:

* The configuration list is generated deterministically from the parameter space, canonicalised and
  de-duplicated; its fingerprint is stored so a resumed run is guaranteed to evaluate the same list.
* Configurations are evaluated in fixed-size chunks. Each finished chunk is written atomically to
  ``<run_dir>/chunks/chunk_NNNNNN.parquet`` by the parent process; an interrupted run resumes by
  skipping chunks that already exist.
* Workers are started with ``forkserver`` (``spawn`` where unavailable) rather than ``fork``, because
  forking a multi-threaded parent (numba, pyarrow) can deadlock. The prepared data is sent to each
  worker once through the pool initializer; numba kernels load from the on-disk cache. Opening ranges
  and ATRs are cached per worker.
* Every configuration is evaluated with fixed 1-contract sizing so compounding cannot distort the
  comparison (RESEARCH_NOTES.md A-20).

Grid-search results are **in-sample**. The best row is labelled "highest historical result", never
"optimal"; the stitched out-of-sample evidence comes from walk-forward analysis (Milestone 3).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..engine.backtest import run_kernel
from ..engine.execution import resolve_costs
from ..engine.metrics import fast_summary
from ..engine.params import ExecutionParams, StrategyParams
from ..engine.session_data import PreparedData
from ..reports.experiment import ExperimentManifest
from .multiple_testing import effective_trials, selection_bias_summary
from .objectives import ObjectiveConfig, add_objectives
from .parameter_space import ExpandedSpace, ParameterSpace, config_frame_row

ProgressFn = Callable[[int, int, float], None]

_STATE: dict = {}


def evaluate_config(
    prep: PreparedData,
    params: StrategyParams,
    execution: ExecutionParams,
    d_lo: int,
    d_hi: int,
    starting_equity: float = 100_000.0,
    return_daily: bool = False,
):
    """Evaluate one configuration with 1-contract sizing. Returns a metrics dict (and daily P&L)."""
    out = run_kernel(prep, params, execution, d_lo, d_hi)
    costs = resolve_costs(prep.instrument, execution)
    a = out.arrays
    tv = prep.instrument.tick_value
    net_ticks = (a["exit_px"] - a["entry_px"]) * a["dir"]
    net_pc = net_ticks * tv - costs.round_trip_fixed
    gross_pc = (net_ticks + a["entry_slip"] + a["exit_slip"]) * tv
    risk_pc = a["risk"] * tv
    r = np.where(risk_pc > 0, net_pc / np.where(risk_pc > 0, risk_pc, 1.0), 0.0)
    day_pos = (a["day"] - d_lo).astype(np.int64)
    dates = pd.DatetimeIndex(prep.dates[d_lo:d_hi].astype("datetime64[ns]"))
    trade_year = dates.year.to_numpy()[day_pos] if len(day_pos) else np.zeros(0, dtype=int)
    hold = (prep.tod[a["exit_idx"]] - prep.tod[a["entry_idx"]]).astype(float)
    summary = fast_summary(net_pc, r, day_pos, a["dir"], trade_year, hold, dates, starting_equity)
    summary["gross_pnl"] = float(gross_pc.sum())
    summary["gross_expectancy_r"] = float(np.mean(gross_pc / np.where(risk_pc > 0, risk_pc, 1.0))) if len(r) else 0.0
    summary["cost_per_trade"] = float(np.mean(gross_pc - net_pc)) if len(r) else 0.0
    summary["ambiguous_exit_pct"] = float(np.mean(a["ambiguous_exit"])) if len(r) else 0.0
    summary["sessions"] = int(d_hi - d_lo)
    if return_daily:
        daily = np.bincount(day_pos, weights=net_pc, minlength=d_hi - d_lo) if len(net_pc) else np.zeros(d_hi - d_lo)
        return summary, daily
    return summary


def _init_worker(state: dict) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _STATE.clear()
    _STATE.update(state)


def _eval_chunk(chunk_id: int, configs: list[StrategyParams]) -> tuple[int, list[dict]]:
    prep = _STATE["prep"]
    rows = []
    for params in configs:
        row = config_frame_row(params)
        row.update(evaluate_config(prep, params, _STATE["execution"], _STATE["d_lo"], _STATE["d_hi"], _STATE["equity"]))
        rows.append(row)
    return chunk_id, rows


@dataclass
class GridResult:
    results: pd.DataFrame
    run_dir: Path
    expanded: ExpandedSpace
    elapsed: float
    n_evaluated: int
    n_resumed_chunks: int
    selection_bias: dict
    experiment_id: str


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def run_grid(
    prep: PreparedData,
    space: ParameterSpace,
    execution: ExecutionParams | None = None,
    *,
    start=None,
    end=None,
    n_workers: int = 1,
    chunk_size: int = 100,
    run_dir: str | Path | None = None,
    resume: bool = True,
    progress: ProgressFn | None = None,
    objective: ObjectiveConfig | None = None,
    starting_equity: float = 100_000.0,
    data_description: dict | None = None,
    n_effective_sample: int = 150,
    seed: int = 12345,
) -> GridResult:
    execution = execution or ExecutionParams()
    objective = objective or ObjectiveConfig()
    expanded = space.expand(prep.base_minutes)
    configs = expanded.configs
    d_lo, d_hi = prep.day_range(start, end)
    if d_hi <= d_lo:
        raise ValueError("Empty date range for grid search")

    if run_dir is None:
        manifest = ExperimentManifest.create(
            "grid",
            data=data_description or {"data_hash": prep.data_hash},
            instrument=prep.instrument.to_dict(),
            search_space=space.to_dict(),
            execution=execution.to_dict(),
            sizing={"mode": "fixed_contracts", "contracts": 1, "starting_equity": starting_equity},
            costs=asdict(resolve_costs(prep.instrument, execution)),
            date_range={"start": str(prep.dates[d_lo]), "end": str(prep.dates[d_hi - 1])},
            objective=objective.to_dict(),
            random_seed=seed,
            n_configs=len(configs),
            notes=[f"fingerprint={expanded.fingerprint}", f"raw={expanded.n_raw} invalid={expanded.n_invalid} duplicates={expanded.n_duplicates}"],
        )
        run_path = manifest.run_dir()
        manifest.save()
        experiment_id = manifest.experiment_id
    else:
        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)
        manifest_file = run_path / "manifest.json"
        if manifest_file.exists():
            experiment_id = ExperimentManifest.load(manifest_file).experiment_id
        else:
            experiment_id = run_path.name
    chunk_dir = run_path / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    state_file = run_path / "grid_state.json"
    state = {"fingerprint": expanded.fingerprint, "n_configs": len(configs), "chunk_size": chunk_size, "d_lo": d_lo, "d_hi": d_hi,
             "execution": execution.to_dict()}
    if state_file.exists():
        previous = json.loads(state_file.read_text())
        if not resume:
            for f in chunk_dir.glob("chunk_*.parquet"):
                f.unlink()
        elif {k: previous.get(k) for k in state} != state:
            raise ValueError(
                "Existing run directory was created for a different configuration list / date range / execution "
                "settings; refusing to mix results. Use a new run directory or resume=False."
            )
    state_file.write_text(json.dumps(state, indent=2))

    chunks = [configs[i : i + chunk_size] for i in range(0, len(configs), chunk_size)]
    done = {int(p.stem.split("_")[1]) for p in chunk_dir.glob("chunk_*.parquet")} if resume else set()
    pending = [k for k in range(len(chunks)) if k not in done]
    n_resumed = len(chunks) - len(pending)

    # compile kernels (populating numba's on-disk cache) before starting workers
    if configs:
        evaluate_config(prep, configs[0], execution, d_lo, min(d_lo + 5, d_hi), starting_equity)
    _STATE.update(prep=prep, execution=execution, d_lo=d_lo, d_hi=d_hi, equity=starting_equity)

    t0 = time.time()
    finished_configs = sum(len(chunks[k]) for k in done)
    total = len(configs)
    if progress:
        progress(finished_configs, total, 0.0)

    def _store(chunk_id: int, rows: list[dict]) -> None:
        nonlocal finished_configs
        _atomic_parquet(pd.DataFrame(rows), chunk_dir / f"chunk_{chunk_id:06d}.parquet")
        finished_configs += len(rows)
        if progress:
            progress(finished_configs, total, time.time() - t0)

    if n_workers <= 1 or len(pending) <= 1:
        for k in pending:
            _store(*_eval_chunk(k, chunks[k]))
    else:
        method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(method)
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx, initializer=_init_worker, initargs=(dict(_STATE),)) as pool:
            futures = [pool.submit(_eval_chunk, k, chunks[k]) for k in pending]
            for fut in as_completed(futures):
                _store(*fut.result())

    frames = [pd.read_parquet(p) for p in sorted(chunk_dir.glob("chunk_*.parquet"))]
    results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    order = {p.key(): i for i, p in enumerate(configs)}
    if len(results):
        results["config_index"] = results["config_key"].map(order)
        results = results.sort_values("config_index").reset_index(drop=True)
        results.insert(0, "market", prep.instrument.symbol)
        results = add_objectives(results, objective)

    # multiple-testing context: effective number of trials from a deterministic sample of configs
    n_eff = None
    if len(configs) >= 2:
        rng = np.random.default_rng(seed)
        sample_idx = np.sort(rng.choice(len(configs), size=min(n_effective_sample, len(configs)), replace=False))
        mat = np.column_stack(
            [evaluate_config(prep, configs[i], execution, d_lo, d_hi, starting_equity, return_daily=True)[1] for i in sample_idx]
        )
        n_eff_sample = effective_trials(mat)
        # scale the sample estimate to the full grid (bounded by the number of configs)
        n_eff = float(min(len(configs), n_eff_sample * len(configs) / len(sample_idx))) if len(sample_idx) < len(configs) else n_eff_sample
    bias = selection_bias_summary(results["sharpe"].to_numpy() if len(results) else np.zeros(0), n_eff)
    if len(results):
        _atomic_parquet(results, run_path / "results.parquet")
    (run_path / "selection_bias.json").write_text(json.dumps(bias, indent=2))
    return GridResult(
        results=results,
        run_dir=run_path,
        expanded=expanded,
        elapsed=time.time() - t0,
        n_evaluated=total,
        n_resumed_chunks=n_resumed,
        selection_bias=bias,
        experiment_id=experiment_id,
    )


RESULT_COLUMNS = [
    ("rank_composite", "Rank"), ("market", "Market"), ("orb_start", "ORB Start"), ("range_minutes", "Range Length"),
    ("entry_tf", "Entry TF"), ("entry_method", "Entry Method"), ("confirmation", "Confirmation"), ("stop", "Stop Method"),
    ("target_r", "R"), ("cutoff", "Cutoff"), ("direction", "Direction"), ("n_trades", "Trades"), ("win_rate", "Win Rate"),
    ("expectancy", "Expectancy $"), ("avg_r", "Expectancy R"), ("profit_factor", "Profit Factor"), ("cagr", "CAGR"),
    ("sharpe", "Sharpe"), ("sortino", "Sortino"), ("max_dd", "Max DD"), ("calmar", "Calmar"), ("total_return", "Net Return"),
    ("t_stat_r", "t-stat R"), ("pct_years_profitable", "Years Profitable"), ("top5_share", "Top-5 Share"),
    ("n_long", "Longs"), ("n_short", "Shorts"), ("pnl_long", "P&L Long"), ("pnl_short", "P&L Short"),
    ("is_score", "IS Score"), ("config_key", "Config Key"),
]


def results_table(results: pd.DataFrame) -> pd.DataFrame:
    """Human-facing results table with the columns requested in the spec."""
    cols = [c for c, _ in RESULT_COLUMNS if c in results.columns]
    table = results[cols].rename(columns=dict(RESULT_COLUMNS))
    return table.sort_values("Rank") if "Rank" in table else table

"""Monthly sufficient-statistics cube: every configuration simulated once, aggregated by calendar month.

The strategies are intraday with no state carried between sessions and are evaluated with fixed
1-contract sizing. The performance of a configuration over any set of whole months is therefore an
exact sum of its monthly statistics. Walk-forward windows (train / validation / OOS) and per-phase
heatmaps are computed from the cube without re-simulating.

The cube lives on disk as a float32 ``.npy`` memmap (aggregated in float64) of shape ``(n_configs, n_months, n_stats)``,
built chunk by chunk with checkpoints. Its directory name is a hash of the data, date range,
configuration list and execution assumptions, so different inputs can never share a cube.

Selection code must only read months it is allowed to see; :class:`CubeView` enforces that.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..engine.backtest import run_kernel
from ..engine.execution import resolve_costs
from ..engine.params import ExecutionParams, StrategyParams
from ..engine.session_data import PreparedData

STATS = ("n", "wins", "pnl", "pnl_day_sq", "r", "r_sq", "gross_win", "gross_loss", "n_long", "r_long", "pnl_long", "gross_pnl", "n_amb")
S = {name: i for i, name in enumerate(STATS)}
PERIODS_PER_YEAR = 252

_WORKER: dict = {}


def month_index(dates: np.ndarray) -> tuple[np.ndarray, list[str]]:
    periods = pd.DatetimeIndex(np.asarray(dates, dtype="datetime64[D]").astype("datetime64[ns]")).to_period("M")
    labels = sorted({str(p) for p in periods})
    lookup = {m: k for k, m in enumerate(labels)}
    return np.array([lookup[str(p)] for p in periods], dtype=np.int64), labels


def config_stats(prep: PreparedData, params: StrategyParams, execution: ExecutionParams, day_month: np.ndarray, n_months: int) -> np.ndarray:
    """(n_months, n_stats) statistics of one configuration over all sessions of ``prep``."""
    out = run_kernel(prep, params, execution, 0, prep.n_days)
    a = out.arrays
    res = np.zeros((n_months, len(STATS)))
    if len(a["day"]) == 0:
        return res
    tv = prep.instrument.tick_value
    costs = resolve_costs(prep.instrument, execution)
    net_ticks = (a["exit_px"] - a["entry_px"]) * a["dir"]
    pnl = net_ticks * tv - costs.round_trip_fixed
    gross = (net_ticks + a["entry_slip"] + a["exit_slip"]) * tv
    r = pnl / (a["risk"] * tv)
    m = day_month[a["day"]]
    longs = a["dir"] > 0
    daily = np.bincount(a["day"], weights=pnl, minlength=prep.n_days)
    traded_days = np.unique(a["day"])
    res[:, S["n"]] = np.bincount(m, minlength=n_months)
    res[:, S["wins"]] = np.bincount(m, weights=(pnl > 0).astype(float), minlength=n_months)
    res[:, S["pnl"]] = np.bincount(m, weights=pnl, minlength=n_months)
    res[:, S["pnl_day_sq"]] = np.bincount(day_month[traded_days], weights=daily[traded_days] ** 2, minlength=n_months)
    res[:, S["r"]] = np.bincount(m, weights=r, minlength=n_months)
    res[:, S["r_sq"]] = np.bincount(m, weights=r * r, minlength=n_months)
    res[:, S["gross_win"]] = np.bincount(m, weights=np.where(pnl > 0, pnl, 0.0), minlength=n_months)
    res[:, S["gross_loss"]] = np.bincount(m, weights=np.where(pnl < 0, -pnl, 0.0), minlength=n_months)
    res[:, S["n_long"]] = np.bincount(m, weights=longs.astype(float), minlength=n_months)
    res[:, S["r_long"]] = np.bincount(m, weights=np.where(longs, r, 0.0), minlength=n_months)
    res[:, S["pnl_long"]] = np.bincount(m, weights=np.where(longs, pnl, 0.0), minlength=n_months)
    res[:, S["gross_pnl"]] = np.bincount(m, weights=gross, minlength=n_months)
    res[:, S["n_amb"]] = np.bincount(m, weights=a["ambiguous_exit"].astype(float), minlength=n_months)
    return res


def _init(state: dict) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _WORKER.clear()
    _WORKER.update(state)


def _chunk(k: int, lo: int, hi: int) -> tuple[int, int, int, np.ndarray]:
    w = _WORKER
    block = np.stack([config_stats(w["prep"], w["configs"][i], w["execution"], w["day_month"], w["n_months"]) for i in range(lo, hi)])
    return k, lo, hi, block


@dataclass
class StatsCube:
    path: Path
    months: list[str]
    sessions_per_month: np.ndarray
    configs: list[StrategyParams]
    data: np.ndarray  # memmap (n_configs, n_months, n_stats)
    key: str

    @property
    def n_configs(self) -> int:
        return len(self.configs)

    def month_pos(self, month: str) -> int:
        return self.months.index(month)

    def aggregate(self, m0: int, m1: int, idx: np.ndarray | None = None) -> np.ndarray:
        block = self.data[:, m0:m1, :] if idx is None else self.data[idx, m0:m1, :]
        return np.asarray(block, dtype=np.float64).sum(axis=1)

    def sessions(self, m0: int, m1: int) -> int:
        return int(self.sessions_per_month[m0:m1].sum())

    def view(self, m0: int, m1: int) -> "CubeView":
        return CubeView(self, m0, m1)


class CubeView:
    """Read access restricted to months ``[m0, m1)``. Any other request raises."""

    def __init__(self, cube: StatsCube, m0: int, m1: int):
        self._cube, self.m0, self.m1 = cube, m0, m1

    @property
    def n_configs(self) -> int:
        return self._cube.n_configs

    def aggregate(self, m0: int, m1: int, idx: np.ndarray | None = None) -> np.ndarray:
        if m0 < self.m0 or m1 > self.m1 or m0 >= m1:
            raise PermissionError(f"months [{m0},{m1}) outside the permitted range [{self.m0},{self.m1})")
        return self._cube.aggregate(m0, m1, idx)

    def sessions(self, m0: int, m1: int) -> int:
        if m0 < self.m0 or m1 > self.m1:
            raise PermissionError("session count outside permitted range")
        return self._cube.sessions(m0, m1)


def metrics_from_stats(agg: np.ndarray, n_sessions: int) -> pd.DataFrame:
    """Per-configuration metrics from aggregated statistics (1 contract, fixed notional)."""
    n = agg[:, S["n"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        exp_r = np.where(n > 0, agg[:, S["r"]] / n, 0.0)
        var_r = np.where(n > 1, (agg[:, S["r_sq"]] - n * exp_r**2) / (n - 1), 0.0)
        std_r = np.sqrt(np.clip(var_r, 0, None))
        t_stat = np.where(std_r > 0, exp_r / std_r * np.sqrt(n), 0.0)
        mean_d = agg[:, S["pnl"]] / max(n_sessions, 1)
        var_d = (agg[:, S["pnl_day_sq"]] - n_sessions * mean_d**2) / max(n_sessions - 1, 1)
        std_d = np.sqrt(np.clip(var_d, 0, None))
        sharpe = np.where(std_d > 0, mean_d / std_d * np.sqrt(PERIODS_PER_YEAR), 0.0)
        gl = agg[:, S["gross_loss"]]
        gw = agg[:, S["gross_win"]]
        pf = np.where(gl > 0, gw / gl, np.where(gw > 0, np.inf, 0.0))
        n_long = agg[:, S["n_long"]]
        n_short = n - n_long
    return pd.DataFrame(
        {
            "n_trades": n,
            "win_rate": np.where(n > 0, agg[:, S["wins"]] / np.where(n > 0, n, 1), 0.0),
            "exp_r": exp_r,
            "t_stat_r": t_stat,
            "sharpe": sharpe,
            "profit_factor": pf,
            "net_pnl": agg[:, S["pnl"]],
            "gross_pnl": agg[:, S["gross_pnl"]],
            "exp_usd": np.where(n > 0, agg[:, S["pnl"]] / np.where(n > 0, n, 1), 0.0),
            "n_long": n_long,
            "n_short": n_short,
            "exp_r_long": np.where(n_long > 0, agg[:, S["r_long"]] / np.where(n_long > 0, n_long, 1), 0.0),
            "exp_r_short": np.where(n_short > 0, (agg[:, S["r"]] - agg[:, S["r_long"]]) / np.where(n_short > 0, n_short, 1), 0.0),
            "pnl_long": agg[:, S["pnl_long"]],
            "pnl_short": agg[:, S["pnl"]] - agg[:, S["pnl_long"]],
            "ambiguous_share": np.where(n > 0, agg[:, S["n_amb"]] / np.where(n > 0, n, 1), 0.0),
        }
    )


def cube_key(prep: PreparedData, configs: list[StrategyParams], execution: ExecutionParams) -> str:
    payload = {
        "data_hash": prep.data_hash,
        "instrument": prep.instrument.to_dict(),
        "first": str(prep.dates[0]),
        "last": str(prep.dates[-1]),
        "n_days": int(prep.n_days),
        "execution": execution.to_dict(),
        "configs": hashlib.sha1("|".join(c.key() for c in configs).encode()).hexdigest(),
        "version": 1,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def build_cube(
    prep: PreparedData,
    configs: list[StrategyParams],
    execution: ExecutionParams,
    root: Path,
    *,
    n_workers: int = 1,
    chunk_size: int = 500,
    progress: Callable[[int, int, float], None] | None = None,
) -> StatsCube:
    day_month, months = month_index(prep.dates)
    n_months = len(months)
    key = cube_key(prep, configs, execution)
    path = Path(root) / f"cube_{key}"
    path.mkdir(parents=True, exist_ok=True)
    data_file = path / "stats.npy"
    done_file = path / "done.npy"
    n_chunks = (len(configs) + chunk_size - 1) // chunk_size
    if data_file.exists():
        data = np.lib.format.open_memmap(data_file, mode="r+")
        done = np.load(done_file) if done_file.exists() else np.zeros(n_chunks, dtype=bool)
    else:
        data = np.lib.format.open_memmap(data_file, mode="w+", dtype=np.float32, shape=(len(configs), n_months, len(STATS)))
        done = np.zeros(n_chunks, dtype=bool)
        (path / "meta.json").write_text(json.dumps({"months": months, "stats": STATS, "n_configs": len(configs), "execution": execution.to_dict(),
                                                    "first": str(prep.dates[0]), "last": str(prep.dates[-1])}, indent=2))
        pd.DataFrame([c.to_dict() for c in configs]).to_parquet(path / "configs.parquet")
    pending = [k for k in range(n_chunks) if not done[k]]
    t0 = time.time()
    total = len(configs)
    finished = total - sum(min(chunk_size, total - k * chunk_size) for k in pending)
    if progress:
        progress(finished, total, 0.0)

    def store(k, lo, hi, block):
        nonlocal finished
        data[lo:hi] = block
        done[k] = True
        if k % 10 == 0 or not pending:
            data.flush()
            np.save(done_file, done)
        finished += hi - lo
        if progress:
            progress(finished, total, time.time() - t0)

    state = dict(prep=prep, configs=configs, execution=execution, day_month=day_month, n_months=n_months)
    if pending:
        config_stats(prep, configs[0], execution, day_month, n_months)  # compile before starting workers
    if n_workers <= 1 or len(pending) <= 1:
        _init(state)
        for k in pending:
            store(*_chunk(k, k * chunk_size, min((k + 1) * chunk_size, total)))
    else:
        method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context(method), initializer=_init, initargs=(state,)) as pool:
            futs = [pool.submit(_chunk, k, k * chunk_size, min((k + 1) * chunk_size, total)) for k in pending]
            for fut in as_completed(futs):
                store(*fut.result())
    data.flush()
    np.save(done_file, done)
    sessions_per_month = np.bincount(day_month, minlength=n_months)
    return StatsCube(path=path, months=months, sessions_per_month=sessions_per_month, configs=configs,
                     data=np.lib.format.open_memmap(data_file, mode="r"), key=key)

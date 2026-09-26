"""Single-configuration backtest orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..strategies.basic_breakout import simulate_basic_breakout
from .execution import (
    AMBIGUITY_CODES,
    DAY_STATUS_NAMES,
    DIRECTION_CODES,
    ENTRY_CODES,
    EXIT_NAMES,
    FILL_CODES,
    REENTRY_CODES,
    STOP_CODES,
    Costs,
    resolve_costs,
)
from .metrics import compute_metrics
from .params import UNLIMITED_TRADES_CAP, ExecutionParams, SizingParams, StrategyParams
from .portfolio import apply_sizing, daily_equity
from .session_data import PreparedData

KERNEL_FIELDS = (
    "day",
    "dir",
    "signal_tod",
    "entry_idx",
    "exit_idx",
    "entry_px",
    "entry_slip",
    "stop",
    "target",
    "exit_px",
    "exit_slip",
    "reason",
    "ambiguous_exit",
    "intrabar_entry",
    "breakeven_moved",
    "mfe",
    "mae",
    "risk",
)


@dataclass
class KernelOutput:
    arrays: dict[str, np.ndarray]
    day_status: np.ndarray
    d_lo: int
    d_hi: int

    @property
    def n(self) -> int:
        return len(self.arrays["day"])


@dataclass
class BacktestResult:
    params: StrategyParams
    execution: ExecutionParams
    sizing: SizingParams
    costs: Costs
    trades: pd.DataFrame
    daily: pd.DataFrame
    metrics: dict
    metrics_gross: dict
    metrics_long: dict
    metrics_short: dict
    day_status: pd.DataFrame
    diagnostics: dict = field(default_factory=dict)


def run_kernel(prep: PreparedData, params: StrategyParams, execution: ExecutionParams, d_lo: int, d_hi: int) -> KernelOutput:
    params.validate(prep.base_minutes)
    execution.validate()
    costs = resolve_costs(prep.instrument, execution)
    or_hi, or_lo, _, or_ok = prep.opening_range(params.orb_start_min, params.range_minutes)
    atr = prep.atr(params.atr_period) if params.needs_atr else np.full(prep.n_days, np.nan)
    exit_arr = prep.session_exit_min
    if params.time_exit_min is not None:
        exit_arr = np.minimum(exit_arr, params.time_exit_min)
    cutoff_arr = np.minimum(np.int64(params.cutoff_min), exit_arr).astype(np.int64)
    stop_param = params.stop_param
    result = simulate_basic_breakout(
        prep.tod,
        prep.open,
        prep.high,
        prep.low,
        prep.close,
        prep.day_start,
        prep.day_end,
        np.int64(d_lo),
        np.int64(d_hi),
        or_hi,
        or_lo,
        or_ok,
        atr,
        cutoff_arr,
        exit_arr.astype(np.int64),
        np.int64(params.range_end_min),
        np.int64(params.entry_tf if params.entry_tf > 0 else prep.base_minutes),
        np.int64(ENTRY_CODES[params.entry_method]),
        float(params.confirm_ticks),
        float(params.confirm_or_frac),
        float(params.entry_buffer_ticks),
        np.int64(STOP_CODES[params.stop_method]),
        float(stop_param),
        float(params.target_r),
        np.int64(DIRECTION_CODES[params.direction]),
        np.int64(params.max_trades),
        np.int64(REENTRY_CODES[params.reentry]),
        float(params.breakeven_r),
        float(params.or_atr_min),
        float(params.or_atr_max),
        float(costs.slippage_ticks),
        np.int64(FILL_CODES[execution.fill_model]),
        np.int64(AMBIGUITY_CODES[execution.ambiguity]),
        np.int64(execution.max_fill_delay_minutes),
        np.int64(UNLIMITED_TRADES_CAP),
    )
    arrays = dict(zip(KERNEL_FIELDS, result[2:]))
    return KernelOutput(arrays=arrays, day_status=result[1], d_lo=d_lo, d_hi=d_hi)


def kernel_to_trades(prep: PreparedData, out: KernelOutput, params: StrategyParams, costs: Costs) -> pd.DataFrame:
    """Per-contract trade log in price units and dollars."""
    a = out.arrays
    inst = prep.instrument
    tick = inst.tick_size
    tv = inst.tick_value
    if out.n == 0:
        return pd.DataFrame(
            columns=[
                "session_date", "direction", "entry_time", "exit_time", "entry_price", "exit_price",
                "gross_pnl_pc", "slippage_pc", "commission_pc", "net_pnl_pc", "risk_per_contract", "holding_minutes",
            ]
        )
    day = a["day"]
    d = a["dir"]
    entry_tod = prep.tod[a["entry_idx"]]
    exit_tod = prep.tod[a["exit_idx"]]
    or_hi, or_lo, _, _ = prep.opening_range(params.orb_start_min, params.range_minutes)
    atr = prep.atr(params.atr_period if params.atr_period > 0 else 14)
    net_ticks_pre_comm = (a["exit_px"] - a["entry_px"]) * d
    slip_ticks = a["entry_slip"] + a["exit_slip"]
    gross_ticks = net_ticks_pre_comm + slip_ticks
    frame = pd.DataFrame(
        {
            "session_date": pd.DatetimeIndex(prep.dates[day].astype("datetime64[ns]")),
            "day_idx": day,
            "direction": np.where(d > 0, "long", "short"),
            "dir": d,
            "signal_time": prep.timestamps(day, a["signal_tod"]),
            "entry_time": prep.timestamps(day, entry_tod),
            "exit_time": prep.timestamps(day, exit_tod),
            "entry_price": a["entry_px"] * tick,
            "stop_price": a["stop"] * tick,
            "target_price": a["target"] * tick,
            "exit_price": a["exit_px"] * tick,
            "exit_reason": [EXIT_NAMES[int(x)] for x in a["reason"]],
            "risk_ticks": a["risk"],
            "entry_slip_ticks": a["entry_slip"],
            "exit_slip_ticks": a["exit_slip"],
            "ambiguous_exit": a["ambiguous_exit"],
            "intrabar_entry": a["intrabar_entry"],
            "breakeven_moved": a["breakeven_moved"],
            "mfe_r": a["mfe"] / a["risk"],
            "mae_r": a["mae"] / a["risk"],
            "or_high": or_hi[day] * tick,
            "or_low": or_lo[day] * tick,
            "or_width": (or_hi[day] - or_lo[day]) * tick,
            "atr_daily": atr[day] * tick,
            "holding_minutes": (exit_tod - entry_tod).astype(float),
            "gross_pnl_pc": gross_ticks * tv,
            "slippage_pc": slip_ticks * tv,
            "commission_pc": np.full(len(day), costs.round_trip_fixed),
        }
    )
    frame["or_atr"] = frame["or_width"] / frame["atr_daily"]
    frame["net_pnl_pc"] = frame["gross_pnl_pc"] - frame["slippage_pc"] - frame["commission_pc"]
    frame["risk_per_contract"] = frame["risk_ticks"] * tv
    frame["r_multiple"] = frame["net_pnl_pc"] / frame["risk_per_contract"]
    frame["weekday"] = frame["session_date"].dt.day_name()
    frame.insert(0, "trade_id", np.arange(1, len(frame) + 1))
    return frame


def run_backtest(
    prep: PreparedData,
    params: StrategyParams,
    execution: ExecutionParams | None = None,
    sizing: SizingParams | None = None,
    start=None,
    end=None,
) -> BacktestResult:
    execution = execution or ExecutionParams()
    sizing = sizing or SizingParams()
    d_lo, d_hi = prep.day_range(start, end)
    costs = resolve_costs(prep.instrument, execution)
    out = run_kernel(prep, params, execution, d_lo, d_hi)
    trades = kernel_to_trades(prep, out, params, costs)
    if len(trades):
        trades = apply_sizing(trades, sizing, prep.instrument, costs)
    else:
        for col in ("qty", "gross_pnl", "slippage_cost", "commission", "net_pnl", "risk_dollars", "equity_before", "equity_after"):
            trades[col] = pd.Series(dtype=float)
        trades["sized_out"] = pd.Series(dtype=bool)
        trades["r_multiple"] = pd.Series(dtype=float)
    dates = prep.dates[d_lo:d_hi]
    daily = daily_equity(trades, dates, sizing.starting_equity)
    metrics = compute_metrics(trades, daily, sizing.starting_equity, "net")
    metrics_gross = compute_metrics(trades, daily, sizing.starting_equity, "gross")
    side = {}
    for name in ("long", "short"):
        sub = trades[trades["direction"] == name] if len(trades) else trades
        side[name] = compute_metrics(sub, daily_equity(sub, dates, sizing.starting_equity), sizing.starting_equity, "net")
    status = out.day_status[d_lo:d_hi]
    day_status = pd.DataFrame(
        {"session_date": pd.DatetimeIndex(dates.astype("datetime64[ns]")), "status": [DAY_STATUS_NAMES.get(int(s), "?") for s in status]}
    )
    n = len(trades)
    diagnostics = {
        "sessions_tested": int(d_hi - d_lo),
        "status_counts": day_status["status"].value_counts().to_dict(),
        "ambiguous_exits": int(trades["ambiguous_exit"].sum()) if n else 0,
        "ambiguous_exit_pct": float(trades["ambiguous_exit"].mean()) if n else 0.0,
        "intrabar_entries": int(trades["intrabar_entry"].sum()) if n else 0,
        "data_end_exits": int((trades["exit_reason"] == "data_end").sum()) if n else 0,
        "sized_out_trades": int(trades["sized_out"].sum()) if n else 0,
        "costs": {"slippage_ticks": costs.slippage_ticks, "round_trip_fixed": costs.round_trip_fixed},
    }
    return BacktestResult(
        params=params,
        execution=execution,
        sizing=sizing,
        costs=costs,
        trades=trades,
        daily=daily,
        metrics=metrics,
        metrics_gross=metrics_gross,
        metrics_long=side["long"],
        metrics_short=side["short"],
        day_status=day_status,
        diagnostics=diagnostics,
    )

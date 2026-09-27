"""Execution stress test: does a configuration survive worse execution than assumed?

Each scenario re-runs the full simulation (fills, stops and targets react to the new assumptions),
except ``adverse entry``, which is an explicitly labelled post-hoc approximation: every entry is k ticks
worse, exits unchanged (stop/target prices fixed), so risk grows by k ticks and P&L falls by k ticks.
It approximates a delayed or poorer entry fill without inventing a new price path.

Verdict:

* ``ROBUST``  expectancy stays positive in every scenario up to the "moderate" level;
* ``FRAGILE`` expectancy is positive at baseline but turns non-positive under a moderate stress
  (+1 tick/side slippage, 2x fixed costs, pessimistic ambiguity or 1 tick adverse entry);
* ``NOT POSITIVE`` expectancy is not positive even at the baseline assumptions.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ..engine.backtest import run_backtest
from ..engine.execution import resolve_costs
from ..engine.params import ExecutionParams, SizingParams, StrategyParams
from ..engine.session_data import PreparedData


def _summary(trades: pd.DataFrame, label: str, group: str, moderate: bool) -> dict:
    t = trades[~trades["sized_out"]] if len(trades) and "sized_out" in trades else trades
    n = len(t)
    r = t["r_multiple"].to_numpy(dtype=float) if n else np.zeros(0)
    pnl = t["net_pnl"].to_numpy(dtype=float) if n else np.zeros(0)
    gl = -pnl[pnl < 0].sum()
    return {
        "group": group,
        "scenario": label,
        "moderate_stress": moderate,
        "trades": n,
        "expectancy_r": float(r.mean()) if n else 0.0,
        "net_pnl": float(pnl.sum()),
        "profit_factor": float(pnl[pnl > 0].sum() / gl) if gl > 0 else (float("inf") if (pnl > 0).any() else 0.0),
        "win_rate": float((pnl > 0).mean()) if n else 0.0,
        "cost_per_trade": float(t["total_cost"].mean()) if n and "total_cost" in t else 0.0,
    }


def adverse_entry(trades: pd.DataFrame, ticks: float, tick_value: float) -> pd.DataFrame:
    """Post-hoc approximation: entry ``ticks`` worse, same exits. Works on a sized trade log."""
    if len(trades) == 0 or ticks == 0:
        return trades
    t = trades.copy()
    per_unit = ticks * tick_value
    t["net_pnl_pc"] = t["net_pnl_pc"] - per_unit
    t["risk_per_contract"] = t["risk_per_contract"] + per_unit
    t["r_multiple"] = t["net_pnl_pc"] / t["risk_per_contract"]
    t["net_pnl"] = t["net_pnl_pc"] * t["qty"]
    t["total_cost"] = t["total_cost"] + per_unit * t["qty"]
    return t


def stress_test(
    prep: PreparedData,
    params: StrategyParams,
    execution: ExecutionParams,
    sizing: SizingParams | None = None,
    start=None,
    end=None,
    slippage_steps: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0),
    cost_multipliers: tuple[float, ...] = (1.5, 2.0, 3.0),
    adverse_ticks: tuple[float, ...] = (1.0, 2.0),
) -> tuple[pd.DataFrame, dict]:
    sizing = sizing or SizingParams(mode="fixed_contracts", contracts=1)
    costs = resolve_costs(prep.instrument, execution)
    base_ex = replace(execution, slippage_ticks=costs.slippage_ticks, commission_per_side=costs.commission_per_side,
                      exchange_fee_per_side=costs.exchange_fee_per_side, friction_per_side=costs.friction_per_side, frictionless=False)
    rows = []

    def run(ex: ExecutionParams):
        return run_backtest(prep, params, ex, sizing, start, end).trades

    base = run(base_ex)
    rows.append(_summary(base, "baseline (your assumptions)", "baseline", False))
    for s in slippage_steps:
        rows.append(_summary(run(replace(base_ex, slippage_ticks=costs.slippage_ticks + s)), f"slippage +{s:g} tick/side", "slippage",
                             s <= 1.0))
    for m in cost_multipliers:
        rows.append(_summary(run(replace(base_ex, commission_per_side=costs.commission_per_side * m, exchange_fee_per_side=costs.exchange_fee_per_side * m,
                                         friction_per_side=costs.friction_per_side * m)), f"fixed costs x{m:g}", "fixed costs", m <= 2.0))
    rows.append(_summary(run(replace(base_ex, ambiguity="pessimistic")), "same-bar ambiguity: pessimistic", "fills", True))
    rows.append(_summary(run(replace(base_ex, fill_model="conservative", ambiguity="conservative")), "conservative limit fills + ambiguity", "fills", True))
    for k in adverse_ticks:
        rows.append(_summary(adverse_entry(base, k, prep.instrument.tick_value), f"adverse entry {k:g} tick (approx. delayed fill)", "entry",
                             k <= 1.0))
    worst = replace(base_ex, slippage_ticks=costs.slippage_ticks + 1.0, commission_per_side=costs.commission_per_side * 2,
                    exchange_fee_per_side=costs.exchange_fee_per_side * 2, friction_per_side=costs.friction_per_side * 2,
                    ambiguity="pessimistic", fill_model="conservative")
    rows.append(_summary(run(worst), "combined: +1 tick, 2x costs, pessimistic", "combined", False))
    table = pd.DataFrame(rows)
    base_exp = table["expectancy_r"].iloc[0]
    moderate = table[table["moderate_stress"]]
    breaks = moderate[moderate["expectancy_r"] <= 0]["scenario"].tolist()
    if table["trades"].iloc[0] == 0:
        verdict = "NO TRADES"
    elif base_exp <= 0:
        verdict = "NOT POSITIVE"
    elif breaks:
        verdict = "FRAGILE"
    else:
        verdict = "ROBUST"
    slip_rows = table[table["group"] == "slippage"]
    breakeven = slip_rows[slip_rows["expectancy_r"] <= 0]["scenario"].head(1).tolist()
    info = {"verdict": verdict, "breaks_under": breaks, "baseline_expectancy_r": float(base_exp),
            "first_non_positive_slippage": breakeven[0] if breakeven else None, "unit": prep.instrument.unit}
    return table, info

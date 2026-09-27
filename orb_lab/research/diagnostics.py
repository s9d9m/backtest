"""Research sanity checks that can be run from the dashboard. Each returns a pass/fail record.

These use synthetic data or controlled manipulations with a KNOWN correct answer. They test the
software, never the strategy; their numbers must never be mixed with real results.

* ``null_control``       driftless random walk: no configuration may show a significant frictionless edge.
* ``planted_edge``       synthetic data with a hidden intraday trend: the engine must detect it.
* ``lookahead_truncation`` deleting all data after a date must not change any trade that exited before it.
* ``random_direction``   on the loaded data: does the candidate beat trades with random direction (same entries/exits)?
* ``cost_units``         a hand-computed cost check for the loaded instrument (per unit, per side).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ..engine.backtest import run_backtest
from ..engine.instruments import Instrument
from ..engine.params import ExecutionParams, SizingParams, StrategyParams
from ..engine.pipeline import build_dataset
from ..engine.sessions import TradingCalendar
from ..engine.synthetic import generate_bars

REFERENCE = [
    StrategyParams(range_minutes=15, entry_tf=5, target_r=1.0, cutoff="11:00"),
    StrategyParams(range_minutes=30, entry_tf=10, target_r=1.5, cutoff="12:00", stop_method="or_mid"),
    StrategyParams(range_minutes=10, entry_method="stop", target_r=1.0, cutoff="11:30"),
]
SYNTH_INST = Instrument(symbol="ES", name="synthetic ES-like test instrument (NOT market data)", tick_size=0.25, tick_value=12.5, multiplier=50,
                        commission_per_side=0.85, exchange_fee_per_side=1.40, slippage_ticks=1.0, calendar="WEEKDAYS")


def _t(r: np.ndarray) -> float:
    return float(r.mean() / r.std(ddof=1) * np.sqrt(len(r))) if len(r) > 1 and r.std(ddof=1) > 0 else 0.0


def _synthetic(trend: float, years: int, seed: int):
    end = f"{2019 + years - 1}-12-30"
    bars = generate_bars(SYNTH_INST, "2019-01-01", end, seed=seed, trend_strength=trend, calendar="WEEKDAYS")
    return build_dataset(bars, SYNTH_INST, calendar=TradingCalendar("WEEKDAYS"))


def null_control(years: int = 2, seed: int = 123) -> dict:
    ds = _synthetic(0.0, years, seed)
    rows = []
    for p in REFERENCE:
        r0 = run_backtest(ds.prep, p, ExecutionParams(frictionless=True)).trades["r_multiple"].to_numpy()
        r1 = run_backtest(ds.prep, p, ExecutionParams()).trades["r_multiple"].to_numpy()
        rows.append({"config": f"{p.entry_method} {p.range_minutes}m {p.stop_method} {p.target_r}R", "trades": len(r0),
                     "frictionless_mean_r": r0.mean(), "t_stat": _t(r0), "net_mean_r": r1.mean()})
    t = pd.DataFrame(rows)
    ok = bool((t["t_stat"].abs() < 3).all() and (t["net_mean_r"] < t["frictionless_mean_r"]).all())
    return {"check": "Null control (random walk, no edge)", "passed": ok, "table": t,
            "expected": "|t| < 3 for every configuration without costs, and costs always lower the result."}


def planted_edge(years: int = 2, seed: int = 123, trend: float = 1.5) -> dict:
    ds = _synthetic(trend, years, seed)
    rows = []
    for p in REFERENCE:
        r0 = run_backtest(ds.prep, p, ExecutionParams(frictionless=True)).trades["r_multiple"].to_numpy()
        rows.append({"config": f"{p.entry_method} {p.range_minutes}m {p.stop_method} {p.target_r}R", "trades": len(r0),
                     "frictionless_mean_r": r0.mean(), "t_stat": _t(r0)})
    t = pd.DataFrame(rows)
    ok = bool((t["t_stat"] > 3).all())
    return {"check": f"Planted edge (synthetic intraday trend {trend} daily sigma)", "passed": ok, "table": t,
            "expected": "t > 3 for every configuration: the engine must find an edge that is really there."}


def lookahead_truncation(prep, params: StrategyParams, execution: ExecutionParams | None = None, cut_share: float = 0.6) -> dict:
    from .lockbox import subset_before

    execution = execution or ExecutionParams()
    if prep.n_days < 5:
        return {"check": "Lookahead: truncate future data", "passed": False, "table": pd.DataFrame(), "expected": "need >= 5 sessions"}
    cut = pd.Timestamp(prep.dates[int(prep.n_days * cut_share)])
    full = run_backtest(prep, params, execution).trades
    part = run_backtest(subset_before(prep, cut), params, execution).trades
    cols = ["session_date", "direction", "entry_time", "entry_price", "stop_price", "target_price", "exit_time", "exit_price", "net_pnl"]
    before = full[full["session_date"] < cut][cols].reset_index(drop=True) if len(full) else pd.DataFrame(columns=cols)
    part = part[cols].reset_index(drop=True) if len(part) else pd.DataFrame(columns=cols)
    ok = len(before) == len(part) and (len(before) == 0 or before.equals(part))
    table = pd.DataFrame({"trades before cut (full data)": [len(before)], "trades with future deleted": [len(part)], "cut": [str(cut.date())]})
    return {"check": "Lookahead: deleting future data changes no earlier trade", "passed": bool(ok), "table": table,
            "expected": "identical trades before the cut date with and without the later data."}


def random_direction(prep, params: StrategyParams, execution: ExecutionParams | None = None, n_sims: int = 2000, seed: int = 7) -> dict:
    """Does the breakout direction beat a coin flip? Same entries, risk and target multiple, direction random.

    Both directions of every trade are re-simulated with the engine's conservative exit rules
    (``phase0.controls``). Break-even stops are not modelled in the control.
    """
    from ..engine.execution import resolve_costs
    from ..phase0.controls import both_direction_outcomes, random_direction_control

    execution = execution or ExecutionParams()
    t = run_backtest(prep, params, execution).trades
    if len(t) < 5:
        return {"check": "Random-direction control", "passed": False, "table": pd.DataFrame(), "expected": "need >= 5 trades"}
    costs = resolve_costs(prep.instrument, execution)
    exit_min = prep.session_exit_min.copy()
    if params.time_exit_min is not None:
        exit_min = np.minimum(exit_min, params.time_exit_min)
    oc = both_direction_outcomes(prep, t, params.target_r, costs.slippage_ticks, exit_min,
                                 cost_ticks_round_trip=costs.round_trip_fixed / prep.instrument.tick_value)
    ctrl = random_direction_control(oc, n_sims=n_sims, seed=seed)
    p = ctrl["p_value_random_ge_strategy"]
    table = pd.DataFrame([{k: ctrl[k] for k in ("n_trades", "strategy_mean_r", "random_mean_r_median", "random_mean_r_p05",
                                                  "random_mean_r_p95", "p_value_random_ge_strategy")}])
    return {"check": "Random-direction control (loaded data)", "passed": p < 0.05, "table": table,
            "expected": "p < 0.05 means the breakout direction adds value beyond chance. FAIL here is a finding about the strategy "
                        "on this data, not a software error."}


def cost_units(instrument: Instrument) -> dict:
    """Check that the engine charges $ per unit per side exactly as a hand calculation."""
    from ..engine.execution import resolve_costs
    from ..engine.portfolio import apply_sizing

    costs = resolve_costs(instrument, ExecutionParams())
    qty = 3.0
    trades = pd.DataFrame({"entry_price": [100.0], "risk_per_contract": [10 * instrument.tick_value], "gross_pnl_pc": [0.0],
                           "slippage_pc": [2 * costs.slippage_ticks * instrument.tick_value], "commission_pc": [2 * costs.commission_per_side],
                           "fees_pc": [2 * costs.exchange_fee_per_side], "friction_pc": [2 * costs.friction_per_side],
                           "fixed_cost_pc": [costs.round_trip_fixed]})
    trades["total_cost_pc"] = trades["slippage_pc"] + trades["fixed_cost_pc"]
    trades["net_pnl_pc"] = -trades["total_cost_pc"]
    out = apply_sizing(trades, SizingParams(contracts=qty, max_leverage=0), instrument, costs)
    hand = qty * 2 * (costs.commission_per_side + costs.exchange_fee_per_side + costs.friction_per_side + costs.slippage_ticks * instrument.tick_value)
    ok = abs(out["total_cost"].iloc[0] - hand) < 1e-9
    table = pd.DataFrame({"unit": [instrument.unit], "quantity": [qty], "engine_total_cost": [out["total_cost"].iloc[0]],
                          "hand_calculation": [hand], "formula": [f"{qty:g} x 2 fills x (commission + fees + friction + slippage ticks x tick value)"]})
    return {"check": f"Cost units ({instrument.symbol}: $ per {instrument.unit} per side)", "passed": bool(ok), "table": table,
            "expected": "engine total = hand calculation, scaling with quantity (never per order)."}

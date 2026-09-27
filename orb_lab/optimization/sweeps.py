"""Robustness sweeps around one candidate configuration.

* **One-at-a-time sweeps**: change one parameter across its range while every other parameter stays at
  the candidate's value. A real effect should degrade gradually (a plateau); a lucky configuration
  sits on a spike whose immediate neighbours are much worse.
* **Pair grids**: two parameters varied together (others fixed) for heatmaps.

Every evaluation uses fixed 1-unit sizing and the same execution assumptions as the candidate, over
the period you choose. These are in-sample descriptions of the neighbourhood, not new evidence.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np
import pandas as pd

from ..engine.params import ParamError, StrategyParams
from ..engine.session_data import PreparedData
from .grid_search import evaluate_config
from .parameter_space import CONFIRMATION_PRESETS, confirmation_label, parse_stop, stop_label

RANGES = [5, 10, 15, 20, 30, 45, 60]
ENTRY_TFS = [1, 2, 3, 5, 10, 15]
TARGETS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
CUTOFFS = ["10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "14:00", "15:00"]
STOPS = ["or_mid", "or_pct:0.25", "or_pct:0.33", "or_pct:0.5", "or_pct:0.67", "or_pct:0.75", "or_opposite", "or_pct:1",
         "atr:0.5", "atr:0.75", "atr:1.0", "atr:1.5", "atr:2.0"]
ORB_STARTS = ["09:00", "09:15", "09:30", "09:45", "10:00"]

AXES = {
    "range_minutes": "Opening-range length (min)",
    "entry_tf": "Entry timeframe (min)",
    "target_r": "Target (R)",
    "stop": "Stop placement",
    "cutoff": "Last entry time (ET)",
    "entry_method": "Entry type",
    "confirmation": "Breakout confirmation",
    "entry_buffer_ticks": "Order buffer (ticks)",
    "direction": "Direction",
    "orb_start": "Opening-range start (ET)",
}
ORDERED = {"range_minutes", "entry_tf", "target_r", "cutoff", "entry_buffer_ticks", "orb_start", "stop"}
METRICS = ("n_trades", "avg_r", "t_stat_r", "sharpe", "profit_factor", "net_pnl", "win_rate", "max_dd")


def axis_values(axis: str, base_minutes: int = 1) -> list:
    if axis == "range_minutes":
        return [r for r in RANGES if r % base_minutes == 0]
    if axis == "entry_tf":
        return [t for t in ENTRY_TFS if t % base_minutes == 0]
    if axis == "target_r":
        return TARGETS
    if axis == "cutoff":
        return CUTOFFS
    if axis == "stop":
        return STOPS
    if axis == "entry_method":
        return ["market", "limit", "stop"]
    if axis == "confirmation":
        return list(CONFIRMATION_PRESETS)
    if axis == "entry_buffer_ticks":
        return [0, 1, 2, 3, 4]
    if axis == "direction":
        return ["both", "long", "short"]
    if axis == "orb_start":
        return [s for s in ORB_STARTS if (int(s[:2]) * 60 + int(s[3:])) % base_minutes == 0]
    raise KeyError(axis)


def axis_value_of(params: StrategyParams, axis: str):
    if axis == "stop":
        return stop_label(params.stop_method, params.stop_param)
    if axis == "confirmation":
        return confirmation_label(params.confirm_ticks, params.confirm_or_frac)
    return getattr(params, axis)


def with_value(params: StrategyParams, axis: str, value) -> StrategyParams:
    if axis == "stop":
        method, p = parse_stop(str(value))
        return replace(params, stop_method=method, stop_param=float(p))
    if axis == "confirmation":
        ticks, frac = CONFIRMATION_PRESETS[value]
        return replace(params, confirm_ticks=ticks, confirm_or_frac=frac)
    if axis == "entry_method" and value != "stop" and params.entry_tf == 0:
        return replace(params, entry_method=value, entry_tf=5)
    return replace(params, **{axis: value})


def values_with_candidate(axis: str, candidate: StrategyParams, base_minutes: int = 1) -> list:
    """Axis values, with the candidate's own value inserted if it is not on the standard list."""
    vals = list(axis_values(axis, base_minutes))
    cv = axis_value_of(candidate, axis)
    if not any(_same(v, cv) for v in vals):
        vals.append(cv)
        if axis in ("range_minutes", "entry_tf", "target_r", "entry_buffer_ticks", "cutoff", "orb_start"):
            vals = sorted(vals, key=lambda v: (float(v) if not isinstance(v, str) else v))
    return vals


def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            return False
    return str(a) == str(b)


def _evaluate(prep, params, execution, d_lo, d_hi) -> dict | None:
    try:
        params.validate(prep.base_minutes)
    except (ParamError, ValueError):
        return None
    m = evaluate_config(prep, params, execution, d_lo, d_hi)
    return {k: m.get(k, np.nan) for k in METRICS}


def one_at_a_time(prep: PreparedData, candidate: StrategyParams, execution, axes: list[str], start=None, end=None,
                  progress: Callable[[int, int], None] | None = None) -> pd.DataFrame:
    d_lo, d_hi = prep.day_range(start, end)
    base = _evaluate(prep, candidate, execution, d_lo, d_hi) or {}
    jobs = [(axis, v) for axis in axes for v in values_with_candidate(axis, candidate, prep.base_minutes)]
    rows = []
    for k, (axis, v) in enumerate(jobs):
        is_cand = _same(v, axis_value_of(candidate, axis))
        m = base if is_cand else _evaluate(prep, with_value(candidate, axis, v), execution, d_lo, d_hi)
        if progress:
            progress(k + 1, len(jobs))
        if m is None:
            continue
        rows.append({"axis": axis, "value": str(v), "is_candidate": is_cand, **m})
    table = pd.DataFrame(rows)
    if len(table):
        for col in ("avg_r", "sharpe", "net_pnl"):
            table[f"{col}_vs_candidate"] = table[col] - base.get(col, np.nan)
    return table


def axis_summary(table: pd.DataFrame, candidate: StrategyParams) -> pd.DataFrame:
    """Per axis: immediate neighbours of the candidate value and a plateau / spike classification."""
    rows = []
    for axis, grp in table.groupby("axis", sort=False):
        grp = grp.reset_index(drop=True)
        pos = grp.index[grp["is_candidate"]].tolist()
        if not pos:
            continue
        i = pos[0]
        c = grp.loc[i, "avg_r"]
        if axis in ORDERED:
            nb = grp.loc[[j for j in (i - 1, i + 1) if 0 <= j < len(grp)]]
        else:
            nb = grp.drop(index=i)
        nb_exp = nb["avg_r"].to_numpy()
        med = float(np.median(nb_exp)) if len(nb_exp) else np.nan
        if len(nb_exp) == 0:
            cls = "no neighbours"
        elif c <= 0:
            cls = "candidate not positive"
        elif np.all(nb_exp > 0) and med >= 0.5 * c:
            cls = "plateau"
        elif med <= 0.25 * c or med <= 0:
            cls = "spike"
        else:
            cls = "mixed"
        rows.append({"axis": axis, "label": AXES.get(axis, axis), "candidate_value": grp.loc[i, "value"], "candidate_exp_r": c,
                     "neighbours": ", ".join(nb["value"].tolist()), "neighbour_median_exp_r": med,
                     "neighbour_min_exp_r": float(np.min(nb_exp)) if len(nb_exp) else np.nan,
                     "share_of_range_positive": float((grp["avg_r"] > 0).mean()), "classification": cls})
    return pd.DataFrame(rows)


def overall_verdict(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "NOT EVALUATED"
    classes = summary["classification"].tolist()
    if "candidate not positive" in classes:
        return "NOT POSITIVE"
    if "spike" in classes:
        return "SPIKE (fragile: at least one parameter's neighbours collapse)"
    if all(c in ("plateau", "no neighbours") for c in classes):
        return "PLATEAU (neighbouring settings behave similarly)"
    return "MIXED"


def pair_grid(prep: PreparedData, candidate: StrategyParams, execution, x_axis: str, y_axis: str, start=None, end=None,
              progress: Callable[[int, int], None] | None = None) -> pd.DataFrame:
    d_lo, d_hi = prep.day_range(start, end)
    xs = values_with_candidate(x_axis, candidate, prep.base_minutes)
    ys = values_with_candidate(y_axis, candidate, prep.base_minutes)
    rows = []
    total = len(xs) * len(ys)
    k = 0
    for y in ys:
        for x in xs:
            k += 1
            p = with_value(with_value(candidate, y_axis, y), x_axis, x)
            m = _evaluate(prep, p, execution, d_lo, d_hi)
            if progress:
                progress(k, total)
            if m is not None:
                rows.append({x_axis: str(x), y_axis: str(y), **m,
                             "is_candidate": _same(x, axis_value_of(candidate, x_axis)) and _same(y, axis_value_of(candidate, y_axis))})
    return pd.DataFrame(rows)

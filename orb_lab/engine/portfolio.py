"""Position sizing and equity accounting.

The simulation kernel produces **per-contract** trade outcomes. Sizing is applied afterwards because it
never changes *which* trades occur (one instrument, non-overlapping trades) — only their quantity.

* ``fixed_contracts``: constant quantity; returns are measured against the running equity so that
  results are comparable without compounding distortions of position size.
* ``fixed_risk``: ``qty = floor(risk_dollars / risk_per_contract)``.
* ``pct_equity``: ``qty = floor(equity_before * risk_pct / risk_per_contract)``.

``risk_per_contract`` is the distance from the actual entry fill to the initial stop, in dollars.
A trade whose risk is too large for even ``min_qty`` contracts is **skipped** (recorded as
``sized_out``) rather than traded at an impossible fractional size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .execution import Costs
from .instruments import Instrument
from .params import SizingParams


def apply_sizing(trades: pd.DataFrame, sizing: SizingParams, instrument: Instrument, costs: Costs) -> pd.DataFrame:
    """Add quantity, dollar P&L and equity columns to a per-contract trade log."""
    sizing.validate()
    out = trades.copy()
    n = len(out)
    qty = np.zeros(n)
    equity_before = np.zeros(n)
    equity_after = np.zeros(n)
    sized_out = np.zeros(n, dtype=bool)
    equity = float(sizing.starting_equity)
    risk_pc = out["risk_per_contract"].to_numpy() if n else np.zeros(0)
    net_pc = out["net_pnl_pc"].to_numpy() if n else np.zeros(0)
    step = float(instrument.qty_step)
    min_qty = float(instrument.min_qty)
    for k in range(n):
        equity_before[k] = equity
        if sizing.mode == "fixed_contracts":
            q = float(sizing.contracts)
        else:
            budget = sizing.risk_dollars if sizing.mode == "fixed_risk" else equity * sizing.risk_pct
            q = np.floor((budget / risk_pc[k]) / step + 1e-9) * step if risk_pc[k] > 0 else 0.0
        q = min(q, float(sizing.max_contracts))
        if q < min_qty - 1e-12 or equity <= 0:
            q = 0.0
            sized_out[k] = True
        qty[k] = q
        equity += q * net_pc[k]
        equity_after[k] = equity
    out["qty"] = qty
    out["sized_out"] = sized_out
    out["gross_pnl"] = out["gross_pnl_pc"] * qty
    out["slippage_cost"] = out["slippage_pc"] * qty
    out["commission"] = out["commission_pc"] * qty
    out["net_pnl"] = out["net_pnl_pc"] * qty
    out["risk_dollars"] = out["risk_per_contract"] * qty
    out["equity_before"] = equity_before
    out["equity_after"] = equity_after
    return out


def daily_equity(trades: pd.DataFrame, session_dates: np.ndarray, starting_equity: float) -> pd.DataFrame:
    """Daily P&L/equity over *every* eligible session in the tested range (flat days included)."""
    index = pd.DatetimeIndex(np.asarray(session_dates, dtype="datetime64[D]").astype("datetime64[ns]"), name="session_date")
    frame = pd.DataFrame(index=index)
    if len(trades):
        executed = trades[~trades["sized_out"]] if "sized_out" in trades else trades
        by_day = executed.groupby("session_date")
        for col in ("net_pnl", "gross_pnl", "commission", "slippage_cost"):
            frame[col] = by_day[col].sum().reindex(index).fillna(0.0)
        frame["n_trades"] = by_day.size().reindex(index).fillna(0).astype(int)
    else:
        for col in ("net_pnl", "gross_pnl", "commission", "slippage_cost"):
            frame[col] = 0.0
        frame["n_trades"] = 0
    for kind in ("net", "gross"):
        equity = starting_equity + frame[f"{kind}_pnl"].cumsum()
        prev = equity.shift(1).fillna(starting_equity)
        frame[f"{kind}_equity"] = equity
        frame[f"{kind}_return"] = np.where(prev > 0, frame[f"{kind}_pnl"] / prev, 0.0)
        frame[f"{kind}_drawdown"] = equity / np.maximum(equity.cummax(), starting_equity) - 1.0
    return frame

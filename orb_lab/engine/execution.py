"""Execution model: order-fill rules, cost resolution and the integer codes shared with numba kernels.

All kernel prices are in **tick units** (price / tick_size), so "1 tick" is exactly 1.0.

Fill rules (RESEARCH_NOTES.md A-08 .. A-14):

* Market orders fill at the open of the first base bar at/after the decision time, plus
  ``slippage_ticks`` against the trader.
* Stop orders (stop entries, protective stops) fill at the stop price plus slippage; if the bar opens
  beyond the stop (gap) they fill at the open plus slippage.
* Limit orders (limit entries, profit targets) never receive slippage but, under the
  ``conservative`` fill model, require the market to trade **through** the limit by one tick.
  ``standard`` accepts a touch.
* Same-bar ambiguity (stop and target both inside one bar) is resolved by the ``ambiguity`` mode:
  ``optimistic`` -> target first; ``pessimistic`` -> stop first; ``conservative`` -> stop first *and*
  a position filled intrabar cannot reach its target on the fill bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .instruments import Instrument
from .params import ExecutionParams

EPS = 1e-7

# entry methods
EM_MARKET, EM_LIMIT, EM_STOP = 0, 1, 2
ENTRY_CODES = {"market": EM_MARKET, "limit": EM_LIMIT, "stop": EM_STOP}
# stop methods
SM_OR_OPPOSITE, SM_OR_MID, SM_OR_PCT, SM_ATR, SM_FIXED = 0, 1, 2, 3, 4
STOP_CODES = {"or_opposite": SM_OR_OPPOSITE, "or_mid": SM_OR_MID, "or_pct": SM_OR_PCT, "atr": SM_ATR, "fixed_ticks": SM_FIXED}
# direction
DIR_BOTH, DIR_LONG, DIR_SHORT = 0, 1, 2
DIRECTION_CODES = {"both": DIR_BOTH, "long": DIR_LONG, "short": DIR_SHORT}
# re-entry
RE_ANY, RE_OPPOSITE_AFTER_LOSS, RE_SAME_AFTER_LOSS, RE_EITHER_AFTER_LOSS = 0, 1, 2, 3
REENTRY_CODES = {
    "any": RE_ANY,
    "opposite_after_loss": RE_OPPOSITE_AFTER_LOSS,
    "same_after_loss": RE_SAME_AFTER_LOSS,
    "either_after_loss": RE_EITHER_AFTER_LOSS,
}
# fills / ambiguity
FILL_STANDARD, FILL_CONSERVATIVE = 0, 1
FILL_CODES = {"standard": FILL_STANDARD, "conservative": FILL_CONSERVATIVE}
AMB_OPTIMISTIC, AMB_PESSIMISTIC, AMB_CONSERVATIVE = 0, 1, 2
AMBIGUITY_CODES = {"optimistic": AMB_OPTIMISTIC, "pessimistic": AMB_PESSIMISTIC, "conservative": AMB_CONSERVATIVE}
# exit reasons
EXIT_STOP, EXIT_TARGET, EXIT_TIME, EXIT_DATA_END, EXIT_BREAKEVEN = 1, 2, 3, 4, 5
EXIT_NAMES = {EXIT_STOP: "stop", EXIT_TARGET: "target", EXIT_TIME: "time", EXIT_DATA_END: "data_end", EXIT_BREAKEVEN: "breakeven_stop"}
# day status
DAY_NO_SIGNAL, DAY_TRADED, DAY_BAD_OR, DAY_FILTERED, DAY_NO_ATR = 0, 1, 2, 3, 4
DAY_NOT_FILLED, DAY_AMBIGUOUS_ENTRY, DAY_INVALID_STOP, DAY_NO_TIME = 5, 6, 7, 8
DAY_STATUS_NAMES = {
    DAY_NO_SIGNAL: "no_signal",
    DAY_TRADED: "traded",
    DAY_BAD_OR: "invalid_or_insufficient_range_data",
    DAY_FILTERED: "filtered_or_atr",
    DAY_NO_ATR: "atr_unavailable",
    DAY_NOT_FILLED: "signal_not_filled",
    DAY_AMBIGUOUS_ENTRY: "ambiguous_both_sides_entry",
    DAY_INVALID_STOP: "invalid_stop",
    DAY_NO_TIME: "cutoff_before_range_end",
}


@dataclass(frozen=True)
class Costs:
    slippage_ticks: float
    commission_per_side: float
    exchange_fee_per_side: float

    @property
    def fixed_per_side(self) -> float:
        return self.commission_per_side + self.exchange_fee_per_side

    @property
    def round_trip_fixed(self) -> float:
        return 2.0 * self.fixed_per_side


def resolve_costs(instrument: Instrument, execution: ExecutionParams) -> Costs:
    if execution.frictionless:
        return Costs(0.0, 0.0, 0.0)
    return Costs(
        slippage_ticks=float(instrument.slippage_ticks if execution.slippage_ticks is None else execution.slippage_ticks),
        commission_per_side=float(
            instrument.commission_per_side if execution.commission_per_side is None else execution.commission_per_side
        ),
        exchange_fee_per_side=float(
            instrument.exchange_fee_per_side if execution.exchange_fee_per_side is None else execution.exchange_fee_per_side
        ),
    )


@njit(cache=True, inline="always")
def compute_stop_target(sig, fill, or_hi, or_lo, width, atr_ticks, stop_method, stop_param, target_r):
    """Protective stop and target (tick units) for a new position.

    Returns ``(stop, target, has_target, risk, ok)``. Rounding is *away from the entry* for both stop and
    target (``floor`` below a long, ``ceil`` above a short), so risk is never understated and targets are
    never easier than the nominal R multiple.
    """
    if stop_method == SM_OR_OPPOSITE:
        raw = or_lo if sig > 0 else or_hi
    elif stop_method == SM_OR_MID:
        raw = 0.5 * (or_hi + or_lo)
    elif stop_method == SM_OR_PCT:
        raw = fill - sig * stop_param * width
    elif stop_method == SM_ATR:
        raw = fill - sig * stop_param * atr_ticks
    else:
        raw = fill - sig * stop_param
    # normalised coordinates: long-equivalent
    stop_n = np.floor(sig * raw + EPS)
    fill_n = sig * fill
    risk = fill_n - stop_n
    if not (risk >= 1.0 - EPS):
        return 0.0, 0.0, False, risk, False
    if target_r > 0.0:
        tgt_n = np.ceil(fill_n + target_r * risk - EPS)
        return sig * stop_n, sig * tgt_n, True, risk, True
    return sig * stop_n, 0.0, False, risk, True


@njit(cache=True, inline="always")
def reentry_allowed(direction_sig, n_today, last_dir, last_loss, reentry):
    if n_today == 0:
        return True
    if reentry == RE_ANY:
        return True
    if not last_loss:
        return False
    if reentry == RE_OPPOSITE_AFTER_LOSS:
        return direction_sig == -last_dir
    if reentry == RE_SAME_AFTER_LOSS:
        return direction_sig == last_dir
    return True

"""Opening-range and daily-indicator computations.

No-lookahead contract:

* The opening range for session *d* uses only base bars whose **start** time lies in
  ``[orb_start, orb_start + range_minutes)``. The range is therefore known at ``orb_start + range_minutes``
  and no trading decision may occur before that time (enforced by the simulation kernel).
* ``daily_atr`` for session *d* is computed from sessions strictly before *d* (it is shifted by one
  session), so it is known before session *d* opens.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def compute_opening_ranges(day_start, day_end, tod, high, low, start_min, length_min, base_min, min_coverage):
    """Per-session OR high/low (tick units), coverage fraction and validity flag."""
    n_days = day_start.shape[0]
    or_hi = np.full(n_days, np.nan)
    or_lo = np.full(n_days, np.nan)
    coverage = np.zeros(n_days)
    ok = np.zeros(n_days, dtype=np.bool_)
    end_min = start_min + length_min
    expected = length_min // base_min
    for d in range(n_days):
        hi = -np.inf
        lo = np.inf
        count = 0
        for i in range(day_start[d], day_end[d]):
            t = tod[i]
            if t < start_min:
                continue
            if t >= end_min:
                break
            if high[i] > hi:
                hi = high[i]
            if low[i] < lo:
                lo = low[i]
            count += 1
        if count > 0:
            or_hi[d] = hi
            or_lo[d] = lo
        cov = count / expected if expected > 0 else 0.0
        coverage[d] = cov
        ok[d] = count > 0 and cov >= min_coverage - 1e-12
    return or_hi, or_lo, coverage, ok


def wilder_atr_known_before(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder ATR where element *d* only uses sessions ``< d`` (NaN until enough history).

    True range of session *k* is ``max(H-L, |H-C[k-1]|, |L-C[k-1]|)``; the first session uses ``H-L``.
    The Wilder average is seeded with the simple mean of the first ``period`` true ranges.
    """
    n = len(high)
    out = np.full(n, np.nan)
    if period < 1 or n == 0:
        return out
    prev_close = np.concatenate(([np.nan], close[:-1]))
    tr = np.maximum.reduce(
        [high - low, np.abs(high - np.nan_to_num(prev_close, nan=high)), np.abs(low - np.nan_to_num(prev_close, nan=low))]
    )
    atr_through = np.full(n, np.nan)
    if n >= period:
        atr_through[period - 1] = tr[:period].mean()
        for k in range(period, n):
            atr_through[k] = (atr_through[k - 1] * (period - 1) + tr[k]) / period
    out[1:] = atr_through[:-1]
    return out


def trailing_percentile_rank(values: np.ndarray, lookback: int, min_history: int) -> np.ndarray:
    """Percentile rank (0..1) of ``values[d]`` within ``values[d-lookback:d]`` (strictly prior data).

    Used for historical-only regime/percentile filters. NaN until ``min_history`` prior values exist.
    """
    n = len(values)
    out = np.full(n, np.nan)
    for d in range(n):
        lo = max(0, d - lookback)
        window = values[lo:d]
        window = window[~np.isnan(window)]
        if len(window) < min_history or np.isnan(values[d]):
            continue
        out[d] = float(np.mean(window < values[d]) + 0.5 * np.mean(window == values[d]))
    return out

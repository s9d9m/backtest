"""Precomputed, per-session arrays consumed by the simulation kernels.

``prepare_data`` does all the expensive, parameter-independent work once:

* assigns every bar to a trading session and computes its ET wall-clock ``tod``
* decides which sessions are tradable (calendar session, has data, no intraday contract change, ...)
* builds daily (full Globex session) OHLC used for ATR and other prior-day indicators
* flattens the intraday analysis window (``analysis_window_start`` .. ``session_close``) into
  contiguous numpy arrays in **tick units** with ``day_start``/``day_end`` offsets

Opening ranges and ATRs are cached per ``(start, length)`` / ``period`` so a grid search never
recomputes the same indicator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .instruments import Instrument
from .orb import compute_opening_ranges, wilder_atr_known_before
from .sessions import TradingCalendar, assign_sessions, hhmm_to_minutes, session_bar_timestamp


@dataclass
class PreparedData:
    instrument: Instrument
    base_minutes: int
    dates: np.ndarray  # datetime64[D], eligible sessions in order
    close_min: np.ndarray  # calendar close (ET minutes) per session
    early_close: np.ndarray  # bool
    session_exit_min: np.ndarray  # latest allowed exit per session (ET minutes)
    day_start: np.ndarray  # int64 offsets into the flattened window arrays
    day_end: np.ndarray
    tod: np.ndarray  # int32
    open: np.ndarray  # float64, tick units
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    daily: pd.DataFrame  # per eligible session: full-session OHLC etc. (tick units)
    excluded: pd.DataFrame  # sessions not tradable, with reason
    data_hash: str = ""
    min_or_coverage: float = 0.8
    _or_cache: dict = field(default_factory=dict, repr=False)
    _atr_cache: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ basics
    @property
    def n_days(self) -> int:
        return len(self.dates)

    @property
    def tick_size(self) -> float:
        return self.instrument.tick_size

    def day_range(self, start=None, end=None) -> tuple[int, int]:
        """Half-open index range ``[d_lo, d_hi)`` of sessions with ``start <= date <= end``."""
        d_lo = 0 if start is None else int(np.searchsorted(self.dates, np.datetime64(pd.Timestamp(start).date(), "D"), "left"))
        d_hi = self.n_days if end is None else int(np.searchsorted(self.dates, np.datetime64(pd.Timestamp(end).date(), "D"), "right"))
        return d_lo, max(d_lo, d_hi)

    # --------------------------------------------------------------- indicators
    def opening_range(self, start_min: int, length_min: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        key = (int(start_min), int(length_min))
        if key not in self._or_cache:
            self._or_cache[key] = compute_opening_ranges(
                self.day_start,
                self.day_end,
                self.tod,
                self.high,
                self.low,
                np.int64(start_min),
                np.int64(length_min),
                np.int64(self.base_minutes),
                float(self.min_or_coverage),
            )
        return self._or_cache[key]

    def atr(self, period: int) -> np.ndarray:
        """Daily Wilder ATR (tick units) known before each session opens; NaN if unavailable."""
        if period <= 0:
            return np.full(self.n_days, np.nan)
        if period not in self._atr_cache:
            self._atr_cache[period] = wilder_atr_known_before(
                self.daily["high"].to_numpy(),
                self.daily["low"].to_numpy(),
                self.daily["close"].to_numpy(),
                int(period),
                prev_close=self.daily["prev_close"].to_numpy(),
            )
        return self._atr_cache[period]

    def timestamps(self, day_idx: np.ndarray, tod: np.ndarray) -> pd.DatetimeIndex:
        return session_bar_timestamp(self.dates[np.asarray(day_idx, dtype=np.int64)], tod, self.instrument.timezone)

    def clear_caches(self) -> None:
        self._or_cache.clear()
        self._atr_cache.clear()


def prepare_data(
    bars: pd.DataFrame,
    instrument: Instrument,
    *,
    base_minutes: int = 1,
    calendar: TradingCalendar | None = None,
    min_or_coverage: float = 0.8,
    exclude_early_close: bool = False,
    data_hash: str = "",
) -> PreparedData:
    """Build :class:`PreparedData` from clean, sorted, UTC-indexed bars (see ``data_quality.check_and_clean``)."""
    if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
        raise ValueError("bars must be sorted and de-duplicated; run data_quality.check_and_clean first")
    calendar = calendar or TradingCalendar(instrument.calendar)
    tick = instrument.tick_size
    session_date, tod = assign_sessions(bars.index, instrument.session_open, instrument.timezone)
    close_limit = hhmm_to_minutes(instrument.session_close)

    work = pd.DataFrame(
        {
            "session_date": session_date,
            "tod": tod,
            "open": bars["open"].to_numpy() / tick,
            "high": bars["high"].to_numpy() / tick,
            "low": bars["low"].to_numpy() / tick,
            "close": bars["close"].to_numpy() / tick,
            "volume": bars["volume"].fillna(0).to_numpy(),
        }
    )
    if "contract" in bars.columns:
        work["contract"] = bars["contract"].to_numpy()
    # Bars after the Globex close but before the next open (maintenance window) are not part of any session.
    work = work[work["tod"] < close_limit]

    data_dates = pd.DatetimeIndex(np.unique(work["session_date"].to_numpy()))
    excluded_rows: list[dict] = []
    if len(data_dates) == 0:
        raise ValueError("No bars inside trading sessions")
    cal = calendar.sessions(data_dates.min(), data_dates.max())
    cal_idx = pd.DatetimeIndex(cal.index)

    eligible = data_dates[data_dates.isin(cal_idx)]
    for d in data_dates[~data_dates.isin(cal_idx)]:
        excluded_rows.append({"session_date": d, "reason": "non_calendar_session"})
    for d in cal_idx[~cal_idx.isin(data_dates)]:
        excluded_rows.append({"session_date": d, "reason": "no_data"})
    if "contract" in work.columns:
        n_contracts = work.groupby("session_date")["contract"].nunique()
        rolled = pd.DatetimeIndex(n_contracts[n_contracts > 1].index)
        for d in rolled:
            if d in eligible:
                excluded_rows.append({"session_date": d, "reason": "intraday_contract_change"})
        eligible = eligible[~eligible.isin(rolled)]
    if exclude_early_close:
        early = pd.DatetimeIndex(cal.index[cal["early_close"]])
        for d in eligible[eligible.isin(early)]:
            excluded_rows.append({"session_date": d, "reason": "early_close_excluded"})
        eligible = eligible[~eligible.isin(early)]

    work = work[pd.DatetimeIndex(work["session_date"]).isin(eligible)]

    # Daily (full-session) bars for prior-day indicators. Sorted by UTC time already.
    grouped = work.groupby("session_date", sort=True)
    daily = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "volume": grouped["volume"].sum(),
            "n_bars": grouped.size(),
        }
    )
    daily.index = pd.DatetimeIndex(daily.index)
    daily["prev_close"] = daily["close"].shift(1)
    daily["prev_range"] = (daily["high"] - daily["low"]).shift(1)

    window_start = hhmm_to_minutes(instrument.analysis_window_start)
    window = work[work["tod"] >= window_start]
    # keep only sessions with at least one bar in the window
    window_dates = pd.DatetimeIndex(np.unique(window["session_date"].to_numpy()))
    for d in eligible[~eligible.isin(window_dates)]:
        excluded_rows.append({"session_date": d, "reason": "no_intraday_window_data"})
    eligible = eligible[eligible.isin(window_dates)]
    daily = daily.loc[daily.index.isin(eligible)]
    window = window[pd.DatetimeIndex(window["session_date"]).isin(eligible)]
    # the prior-day columns must be recomputed after dropping sessions
    daily["prev_close"] = daily["close"].shift(1)
    daily["prev_range"] = (daily["high"] - daily["low"]).shift(1)
    if "contract" in window.columns and len(daily):
        # never compare prices of different contracts: the previous close is unknown on roll sessions
        session_contract = work.groupby("session_date")["contract"].first()
        session_contract.index = pd.DatetimeIndex(session_contract.index)
        daily["contract"] = session_contract.reindex(daily.index).to_numpy()
        rolled_in = daily["contract"].ne(daily["contract"].shift(1))
        rolled_in.iloc[0] = False
        daily.loc[rolled_in, "prev_close"] = np.nan
        daily["roll_session"] = rolled_in

    dates = eligible.to_numpy().astype("datetime64[D]")
    wdates = window["session_date"].to_numpy().astype("datetime64[D]")
    day_start = np.searchsorted(wdates, dates, "left").astype(np.int64)
    day_end = np.searchsorted(wdates, dates, "right").astype(np.int64)

    cal_e = cal.reindex(eligible)
    close_min = cal_e["close_min"].to_numpy().astype(np.int64)
    early_flags = cal_e["early_close"].to_numpy().astype(bool)
    session_exit = hhmm_to_minutes(instrument.session_exit)
    exit_min = np.minimum(session_exit, close_min - int(instrument.early_close_exit_buffer_min)).astype(np.int64)

    excluded = pd.DataFrame(excluded_rows, columns=["session_date", "reason"]).sort_values("session_date").reset_index(drop=True)
    return PreparedData(
        instrument=instrument,
        base_minutes=int(base_minutes),
        dates=dates,
        close_min=close_min,
        early_close=early_flags,
        session_exit_min=exit_min,
        day_start=day_start,
        day_end=day_end,
        tod=window["tod"].to_numpy().astype(np.int32),
        open=window["open"].to_numpy().astype(np.float64),
        high=window["high"].to_numpy().astype(np.float64),
        low=window["low"].to_numpy().astype(np.float64),
        close=window["close"].to_numpy().astype(np.float64),
        volume=window["volume"].to_numpy().astype(np.float64),
        daily=daily,
        excluded=excluded,
        data_hash=data_hash,
        min_or_coverage=float(min_or_coverage),
    )

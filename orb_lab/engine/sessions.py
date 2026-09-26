"""Timezone, trading-session and exchange-calendar handling.

Conventions (see RESEARCH_NOTES.md A-01 .. A-06):

* Internally every bar is stored with a tz-aware **UTC** timestamp marking the **start** of the bar.
* Strategy clocks are **America/New_York wall-clock** times. DST is handled by ``zoneinfo`` via pandas,
  so "09:30" means 09:30 EST in winter and 09:30 EDT in summer.
* A bar belongs to a **trading session** (not a calendar date). For CME Globex products the session for
  date D runs from D-1 18:00 ET to D 17:00 ET, so a bar at Sunday 18:05 ET belongs to Monday's session.
* ``tod`` ("time of day") is the bar-start wall-clock time in minutes after midnight ET **of the session
  date**. Bars from the previous evening therefore have negative ``tod`` (18:00 the day before -> -360).
* Trading-day eligibility (holidays, early closes) comes from an exchange calendar. The default is the
  NYSE calendar (XNYS) because the research anchor is the U.S. cash-equity open.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

ET_TZ = "America/New_York"
MINUTES_PER_DAY = 1440


def hhmm_to_minutes(value: str | int) -> int:
    """Convert ``"09:30"`` to 570. Integers are returned unchanged (already minutes)."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Time must be HH:MM, got {value!r}")
    hours, minutes = int(parts[0]), int(parts[1])
    if not (0 <= hours <= 23 and 0 <= minutes <= 59):
        raise ValueError(f"Invalid time {value!r}")
    return hours * 60 + minutes


def minutes_to_hhmm(minutes: int) -> str:
    minutes = int(minutes)
    sign = "-" if minutes < 0 else ""
    minutes = abs(minutes)
    return f"{sign}{minutes // 60:02d}:{minutes % 60:02d}"


def to_utc_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        raise ValueError("Timestamps are timezone-naive; declare the source timezone explicitly.")
    return index.tz_convert("UTC")


def assign_sessions(
    ts_utc: pd.DatetimeIndex, session_open: str = "18:00", tz: str = ET_TZ
) -> tuple[np.ndarray, np.ndarray]:
    """Map UTC bar-start timestamps to (session_date, tod).

    Returns
    -------
    session_date : ndarray[datetime64[D]]
        Trading-session date each bar belongs to.
    tod : ndarray[int32]
        Wall-clock minutes after midnight (local ``tz``) of the session date; negative for bars from
        the previous evening.
    """
    local = to_utc_index(ts_utc).tz_convert(tz)
    wall = (local.hour * 60 + local.minute).to_numpy().astype(np.int32)
    local_date = local.tz_localize(None).normalize().to_numpy().astype("datetime64[D]")
    open_min = hhmm_to_minutes(session_open)
    rolled = wall >= open_min
    session_date = local_date + rolled.astype("timedelta64[D]")
    tod = np.where(rolled, wall - MINUTES_PER_DAY, wall).astype(np.int32)
    return session_date, tod


def session_bar_timestamp(session_dates: np.ndarray, tod: np.ndarray, tz: str = ET_TZ) -> pd.DatetimeIndex:
    """Inverse of :func:`assign_sessions`: wall-clock ET timestamps for (session_date, tod) pairs.

    Only unambiguous for daytime ``tod`` (the DST fall-back hour is 01:00-02:00 ET, outside the
    intraday analysis window).
    """
    naive = pd.DatetimeIndex(np.asarray(session_dates, dtype="datetime64[D]").astype("datetime64[ns]")) + pd.to_timedelta(
        np.asarray(tod, dtype=np.int64), unit="min"
    )
    return naive.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")


@dataclass(frozen=True)
class SessionInfo:
    """Calendar facts for one trading date (ET wall-clock minutes)."""

    date: date
    open_min: int
    close_min: int
    early_close: bool


class TradingCalendar:
    """Thin wrapper around ``exchange_calendars`` with a weekday-only fallback.

    ``name="WEEKDAYS"`` treats every Monday-Friday as a full 09:30-16:00 session (useful for synthetic
    data and instruments without a calendar). Any other name is looked up in ``exchange_calendars``.
    """

    def __init__(self, name: str = "XNYS"):
        self.name = name

    def sessions(self, start: date | str, end: date | str) -> pd.DataFrame:
        """Return a DataFrame indexed by session date with open/close ET minutes and an early-close flag."""
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        if end_ts < start_ts:
            return pd.DataFrame(columns=["open_min", "close_min", "early_close"])
        if self.name.upper() == "WEEKDAYS":
            days = pd.bdate_range(start_ts, end_ts)
            frame = pd.DataFrame(index=days)
            frame["open_min"] = 570
            frame["close_min"] = 960
            frame["early_close"] = False
            frame.index.name = "session_date"
            return frame
        schedule = _schedule(self.name, start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
        opens = schedule["open"].dt.tz_convert(ET_TZ)
        closes = schedule["close"].dt.tz_convert(ET_TZ)
        frame = pd.DataFrame(index=pd.DatetimeIndex(schedule.index.tz_localize(None) if schedule.index.tz else schedule.index))
        frame["open_min"] = (opens.dt.hour * 60 + opens.dt.minute).to_numpy()
        frame["close_min"] = (closes.dt.hour * 60 + closes.dt.minute).to_numpy()
        regular_close = int(pd.Series(frame["close_min"]).mode().iloc[0]) if len(frame) else 960
        frame["early_close"] = frame["close_min"] < regular_close
        frame.index.name = "session_date"
        return frame


@lru_cache(maxsize=32)
def _schedule(name: str, start: str, end: str) -> pd.DataFrame:
    import exchange_calendars as xcals

    cal_start = pd.Timestamp(start) - pd.Timedelta(days=7)
    cal_end = pd.Timestamp(end) + pd.Timedelta(days=7)
    calendar = xcals.get_calendar(name, start=cal_start.strftime("%Y-%m-%d"), end=cal_end.strftime("%Y-%m-%d"))
    schedule = calendar.schedule
    mask = (schedule.index >= pd.Timestamp(start)) & (schedule.index <= pd.Timestamp(end))
    return schedule.loc[mask]

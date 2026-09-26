import numpy as np
import pandas as pd

from orb_lab.engine.sessions import TradingCalendar, assign_sessions, hhmm_to_minutes, session_bar_timestamp


def utc(*stamps):
    return pd.DatetimeIndex(pd.to_datetime(list(stamps), utc=True))


def test_hhmm_parsing():
    assert hhmm_to_minutes("09:30") == 570
    assert hhmm_to_minutes("00:00") == 0
    assert hhmm_to_minutes(600) == 600


def test_0930_et_maps_to_different_utc_across_dst():
    # 2024-03-08 is EST (UTC-5); 2024-03-11 is EDT (UTC-4). DST began 2024-03-10.
    idx = utc("2024-03-08 14:30", "2024-03-11 13:30", "2024-11-01 13:30", "2024-11-04 14:30")
    dates, tod = assign_sessions(idx)
    assert list(tod) == [570, 570, 570, 570]
    assert [str(d) for d in dates] == ["2024-03-08", "2024-03-11", "2024-11-01", "2024-11-04"]


def test_naive_utc_offset_mistake_is_visible():
    # 14:30 UTC in summer is 10:30 EDT, not 09:30.
    _, tod = assign_sessions(utc("2024-07-01 14:30"))
    assert tod[0] == 630


def test_globex_evening_bars_belong_to_next_session():
    # Sunday 2024-03-10 18:00 EDT -> Monday session; Monday 17:59 EDT would be Tuesday's session
    idx = utc("2024-03-10 22:00", "2024-03-11 20:59", "2024-03-11 22:00")
    dates, tod = assign_sessions(idx)
    assert [str(d) for d in dates] == ["2024-03-11", "2024-03-11", "2024-03-12"]
    assert list(tod) == [-360, 1019, -360]


def test_friday_evening_rolls_to_saturday_not_monday():
    # A (bad) bar on Friday 18:30 ET is assigned to Saturday, which no calendar trades -> excluded later.
    dates, _ = assign_sessions(utc("2024-03-08 23:30"))
    assert str(dates[0]) == "2024-03-09"


def test_session_bar_timestamp_roundtrip():
    idx = utc("2024-03-08 14:30", "2024-03-11 13:30")
    dates, tod = assign_sessions(idx)
    back = session_bar_timestamp(dates, tod)
    assert (back.tz_convert("UTC") == idx).all()


def test_xnys_holidays_and_early_close():
    cal = TradingCalendar("XNYS").sessions("2024-07-01", "2024-07-08")
    dates = [d.strftime("%Y-%m-%d") for d in cal.index]
    assert "2024-07-04" not in dates  # Independence Day
    assert "2024-07-06" not in dates  # Saturday
    assert cal.loc["2024-07-03", "early_close"]
    assert cal.loc["2024-07-03", "close_min"] == 13 * 60
    assert cal.loc["2024-07-05", "close_min"] == 16 * 60
    thanks = TradingCalendar("XNYS").sessions("2024-11-25", "2024-11-29")
    assert "2024-11-28" not in [d.strftime("%Y-%m-%d") for d in thanks.index]
    assert thanks.loc["2024-11-29", "early_close"]


def test_weekday_calendar():
    cal = TradingCalendar("WEEKDAYS").sessions("2024-03-08", "2024-03-12")
    assert [d.strftime("%a") for d in cal.index] == ["Fri", "Mon", "Tue"]
    assert (cal["close_min"] == 960).all()

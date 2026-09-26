"""Synthetic 1-minute futures data for validation.

Two uses:

1. **Null model** (``trend_strength=0``): a driftless random walk with realistic intraday volatility.
   No ORB variant can have a true edge here; after costs every configuration should lose roughly the
   transaction costs. A research pipeline that reports a robust edge on this data is broken or overfit.
2. **Planted edge** (``trend_strength>0``): on each session a hidden random direction adds drift
   between 09:30 and 12:00 ET totalling ``trend_strength`` daily standard deviations. The direction is unknowable in advance, but an ORB breakout tends to
   align with it, so a correct engine must detect a positive frictionless edge.

Bars are generated for every session of the chosen calendar, Globex hours (18:00 ET previous day to
17:00 ET), on the instrument's tick grid, with UTC bar-start timestamps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .instruments import Instrument
from .sessions import ET_TZ, TradingCalendar


def generate_bars(
    instrument: Instrument,
    start: str = "2020-01-01",
    end: str = "2021-12-31",
    *,
    seed: int = 7,
    start_price: float | None = None,
    daily_vol_ticks: float | None = None,
    trend_strength: float = 0.0,
    trend_start: str = "09:30",
    trend_end: str = "12:00",
    calendar: str = "XNYS",
    substeps: int = 32,
) -> pd.DataFrame:
    """Return a UTC-indexed OHLCV frame of 1-minute bars (bar-start timestamps)."""
    rng = np.random.default_rng(seed)
    tick = instrument.tick_size
    defaults = {"ES": (4000.0, 160.0), "NQ": (14000.0, 800.0), "GC": (1900.0, 200.0), "6E": (1.10, 140.0)}
    p0, vol0 = defaults.get(instrument.symbol, (100.0, 150.0))
    price_ticks = round((start_price or p0) / tick)
    daily_vol = daily_vol_ticks or vol0

    sessions = TradingCalendar(calendar).sessions(start, end)
    minutes = np.arange(-360, 1020)  # 18:00 prev day .. 16:59
    n_min = len(minutes)
    # intraday volatility profile: quiet overnight, spike at 09:30, U-shape through the day
    profile = np.full(n_min, 0.35)
    rth = (minutes >= 570) & (minutes < 960)
    x = (minutes[rth] - 570) / 390.0
    profile[rth] = 0.8 + 1.6 * np.exp(-x * 8) + 0.6 * x**2
    pre = (minutes >= 480) & (minutes < 570)
    profile[pre] = 0.6
    profile /= np.sqrt(np.mean(profile**2))
    sigma_min = daily_vol / np.sqrt(n_min)
    vol_profile = 1000 * profile**2

    t_start = int(trend_start[:2]) * 60 + int(trend_start[3:])
    t_end = int(trend_end[:2]) * 60 + int(trend_end[3:])
    trend_mask = (minutes >= t_start) & (minutes < t_end)

    n_trend = max(int(trend_mask.sum()), 1)
    frames = []
    price = float(price_ticks)
    for session_date, row in sessions.iterrows():
        close_min = int(row["close_min"])
        day_minutes = minutes[minutes < min(1020, close_min + 60)]
        m = len(day_minutes)
        regime = rng.lognormal(0.0, 0.25)
        scale = price / price_ticks  # geometric: volatility in ticks scales with the price level
        steps = rng.standard_normal((m, substeps)) * (scale * sigma_min * regime * profile[:m, None] / np.sqrt(substeps))
        if trend_strength > 0:
            direction = rng.choice([-1.0, 1.0])
            drift = trend_strength * daily_vol * scale * regime * direction / (n_trend * substeps)
            steps[trend_mask[:m]] += drift
        path = price + np.cumsum(steps.ravel())
        path = np.round(path).reshape(m, substeps)
        opens = np.concatenate(([round(price)], path[:-1, -1]))
        highs = np.maximum(path.max(axis=1), opens)
        lows = np.minimum(path.min(axis=1), opens)
        closes = path[:, -1]
        price = closes[-1]
        volume = rng.poisson(vol_profile[:m] * regime) + 1
        local = pd.Timestamp(session_date) + pd.to_timedelta(day_minutes, unit="min")
        ts = local.tz_localize(ET_TZ, ambiguous="NaT", nonexistent="NaT")
        frame = pd.DataFrame(
            {"open": opens * tick, "high": highs * tick, "low": lows * tick, "close": closes * tick, "volume": volume},
            index=ts,
        )
        frames.append(frame[~frame.index.isna()])
    bars = pd.concat(frames)
    bars.index = pd.DatetimeIndex(bars.index).tz_convert("UTC")
    bars.index.name = "ts"
    # drop the Sunday/holiday-eve evening bars that would fall on a weekend market closure day structure
    bars = bars[~bars.index.duplicated(keep="first")].sort_index()
    for col in ("open", "high", "low", "close"):
        bars[col] = np.round(bars[col] / tick) * tick
    return bars

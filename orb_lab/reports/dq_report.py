"""Per-instrument data-quality report for real market data.

On top of :func:`orb_lab.engine.data_quality.check_and_clean`, this verifies across the WHOLE history
that 09:30 in our data is 09:30 New York time:

* **Month-by-month volume timing.** The minute with the highest average volume (07:00-12:00 ET) must
  not jump by ~60 minutes relative to the instrument's usual peak, and the volume step-up at 09:30 ET
  (the U.S. cash open) must stay where it is. A one-hour displacement in some months is the
  signature of a timezone / DST error.
* **DST transitions.** For every U.S. DST change, the 09:30 step-up must appear at 09:30 ET both in
  the week before and in the week after.
* **Early closes.** Calendar early-close sessions must actually stop trading near the calendar close.
* **Year-by-year completeness.** Sessions, bars, missing RTH minutes, zero-volume bars, gaps.

Verdict:

* ``STOP``: unresolved DQ errors or timing misalignment. Research on the instrument must stop
  until this is explained.
* ``REVIEW``: warnings that need a human decision.
* ``PASS``: none of the above.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..engine.pipeline import Dataset
from ..engine.sessions import TradingCalendar, assign_sessions

OPEN_MIN = 570


@dataclass
class RealDataQuality:
    symbol: str
    verdict: str
    reasons: list[str]
    monthly_timing: pd.DataFrame
    dst_checks: pd.DataFrame
    early_close: pd.DataFrame
    yearly: pd.DataFrame
    summary: dict = field(default_factory=dict)


def _minute_frame(bars: pd.DataFrame, instrument) -> pd.DataFrame:
    et = bars.index.tz_convert(instrument.timezone)
    session_date, tod = assign_sessions(bars.index, instrument.session_open, instrument.timezone)
    return pd.DataFrame(
        {"session": pd.to_datetime(session_date), "tod": tod, "volume": bars["volume"].fillna(0).to_numpy(),
         "month": et.tz_localize(None).to_period("M").astype(str)},
        index=bars.index,
    )


def monthly_timing(frame: pd.DataFrame) -> pd.DataFrame:
    day = frame[(frame["tod"] >= 420) & (frame["tod"] < 720)]
    prof = day.groupby(["month", "tod"])["volume"].mean().unstack("tod").fillna(0.0)
    rows = []
    for month, row in prof.iterrows():
        pre = row.reindex(range(OPEN_MIN - 10, OPEN_MIN), fill_value=0).mean()
        post = row.reindex(range(OPEN_MIN, OPEN_MIN + 10), fill_value=0).mean()
        pre_shift = row.reindex(range(OPEN_MIN + 50, OPEN_MIN + 60), fill_value=0).mean()
        post_shift = row.reindex(range(OPEN_MIN + 60, OPEN_MIN + 70), fill_value=0).mean()
        early_pre = row.reindex(range(OPEN_MIN - 70, OPEN_MIN - 60), fill_value=0).mean()
        early_post = row.reindex(range(OPEN_MIN - 60, OPEN_MIN - 50), fill_value=0).mean()
        rows.append(
            {
                "month": month,
                "peak_minute": int(row.idxmax()) if row.max() > 0 else -1,
                "step_0930": post / pre if pre > 0 else np.nan,
                "step_1030": post_shift / pre_shift if pre_shift > 0 else np.nan,
                "step_0830": early_post / early_pre if early_pre > 0 else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    modal_peak = int(out["peak_minute"].mode().iloc[0])
    median_step = float(out["step_0930"].median())
    out["peak_shift_min"] = out["peak_minute"] - modal_peak
    # a month is misaligned if its peak moved by ~an hour, or its open step-up appears an hour away instead of at 09:30
    moved = out["peak_shift_min"].abs().between(55, 65)
    lost_step = (median_step > 1.5) & (out["step_0930"] < 1.0 + 0.25 * (median_step - 1.0)) & (
        (out["step_1030"] > 1.0 + 0.5 * (median_step - 1.0)) | (out["step_0830"] > 1.0 + 0.5 * (median_step - 1.0))
    )
    out["flag"] = np.where(moved | lost_step, "MISALIGNED?", "")
    out.attrs["modal_peak"] = modal_peak
    out.attrs["median_step_0930"] = median_step
    return out


def dst_transition_checks(frame: pd.DataFrame, years: range) -> pd.DataFrame:
    rows = []
    for year in years:
        for label, when in (("spring", pd.Timestamp(f"{year}-03-01")), ("fall", pd.Timestamp(f"{year}-11-01"))):
            # second Sunday of March / first Sunday of November
            sundays = pd.date_range(when, when + pd.Timedelta(days=13), freq="W-SUN")
            change = sundays[1] if label == "spring" else sundays[0]
            for side, lo, hi in (("before", change - pd.Timedelta(days=7), change), ("after", change + pd.Timedelta(days=1), change + pd.Timedelta(days=8))):
                wk = frame[(frame["session"] >= lo) & (frame["session"] < hi)]
                if wk.empty:
                    continue
                prof = wk.groupby("tod")["volume"].mean()
                pre = prof.reindex(range(OPEN_MIN - 10, OPEN_MIN), fill_value=0).mean()
                post = prof.reindex(range(OPEN_MIN, OPEN_MIN + 10), fill_value=0).mean()
                rows.append({"transition": f"{year}-{label}", "change_date": str(change.date()), "week": side,
                             "step_0930": post / pre if pre > 0 else np.nan})
    return pd.DataFrame(rows)


def early_close_check(frame: pd.DataFrame, calendar: TradingCalendar) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    cal = calendar.sessions(frame["session"].min(), frame["session"].max())
    early = cal[cal["early_close"]]
    rows = []
    for d, info in early.iterrows():
        day = frame[frame["session"] == d]
        if day.empty:
            continue
        close = int(info["close_min"])
        late_vol = day.loc[day["tod"] >= close + 30, "volume"].sum()
        rows.append({"session": str(d.date()), "calendar_close_et": f"{close // 60:02d}:{close % 60:02d}",
                     "last_bar_et": f"{int(day['tod'].max()) // 60:02d}:{int(day['tod'].max()) % 60:02d}",
                     "volume_30m_after_close_share": float(late_vol / max(day["volume"].sum(), 1))})
    return pd.DataFrame(rows)


def yearly_summary(ds: Dataset) -> pd.DataFrame:
    ps = ds.report.per_session.copy()
    if ps.empty:
        return ps
    ps["year"] = pd.DatetimeIndex(ps.index).year
    cal = ps[ps.get("calendar_session", True)] if "calendar_session" in ps else ps
    return cal.groupby("year").agg(
        sessions=("n_bars", "size"),
        bars=("n_bars", "sum"),
        rth_expected=("rth_expected", "sum"),
        rth_missing=("rth_missing", "sum"),
        zero_volume_bars=("zero_volume_bars", "sum"),
        max_rth_gap_min=("max_rth_gap_min", "max"),
    ).assign(rth_missing_pct=lambda f: f["rth_missing"] / f["rth_expected"].clip(lower=1))


def evaluate(ds: Dataset, calendar: TradingCalendar | None = None) -> RealDataQuality:
    inst = ds.prep.instrument
    calendar = calendar or TradingCalendar(inst.calendar)
    frame = _minute_frame(ds.bars, inst)
    timing = monthly_timing(frame)
    years = range(pd.Timestamp(ds.bars.index.min()).year, pd.Timestamp(ds.bars.index.max()).year + 1)
    dst = dst_transition_checks(frame, years)
    early = early_close_check(frame, calendar)
    yearly = yearly_summary(ds)
    reasons: list[str] = []
    verdict = "PASS"
    if not ds.report.ok:
        verdict = "STOP"
        reasons += [f"DQ error [{i.code}]: {i.message} (n={i.count})" for i in ds.report.errors]
    bad_months = timing[timing["flag"] != ""] if len(timing) else timing
    if len(bad_months):
        verdict = "STOP"
        reasons.append(f"{len(bad_months)} month(s) with volume timing displaced by ~1h: {', '.join(bad_months['month'].head(12))}")
    if len(dst):
        median_step = timing.attrs.get("median_step_0930", np.nan) if len(timing) else np.nan
        weak = dst[dst["step_0930"] < 1.0 + 0.25 * (median_step - 1.0)] if np.isfinite(median_step) and median_step > 1.5 else dst.iloc[0:0]
        if len(weak):
            verdict = "STOP" if verdict == "STOP" else "REVIEW"
            reasons.append(f"09:30 volume step missing around DST changes: {', '.join(weak['transition'] + ' ' + weak['week'])}")
    if len(early):
        suspicious = early[early["volume_30m_after_close_share"] > 0.10]
        if len(suspicious):
            verdict = "STOP" if verdict == "STOP" else "REVIEW"
            reasons.append(f"early-close sessions trading well past the calendar close: {', '.join(suspicious['session'])}")
    warnings = [i for i in ds.report.issues if i.severity == "warning"]
    if warnings and verdict == "PASS":
        verdict = "REVIEW"
    reasons += [f"warning [{i.code}]: {i.message} (n={i.count})" for i in warnings]
    summary = {
        "symbol": inst.symbol,
        "verdict": verdict,
        "rows": int(len(ds.bars)),
        "first_bar_utc": str(ds.bars.index.min()),
        "last_bar_utc": str(ds.bars.index.max()),
        "tradable_sessions": int(ds.prep.n_days),
        "excluded_sessions": ds.prep.excluded["reason"].value_counts().to_dict(),
        "modal_peak_minute_et": timing.attrs.get("modal_peak") if len(timing) else None,
        "median_step_0930": timing.attrs.get("median_step_0930") if len(timing) else None,
        "contract_rolls": int(ds.prep.daily.get("roll_session", pd.Series(dtype=bool)).sum()) if "roll_session" in ds.prep.daily else None,
    }
    return RealDataQuality(inst.symbol, verdict, reasons, timing, dst, early, yearly, summary)


def write_report(rdq: RealDataQuality, ds: Dataset, out_dir: Path, provenance: dict | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = lambda f: f.to_markdown(index=False, floatfmt=".3f") if len(f) else "_none_"  # noqa: E731
    issues = ds.report.to_frame()
    lines = [
        f"# Data-quality report: {rdq.symbol}",
        "",
        f"**Verdict: {rdq.verdict}**",
        "",
        *[f"- {r}" for r in rdq.reasons],
        "",
        "## Source",
        "```json",
        json.dumps({**ds.loaded.describe(), **({"provenance": provenance} if provenance else {})}, indent=2, default=str),
        "```",
        "## Summary",
        "```json",
        json.dumps(rdq.summary, indent=2, default=str),
        "```",
        "## Issues",
        fmt(issues),
        "",
        "## Year by year",
        rdq.yearly.reset_index().to_markdown(index=False, floatfmt=".4f") if len(rdq.yearly) else "_none_",
        "",
        "## 09:30 ET alignment, month by month",
        "`peak_minute` = ET minute of highest mean volume (07:00-12:00); `step_0930` = mean volume 09:30-09:39 / 09:20-09:29. "
        "The same ratio one hour later (`step_1030`) or earlier (`step_0830`) should NOT carry the open step.",
        "",
        fmt(rdq.monthly_timing),
        "",
        "## DST transitions",
        fmt(rdq.dst_checks),
        "",
        "## Early-close sessions",
        fmt(rdq.early_close),
    ]
    path = out_dir / f"{rdq.symbol}_data_quality.md"
    path.write_text("\n".join(lines))
    (out_dir / f"{rdq.symbol}_data_quality.json").write_text(json.dumps({"summary": rdq.summary, "reasons": rdq.reasons}, indent=2, default=str))
    return path

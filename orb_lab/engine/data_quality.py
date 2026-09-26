"""Data-quality report and explicit cleaning policy.

Rule: corrupted data is never silently backtested. Every problem found is recorded as an issue with a
severity. ``error`` issues block backtesting unless they were resolved by an explicit, recorded
cleaning policy (e.g. ``duplicate_policy="keep_first"``) or the caller passes ``allow_errors=True``
(which is stored in the experiment manifest).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .instruments import Instrument
from .sessions import TradingCalendar, assign_sessions, hhmm_to_minutes

RTH_START = 570  # 09:30 ET
RTH_END = 960  # 16:00 ET


@dataclass
class DQIssue:
    severity: str  # "error" | "warning" | "info"
    code: str
    message: str
    count: int = 0
    examples: list[str] = field(default_factory=list)


@dataclass
class CleaningPolicy:
    duplicate_policy: str = "error"  # error | keep_first | keep_last  (conflicting duplicates)
    invalid_policy: str = "error"  # error | drop  (invalid OHLC rows)


@dataclass
class DataQualityReport:
    n_rows_raw: int
    n_rows_clean: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    bar_minutes: int
    issues: list[DQIssue]
    per_session: pd.DataFrame
    actions: list[str]

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def errors(self) -> list[DQIssue]:
        return [i for i in self.issues if i.severity == "error"]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "severity": i.severity,
                    "code": i.code,
                    "count": i.count,
                    "message": i.message,
                    "examples": ", ".join(i.examples[:5]),
                }
                for i in self.issues
            ],
            columns=["severity", "code", "count", "message", "examples"],
        )

    def summary(self) -> dict:
        return {
            "ok": self.ok,
            "rows_raw": self.n_rows_raw,
            "rows_clean": self.n_rows_clean,
            "start": str(self.start),
            "end": str(self.end),
            "bar_minutes": self.bar_minutes,
            "n_errors": sum(i.severity == "error" for i in self.issues),
            "n_warnings": sum(i.severity == "warning" for i in self.issues),
            "actions": list(self.actions),
        }


def _examples(index: pd.Index, limit: int = 5) -> list[str]:
    return [str(x) for x in list(index[:limit])]


def check_and_clean(
    raw: pd.DataFrame,
    instrument: Instrument,
    *,
    bar_minutes: int = 1,
    policy: CleaningPolicy | None = None,
    calendar: TradingCalendar | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Run all data-quality checks on the canonical (unsorted) frame and return (clean_bars, report)."""
    policy = policy or CleaningPolicy()
    calendar = calendar or TradingCalendar(instrument.calendar)
    issues: list[DQIssue] = []
    actions: list[str] = []
    frame = raw.copy()
    n_raw = len(frame)

    if n_raw == 0:
        issues.append(DQIssue("error", "empty", "No rows loaded"))
        return frame, DataQualityReport(0, 0, None, None, bar_minutes, issues, pd.DataFrame(), actions)

    # ---- ordering
    ts_ns = frame.index.asi8
    backwards = int(np.sum(np.diff(ts_ns) < 0))
    if backwards:
        issues.append(DQIssue("warning", "unsorted", "Rows were not in chronological order; sorted", backwards))
        actions.append("sorted rows chronologically")
    frame = frame.sort_index(kind="stable")

    # ---- duplicates
    dup_ts = frame.index.duplicated(keep=False)
    if dup_ts.any():
        dups = frame[dup_ts]
        value_cols = [c for c in ("open", "high", "low", "close", "volume") if c in frame.columns]
        grouped = dups.groupby(level=0)[value_cols].nunique(dropna=False)
        conflicting = grouped[(grouped > 1).any(axis=1)].index
        non_conflicting = dups[~dups.index.isin(conflicting)]
        exact_only = int(len(non_conflicting) - non_conflicting.index.nunique())
        if len(conflicting):
            severity = "error" if policy.duplicate_policy == "error" else "warning"
            msg = f"{len(conflicting)} timestamps have conflicting duplicate bars"
            if severity == "warning":
                msg += f" (resolved by explicit policy {policy.duplicate_policy})"
            issues.append(DQIssue(severity, "duplicate_conflict", msg, len(conflicting), _examples(conflicting)))
        if exact_only > 0:
            issues.append(DQIssue("warning", "duplicate_exact", "Exact duplicate bars dropped", exact_only))
        keep = "last" if policy.duplicate_policy == "keep_last" else "first"
        frame = frame[~frame.index.duplicated(keep=keep)]
        actions.append(f"dropped duplicate timestamps (keep={keep})")

    # ---- invalid OHLC
    o, h, l, c, v = (frame[k].to_numpy() for k in ("open", "high", "low", "close", "volume"))
    nan_rows = np.isnan(o) | np.isnan(h) | np.isnan(l) | np.isnan(c)
    bad = (
        nan_rows
        | (h < np.maximum(o, c) - 1e-12)
        | (l > np.minimum(o, c) + 1e-12)
        | (h < l)
        | (np.minimum.reduce([o, h, l, c]) <= 0)
        | (np.nan_to_num(v, nan=0.0) < 0)
    )
    if bad.any():
        severity = "error" if policy.invalid_policy == "error" else "warning"
        msg = "Rows with invalid OHLC (NaN, high<max(open,close), low>min(open,close), non-positive price or negative volume)"
        if severity == "warning":
            msg += "; dropped by explicit policy"
        issues.append(DQIssue(severity, "invalid_ohlc", msg, int(bad.sum()), _examples(frame.index[bad])))
        if policy.invalid_policy == "drop":
            frame = frame[~bad]
            actions.append(f"dropped {int(bad.sum())} invalid OHLC rows")

    # ---- zero volume
    zero_vol = frame["volume"].fillna(0).to_numpy() == 0
    if zero_vol.any():
        issues.append(
            DQIssue("warning", "zero_volume", "Bars with zero volume (kept)", int(zero_vol.sum()), _examples(frame.index[zero_vol]))
        )

    # ---- timestamp alignment to the bar grid
    ns_per_bar = bar_minutes * 60 * 1_000_000_000
    misaligned = (frame.index.as_unit("ns").asi8 % ns_per_bar) != 0
    if misaligned.any():
        issues.append(
            DQIssue(
                "error",
                "misaligned_timestamps",
                f"Timestamps not aligned to the {bar_minutes}-minute grid (irregular or tick data?)",
                int(misaligned.sum()),
                _examples(frame.index[misaligned]),
            )
        )

    # ---- off-tick prices
    ticks = frame[["open", "high", "low", "close"]].to_numpy() / instrument.tick_size
    off_grid = np.abs(ticks - np.round(ticks)) > 1e-6
    frac_off = float(off_grid.any(axis=1).mean()) if len(frame) else 0.0
    if frac_off > 0.01:
        issues.append(
            DQIssue(
                "warning",
                "off_tick_prices",
                f"{frac_off:.1%} of bars have prices off the {instrument.tick_size} tick grid "
                "(ratio back-adjusted continuous contract?). Tick-based order levels become approximate.",
                int(off_grid.any(axis=1).sum()),
            )
        )

    # ---- sessions / timezone sanity
    session_date, tod = assign_sessions(frame.index, instrument.session_open, instrument.timezone)
    et = frame.index.tz_convert(instrument.timezone)
    weekday = et.dayofweek.to_numpy()
    wall = (et.hour * 60 + et.minute).to_numpy()
    close_min = hhmm_to_minutes(instrument.session_close)
    open_min = hhmm_to_minutes(instrument.session_open)
    saturday = (weekday == 5) | ((weekday == 4) & (wall >= close_min)) | ((weekday == 6) & (wall < open_min))
    if saturday.any():
        issues.append(
            DQIssue(
                "warning",
                "weekend_bars",
                "Bars during the weekend market closure (Fri close to Sun open ET): timezone problem?",
                int(saturday.sum()),
                _examples(frame.index[saturday]),
            )
        )
    maintenance = (wall >= close_min) & (wall < open_min) & (weekday < 5)
    frac_maint = float(maintenance.mean())
    if frac_maint > 0.005:
        issues.append(
            DQIssue(
                "warning",
                "maintenance_window_bars",
                f"{frac_maint:.2%} of bars fall inside the {instrument.session_close}-{instrument.session_open} ET "
                "daily halt; the declared source timezone may be wrong",
                int(maintenance.sum()),
            )
        )
    _check_dst_volume_profile(frame, et, issues)

    # ---- price spikes
    closes = frame["close"].to_numpy()
    if len(closes) > 100:
        rets = np.abs(np.diff(np.log(closes)))
        med = np.median(rets[rets > 0]) if np.any(rets > 0) else 0.0
        if med > 0:
            spikes = np.where(rets > 50 * med)[0]
            if len(spikes):
                issues.append(
                    DQIssue(
                        "warning",
                        "price_spikes",
                        "Bar-to-bar moves > 50x the median absolute move (bad ticks, rolls or gaps)",
                        len(spikes),
                        _examples(frame.index[spikes + 1]),
                    )
                )

    # ---- per-session completeness, gaps, calendar alignment
    per_session = _per_session_stats(frame, session_date, tod, bar_minutes)
    if len(per_session):
        cal = calendar.sessions(per_session.index.min(), per_session.index.max())
        cal_dates = set(cal.index.date)
        data_dates = set(per_session.index.date)
        missing_sessions = sorted(cal_dates - data_dates)
        extra_sessions = sorted(data_dates - cal_dates)
        if missing_sessions:
            issues.append(
                DQIssue(
                    "warning",
                    "missing_sessions",
                    f"Calendar ({calendar.name}) sessions with no data",
                    len(missing_sessions),
                    [str(d) for d in missing_sessions[:5]],
                )
            )
        if extra_sessions:
            issues.append(
                DQIssue(
                    "info",
                    "non_calendar_sessions",
                    f"Sessions with data that are not {calendar.name} trading days (holidays/weekends); excluded from trading",
                    len(extra_sessions),
                    [str(d) for d in extra_sessions[:5]],
                )
            )
        per_session["calendar_session"] = [d in cal_dates for d in per_session.index.date]
        in_cal = per_session[per_session["calendar_session"]]
        if len(cal):
            early = cal[cal["early_close"]]
            per_session["early_close"] = per_session.index.isin(early.index)
            # expected RTH bars respect early closes
            close_lookup = cal["close_min"].to_dict()
            expected = []
            for dt in per_session.index:
                close = close_lookup.get(dt, RTH_END)
                expected.append(max(0, (min(close, RTH_END) - RTH_START) // bar_minutes))
            per_session["rth_expected"] = expected
            per_session["rth_missing"] = (per_session["rth_expected"] - per_session["rth_present"]).clip(lower=0)
            in_cal = per_session[per_session["calendar_session"]]
        incomplete = in_cal[in_cal["rth_missing"] > 0.05 * in_cal["rth_expected"].clip(lower=1)]
        if len(incomplete):
            issues.append(
                DQIssue(
                    "warning",
                    "missing_rth_bars",
                    "Sessions missing more than 5% of 09:30-16:00 ET bars",
                    len(incomplete),
                    [str(d.date()) for d in incomplete.index[:5]],
                )
            )
        big_gaps = in_cal[in_cal["max_rth_gap_min"] > 15]
        if len(big_gaps):
            issues.append(
                DQIssue(
                    "warning",
                    "large_gaps",
                    "Sessions with an intraday (RTH) timestamp gap longer than 15 minutes",
                    len(big_gaps),
                    [str(d.date()) for d in big_gaps.index[:5]],
                )
            )
        if "contract" in frame.columns:
            rolls = per_session[per_session["n_contracts"] > 1]
            if len(rolls):
                issues.append(
                    DQIssue(
                        "warning",
                        "intraday_contract_change",
                        "Sessions containing more than one contract; these sessions are excluded from trading",
                        len(rolls),
                        [str(d.date()) for d in rolls.index[:5]],
                    )
                )

    report = DataQualityReport(
        n_rows_raw=n_raw,
        n_rows_clean=len(frame),
        start=frame.index.min() if len(frame) else None,
        end=frame.index.max() if len(frame) else None,
        bar_minutes=bar_minutes,
        issues=issues,
        per_session=per_session,
        actions=actions,
    )
    return frame, report


def _per_session_stats(frame: pd.DataFrame, session_date: np.ndarray, tod: np.ndarray, bar_minutes: int) -> pd.DataFrame:
    work = pd.DataFrame(
        {
            "session_date": pd.to_datetime(session_date),
            "tod": tod,
            "volume": frame["volume"].to_numpy(),
        }
    )
    if "contract" in frame.columns:
        work["contract"] = frame["contract"].to_numpy()
    rth = work[(work["tod"] >= RTH_START) & (work["tod"] < RTH_END)].copy()
    rth["gap"] = rth.groupby("session_date")["tod"].diff() - bar_minutes
    grouped = work.groupby("session_date")
    out = pd.DataFrame(
        {
            "n_bars": grouped.size(),
            "zero_volume_bars": grouped["volume"].apply(lambda s: int((s.fillna(0) == 0).sum())),
        }
    )
    out["rth_present"] = rth.groupby("session_date").size().reindex(out.index).fillna(0).astype(int)
    out["max_rth_gap_min"] = rth.groupby("session_date")["gap"].max().reindex(out.index).fillna(0.0)
    if "contract" in work.columns:
        out["n_contracts"] = grouped["contract"].nunique()
    out["rth_expected"] = (RTH_END - RTH_START) // bar_minutes
    out["rth_missing"] = (out["rth_expected"] - out["rth_present"]).clip(lower=0)
    out.index.name = "session_date"
    return out


def _check_dst_volume_profile(frame: pd.DataFrame, et: pd.DatetimeIndex, issues: list[DQIssue]) -> None:
    """Detect data labeled in a fixed offset: the volume peak shifts by 60 min between EST and EDT."""
    if len(frame) < 5000:
        return
    offset_min = ((et.tz_localize(None) - frame.index.tz_localize(None)) / pd.Timedelta(minutes=1)).to_numpy()
    is_dst = offset_min == offset_min.max()
    if offset_min.max() == offset_min.min():
        return
    wall = (et.hour * 60 + et.minute).to_numpy()
    vol = frame["volume"].fillna(0).to_numpy()
    peaks = {}
    for flag in (True, False):
        mask = is_dst == flag
        if mask.sum() < 1000:
            return
        profile = pd.Series(vol[mask]).groupby(wall[mask]).mean()
        if profile.empty or profile.max() <= 0:
            return
        peaks[flag] = int(profile.idxmax())
    if abs(abs(peaks[True] - peaks[False]) - 60) <= 2:
        issues.append(
            DQIssue(
                "error",
                "dst_misalignment",
                f"Volume peak is at {peaks[True]//60:02d}:{peaks[True]%60:02d} ET in summer but "
                f"{peaks[False]//60:02d}:{peaks[False]%60:02d} ET in winter: timestamps were probably recorded "
                "in a fixed UTC offset or the wrong timezone was declared.",
            )
        )

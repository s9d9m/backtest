"""Load -> quality-check -> prepare, in one call, refusing to continue on unresolved data errors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data_loader import LoadedData, from_dataframe, load_bars
from .data_quality import CleaningPolicy, DataQualityReport, check_and_clean
from .instruments import Instrument
from .session_data import PreparedData, prepare_data
from .sessions import TradingCalendar


class DataQualityError(RuntimeError):
    def __init__(self, report: DataQualityReport):
        self.report = report
        lines = "; ".join(f"[{i.code}] {i.message} (n={i.count})" for i in report.errors)
        super().__init__(f"Data-quality errors block backtesting: {lines}")


@dataclass
class Dataset:
    loaded: LoadedData
    report: DataQualityReport
    bars: pd.DataFrame
    prep: PreparedData
    policy: CleaningPolicy
    allow_errors: bool

    def describe(self) -> dict:
        return {
            **self.loaded.describe(),
            "cleaning_policy": vars(self.policy),
            "allow_errors": self.allow_errors,
            "dq_summary": self.report.summary(),
            "sessions": int(self.prep.n_days),
            "first_session": str(self.prep.dates[0]) if self.prep.n_days else None,
            "last_session": str(self.prep.dates[-1]) if self.prep.n_days else None,
        }


def build_dataset(
    source: str | Path | pd.DataFrame,
    instrument: Instrument,
    *,
    source_tz: str | None = None,
    timestamp_convention: str = "start",
    policy: CleaningPolicy | None = None,
    allow_errors: bool = False,
    min_or_coverage: float = 0.8,
    exclude_early_close: bool = False,
    calendar: TradingCalendar | None = None,
    column_map: dict[str, str] | None = None,
) -> Dataset:
    policy = policy or CleaningPolicy()
    if isinstance(source, pd.DataFrame):
        loaded = from_dataframe(source, source_tz=source_tz, timestamp_convention=timestamp_convention)
    else:
        loaded = load_bars(source, source_tz=source_tz, timestamp_convention=timestamp_convention, column_map=column_map)
    calendar = calendar or TradingCalendar(instrument.calendar)
    bars, report = check_and_clean(loaded.raw, instrument, bar_minutes=loaded.bar_minutes, policy=policy, calendar=calendar)
    if not report.ok and not allow_errors:
        raise DataQualityError(report)
    prep = prepare_data(
        bars,
        instrument,
        base_minutes=loaded.bar_minutes,
        calendar=calendar,
        min_or_coverage=min_or_coverage,
        exclude_early_close=exclude_early_close,
        data_hash=loaded.file_hash,
    )
    return Dataset(loaded=loaded, report=report, bars=bars, prep=prep, policy=policy, allow_errors=allow_errors)

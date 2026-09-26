"""Load intraday OHLCV bars from CSV / Parquet (or any registered adapter) into a canonical frame.

Canonical frame (returned *unsorted and uncleaned* so that the data-quality report can see problems):

* index ``ts`` : tz-aware UTC ``DatetimeIndex`` marking the **start** of each bar
* columns      : ``open high low close volume`` (+ optional ``bid ask contract symbol``)

Cleaning (sorting, duplicate handling, invalid-row policy) happens in :mod:`orb_lab.engine.data_quality`
so that nothing is silently repaired.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

REQUIRED = ("open", "high", "low", "close", "volume")
OPTIONAL = ("bid", "ask", "contract", "symbol")

ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "datetime", "date_time", "time_stamp", "ts", "time", "date"),
    "open": ("open", "o", "open_price"),
    "high": ("high", "h", "high_price"),
    "low": ("low", "l", "low_price"),
    "close": ("close", "c", "last", "close_price"),
    "volume": ("volume", "vol", "v"),
    "bid": ("bid", "bid_price"),
    "ask": ("ask", "ask_price", "offer"),
    "contract": ("contract", "expiry", "contract_month", "instrument_id"),
    "symbol": ("symbol", "ticker", "root"),
}


class DataError(ValueError):
    pass


class DataAdapter(Protocol):
    """Anything that can turn a path into a raw DataFrame."""

    def read(self, path: Path) -> pd.DataFrame: ...


class CsvAdapter:
    def read(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)


class ParquetAdapter:
    def read(self, path: Path) -> pd.DataFrame:
        frame = pd.read_parquet(path)
        if isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.reset_index()
        return frame


_ADAPTERS: dict[str, DataAdapter] = {
    ".csv": CsvAdapter(),
    ".txt": CsvAdapter(),
    ".parquet": ParquetAdapter(),
    ".pq": ParquetAdapter(),
}


def register_adapter(extension: str, adapter: DataAdapter) -> None:
    """Register a reader for a new file type / data provider."""
    _ADAPTERS[extension.lower()] = adapter


@dataclass
class LoadedData:
    raw: pd.DataFrame
    source: str
    file_hash: str
    source_tz: str | None
    timestamp_convention: str
    bar_minutes: int
    raw_rows: int
    notes: list[str] = field(default_factory=list)

    def describe(self) -> dict:
        return {
            "source": self.source,
            "file_hash": self.file_hash,
            "source_tz": self.source_tz,
            "timestamp_convention": self.timestamp_convention,
            "bar_minutes": self.bar_minutes,
            "raw_rows": self.raw_rows,
            "notes": list(self.notes),
        }


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def frame_sha256(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _resolve_columns(frame: pd.DataFrame, column_map: dict[str, str] | None) -> dict[str, str]:
    lower = {str(col).strip().lower(): col for col in frame.columns}
    resolved: dict[str, str] = {}
    if column_map:
        for canonical, source in column_map.items():
            if source not in frame.columns:
                raise DataError(f"column_map refers to missing column {source!r}")
            resolved[canonical] = source
    for canonical, aliases in ALIASES.items():
        if canonical in resolved:
            continue
        for alias in aliases:
            if alias in lower:
                resolved[canonical] = lower[alias]
                break
    return resolved


def _parse_timestamps(values: pd.Series, source_tz: str | None, notes: list[str]) -> pd.DatetimeIndex:
    if pd.api.types.is_numeric_dtype(values):
        numbers = values.astype("int64")
        magnitude = float(np.nanmedian(np.abs(numbers.to_numpy()))) if len(numbers) else 0.0
        unit = "s" if magnitude < 1e11 else "ms" if magnitude < 1e14 else "us" if magnitude < 1e17 else "ns"
        notes.append(f"numeric timestamps interpreted as Unix epoch ({unit}, UTC)")
        return pd.DatetimeIndex(pd.to_datetime(numbers, unit=unit, utc=True))
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        parsed = pd.DatetimeIndex(values)
    elif pd.api.types.is_datetime64_dtype(values):
        parsed = pd.DatetimeIndex(values)
    else:
        text = values.astype(str)
        has_offset = text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
        if has_offset.all():
            parsed = pd.DatetimeIndex(pd.to_datetime(text, utc=True, format="mixed"))
        elif has_offset.any():
            raise DataError("Timestamp column mixes offset-aware and naive values; refusing to guess.")
        else:
            parsed = pd.DatetimeIndex(pd.to_datetime(text, format="mixed"))
    if parsed.tz is None:
        if not source_tz:
            raise DataError(
                "Timestamps are timezone-naive. Specify source_tz (e.g. 'America/New_York', "
                "'America/Chicago' or 'UTC'); the loader never guesses a timezone."
            )
        try:
            localized = parsed.tz_localize(source_tz, ambiguous="infer", nonexistent="NaT")
        except Exception:  # ambiguous hour cannot be inferred (e.g. duplicated fall-back hour)
            localized = parsed.tz_localize(source_tz, ambiguous="NaT", nonexistent="NaT")
            notes.append("ambiguous DST fall-back timestamps could not be inferred and were set to NaT")
        n_nat = int(localized.isna().sum() - parsed.isna().sum())
        if n_nat:
            notes.append(f"{n_nat} timestamps invalid in {source_tz} (DST gap/ambiguity) were dropped")
        notes.append(f"naive timestamps localized as {source_tz}")
        parsed = localized
    return parsed.tz_convert("UTC")


def detect_bar_minutes(index: pd.DatetimeIndex) -> int:
    if len(index) < 3:
        return 1
    diffs = np.diff(np.sort(index.as_unit("ns").asi8)) / 60e9
    diffs = diffs[(diffs > 0) & (diffs < 240)]
    if len(diffs) == 0:
        return 1
    values, counts = np.unique(np.round(diffs).astype(int), return_counts=True)
    mode = int(values[np.argmax(counts)])
    return max(mode, 1)


def normalize_frame(
    frame: pd.DataFrame,
    *,
    source_tz: str | None = None,
    timestamp_convention: str = "start",
    bar_minutes: int | None = None,
    column_map: dict[str, str] | None = None,
    date_column: str | None = None,
    time_column: str | None = None,
    notes: list[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Convert an arbitrary raw bar frame into the canonical (unsorted) form."""
    notes = notes if notes is not None else []
    if timestamp_convention not in ("start", "end"):
        raise DataError("timestamp_convention must be 'start' (bar open time) or 'end' (bar close time)")
    columns = _resolve_columns(frame, column_map)
    if date_column and time_column:
        ts_source = frame[date_column].astype(str) + " " + frame[time_column].astype(str)
    elif "timestamp" in columns:
        ts_source = frame[columns["timestamp"]]
    else:
        raise DataError(f"No timestamp column found. Columns: {list(frame.columns)}")
    missing = [col for col in REQUIRED if col not in columns]
    if missing:
        raise DataError(f"Missing required columns {missing}. Columns: {list(frame.columns)}")

    ts = _parse_timestamps(pd.Series(ts_source).reset_index(drop=True), source_tz, notes)
    out = pd.DataFrame(index=pd.RangeIndex(len(frame)))
    for col in REQUIRED:
        out[col] = pd.to_numeric(frame[columns[col]].reset_index(drop=True), errors="coerce").astype("float64")
    for col in ("bid", "ask"):
        if col in columns:
            out[col] = pd.to_numeric(frame[columns[col]].reset_index(drop=True), errors="coerce").astype("float64")
    for col in ("contract", "symbol"):
        if col in columns:
            out[col] = frame[columns[col]].reset_index(drop=True).astype(str)
    out.index = ts
    out.index.name = "ts"
    out = out[~out.index.isna()]

    minutes = bar_minutes or detect_bar_minutes(out.index)
    if timestamp_convention == "end":
        out.index = out.index - pd.Timedelta(minutes=minutes)
        notes.append(f"timestamps shifted from bar-end to bar-start by {minutes} min")
    return out, minutes


def load_bars(
    path: str | Path,
    *,
    source_tz: str | None = None,
    timestamp_convention: str = "start",
    bar_minutes: int | None = None,
    column_map: dict[str, str] | None = None,
    date_column: str | None = None,
    time_column: str | None = None,
    reader: Callable[[Path], pd.DataFrame] | None = None,
) -> LoadedData:
    """Load a bar file. See module docstring for the canonical output."""
    path = Path(path)
    if not path.exists():
        raise DataError(f"File not found: {path}")
    if reader is None:
        adapter = _ADAPTERS.get(path.suffix.lower())
        if adapter is None:
            raise DataError(f"No adapter registered for {path.suffix!r}; use register_adapter()")
        raw_frame = adapter.read(path)
    else:
        raw_frame = reader(path)
    notes: list[str] = []
    canonical, minutes = normalize_frame(
        raw_frame,
        source_tz=source_tz,
        timestamp_convention=timestamp_convention,
        bar_minutes=bar_minutes,
        column_map=column_map,
        date_column=date_column,
        time_column=time_column,
        notes=notes,
    )
    return LoadedData(
        raw=canonical,
        source=str(path),
        file_hash=file_sha256(path),
        source_tz=source_tz,
        timestamp_convention=timestamp_convention,
        bar_minutes=minutes,
        raw_rows=len(raw_frame),
        notes=notes,
    )


def from_dataframe(
    frame: pd.DataFrame,
    *,
    source_tz: str | None = None,
    timestamp_convention: str = "start",
    bar_minutes: int | None = None,
    name: str = "<dataframe>",
) -> LoadedData:
    """Wrap an in-memory frame (e.g. synthetic data) exactly like a file load."""
    notes: list[str] = []
    source = frame.reset_index() if isinstance(frame.index, pd.DatetimeIndex) else frame
    if "ts" in source.columns and "timestamp" not in source.columns:
        source = source.rename(columns={"ts": "timestamp"})
    canonical, minutes = normalize_frame(
        source, source_tz=source_tz, timestamp_convention=timestamp_convention, bar_minutes=bar_minutes, notes=notes
    )
    return LoadedData(
        raw=canonical,
        source=name,
        file_hash=frame_sha256(canonical),
        source_tz=source_tz,
        timestamp_convention=timestamp_convention,
        bar_minutes=minutes,
        raw_rows=len(frame),
        notes=notes,
    )

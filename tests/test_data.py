import numpy as np
import pandas as pd
import pytest

from orb_lab.engine.data_loader import DataError, load_bars
from orb_lab.engine.data_quality import CleaningPolicy, check_and_clean
from orb_lab.engine.pipeline import DataQualityError, build_dataset
from orb_lab.engine.sessions import TradingCalendar

from .conftest import day_bars, make_instrument

CAL = TradingCalendar("WEEKDAYS")


def _write_csv(tmp_path, frame, name="bars.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def test_naive_timestamps_require_declared_timezone(tmp_path):
    frame = pd.DataFrame(
        {"timestamp": ["2024-01-03 09:30:00", "2024-01-03 09:31:00"], "open": [1, 1], "high": [1, 1], "low": [1, 1], "close": [1, 1], "volume": [1, 1]}
    )
    path = _write_csv(tmp_path, frame)
    with pytest.raises(DataError, match="timezone-naive"):
        load_bars(path)
    loaded = load_bars(path, source_tz="America/New_York")
    assert str(loaded.raw.index[0]) == "2024-01-03 14:30:00+00:00"


def test_chicago_source_timezone(tmp_path):
    frame = pd.DataFrame({"Date": ["2024-07-01 08:30"], "Open": [1], "High": [1], "Low": [1], "Close": [1], "Vol": [5]})
    loaded = load_bars(_write_csv(tmp_path, frame), source_tz="America/Chicago")
    # 08:30 CDT == 09:30 EDT == 13:30 UTC
    assert str(loaded.raw.index[0]) == "2024-07-01 13:30:00+00:00"
    assert loaded.raw["volume"].iloc[0] == 5


def test_offset_aware_strings_and_end_convention(tmp_path):
    frame = pd.DataFrame(
        {
            "datetime": ["2024-01-03T14:31:00Z", "2024-01-03T14:32:00Z", "2024-01-03T14:33:00Z"],
            "open": [1, 1, 1], "high": [1, 1, 1], "low": [1, 1, 1], "close": [1, 1, 1], "volume": [1, 1, 1],
        }
    )
    loaded = load_bars(_write_csv(tmp_path, frame), timestamp_convention="end")
    assert loaded.bar_minutes == 1
    assert str(loaded.raw.index[0]) == "2024-01-03 14:30:00+00:00"


def test_parquet_roundtrip_and_optional_columns(tmp_path):
    bars = day_bars("2024-01-03", {}).reset_index()
    bars["contract"] = "ESH4"
    bars["bid"] = bars["close"] - 0.25
    path = tmp_path / "bars.parquet"
    bars.to_parquet(path)
    loaded = load_bars(path)
    assert {"contract", "bid"} <= set(loaded.raw.columns)
    assert len(loaded.raw) == len(bars)
    assert len(loaded.file_hash) == 64


def test_missing_required_column(tmp_path):
    frame = pd.DataFrame({"timestamp": ["2024-01-03T14:31:00Z"], "open": [1], "high": [1], "low": [1], "close": [1]})
    with pytest.raises(DataError, match="volume"):
        load_bars(_write_csv(tmp_path, frame))


def test_dq_detects_unsorted_exact_and_conflicting_duplicates():
    inst = make_instrument()
    bars = day_bars("2024-01-03", {})
    shuffled = pd.concat([bars.iloc[10:], bars.iloc[:10], bars.iloc[[5]]])  # unsorted + exact duplicate
    clean, report = check_and_clean(shuffled, inst, calendar=CAL)
    codes = {i.code for i in report.issues}
    assert {"unsorted", "duplicate_exact"} <= codes
    assert report.ok and clean.index.is_monotonic_increasing and len(clean) == len(bars)

    conflict = bars.iloc[[5]].copy()
    conflict["close"] += 1.0
    conflict["high"] += 1.0
    _, report = check_and_clean(pd.concat([bars, conflict]), inst, calendar=CAL)
    assert not report.ok and report.errors[0].code == "duplicate_conflict"
    _, report = check_and_clean(pd.concat([bars, conflict]), inst, calendar=CAL, policy=CleaningPolicy(duplicate_policy="keep_first"))
    assert report.ok


def test_dq_invalid_ohlc_blocks_backtest():
    inst = make_instrument()
    bars = day_bars("2024-01-03", {})
    bars.iloc[20, bars.columns.get_loc("high")] = 50.0  # high below open/close
    with pytest.raises(DataQualityError, match="invalid_ohlc"):
        build_dataset(bars, inst, calendar=CAL)
    ds = build_dataset(bars, inst, calendar=CAL, policy=CleaningPolicy(invalid_policy="drop"))
    assert ds.report.ok and len(ds.bars) == len(bars) - 1


def test_dq_gaps_zero_volume_and_misalignment():
    inst = make_instrument()
    bars = day_bars("2024-01-03", {})
    bars = bars.drop(bars.index[100:130])  # 30-minute RTH hole
    bars.iloc[5, bars.columns.get_loc("volume")] = 0
    _, report = check_and_clean(bars, inst, calendar=CAL)
    codes = {i.code for i in report.issues}
    assert {"large_gaps", "zero_volume", "missing_rth_bars"} <= codes

    shifted = bars.copy()
    shifted.index = shifted.index + pd.Timedelta(seconds=30)
    _, report = check_and_clean(shifted, inst, calendar=CAL)
    assert "misaligned_timestamps" in {i.code for i in report.errors}


def test_dq_detects_fixed_offset_timestamps():
    """Data recorded in fixed UTC-5 all year: the volume spike moves by an hour in summer."""
    inst = make_instrument()
    frames = []
    for date in pd.bdate_range("2024-01-02", "2024-08-30")[::3]:
        idx = pd.date_range(f"{date.date()} 08:00", f"{date.date()} 15:59", freq="1min")
        vol = np.where((idx.hour == 9) & (idx.minute == 30), 5000, 100)
        frame = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": vol}, index=idx)
        frame.index = (frame.index + pd.Timedelta(hours=5)).tz_localize("UTC")  # "EST" all year
        frames.append(frame)
    _, report = check_and_clean(pd.concat(frames), inst, calendar=CAL)
    assert "dst_misalignment" in {i.code for i in report.errors}


def test_dq_intraday_contract_change_excluded():
    inst = make_instrument()
    bars = day_bars("2024-01-03", {"09:30": (100.0, 101.0, 99.0, 100.0)})
    bars["contract"] = ["ESH4"] * 200 + ["ESM4"] * (len(bars) - 200)
    ds = build_dataset(bars, inst, calendar=CAL)
    assert "intraday_contract_change" in {i.code for i in ds.report.issues}
    assert ds.prep.n_days == 0 or "intraday_contract_change" in set(ds.prep.excluded["reason"])


def test_holidays_are_excluded_from_trading():
    inst = make_instrument(calendar="XNYS")
    frames = [day_bars(d, {}) for d in ("2024-07-03", "2024-07-04", "2024-07-05")]
    ds = build_dataset(pd.concat(frames), inst)
    assert [str(d) for d in ds.prep.dates] == ["2024-07-03", "2024-07-05"]
    assert "non_calendar_session" in set(ds.prep.excluded["reason"])
    # 2024-07-03 is an early close (13:00 ET): forced exit 5 minutes earlier
    assert ds.prep.session_exit_min[0] == 13 * 60 - 5
    assert ds.prep.session_exit_min[1] == 15 * 60 + 55

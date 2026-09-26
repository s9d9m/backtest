"""Front-month construction and Databento integration (mocked client; no network, no fabricated research data)."""

import json

import numpy as np
import pandas as pd
import pytest

from orb_lab.data_sources import databento_source as dbs
from orb_lab.data_sources.futures_roll import build_front_month, expiry_key, is_outright
from orb_lab.engine.pipeline import build_dataset
from orb_lab.engine.sessions import TradingCalendar

from .conftest import make_instrument


def _contract_bars(contract, dates, price, volume):
    rows = []
    for d, v in zip(dates, volume):
        idx = pd.date_range(f"{d} 09:30", f"{d} 10:29", freq="1min", tz="America/New_York").tz_convert("UTC")
        rows.append(pd.DataFrame({"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": v, "contract": contract}, index=idx))
    return pd.concat(rows)


def test_outright_filter_and_expiry():
    assert is_outright("ESH4", "ES") and is_outright("ESZ24", "ES")
    assert not is_outright("ESH4-ESM4", "ES") and not is_outright("ESH4:BF", "ES") and not is_outright("NQH4", "ES")
    assert not is_outright("6EH4", "E")
    assert expiry_key("ESH4", "ES", 2024) < expiry_key("ESM4", "ES", 2024) < expiry_key("ESH5", "ES", 2024)
    assert expiry_key("ESZ9", "ES", 2020) == 2019 * 12 + 11 and expiry_key("ESZ9", "ES", 2021) == 2029 * 12 + 11


def test_front_month_roll_is_causal_forward_only_and_never_mixes_a_session():
    dates = list(pd.bdate_range("2024-03-04", "2024-03-15").strftime("%Y-%m-%d"))
    # H4 dominates until 03-08; on 03-08 M4 volume overtakes -> M4 traded from the NEXT session (03-11)
    vol_h = [900, 900, 800, 700, 100, 50, 60, 40, 30, 20]
    vol_m = [100, 100, 200, 300, 900, 950, 940, 960, 970, 980]
    # a noisy day where the old contract briefly leads again must NOT roll back
    vol_h[7], vol_m[7] = 2000, 10
    bars = pd.concat([
        _contract_bars("ESH4", dates, 5000.0, vol_h),
        _contract_bars("ESM4", dates, 5050.0, vol_m),
        _contract_bars("ESH4-ESM4", dates, -50.0, [5] * len(dates)),
    ])
    res = build_front_month(bars, "ES")
    assert res.dropped_spread_rows == 60 * len(dates)
    assert res.dropped_first_session == "2024-03-04"
    per_session = res.bars.groupby(res.bars.index.tz_convert("America/New_York").date)["contract"].agg(["nunique", "first"])
    assert (per_session["nunique"] == 1).all()
    chosen = per_session["first"].astype(str).to_dict()
    days = [pd.Timestamp(d).date() for d in dates]
    assert [chosen[d] for d in days[1:5]] == ["ESH4"] * 4
    assert chosen[days[5]] == "ESM4"  # 03-11: first session after M4 led on 03-08
    assert all(chosen[d] == "ESM4" for d in days[5:])  # no roll back despite 03-13 volume
    assert list(res.rolls["to_contract"]) == ["ESM4"]
    assert str(res.rolls["session_date"].iloc[0].date()) == "2024-03-11"


def test_atr_true_range_does_not_span_contracts():
    inst = make_instrument(symbol="ES")
    dates = list(pd.bdate_range("2024-03-04", "2024-03-15").strftime("%Y-%m-%d"))
    a = _contract_bars("ESH4", dates[:5], 5000.0, [100] * 5)
    b = _contract_bars("ESM4", dates[5:], 5100.0, [100] * 5)  # 100-point roll gap
    ds = build_dataset(pd.concat([a, b]), inst, calendar=TradingCalendar("WEEKDAYS"))
    daily = ds.prep.daily
    assert daily["roll_session"].sum() == 1 and np.isnan(daily.loc["2024-03-11", "prev_close"])
    atr = ds.prep.atr(2)
    # every session range is 2 points = 8 ticks; the roll gap must not inflate ATR
    assert np.nanmax(atr) == pytest.approx(8.0)


class _Store:
    def __init__(self, frame):
        self.frame = frame

    def to_df(self, **kwargs):
        return self.frame.copy()


class _MockClient:
    def __init__(self, frame):
        self.frame = frame
        self.calls = []
        client = self

        class Timeseries:
            def get_range(self, **kw):
                client.calls.append(kw)
                s, e = pd.Timestamp(kw["start"]).tz_localize("UTC"), pd.Timestamp(kw["end"]).tz_localize("UTC")
                f = client.frame
                return _Store(f[(f.index >= s) & (f.index < e)].rename_axis("ts_event"))

        class Metadata:
            def get_cost(self, **kw):
                return 1.25

        self.timeseries = Timeseries()
        self.metadata = Metadata()


def test_databento_fetch_with_mock_client(tmp_path):
    dates = list(pd.bdate_range("2023-12-26", "2024-01-05").strftime("%Y-%m-%d"))
    bars = pd.concat([_contract_bars("ESH4", dates, 5000.0, [100] * len(dates)), _contract_bars("ESH4-ESM4", dates, -50.0, [1] * len(dates))])
    bars = bars.rename(columns={"contract": "symbol"})
    bars["instrument_id"] = 42
    client = _MockClient(bars)
    res = dbs.fetch("ES", "2023-12-20", "2024-01-10", client=client, raw_dir=tmp_path / "raw", out_dir=tmp_path)
    assert len(client.calls) == 2  # one request per calendar year
    assert all(c["stype_in"] == "parent" and c["symbols"] == "ES.FUT" and c["schema"] == "ohlcv-1m" for c in client.calls)
    assert set(res.front["contract"]) == {"ESH4"}
    prov = json.loads((tmp_path / "ES_databento_provenance.json").read_text())
    assert prov["dataset"] == "GLBX.MDP3" and prov["output_rows"] == len(res.front) and len(prov["raw_files"]) == 2
    # resumable: a second fetch reuses the cache and makes no API calls
    again = _MockClient(bars.iloc[:0])
    dbs.fetch("ES", "2023-12-20", "2024-01-10", client=again, raw_dir=tmp_path / "raw", out_dir=tmp_path)
    assert again.calls == []
    # output loads through the normal pipeline with the contract column intact
    ds = build_dataset(tmp_path / "ES_databento_front_1m.parquet", make_instrument(symbol="ES"), calendar=TradingCalendar("XNYS"))
    assert ds.report.ok and "contract" in ds.bars.columns
    cost = dbs.estimate_cost("ES", "2023-12-20", "2024-01-10", client=client)
    assert cost["total_usd"] == pytest.approx(2.5)


def test_missing_key_is_explained(monkeypatch):
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    with pytest.raises(dbs.DatabentoUnavailable, match="DATABENTO_API_KEY"):
        dbs.get_client()


def test_budget_cap_blocks_paid_download(tmp_path, monkeypatch):
    monkeypatch.setattr(dbs, "DATA_DIR", tmp_path)
    dates = list(pd.bdate_range("2023-12-26", "2024-01-05").strftime("%Y-%m-%d"))
    bars = _contract_bars("ESH4", dates, 5000.0, [100] * len(dates)).rename(columns={"contract": "symbol"})
    client = _MockClient(bars)  # quotes $1.25 per year-chunk
    with pytest.raises(dbs.BudgetExceeded, match="Nothing was downloaded"):
        dbs.fetch("ES", "2023-12-20", "2024-01-10", client=client, out_dir=tmp_path, budget_usd=2.0)
    assert client.calls == []  # no paid request was made
    dbs.fetch("ES", "2023-12-20", "2024-01-10", client=client, out_dir=tmp_path, budget_usd=2.5)
    assert dbs.spend_ledger()["spent_estimate_usd"] == pytest.approx(2.5)
    # cached chunks cost nothing: a re-run within a zero remaining budget still works
    dbs.fetch("ES", "2023-12-20", "2024-01-10", client=_MockClient(bars.iloc[:0]), out_dir=tmp_path, budget_usd=2.5)
    # any new download beyond the cap is refused
    with pytest.raises(dbs.BudgetExceeded):
        dbs.fetch("ES", "2024-01-10", "2024-02-01", client=client, out_dir=tmp_path, budget_usd=2.5)

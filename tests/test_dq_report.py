"""Real-data DQ report: must PASS aligned data and STOP data whose timestamps are an hour off in part of the history."""

import pandas as pd

from orb_lab.engine.pipeline import build_dataset
from orb_lab.engine.synthetic import generate_bars
from orb_lab.reports.dq_report import evaluate, write_report

from .conftest import make_instrument

INST = make_instrument(symbol="ES", calendar="XNYS")


def _bars():
    return generate_bars(INST, "2021-01-04", "2022-06-30", seed=5)


def test_aligned_data_passes_timing_checks(tmp_path):
    ds = build_dataset(_bars(), INST)
    rdq = evaluate(ds)
    assert (rdq.monthly_timing["flag"] == "").all()
    assert rdq.summary["modal_peak_minute_et"] == 570
    assert rdq.verdict in ("PASS", "REVIEW") and not any("displaced" in r for r in rdq.reasons)
    assert len(rdq.dst_checks) >= 4
    path = write_report(rdq, ds, tmp_path)
    assert "09:30 ET alignment" in path.read_text()


def test_one_hour_error_in_part_of_history_stops_research():
    bars = _bars()
    # simulate a vendor that stamped summer 2021 in EST all along (bars appear one hour late in ET)
    ts = bars.index.to_series()
    bad = (ts >= "2021-03-15") & (ts < "2021-11-01")
    new_index = ts.where(~bad, ts + pd.Timedelta(hours=1))
    shifted = bars.set_axis(pd.DatetimeIndex(new_index.values, tz="UTC")).sort_index()
    shifted = shifted[~shifted.index.duplicated()]
    ds = build_dataset(shifted, INST, allow_errors=True)
    rdq = evaluate(ds)
    assert rdq.verdict == "STOP"
    flagged = set(rdq.monthly_timing.loc[rdq.monthly_timing["flag"] != "", "month"])
    assert {"2021-05", "2021-07", "2021-09"} <= flagged
    assert "2022-02" not in flagged

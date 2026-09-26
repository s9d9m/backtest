"""Lockbox, registry and research gates."""

import json

import numpy as np
import pandas as pd
import pytest

from orb_lab.engine.backtest import run_backtest
from orb_lab.engine.params import StrategyParams
from orb_lab.engine.synthetic import generate_bars
from orb_lab.research import lockbox, registry
from orb_lab.research.gates import ResearchGateError, research_dataset, run_dq_stage

from .conftest import make_instrument

INST = make_instrument(symbol="ES", calendar="XNYS")


@pytest.fixture(scope="module")
def data_file(tmp_path_factory):
    bars = generate_bars(INST, "2018-01-02", "2024-06-28", seed=8)
    path = tmp_path_factory.mktemp("d") / "es.parquet"
    bars.reset_index().rename(columns={"ts": "timestamp"}).to_parquet(path)
    return path


def test_gate_requires_dq_report_then_withholds_lockbox(data_file):
    with pytest.raises(ResearchGateError, match="no data-quality report"):
        research_dataset(data_file, INST)
    ds, rdq, report, lock = run_dq_stage(data_file, INST)
    assert rdq.verdict in ("PASS", "REVIEW") and report.exists()
    assert lock["status"] == "sealed" and lock["lockbox_start"] == "2023-06-29"
    dev, lock2 = research_dataset(data_file, INST)
    assert str(dev.prep.dates[-1]) < "2023-06-29" and lock2 == lock
    # every strategy result on the development view ends before the lockbox
    res = run_backtest(dev.prep, StrategyParams())
    assert res.trades.session_date.max() < pd.Timestamp("2023-06-29")
    # the lockbox is immutable: re-running DQ does not move it
    _, _, _, again = run_dq_stage(data_file, INST)
    assert again["lockbox_start"] == "2023-06-29"
    kinds = [e["kind"] for e in registry.read_events()]
    assert kinds.count("lockbox_define") == 1 and kinds.count("dq_report") == 2


def test_stop_verdict_blocks_research(data_file, monkeypatch):
    registry.log_event("dq_report", symbol="ES", data_hash=__import__("hashlib").sha256(data_file.read_bytes()).hexdigest(),
                       verdict="STOP", reasons=["timezone"], source=str(data_file))
    with pytest.raises(ResearchGateError, match="STOP"):
        research_dataset(data_file, INST)


def test_lockbox_unlock_once_with_frozen_candidate(data_file, tmp_path):
    ds, _, _, lock = run_dq_stage(data_file, INST)
    with pytest.raises(lockbox.LockboxError):
        lockbox.run_lockbox_test("ES", ds.prep, lambda p, s: {})
    with pytest.raises(lockbox.LockboxError, match="frozen candidate"):
        lockbox.unlock("ES", tmp_path / "missing.json")
    frozen = tmp_path / "candidate.json"
    frozen.write_text(json.dumps(StrategyParams().to_dict()))
    lockbox.unlock("ES", frozen)
    calls = []

    def evaluate(prep, start):
        calls.append(start)
        r = run_backtest(prep, StrategyParams(), start=start)
        return {"n_trades": r.metrics["n_trades"], "first": str(r.trades.session_date.min().date())}

    first = lockbox.run_lockbox_test("ES", ds.prep, evaluate)
    second = lockbox.run_lockbox_test("ES", ds.prep, evaluate)
    assert calls == ["2023-06-29"] and first == second and first["first"] >= "2023-06-29"
    with pytest.raises(lockbox.LockboxError):
        lockbox.unlock("ES", frozen)


def test_frozen_candidate_cannot_change_after_unlock(data_file, tmp_path):
    ds, _, _, _ = run_dq_stage(data_file, INST)
    frozen = tmp_path / "c.json"
    frozen.write_text("{}")
    lockbox.unlock("ES", frozen)
    frozen.write_text('{"target_r": 1.2}')
    with pytest.raises(lockbox.LockboxError, match="changed"):
        lockbox.run_lockbox_test("ES", ds.prep, lambda p, s: {})


def test_short_history_does_not_use_a_lockbox():
    from orb_lab.engine.pipeline import build_dataset

    ds = build_dataset(generate_bars(INST, "2022-01-03", "2023-12-29", seed=1), INST)
    entry = lockbox.define_lockbox("ES", ds.prep)
    assert entry["status"] == "not_used_insufficient_history" and entry["lockbox_start"] is None


def test_hypothesis_lifecycle():
    h = registry.add_hypothesis("6E OR 10-20m beats 5m OOS", status="EXPLORATORY", source="wfo-x OOS heatmap", data_seen="OOS 2016-2024")
    hid = h["hypothesis_id"]
    with pytest.raises(ValueError):
        registry.add_hypothesis("x", status="CONFIRMED_OOS", source="", data_seen="")
    with pytest.raises(ValueError, match="untouched"):
        registry.update_hypothesis(hid, "CONFIRMED_OOS", "t=3", "")
    registry.update_hypothesis(hid, "REJECTED", "no effect in lockbox", "lockbox 2025")
    state = registry.hypotheses()[hid]
    assert state["status"] == "REJECTED" and state["history"] == ["EXPLORATORY", "REJECTED"]
    assert state["ts_utc"] < registry.read_events("hypothesis_update")[-1]["ts_utc"]

"""Monte Carlo, execution stress, robustness sweeps, session-based walk-forward (leakage), pipeline state
(split / freeze / blind ledger), background jobs, final-report language, metric help and diagnostics."""

from __future__ import annotations

import json
import stat
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from orb_lab import jobs
from orb_lab.engine.params import ExecutionParams, StrategyParams
from orb_lab.engine.pipeline import build_dataset
from orb_lab.engine.sessions import TradingCalendar
from orb_lab.engine.synthetic import generate_bars
from orb_lab.optimization.monte_carlo import MCConfig, equity_paths, longest_losing_streak, max_drawdown, run_monte_carlo, trade_r
from orb_lab.optimization.parameter_space import ParameterSpace
from orb_lab.optimization.robustness import neighbour_index
from orb_lab.optimization.stats_cube import build_session_cube, session_blocks
from orb_lab.optimization.stress import adverse_entry, stress_test
from orb_lab.optimization.sweeps import axis_summary, one_at_a_time, overall_verdict, pair_grid
from orb_lab.optimization.walk_forward import SelectionConfig, WFOStructure, run_wfo
from orb_lab.optimization.wfo_runner import run_experiment
from orb_lab.reports.final_report import build_markdown, classify
from orb_lab.reports.metric_help import cautions, metric_table
from orb_lab.research import diagnostics as dg
from orb_lab.research import pipeline_state as ps

from .conftest import make_instrument

INST = make_instrument(symbol="ES", commission_per_side=0.85, exchange_fee_per_side=1.40, slippage_ticks=1.0)
CAND = StrategyParams(range_minutes=15, entry_tf=5, target_r=1.0, cutoff="11:00")
SMALL = ParameterSpace("small", {"range_minutes": [10, 15, 30], "entry_tf": [5], "stop": ["or_mid", "or_opposite"], "target_r": [1.0, 1.5],
                                 "cutoff": ["11:00", "12:00"]})


def _bars(seed=5, start="2021-01-04", end="2021-12-30", trend=0.3):
    return generate_bars(INST, start, end, seed=seed, trend_strength=trend, calendar="WEEKDAYS")


def _prep(bars):
    return build_dataset(bars, INST, calendar=TradingCalendar("WEEKDAYS")).prep


@pytest.fixture(scope="module")
def prep():
    return _prep(_bars())


# ----------------------------------------------------------------------------------------- Monte Carlo
R = np.array([1.0, -1.0, 1.5, -1.0, -1.0, 2.0, -1.0, 0.5, -1.0, 1.0])


@pytest.mark.parametrize("method", ["reshuffle", "bootstrap", "block_bootstrap", "missed_trades"])
def test_monte_carlo_is_reproducible_with_a_seed(method):
    cfg = MCConfig(method=method, n_sims=300, seed=42)
    a, b = run_monte_carlo(R, cfg), run_monte_carlo(R, cfg)
    pd.testing.assert_frame_equal(a.sims, b.sims)
    c = run_monte_carlo(R, MCConfig(method=method, n_sims=300, seed=43))
    if method != "reshuffle":
        assert not a.sims["sum_r"].equals(c.sims["sum_r"])


def test_reshuffle_keeps_the_same_trades_and_fixed_risk_ending_equity():
    res = run_monte_carlo(R, MCConfig(method="reshuffle", n_sims=500, seed=1, starting_equity=40_000, risk_pct=0.01))
    assert np.allclose(res.sims["sum_r"], R.sum())
    assert np.allclose(res.sims["ending_equity"], 40_000 + R.sum() * 400)  # $400 per R, no compounding
    assert res.historical["ending_equity"] == pytest.approx(40_000 + R.sum() * 400)
    assert (res.sims["max_drawdown"] <= 0).all()


def test_bootstrap_resamples_only_historical_outcomes_and_cost_stress_shifts_them():
    res = run_monte_carlo(R, MCConfig(method="bootstrap", n_sims=200, seed=3, extra_cost_r=0.1))
    stressed = res.sims["sum_r"].to_numpy()
    plain = run_monte_carlo(R, MCConfig(method="bootstrap", n_sims=200, seed=3)).sims["sum_r"].to_numpy()
    assert np.allclose(stressed, plain - 0.1 * len(R))


def test_missed_trades_and_block_bootstrap_shapes():
    rng_cfg = MCConfig(method="missed_trades", n_sims=1000, seed=0, skip_prob=0.5)
    res = run_monte_carlo(R, rng_cfg)
    assert res.sims["sum_r"].std() > 0
    block = run_monte_carlo(np.arange(1.0, 21.0), MCConfig(method="block_bootstrap", n_sims=50, seed=0, block_size=5))
    assert block.sample_paths.shape == (50, 21)


def test_equity_path_math_compounding_and_drawdown():
    cfg = MCConfig(sizing="compounding", starting_equity=100.0, risk_pct=0.1)
    path = equity_paths(np.array([[1.0, -1.0]]), cfg)[0]
    assert np.allclose(path, [100.0, 110.0, 99.0])
    assert max_drawdown(np.array([[100.0, 120.0, 90.0, 130.0]]))[0] == pytest.approx(-0.25)
    assert longest_losing_streak(np.array([[1, -1, -1, 0, -1, 1, -1]]))[0] == 3  # a scratch (0 R) does not reset a streak


def test_monte_carlo_rejects_bad_inputs_and_trade_r_skips_sized_out():
    with pytest.raises(ValueError):
        run_monte_carlo(np.array([1.0]), MCConfig())
    with pytest.raises(ValueError):
        MCConfig(n_sims=0).validate()
    with pytest.raises(ValueError):
        MCConfig(method="magic").validate()
    t = pd.DataFrame({"r_multiple": [1.0, -1.0, 2.0], "sized_out": [False, True, False],
                      "entry_time": pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])})
    assert trade_r(t).tolist() == [2.0, 1.0]


# ----------------------------------------------------------------------------------------- stress
def test_stress_test_scenarios_are_worse_than_baseline(prep):
    table, info = stress_test(prep, CAND, ExecutionParams())
    base = table.iloc[0]
    assert base["scenario"].startswith("baseline")
    for _, row in table.iloc[1:].iterrows():
        assert row["expectancy_r"] <= base["expectancy_r"] + 1e-9, row["scenario"]
    slip = table[table["group"] == "slippage"]["expectancy_r"].to_numpy()
    assert np.all(np.diff(slip) <= 1e-12)  # monotone in slippage
    assert info["verdict"] in ("ROBUST", "FRAGILE", "NOT POSITIVE")


def test_stress_verdicts_on_planted_edge_and_null():
    edge = _prep(_bars(trend=1.5))
    _, info = stress_test(edge, CAND, ExecutionParams())
    assert info["verdict"] == "ROBUST"
    null = _prep(_bars(trend=0.0, seed=11))
    _, info = stress_test(null, CAND, ExecutionParams())
    assert info["verdict"] in ("NOT POSITIVE", "FRAGILE")


def test_adverse_entry_approximation_accounting():
    t = pd.DataFrame({"net_pnl_pc": [100.0], "risk_per_contract": [100.0], "qty": [2.0], "total_cost": [10.0], "net_pnl": [200.0],
                      "r_multiple": [1.0]})
    out = adverse_entry(t, 2.0, 12.5)
    assert out["net_pnl_pc"].iloc[0] == 75.0 and out["risk_per_contract"].iloc[0] == 125.0
    assert out["r_multiple"].iloc[0] == pytest.approx(0.6) and out["net_pnl"].iloc[0] == 150.0


# ----------------------------------------------------------------------------------------- robustness sweeps
def test_one_at_a_time_sweep_marks_candidate_and_classifies(prep):
    table = one_at_a_time(prep, CAND, ExecutionParams(), ["target_r", "range_minutes", "stop"])
    for axis in ("target_r", "range_minutes", "stop"):
        g = table[table["axis"] == axis]
        assert g["is_candidate"].sum() == 1
        assert g.loc[g["is_candidate"], "avg_r_vs_candidate"].iloc[0] == pytest.approx(0.0)
    summary = axis_summary(table, CAND)
    assert set(summary["classification"]) <= {"plateau", "spike", "mixed", "candidate not positive", "no neighbours"}
    assert isinstance(overall_verdict(summary), str)


def test_axis_summary_rules():
    table = pd.DataFrame({"axis": ["target_r"] * 5, "value": ["0.5", "0.75", "1.0", "1.25", "1.5"],
                          "is_candidate": [False, False, True, False, False], "avg_r": [0.08, 0.09, 0.10, 0.09, 0.07]})
    assert axis_summary(table, CAND)["classification"].iloc[0] == "plateau"
    table["avg_r"] = [-0.05, -0.02, 0.30, -0.01, -0.04]
    assert axis_summary(table, CAND)["classification"].iloc[0] == "spike"
    assert overall_verdict(axis_summary(table, CAND)).startswith("SPIKE")


def test_pair_grid_contains_candidate_cell(prep):
    grid = pair_grid(prep, CAND, ExecutionParams(), "target_r", "range_minutes")
    assert grid["is_candidate"].sum() == 1
    assert grid[grid["is_candidate"]].iloc[0]["target_r"] == "1.0"


def test_sweep_inserts_off_list_candidate_value(prep):
    odd = StrategyParams(range_minutes=15, entry_tf=5, target_r=1.1, cutoff="11:00")
    table = one_at_a_time(prep, odd, ExecutionParams(), ["target_r"])
    assert table["is_candidate"].sum() == 1


# ----------------------------------------------------------------------------------------- session-based WFO
def test_session_blocks_and_pre_group():
    prep = _prep(_bars(end="2021-03-31"))
    groups, labels, bounds = session_blocks(prep, 5, start=str(prep.dates[10]))
    assert labels[0] == "pre" and (groups[:10] == 0).all() and (groups[10:15] == 1).all()
    assert bounds[1][0] == str(prep.dates[10]) and bounds[1][1] == str(prep.dates[14])


def test_session_wfo_perturbing_oos_never_changes_earlier_selections():
    bars = _bars()
    prep = _prep(bars)
    configs = SMALL.expand().configs
    nbrs = neighbour_index(configs)
    struct = WFOStructure("s", 12, 4, 4, 4, "sessions")  # blocks of 4 sessions
    sel = SelectionConfig(n_finalists=5, min_trades_train=5, min_trades_val=2)
    cube = build_session_cube(prep, configs, ExecutionParams(), 4)
    res = run_wfo(prep, cube, nbrs, struct, ExecutionParams(), selection=sel, min_first=1, slippage_grid=(1.0,))
    assert len(res.windows) > 10
    for _, w in res.windows.iterrows():
        assert pd.Timestamp(w["train_end"]) < pd.Timestamp(w["val_start"]) <= pd.Timestamp(w["val_end"]) < pd.Timestamp(w["oos_start"])
    k = 4
    cut = pd.Timestamp(res.windows.loc[k, "oos_start"])
    alt = _bars(seed=99)
    utc_cut = cut.tz_localize("America/New_York").tz_convert("UTC")
    prep2 = _prep(pd.concat([bars[bars.index < utc_cut], alt[alt.index >= utc_cut]]))
    res2 = run_wfo(prep2, build_session_cube(prep2, configs, ExecutionParams(), 4), nbrs, struct, ExecutionParams(), selection=sel,
                   min_first=1, slippage_grid=(1.0,))
    assert list(res.windows.loc[:k, "config_key"]) == list(res2.windows.loc[:k, "config_key"])
    assert list(res.windows.loc[:k, "frozen_sha256"]) == list(res2.windows.loc[:k, "frozen_sha256"])
    pd.testing.assert_frame_equal(res.finalists[res.finalists.window <= k].reset_index(drop=True),
                                  res2.finalists[res2.finalists.window <= k].reset_index(drop=True))
    assert not res.oos_trades[res.oos_trades.session_date >= cut].equals(res2.oos_trades[res2.oos_trades.session_date >= cut])


def test_run_experiment_sessions_with_start_and_custom_run_dir(tmp_path):
    prep = _prep(_bars())
    struct = WFOStructure("sess", 40, 20, 20, 20, "sessions")
    start = str(prep.dates[30])
    run_dir, results = run_experiment(prep, SMALL, ExecutionParams(), structures=[struct], families=["all"], root=tmp_path,
                                      selection=SelectionConfig(n_finalists=5, min_trades_train=5, min_trades_val=2), start=start,
                                      run_dir=tmp_path / "out", starting_equity=40_000, risk_pct=0.005)
    assert run_dir == tmp_path / "out" and (run_dir / "wfo_overview.csv").exists()
    w = results[("sess", "all")].windows
    assert pd.Timestamp(w["train_start"].min()) >= pd.Timestamp(start)
    with pytest.raises(ValueError):
        run_experiment(prep, SMALL, ExecutionParams(), structures=[struct, WFOStructure("m", 3, 1, 1, 1)], families=["all"], root=tmp_path)


def test_month_structures_unchanged_presets():
    from orb_lab.optimization.walk_forward import STRUCTURES

    assert asdict(STRUCTURES["primary_12_3_3"]) == {"name": "primary_12_3_3", "train_months": 12, "val_months": 3, "oos_months": 3,
                                                     "roll_months": 3, "unit": "months"}
    for k, v in (("sens_24_3_3", (24, 3, 3)), ("sens_24_6_6", (24, 6, 6)), ("sens_36_6_6", (36, 6, 6))):
        s = STRUCTURES[k]
        assert (s.train_months, s.val_months, s.oos_months, s.roll_months) == (*v, v[2])
    with pytest.raises(ValueError):
        WFOStructure("bad", 10, 2, 2, 3).validate()


# ----------------------------------------------------------------------------------------- pipeline state
def test_split_is_chronological_and_complete():
    dates = pd.bdate_range("2024-01-01", periods=42).values.astype("datetime64[D]")
    sp = ps.make_split(dates, 0.6, 0.2)
    assert sp.n_train + sp.n_val + sp.n_holdout == 42 and sp.n_holdout >= 1
    assert sp.train_end < sp.val_start <= sp.val_end < sp.holdout_start
    with pytest.raises(ValueError):
        ps.make_split(dates[:2])
    with pytest.raises(ValueError):
        ps.make_split(dates, 0.8, 0.3)


def test_freeze_is_hashed_immutable_and_blind_test_is_one_shot():
    rec = ps.freeze_candidate("abc123", "cand", CAND.to_dict(), ExecutionParams().to_dict(), {"mode": "fixed_contracts"}, INST.to_dict())
    assert len(rec["sha256"]) == 64 and ps.verify_frozen(rec)
    assert not (stat.S_IMODE((__import__("os").stat(rec["path"]).st_mode)) & stat.S_IWUSR)  # read-only
    again = ps.freeze_candidate("abc123", "cand", CAND.to_dict(), ExecutionParams().to_dict(), {"mode": "fixed_contracts"}, INST.to_dict())
    assert again["path"] == rec["path"]
    tampered = {**rec, "strategy": {**rec["strategy"], "target_r": 9.0}}
    assert not ps.verify_frozen(tampered)
    first = ps.record_blind_test("abc123", rec, ("2024-03-01", "2024-03-31"), {"n_trades": 5})
    second = ps.record_blind_test("abc123", rec, ("2024-03-01", "2024-03-31"), {"n_trades": 5})
    assert first["status"] == "BLIND" and second["status"].startswith("NOT BLIND")
    ps.save_split("abc123", ps.make_split(pd.bdate_range("2024-01-01", periods=30).values.astype("datetime64[D]")))
    assert ps.load_split("abc123")["holdout_already_used"] is True


# ----------------------------------------------------------------------------------------- jobs
def _job_spec():
    return {"structures": [asdict(WFOStructure("s", 40, 20, 20, 20, "sessions"))], "space": {"name": "small", "grid": SMALL.grid},
            "execution": {}, "selection": {"n_finalists": 5, "min_trades_train": 5, "min_trades_val": 2}, "families": ["all"],
            "start": None, "end": None, "n_workers": 1, "data": {"data_hash": "x"}}


def _wait(job_dir, states, timeout=240):
    for _ in range(timeout * 5):
        s = jobs.read_status(job_dir)
        if s["state"] in states:
            return s
        time.sleep(0.2)
    raise AssertionError(jobs.read_status(job_dir))


def test_background_job_runs_to_completion(prep):
    job = jobs.submit("wfo", _job_spec(), prep)
    s = _wait(job, jobs.TERMINAL)
    assert s["state"] == "done", s
    assert (job / "result" / "s" / "all" / "windows.csv").exists()
    assert jobs.list_jobs(kind="wfo")[0] == job


def test_job_stop_and_resume(prep):
    job = jobs.create_job("wfo", _job_spec(), prep)
    (job / "STOP").write_text("x")  # stop requested before the worker reports progress
    s = jobs.run_job(job)
    assert s["state"] == "stopped"
    jobs.resume(job)  # clears the stop flag and relaunches
    assert _wait(job, jobs.TERMINAL)["state"] == "done"


def test_failed_job_records_error(prep):
    spec = _job_spec()
    spec["structures"][0]["train_months"] = 100_000
    job = jobs.create_job("wfo", spec, prep)
    s = jobs.run_job(job)
    assert s["state"] == "failed" and "not enough history" in s["error"]


# ----------------------------------------------------------------------------------------- report / metric help
def test_final_report_verdict_language():
    assert classify({})[0] == "INSUFFICIENT EVIDENCE"
    wfo = {"stitched_always_trade": {"n_trades": 20, "avg_r": 0.3, "t_stat_r": 3.0}}
    assert classify({"wfo": wfo})[0] == "INSUFFICIENT EVIDENCE"
    wfo = {"stitched_always_trade": {"n_trades": 200, "avg_r": -0.05, "t_stat_r": -1.0}}
    assert classify({"wfo": wfo})[0] == "NO PRELIMINARY EVIDENCE"
    wfo = {"stitched_always_trade": {"n_trades": 200, "avg_r": 0.05, "t_stat_r": 1.0}}
    assert classify({"wfo": wfo})[0] == "MIXED"
    wfo = {"stitched_always_trade": {"n_trades": 400, "avg_r": 0.12, "t_stat_r": 3.1}, "overfit_flags": []}
    assert classify({"wfo": wfo})[0] == "PROMISING"
    assert classify({"wfo": wfo, "stress": {"verdict": "FRAGILE"}})[0] == "MIXED"
    blind = {"status": "NOT BLIND (holdout already used)", "metrics": {"n_trades": 400, "avg_r": 0.5, "t_stat_r": 5}}
    assert classify({"blind": blind})[0] == "INSUFFICIENT EVIDENCE"  # a reused holdout is not evidence
    md = build_markdown({"wfo": wfo, "data": {"proxy": True}, "train": {"n_trades": 10, "avg_r": 0.5}})
    assert "NOT FUTURES VALIDATION" in md and "NOT evidence" in md and "PROMISING" in md


def test_metric_help_cautions_short_samples():
    m = {"n_trades": 12, "years": 0.2, "t_stat_r": 1.1, "avg_r": 0.2, "sharpe": 3.0, "cagr": 2.5, "top5_share": 0.8, "n_long": 6,
         "n_short": 6, "pnl_long": 100.0, "pnl_short": -50.0, "ambiguous_exit_pct": 0.2}
    text = " ".join(cautions(m))
    for phrase in ("Only 12 trades", "Annualised", "not statistically distinguishable", "5 best trades", "one side", "ambiguous"):
        assert phrase.lower() in text.lower(), phrase
    table = metric_table(m, ["sharpe", "cagr", "avg_r"])
    assert (table.loc[table["metric"].str.contains("Sharpe|CAGR"), "caution"] != "").all()
    assert cautions({"n_trades": 0}) == ["No trades: nothing can be concluded."]


# ----------------------------------------------------------------------------------------- diagnostics
def test_diagnostics_on_loaded_data(prep):
    assert dg.lookahead_truncation(prep, CAND)["passed"]
    assert dg.cost_units(INST)["passed"]
    rd = dg.random_direction(prep, CAND, n_sims=300)
    assert 0.0 <= rd["table"]["p_value_random_ge_strategy"].iloc[0] <= 1.0


def test_synthetic_null_and_planted_edge_checks_pass():
    assert dg.null_control(years=1)["passed"]
    assert dg.planted_edge(years=1)["passed"]


# ----------------------------------------------------------------------------------------- malformed data
def test_malformed_and_missing_data_are_reported(tmp_path):
    from orb_lab.engine.pipeline import DataQualityError

    bars = _bars(end="2021-01-29").reset_index().rename(columns={"index": "timestamp"})
    ts_col = bars.columns[0]
    bad = bars.copy()
    bad.loc[5, "high"] = bad.loc[5, "low"] - 10  # high below low
    f = tmp_path / "bad.csv"
    bad.rename(columns={ts_col: "timestamp"}).to_csv(f, index=False)
    with pytest.raises(DataQualityError):
        build_dataset(f, INST, calendar=TradingCalendar("WEEKDAYS"))
    missing = tmp_path / "missing.csv"
    bars.drop(columns=["close"]).rename(columns={ts_col: "timestamp"}).to_csv(missing, index=False)
    with pytest.raises(Exception):
        build_dataset(missing, INST, calendar=TradingCalendar("WEEKDAYS"))

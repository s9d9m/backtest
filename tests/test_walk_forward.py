"""Walk-forward segmentation, cube correctness and leakage tests."""

import numpy as np
import pandas as pd
import pytest

from orb_lab.engine.backtest import run_backtest
from orb_lab.engine.params import ExecutionParams, SizingParams, StrategyParams
from orb_lab.engine.pipeline import build_dataset
from orb_lab.engine.sessions import TradingCalendar
from orb_lab.engine.synthetic import generate_bars
from orb_lab.optimization.parameter_space import ParameterSpace
from orb_lab.optimization.robustness import classify_plateau, neighbour_index, neighbour_stats
from orb_lab.optimization.stats_cube import StatsCube, build_cube, metrics_from_stats
from orb_lab.optimization.walk_forward import (
    STRUCTURES,
    SelectionConfig,
    WFOStructure,
    generate_windows,
    phase_metrics,
    run_wfo,
    select_window,
)

from .conftest import make_instrument

INST = make_instrument(symbol="ES", commission_per_side=0.85, exchange_fee_per_side=1.40, slippage_ticks=1.0)
SPACE = ParameterSpace("wfo_test", {
    "range_minutes": [10, 15, 30], "entry_tf": [5, 10], "entry_method": ["market", "stop"],
    "stop": ["or_mid", "or_opposite"], "target_r": [0.75, 1.0, 1.5], "cutoff": ["11:00", "12:00"], "direction": ["both", "long"],
})
EXEC = ExecutionParams()
SEL = SelectionConfig(n_finalists=10, min_trades_train=30, min_trades_val=5)


def _bars(seed=21, start="2019-01-01", end="2021-12-31"):
    return generate_bars(INST, start, end, seed=seed, trend_strength=0.4, calendar="WEEKDAYS")


def _prep(bars):
    return build_dataset(bars, INST, calendar=TradingCalendar("WEEKDAYS")).prep


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    prep = _prep(_bars())
    configs = SPACE.expand().configs
    cube = build_cube(prep, configs, EXEC, tmp_path_factory.mktemp("cubes"))
    return prep, configs, cube, neighbour_index(configs)


def test_window_generation_matches_research_plan_example():
    months = [str(p) for p in pd.period_range("2020-01", "2022-12", freq="M")]
    w = generate_windows(months, STRUCTURES["primary_12_3_3"])
    assert w[0].labels == {"train": "2020-01..2020-12", "val": "2021-01..2021-03", "oos": "2021-04..2021-06"}
    assert w[1].labels == {"train": "2020-04..2021-03", "val": "2021-04..2021-06", "oos": "2021-07..2021-09"}
    assert len(w) == 7
    for a, b in zip(w, w[1:]):
        assert a.oos[1] == b.oos[0]  # contiguous, non-overlapping OOS
    for win in w:
        assert win.train[1] == win.val[0] and win.val[1] == win.oos[0] and win.train[1] - win.train[0] == 12


@pytest.mark.parametrize("name,lengths", [("sens_24_3_3", (24, 3, 3)), ("sens_24_6_6", (24, 6, 6)), ("sens_36_6_6", (36, 6, 6))])
def test_sensitivity_structures(name, lengths):
    months = [str(p) for p in pd.period_range("2015-01", "2024-12", freq="M")]
    w = generate_windows(months, STRUCTURES[name])
    assert all((x.train[1] - x.train[0], x.val[1] - x.val[0], x.oos[1] - x.oos[0]) == lengths for x in w)
    assert w[-1].oos[1] <= len(months)
    with pytest.raises(ValueError):
        WFOStructure("bad", 12, 3, 3, 1).validate()


def test_cube_view_refuses_oos_months(setup):
    _, _, cube, _ = setup
    view = cube.view(0, 15)
    view.aggregate(0, 12)
    with pytest.raises(PermissionError):
        view.aggregate(12, 18)
    with pytest.raises(PermissionError):
        view.sessions(15, 18)


def test_cube_equals_direct_backtest(setup):
    prep, configs, cube, _ = setup
    for i in (0, 7, len(configs) - 1):
        m0, m1 = 3, 9
        start = pd.Period(cube.months[m0], "M").start_time
        end = pd.Period(cube.months[m1 - 1], "M").end_time
        direct = run_backtest(prep, configs[i], EXEC, SizingParams(), start=start, end=end).metrics
        fromcube = metrics_from_stats(cube.aggregate(m0, m1, idx=np.array([i])), cube.sessions(m0, m1)).iloc[0]
        assert fromcube["n_trades"] == direct["n_trades"]
        assert fromcube["net_pnl"] == pytest.approx(direct["net_pnl"], rel=1e-5, abs=0.01)
        assert fromcube["exp_r"] == pytest.approx(direct["avg_r"], rel=1e-5, abs=1e-6)
        assert fromcube["win_rate"] == pytest.approx(direct["win_rate"])
        assert fromcube["t_stat_r"] == pytest.approx(direct["t_stat_r"], rel=1e-4, abs=1e-6)


def test_selection_is_unaffected_by_scrambled_oos_and_future_months(setup, tmp_path):
    _, configs, cube, nbrs = setup
    windows = generate_windows(cube.months, STRUCTURES["primary_12_3_3"])
    rng = np.random.default_rng(0)
    for w in windows:
        base = select_window(cube.view(w.train[0], w.val[1]), w, nbrs, np.ones(len(configs), bool), SEL)
        corrupted = np.array(cube.data, dtype=np.float32)
        corrupted[:, w.val[1]:, :] = rng.normal(size=corrupted[:, w.val[1]:, :].shape) * 1e4
        fake = StatsCube(tmp_path, cube.months, cube.sessions_per_month, configs, corrupted, "x")
        other = select_window(fake.view(w.train[0], w.val[1]), w, nbrs, np.ones(len(configs), bool), SEL)
        assert base["selected"] == other["selected"]
        pd.testing.assert_frame_equal(base["finalists"], other["finalists"])


def test_end_to_end_perturbing_oos_bars_never_changes_earlier_selections(tmp_path):
    bars = _bars()
    prep = _prep(bars)
    configs = SPACE.expand().configs
    nbrs = neighbour_index(configs)
    res = run_wfo(prep, build_cube(prep, configs, EXEC, tmp_path / "a"), nbrs, STRUCTURES["primary_12_3_3"], EXEC, selection=SEL,
                  slippage_grid=(1.0,))
    k = 2
    cut = pd.Timestamp(res.windows.loc[k, "oos_start"])
    alt = _bars(seed=99)
    mixed = pd.concat([bars[bars.index < cut.tz_localize("America/New_York").tz_convert("UTC")],
                       alt[alt.index >= cut.tz_localize("America/New_York").tz_convert("UTC")]])
    prep2 = _prep(mixed)
    res2 = run_wfo(prep2, build_cube(prep2, configs, EXEC, tmp_path / "b"), nbrs, STRUCTURES["primary_12_3_3"], EXEC, selection=SEL,
                   slippage_grid=(1.0,))
    # windows 0..k were selected using data strictly before `cut`: identical choices AND identical finalist scores
    assert list(res.windows.loc[:k, "config_key"]) == list(res2.windows.loc[:k, "config_key"])
    f1 = res.finalists[res.finalists.window <= k].reset_index(drop=True)
    f2 = res2.finalists[res2.finalists.window <= k].reset_index(drop=True)
    pd.testing.assert_frame_equal(f1, f2)
    # ... while window k+1 (validation after the cut) is allowed to, and here does, see different data
    assert not res.finalists[res.finalists.window == k + 1].reset_index(drop=True).equals(res2.finalists[res2.finalists.window == k + 1].reset_index(drop=True))
    before = res.oos_trades[res.oos_trades.session_date < cut]
    before2 = res2.oos_trades[res2.oos_trades.session_date < cut]
    pd.testing.assert_frame_equal(before[["entry_time", "net_pnl"]].reset_index(drop=True), before2[["entry_time", "net_pnl"]].reset_index(drop=True))
    assert not res.oos_trades[res.oos_trades.session_date >= cut].equals(res2.oos_trades[res2.oos_trades.session_date >= cut])


def test_stitched_curve_contains_only_blind_oos(setup):
    prep, configs, cube, nbrs = setup
    res = run_wfo(prep, cube, nbrs, STRUCTURES["primary_12_3_3"], EXEC, selection=SEL, slippage_grid=(0.5, 1.0, 3.0))
    t = res.oos_trades
    assert len(t) > 0
    for _, w in res.windows.iterrows():
        sub = t[t.window == w.window]
        assert (sub.session_date >= pd.Timestamp(w.oos_start)).all() and (sub.session_date <= pd.Timestamp(w.oos_end)).all()
    first_oos = pd.Timestamp(res.windows.oos_start.min())
    assert res.stitched_daily.index.min() >= first_oos
    assert len(res.stitched_daily) == res.windows.oos_sessions.sum()
    assert res.stitched_metrics["n_trades"] == len(t)
    assert (res.windows["oos_exp_r"] - res.windows["oos_exp_r_check_cube"]).abs().max() < 1e-5
    assert list(res.slippage_sensitivity.slippage_ticks) == [0.5, 1.0, 3.0]
    assert res.slippage_sensitivity.net_pnl.is_monotonic_decreasing
    assert res.stitched_metrics_stand_aside["n_trades"] <= res.stitched_metrics["n_trades"]
    for key in ("pct_windows_oos_profitable", "median_train_exp_r", "median_oos_exp_r", "parameter_stability", "overfit_flags"):
        assert key in res.summary


def test_family_restriction(setup):
    prep, configs, cube, nbrs = setup
    mask = np.array([c.entry_method == "stop" for c in configs])
    res = run_wfo(prep, cube, nbrs, STRUCTURES["primary_12_3_3"], EXEC, selection=SEL, eligible=mask, slippage_grid=(1.0,))
    chosen = res.windows[res.windows.selected >= 0]
    assert (chosen["param_entry_method"] == "stop").all()


def test_neighbours_and_plateau_classification():
    configs = SPACE.expand().configs
    nbrs = neighbour_index(configs)
    i = next(k for k, c in enumerate(configs) if c.range_minutes == 15 and c.target_r == 1.0 and c.entry_method == "market"
             and c.stop_method == "or_mid" and c.entry_tf == 5 and c.cutoff == "11:00" and c.direction == "both")
    got = {(configs[j].range_minutes, configs[j].target_r, configs[j].stop_method, configs[j].entry_tf, configs[j].cutoff) for j in nbrs[i] if j >= 0}
    assert got == {(10, 1.0, "or_mid", 5, "11:00"), (30, 1.0, "or_mid", 5, "11:00"), (15, 0.75, "or_mid", 5, "11:00"),
                   (15, 1.5, "or_mid", 5, "11:00"), (15, 1.0, "or_opposite", 5, "11:00"), (15, 1.0, "or_mid", 10, "11:00"),
                   (15, 1.0, "or_mid", 5, "12:00")}
    score = np.array([1.0, 1.0, 1.0, 0.1])
    nb = np.array([[1, 2, -1], [0, 2, -1], [0, 1, -1], [-1, -1, -1]])
    med, mn, _ = neighbour_stats(score, nb)
    assert list(classify_plateau(score, med, mn)) == ["plateau", "plateau", "plateau", "isolated"]
    spike = np.array([1.0, -0.2, 0.05])
    med, mn, _ = neighbour_stats(spike, np.array([[1, 2], [0, 2], [0, 1]]))
    assert classify_plateau(spike, med, mn)[0] == "spike"


def test_phase_metrics(setup):
    _, configs, cube, _ = setup
    pm = phase_metrics(cube, STRUCTURES["primary_12_3_3"])
    assert len(pm) == len(configs)
    assert {"train_exp_r", "val_exp_r", "oos_exp_r", "stop", "confirmation"} <= set(pm.columns)


def test_run_experiment_all_structures_and_families(tmp_path):
    from orb_lab.optimization.wfo_runner import run_experiment
    from orb_lab.research import registry

    prep = _prep(_bars(start="2018-01-01", end="2021-12-31"))
    run_dir, results = run_experiment(prep, SPACE, EXEC, structures=["primary_12_3_3", "sens_24_6_6"], families=["all", "stop"],
                                      selection=SEL, research=True, root=tmp_path)
    overview = pd.read_csv(run_dir / "wfo_overview.csv")
    assert set(zip(overview.structure, overview.family)) == {(s, f) for s in ("primary_12_3_3", "sens_24_6_6") for f in ("all", "stop")}
    for s in ("primary_12_3_3", "sens_24_6_6"):
        for f in ("all", "stop"):
            d = run_dir / s / f
            assert {"windows.csv", "summary.json", "stitched_oos_daily.csv", "finalists.csv", "slippage_sensitivity.csv"} <= {p.name for p in d.iterdir()}
    assert (run_dir / "phase_metrics.parquet").exists()
    events = registry.read_events("experiment_wfo")
    assert events and events[-1]["n_configs"] == len(SPACE.expand().configs)
    # the 24-month structure starts its OOS one year later than the 12-month one
    o = overview.set_index(["structure", "family"])
    assert o.loc[("sens_24_6_6", "all"), "oos_first"] > o.loc[("primary_12_3_3", "all"), "oos_first"]

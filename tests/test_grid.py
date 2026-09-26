import json

import numpy as np
import pandas as pd
import pytest

from orb_lab.engine.backtest import run_backtest
from orb_lab.engine.params import ExecutionParams, SizingParams, StrategyParams
from orb_lab.engine.pipeline import build_dataset
from orb_lab.engine.sessions import TradingCalendar
from orb_lab.engine.synthetic import generate_bars
from orb_lab.optimization.grid_search import evaluate_config, results_table, run_grid
from orb_lab.optimization.heatmaps import heatmap_table, varying_parameters
from orb_lab.optimization.multiple_testing import effective_trials, expected_max_sharpe
from orb_lab.optimization.parameter_space import ParameterSpace, load_search_spaces, parse_stop

from .conftest import make_instrument

INST = make_instrument(symbol="ES", commission_per_side=0.85, exchange_fee_per_side=1.40, slippage_ticks=1.0)


@pytest.fixture(scope="module")
def prep():
    bars = generate_bars(INST, "2021-01-04", "2022-06-30", seed=3, trend_strength=0.8, calendar="WEEKDAYS")
    return build_dataset(bars, INST, calendar=TradingCalendar("WEEKDAYS")).prep


SPACE = ParameterSpace(
    "test",
    {
        "range_minutes": [10, 15],
        "entry_tf": [5, 10],
        "stop": ["or_opposite", "or_mid", "or_pct:0.5"],
        "target_r": [1.0, 1.5],
        "cutoff": ["11:00"],
        "direction": ["both", "long"],
    },
)


def test_parse_stop():
    assert parse_stop("or_pct:0.67") == ("or_pct", 0.67)
    assert parse_stop("or_mid") == ("or_mid", 0.0)


def test_expansion_dedupes_irrelevant_parameters():
    space = ParameterSpace("x", {"entry_method": ["market", "stop"], "entry_tf": [5, 10], "entry_buffer_ticks": [0, 1], "cutoff": ["09:40", "11:00"]})
    exp = space.expand()
    # cutoff 09:40 is before the 09:45 range end -> invalid (2*2*2 = 8 combos)
    assert exp.n_invalid == 8
    # market ignores buffer (2 -> 1); stop ignores entry_tf (2 -> 1)
    methods = [c.entry_method for c in exp.configs]
    assert methods.count("market") == 2 and methods.count("stop") == 2
    assert exp.n_duplicates == 4
    assert len({c.key() for c in exp.configs}) == len(exp.configs)


def test_builtin_spaces_expand():
    spaces = load_search_spaces()
    primary = spaces["primary_930"]
    exp = primary.expand()
    assert primary.raw_size() == 4 * 4 * 6 * 13 * 6 * 3 * 5
    # the 30-minute range ends at 10:00, so the 10:00 cutoff is invalid for it: 4*6*13*3*5 combos
    assert exp.n_invalid == 4 * 6 * 13 * 3 * 5
    assert len(exp.configs) == primary.raw_size() - exp.n_invalid
    assert all(c.orb_start == "09:30" and c.entry_method == "market" for c in exp.configs)


def test_fast_summary_matches_full_backtest(prep):
    params = StrategyParams(range_minutes=15, entry_tf=5, target_r=1.25, stop_method="or_mid", cutoff="12:00")
    d_lo, d_hi = prep.day_range()
    fast = evaluate_config(prep, params, ExecutionParams(), d_lo, d_hi, 100_000)
    full = run_backtest(prep, params, ExecutionParams(), SizingParams(mode="fixed_contracts", contracts=1)).metrics
    for key in ("n_trades", "net_pnl", "expectancy", "win_rate", "profit_factor", "sharpe", "sortino", "max_dd", "cagr", "calmar",
                "avg_r", "median_r", "t_stat_r", "longest_loss_streak", "top5_share", "pct_years_profitable", "max_dd_duration_days"):
        assert fast[key] == pytest.approx(full[key], rel=1e-9, abs=1e-9), key


def test_grid_resume_after_interruption(prep, tmp_path):
    full = run_grid(prep, SPACE, run_dir=tmp_path / "a", chunk_size=4)
    n = len(SPACE.expand().configs)
    assert len(full.results) == n == full.n_evaluated

    # simulate an interrupted run: evaluate, then delete some chunks, then resume
    partial_dir = tmp_path / "b"
    run_grid(prep, SPACE, run_dir=partial_dir, chunk_size=4)
    chunks = sorted((partial_dir / "chunks").glob("chunk_*.parquet"))
    for c in chunks[2:5]:
        c.unlink()
    calls = []
    resumed = run_grid(prep, SPACE, run_dir=partial_dir, chunk_size=4, progress=lambda d, t, e: calls.append(d))
    assert resumed.n_resumed_chunks == len(chunks) - 3
    assert calls[0] == n - 3 * 4 and calls[-1] == n  # progress starts from the checkpointed count
    cols = ["config_key", "net_pnl", "sharpe", "n_trades"]
    pd.testing.assert_frame_equal(full.results[cols], resumed.results[cols])


def test_resume_refuses_different_space(prep, tmp_path):
    run_grid(prep, SPACE, run_dir=tmp_path / "c", chunk_size=8)
    other = ParameterSpace("other", {**SPACE.grid, "target_r": [2.0]})
    with pytest.raises(ValueError, match="different configuration list"):
        run_grid(prep, other, run_dir=tmp_path / "c", chunk_size=8)


def test_parallel_equals_serial(prep, tmp_path):
    serial = run_grid(prep, SPACE, run_dir=tmp_path / "s", chunk_size=3, n_workers=1)
    parallel = run_grid(prep, SPACE, run_dir=tmp_path / "p", chunk_size=3, n_workers=3)
    cols = ["config_key", "net_pnl", "sharpe", "n_trades", "is_score", "rank_composite"]
    pd.testing.assert_frame_equal(serial.results[cols], parallel.results[cols])
    assert serial.selection_bias == parallel.selection_bias


def test_results_table_and_heatmap(prep, tmp_path):
    res = run_grid(prep, SPACE, run_dir=tmp_path / "h", chunk_size=50).results
    table = results_table(res)
    for col in ("Rank", "Market", "ORB Start", "Range Length", "Entry TF", "Stop Method", "R", "Cutoff", "Trades", "Win Rate",
                "Expectancy $", "Profit Factor", "CAGR", "Sharpe", "Sortino", "Max DD", "Calmar", "Net Return", "IS Score"):
        assert col in table.columns
    hm = heatmap_table(res, "target_r", "range_minutes", "sharpe", "median")
    assert hm.shape == (2, 2)
    cell = res[(res.range_minutes == 10) & (res.target_r == 1.5)]["sharpe"].median()
    assert hm.loc[10, 1.5] == pytest.approx(cell)
    assert set(varying_parameters(res)) == {"range_minutes", "entry_tf", "stop", "target_r", "direction"}
    assert res["rank_composite"].min() == 1


def test_multiple_testing_helpers():
    assert expected_max_sharpe(1, 1.0) == 0.0
    assert expected_max_sharpe(1000, 0.5) > expected_max_sharpe(10, 0.5) > 0
    rng = np.random.default_rng(0)
    base = rng.normal(size=(500, 1))
    identical = np.hstack([base] * 20)
    assert effective_trials(identical) == pytest.approx(1.0, abs=1e-6)
    independent = rng.normal(size=(5000, 20))
    assert effective_trials(independent) > 17

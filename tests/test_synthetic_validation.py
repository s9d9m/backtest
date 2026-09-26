"""Statistical sanity checks of the whole pipeline on synthetic data with a known answer.

* Null (driftless random walk): no configuration has a true edge. The frictionless mean R must be
  statistically indistinguishable from zero and costs must make the result negative on average.
* Planted edge (hidden intraday trend): a correct engine must detect a clearly positive edge.
"""

import numpy as np
import pytest

from orb_lab.engine.backtest import run_backtest
from orb_lab.engine.params import ExecutionParams, StrategyParams
from orb_lab.engine.pipeline import build_dataset
from orb_lab.engine.sessions import TradingCalendar
from orb_lab.engine.synthetic import generate_bars

from .conftest import make_instrument

INST = make_instrument(symbol="ES", commission_per_side=0.85, exchange_fee_per_side=1.40, slippage_ticks=1.0)
PARAMS = [
    StrategyParams(range_minutes=15, entry_tf=5, target_r=1.0, cutoff="11:00"),
    StrategyParams(range_minutes=30, entry_tf=10, target_r=1.5, cutoff="12:00", stop_method="or_mid"),
    StrategyParams(range_minutes=10, entry_method="stop", target_r=1.0, cutoff="11:30"),
]


def _dataset(trend):
    bars = generate_bars(INST, "2019-01-01", "2022-12-30", seed=123, trend_strength=trend, calendar="WEEKDAYS")
    return build_dataset(bars, INST, calendar=TradingCalendar("WEEKDAYS"))


@pytest.fixture(scope="module")
def null_ds():
    return _dataset(0.0)


@pytest.fixture(scope="module")
def edge_ds():
    return _dataset(1.5)


def _r(ds, params, frictionless):
    res = run_backtest(ds.prep, params, ExecutionParams(frictionless=frictionless))
    return res.trades["r_multiple"].to_numpy()


@pytest.mark.parametrize("params", PARAMS)
def test_null_model_has_no_frictionless_edge_and_loses_after_costs(null_ds, params):
    r0 = _r(null_ds, params, True)
    assert len(r0) > 700
    t = r0.mean() / r0.std(ddof=1) * np.sqrt(len(r0))
    assert abs(t) < 3.0, f"spurious edge on a random walk: mean R {r0.mean():.3f}, t={t:.2f}"
    r1 = _r(null_ds, params, False)
    assert r1.mean() < r0.mean()  # costs always hurt


@pytest.mark.parametrize("params", PARAMS)
def test_planted_trend_edge_is_detected(edge_ds, params):
    r0 = _r(edge_ds, params, True)
    t = r0.mean() / r0.std(ddof=1) * np.sqrt(len(r0))
    assert t > 3.0, f"engine failed to detect planted edge: mean R {r0.mean():.3f}, t={t:.2f}"

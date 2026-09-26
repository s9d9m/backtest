"""Milestone 3: rolling walk-forward optimisation with frozen out-of-sample evaluation.

For every window:

1. **Train**: every configuration is scored on the training months only (composite of Sharpe,
   t-stat of mean R and profit factor, shrunk for low trade counts). The score is blended with the
   median training score of the configuration's immediate parameter neighbours, so isolated spikes
   are penalised.
2. **Finalists**: the top ``n_finalists`` by that robust training score.
3. **Validation**: finalists are scored on the validation months only. The selection score blends
   the validation score and the robust training score.
4. **Freeze**: the selected configuration is fixed.
5. **OOS**: the frozen configuration is simulated trade by trade on the blind OOS months.

Only the OOS segments are stitched into the headline equity curve. Selection functions receive a
:class:`~orb_lab.optimization.stats_cube.CubeView` limited to ``[train_start, validation_end)``, so they
cannot read OOS months even by mistake (enforced and tested).

Two OOS variants are always reported. Both use only pre-OOS information:

* ``always_trade`` (primary): trade the selected configuration in every OOS window.
* ``stand_aside``: stay flat in windows where the selected configuration's validation expectancy
  was not positive.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..engine.backtest import run_backtest
from ..engine.execution import resolve_costs
from ..engine.metrics import compute_metrics
from ..engine.params import ExecutionParams, SizingParams, StrategyParams
from ..engine.portfolio import apply_sizing, daily_equity
from ..engine.session_data import PreparedData
from .parameter_space import config_frame_row
from .robustness import classify_plateau, neighbour_stats
from .stats_cube import CubeView, StatsCube, metrics_from_stats


@dataclass(frozen=True)
class WFOStructure:
    name: str
    train_months: int
    val_months: int
    oos_months: int
    roll_months: int

    def validate(self) -> None:
        if min(self.train_months, self.val_months, self.oos_months, self.roll_months) <= 0:
            raise ValueError("all WFO lengths must be positive")
        if self.roll_months != self.oos_months:
            raise ValueError("roll must equal the OOS length so stitched OOS segments neither overlap nor leave gaps")


STRUCTURES = {
    "primary_12_3_3": WFOStructure("primary_12_3_3", 12, 3, 3, 3),
    "sens_24_3_3": WFOStructure("sens_24_3_3", 24, 3, 3, 3),
    "sens_24_6_6": WFOStructure("sens_24_6_6", 24, 6, 6, 6),
    "sens_36_6_6": WFOStructure("sens_36_6_6", 36, 6, 6, 6),
    "model_C_48_12_12": WFOStructure("model_C_48_12_12", 48, 12, 12, 12),
}


@dataclass(frozen=True)
class SelectionConfig:
    n_finalists: int = 25
    min_trades_train: int = 40
    min_trades_val: int = 10
    neighbour_weight: float = 0.5
    validation_weight: float = 0.5
    w_sharpe: float = 0.35
    w_tstat: float = 0.35
    w_pf: float = 0.30


@dataclass
class Window:
    idx: int
    train: tuple[int, int]
    val: tuple[int, int]
    oos: tuple[int, int]
    labels: dict = field(default_factory=dict)


def usable_month_range(sessions_per_month: np.ndarray, min_share: float = 0.7) -> tuple[int, int]:
    """Drop a partial first/last month (fewer sessions than ``min_share`` of the median month)."""
    med = float(np.median(sessions_per_month[sessions_per_month > 0])) if np.any(sessions_per_month > 0) else 0.0
    lo, hi = 0, len(sessions_per_month)
    while lo < hi and sessions_per_month[lo] < min_share * med:
        lo += 1
    while hi > lo and sessions_per_month[hi - 1] < min_share * med:
        hi -= 1
    return lo, hi


def generate_windows(months: list[str], structure: WFOStructure, first: int = 0, last: int | None = None) -> list[Window]:
    structure.validate()
    last = len(months) if last is None else last
    windows = []
    start = first
    k = 0
    while True:
        t0, t1 = start, start + structure.train_months
        v1 = t1 + structure.val_months
        o1 = v1 + structure.oos_months
        if o1 > last:
            break
        windows.append(
            Window(
                k,
                (t0, t1),
                (t1, v1),
                (v1, o1),
                {
                    "train": f"{months[t0]}..{months[t1 - 1]}",
                    "val": f"{months[t1]}..{months[v1 - 1]}",
                    "oos": f"{months[v1]}..{months[o1 - 1]}",
                },
            )
        )
        start += structure.roll_months
        k += 1
    return windows


def objective(m: pd.DataFrame, min_trades: int, sel: SelectionConfig) -> np.ndarray:
    pf = np.clip(m["profit_factor"].replace(np.inf, 10.0).to_numpy(), 1e-3, 10.0)
    base = (
        sel.w_sharpe * np.clip(m["sharpe"].to_numpy(), -3, 3) / 3
        + sel.w_tstat * np.clip(m["t_stat_r"].to_numpy(), -5, 5) / 5
        + sel.w_pf * np.clip(np.log(pf), -1, 1)
    )
    n = m["n_trades"].to_numpy()
    factor = np.clip(n / max(min_trades, 1), 0, 1)
    score = np.where(base > 0, base * factor, base)
    return np.where(n > 0, score, -1.0)


def select_window(view: CubeView, window: Window, nbrs: np.ndarray, eligible: np.ndarray, sel: SelectionConfig) -> dict:
    """Choose the configuration for ``window`` from training and validation months only."""
    t0, t1 = window.train
    v0, v1 = window.val
    tr = metrics_from_stats(view.aggregate(t0, t1), view.sessions(t0, t1))
    s_tr = objective(tr, sel.min_trades_train, sel)
    valid = eligible & (tr["n_trades"].to_numpy() >= sel.min_trades_train)
    nbr_med, nbr_min, _ = neighbour_stats(s_tr, nbrs)
    plateau = np.where(np.isnan(nbr_med), 0.0, nbr_med)
    robust = (1 - sel.neighbour_weight) * s_tr + sel.neighbour_weight * plateau
    ranked = np.where(valid, robust, -np.inf)
    order = np.argsort(-ranked, kind="stable")
    finalists = order[: sel.n_finalists]
    finalists = finalists[np.isfinite(ranked[finalists])]
    best_train = int(np.argmax(np.where(valid, s_tr, -np.inf))) if valid.any() else -1
    if len(finalists) == 0:
        return {"selected": -1, "best_train": best_train, "finalists": pd.DataFrame(), "reason": "no configuration met the training trade minimum"}
    va = metrics_from_stats(view.aggregate(v0, v1, idx=finalists), view.sessions(v0, v1))
    s_va = objective(va, sel.min_trades_val, sel)
    s_va = np.where(va["n_trades"].to_numpy() >= sel.min_trades_val, s_va, s_va - 1.0)
    sel_score = sel.validation_weight * s_va + (1 - sel.validation_weight) * robust[finalists]
    pick = int(np.argmax(sel_score))
    table = pd.DataFrame(
        {
            "config_index": finalists,
            "train_score": s_tr[finalists],
            "neighbour_median_train_score": nbr_med[finalists],
            "robust_train_score": robust[finalists],
            "train_exp_r": tr["exp_r"].to_numpy()[finalists],
            "train_trades": tr["n_trades"].to_numpy()[finalists],
            "val_score": s_va,
            "val_exp_r": va["exp_r"].to_numpy(),
            "val_trades": va["n_trades"].to_numpy(),
            "selection_score": sel_score,
        }
    )
    chosen = int(finalists[pick])
    return {
        "selected": chosen,
        "best_train": best_train,
        "finalists": table,
        "train_metrics": tr.iloc[chosen].to_dict(),
        "val_metrics": va.iloc[pick].to_dict(),
        "best_train_metrics": tr.iloc[best_train].to_dict() if best_train >= 0 else {},
        "plateau_class": str(classify_plateau(s_tr[[chosen]], nbr_med[[chosen]], nbr_min[[chosen]])[0]),
        "stand_aside": bool(va["exp_r"].to_numpy()[pick] <= 0),
        "reason": "",
    }


def _month_dates(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    p = pd.Period(month, "M")
    return p.start_time.normalize(), p.end_time.normalize()


@dataclass
class WFOResult:
    structure: WFOStructure
    selection: SelectionConfig
    windows: pd.DataFrame
    oos_trades: pd.DataFrame
    stitched_daily: pd.DataFrame
    stitched_metrics: dict
    stitched_metrics_stand_aside: dict
    stitched_metrics_pct_equity: dict
    slippage_sensitivity: pd.DataFrame
    summary: dict
    finalists: pd.DataFrame = field(default_factory=pd.DataFrame)
    run_dir: Path | None = None


def run_wfo(
    prep: PreparedData,
    cube: StatsCube,
    nbrs: np.ndarray,
    structure: WFOStructure,
    execution: ExecutionParams,
    *,
    selection: SelectionConfig | None = None,
    eligible: np.ndarray | None = None,
    family_label: str = "all",
    slippage_grid: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0),
    starting_equity: float = 100_000.0,
    run_dir: Path | None = None,
) -> WFOResult:
    sel = selection or SelectionConfig()
    eligible = np.ones(cube.n_configs, dtype=bool) if eligible is None else eligible
    first, last = usable_month_range(cube.sessions_per_month)
    windows = generate_windows(cube.months, structure, first, last)
    if not windows:
        raise ValueError(f"not enough history for {structure.name}: {last - first} usable months")
    fixed = SizingParams(mode="fixed_contracts", contracts=1, starting_equity=starting_equity)
    rows, trade_frames, oos_dates, finalist_frames = [], [], [], []
    selected_params: list[StrategyParams | None] = []
    for w in windows:
        view = cube.view(w.train[0], w.val[1])
        pick = select_window(view, w, nbrs, eligible, sel)
        if len(pick["finalists"]):
            finalist_frames.append(pick["finalists"].assign(window=w.idx, config_key=[cube.configs[i].key() for i in pick["finalists"]["config_index"]]))
        o0, o1 = w.oos
        start, _ = _month_dates(cube.months[o0])
        _, end = _month_dates(cube.months[o1 - 1])
        d_lo, d_hi = prep.day_range(start, end)
        oos_dates.append(prep.dates[d_lo:d_hi])
        row = {"window": w.idx, **{f"{k}_months": v for k, v in w.labels.items()}, "oos_start": str(start.date()), "oos_end": str(end.date()),
               "oos_sessions": int(d_hi - d_lo), "selected": pick["selected"], "reason": pick["reason"]}
        if pick["selected"] < 0:
            selected_params.append(None)
            rows.append(row)
            continue
        params = cube.configs[pick["selected"]]
        selected_params.append(params)
        res = run_backtest(prep, params, execution, fixed, start=start, end=end)
        trades = res.trades.assign(window=w.idx, stand_aside=pick["stand_aside"])
        trade_frames.append(trades)
        oos_best = metrics_from_stats(cube.aggregate(o0, o1, idx=np.array([pick["best_train"], pick["selected"]])), cube.sessions(o0, o1))
        row.update(
            {
                **{f"param_{k}": v for k, v in config_frame_row(params).items() if k not in ("config_key",)},
                "config_key": params.key(),
                "plateau_class": pick["plateau_class"],
                "stand_aside": pick["stand_aside"],
                "train_trades": pick["train_metrics"]["n_trades"],
                "train_exp_r": pick["train_metrics"]["exp_r"],
                "train_sharpe": pick["train_metrics"]["sharpe"],
                "val_trades": pick["val_metrics"]["n_trades"],
                "val_exp_r": pick["val_metrics"]["exp_r"],
                "val_sharpe": pick["val_metrics"]["sharpe"],
                "oos_trades": res.metrics["n_trades"],
                "oos_exp_r": res.metrics["avg_r"],
                "oos_net_pnl": res.metrics["net_pnl"],
                "oos_sharpe": res.metrics["sharpe"],
                "oos_exp_r_check_cube": float(oos_best["exp_r"].iloc[1]),
                "best_train_exp_r": pick["best_train_metrics"].get("exp_r", np.nan),
                "best_train_oos_exp_r": float(oos_best["exp_r"].iloc[0]),
            }
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    all_dates = np.concatenate(oos_dates) if oos_dates else np.array([], dtype="datetime64[D]")
    stitched = _stitch(trades, all_dates, starting_equity)
    stand = trades[~trades["stand_aside"]] if len(trades) else trades
    stitched_sa = _stitch(stand, all_dates, starting_equity)
    pct = _stitch(trades, all_dates, starting_equity, SizingParams(mode="pct_equity", risk_pct=0.01, starting_equity=starting_equity), prep)
    sens = _slippage_sensitivity(prep, windows, selected_params, cube, execution, fixed, all_dates, starting_equity, slippage_grid)
    summary = _summarise(structure, sel, table, stitched["metrics"], stitched_sa["metrics"], sens, family_label, execution, prep)
    finalists = pd.concat(finalist_frames, ignore_index=True) if finalist_frames else pd.DataFrame()
    result = WFOResult(structure, sel, table, trades, stitched["daily"], stitched["metrics"], stitched_sa["metrics"], pct["metrics"], sens, summary, finalists)
    if run_dir is not None:
        save_wfo(result, Path(run_dir))
        result.run_dir = Path(run_dir)
    return result


def _stitch(trades: pd.DataFrame, dates: np.ndarray, equity: float, sizing: SizingParams | None = None, prep: PreparedData | None = None) -> dict:
    if len(trades) and sizing is not None and prep is not None:
        cols = [c for c in trades.columns if c not in ("qty", "sized_out", "gross_pnl", "slippage_cost", "commission", "net_pnl", "risk_dollars", "equity_before", "equity_after")]
        costs = resolve_costs(prep.instrument, ExecutionParams())
        trades = apply_sizing(trades[cols].sort_values("entry_time").reset_index(drop=True), sizing, prep.instrument, costs)
    daily = daily_equity(trades, dates, equity)
    metrics = compute_metrics(trades, daily, equity, "net") if len(daily) else {}
    return {"daily": daily, "metrics": metrics}


def _slippage_sensitivity(prep, windows, selected_params, cube, execution, fixed, dates, equity, grid) -> pd.DataFrame:
    rows = []
    for slip in grid:
        frames = []
        ex = ExecutionParams(**{**execution.to_dict(), "slippage_ticks": float(slip), "frictionless": False})
        for w, params in zip(windows, selected_params):
            if params is None:
                continue
            start, _ = _month_dates(cube.months[w.oos[0]])
            _, end = _month_dates(cube.months[w.oos[1] - 1])
            frames.append(run_backtest(prep, params, ex, fixed, start=start, end=end).trades)
        trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        m = _stitch(trades, dates, equity)["metrics"]
        rows.append({"slippage_ticks": slip, "trades": m.get("n_trades", 0), "net_pnl": m.get("net_pnl", 0.0), "exp_r": m.get("avg_r", 0.0),
                     "t_stat_r": m.get("t_stat_r", 0.0), "profit_factor": m.get("profit_factor", 0.0), "sharpe": m.get("sharpe", 0.0),
                     "max_dd": m.get("max_dd", 0.0)})
    return pd.DataFrame(rows)


def _summarise(structure, sel, table, m, m_sa, sens, family, execution, prep) -> dict:
    traded = table[table["selected"] >= 0] if len(table) else table
    base_slip = resolve_costs(prep.instrument, execution).slippage_ticks
    out = {
        "structure": asdict(structure),
        "selection": asdict(sel),
        "family": family,
        "windows": int(len(table)),
        "windows_with_selection": int(len(traded)),
        "oos_first": str(table["oos_start"].min()) if len(table) else None,
        "oos_last": str(table["oos_end"].max()) if len(table) else None,
        "stitched_always_trade": {k: m.get(k) for k in ("n_trades", "net_pnl", "avg_r", "t_stat_r", "profit_factor", "sharpe", "max_dd", "win_rate", "cagr")},
        "stitched_stand_aside": {k: m_sa.get(k) for k in ("n_trades", "net_pnl", "avg_r", "t_stat_r", "profit_factor", "sharpe", "max_dd")},
    }
    if len(traded):
        out.update(
            {
                "pct_windows_oos_profitable": float((traded["oos_net_pnl"] > 0).mean()),
                "median_train_exp_r": float(traded["train_exp_r"].median()),
                "median_val_exp_r": float(traded["val_exp_r"].median()),
                "median_oos_exp_r": float(traded["oos_exp_r"].median()),
                "median_best_train_exp_r": float(traded["best_train_exp_r"].median()),
                "median_best_train_oos_exp_r": float(traded["best_train_oos_exp_r"].median()),
                "stand_aside_windows": int(traded["stand_aside"].sum()),
                "plateau_classes": traded["plateau_class"].value_counts().to_dict(),
                "parameter_stability": {
                    p: float(traded[f"param_{p}"].astype(str).value_counts(normalize=True).iloc[0])
                    for p in ("range_minutes", "entry_tf", "entry_method", "confirmation", "stop", "target_r", "cutoff", "direction")
                    if f"param_{p}" in traded
                },
                "distinct_configs_selected": int(traded["config_key"].nunique()),
            }
        )
        train_pos = (traded["train_exp_r"] > 0).mean()
        collapse = out["median_oos_exp_r"] <= 0 < out["median_train_exp_r"]
        best_collapse = out["median_best_train_oos_exp_r"] <= 0 < out["median_best_train_exp_r"]
        out["overfit_flags"] = [
            flag
            for flag, cond in (
                ("selected configurations are profitable in training but not OOS (median)", collapse),
                ("the best raw training configuration collapses OOS (median)", best_collapse),
                ("fewer than half of OOS windows profitable", out["pct_windows_oos_profitable"] < 0.5),
                ("selected parameters change in most windows (low stability)", out["distinct_configs_selected"] > 0.75 * len(traded)),
            )
            if cond
        ]
        out["train_windows_positive_share"] = float(train_pos)
    if len(sens):
        base_row = sens.iloc[(sens["slippage_ticks"] - base_slip).abs().argsort()[:1]]
        base_exp = float(base_row["exp_r"].iloc[0])
        worse = sens[(sens["slippage_ticks"] > base_slip) & (sens["slippage_ticks"] <= base_slip + 2.0)]
        out["execution_sensitive"] = bool(base_exp > 0 and (worse["exp_r"] <= 0).any())
    return out


def save_wfo(result: WFOResult, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    result.windows.to_csv(run_dir / "windows.csv", index=False)
    if len(result.oos_trades):
        result.oos_trades.to_csv(run_dir / "oos_trades.csv", index=False)
    result.stitched_daily.to_csv(run_dir / "stitched_oos_daily.csv")
    result.slippage_sensitivity.to_csv(run_dir / "slippage_sensitivity.csv", index=False)
    result.finalists.to_csv(run_dir / "finalists.csv", index=False)
    (run_dir / "summary.json").write_text(json.dumps({**result.summary, "stitched_pct_equity_1pct": {
        k: result.stitched_metrics_pct_equity.get(k) for k in ("n_trades", "net_pnl", "cagr", "sharpe", "max_dd")}}, indent=2, default=str))


def phase_metrics(cube: StatsCube, structure: WFOStructure) -> pd.DataFrame:
    """Per-configuration descriptive metrics by phase, for heatmaps (NOT used for selection).

    * train: mean over windows of the per-window training expectancy / Sharpe (training windows overlap)
    * val: pooled over all validation months
    * oos: pooled over all OOS months, i.e. each configuration held fixed through the OOS periods
    """
    first, last = usable_month_range(cube.sessions_per_month)
    windows = generate_windows(cube.months, structure, first, last)
    tr_exp = np.zeros(cube.n_configs)
    tr_sharpe = np.zeros(cube.n_configs)
    for w in windows:
        m = metrics_from_stats(cube.aggregate(*w.train), cube.sessions(*w.train))
        tr_exp += m["exp_r"].to_numpy()
        tr_sharpe += m["sharpe"].to_numpy()
    tr_exp /= max(len(windows), 1)
    tr_sharpe /= max(len(windows), 1)
    out = {"train_exp_r": tr_exp, "train_sharpe": tr_sharpe}
    for phase, attr in (("val", "val"), ("oos", "oos")):
        months = np.concatenate([np.arange(*getattr(w, attr)) for w in windows])
        agg = np.asarray(cube.data[:, months, :], dtype=np.float64).sum(axis=1)
        m = metrics_from_stats(agg, int(cube.sessions_per_month[months].sum()))
        for col in ("exp_r", "sharpe", "t_stat_r", "n_trades", "net_pnl", "profit_factor"):
            out[f"{phase}_{col}"] = m[col].to_numpy()
    frame = pd.DataFrame([config_frame_row(c) for c in cube.configs])
    return pd.concat([frame, pd.DataFrame(out)], axis=1)

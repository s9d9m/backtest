"""PHASE0_FREE_PROXY: free Yahoo intraday ETF data through the ORB research pipeline.

FREE PROXY EXPERIMENT — NOT FUTURES VALIDATION. SPY/QQQ are proxies; nothing here validates ES/NQ.

Order of operations (enforced in code):

1. Acquire (cached) Yahoo bars; measured availability picks the interval.
2. Data-quality gate (STOP blocks everything).
3. Chronological split by session: train / validation / final test. No shuffling.
4. Selection on a dataset physically truncated before the first test session (the final test's bars
   are not in memory during selection). Neighbourhood-robust training score plus validation score
   (same rule as the futures walk-forward).
5. Freeze the selection to ``selection_frozen.json`` (SHA-256 recorded).
6. Only then is the full dataset used: the frozen candidate is run once on the final test, followed
   by descriptive analyses (phase heatmaps, neighbourhood, friction sensitivity, controls,
   concentration, charts).

Nothing is written to the futures research registry or lockbox. The experiment log is
``phase0_results/phase0_log.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..data_sources import yahoo
from ..engine.backtest import run_backtest
from ..engine.data_quality import check_and_clean
from ..engine.instruments import CONFIG_DIR, Instrument
from ..engine.params import ExecutionParams, SizingParams, StrategyParams
from ..engine.pipeline import build_dataset
from ..engine.sessions import TradingCalendar
from ..optimization.heatmaps import heatmap_table
from ..optimization.multiple_testing import selection_bias_summary
from ..optimization.parameter_space import ParameterSpace, config_frame_row
from ..optimization.robustness import classify_plateau, neighbour_index, neighbour_stats
from ..optimization.stats_cube import grouped_stats, metrics_from_stats
from ..optimization.walk_forward import SelectionConfig, Window, objective, select_window
from ..reports.dq_report import evaluate as dq_evaluate
from ..research.lockbox import subset_before
from .charts import chart_trade, heatmap_png
from .controls import both_direction_outcomes, buy_and_hold, random_direction_control

LABEL = yahoo.PHASE0_LABEL
FAMILY = yahoo.EXPERIMENT_FAMILY
RESULTS_DIR = Path(__file__).resolve().parents[2] / "phase0_results"
HEATMAP_PAIRS = [
    ("range_minutes", "target_r", "ORB duration x R"),
    ("range_minutes", "stop", "ORB duration x stop"),
    ("target_r", "cutoff", "R x cutoff"),
    ("confirmation", "target_r", "confirmation x R"),
    ("range_minutes", "entry_tf", "ORB duration x entry TF"),
]
PHASES = ("train", "val", "test")


def load_config(path: Path | None = None) -> dict:
    with open(path or CONFIG_DIR / "phase0.yaml") as fh:
        return yaml.safe_load(fh)


def etf_instrument(symbol: str, cfg: dict, friction_usd: float) -> Instrument:
    """ETF spec with friction modelled as a per-share cost on EVERY fill (entry and exit, any order type).

    Implemented through the engine's ``friction_per_side`` ($ per unit per fill; 1 unit = 1 share), so each round
    trip pays 2 x friction per share whatever the path. Stop/target distances are unaffected. This is
    more conservative for limit fills than the engine's path-slippage model, which only charges
    market/stop fills and moves brackets with the fill (run 1; see the report).
    """
    spec = dict(cfg["etf_spec"])
    spec["friction_per_side"] = float(friction_usd)
    spec.setdefault("asset_class", "etf")
    return Instrument(symbol=symbol, name=cfg["instruments"].get(symbol, {}).get("name", symbol), slippage_ticks=0.0, **spec)


def build_space(cfg: dict, base_minutes: int) -> tuple[list[StrategyParams], dict]:
    """Expand the Phase-0 grid; drop configurations that are mathematically identical to another.

    Canonicalisation (engine) removes irrelevant parameters. In addition, a limit entry fills exactly
    at the broken boundary (buffer 0, conservative fill), so its "50 % of OR width from entry" stop
    equals the OR midpoint stop. Those duplicates are removed here.
    """
    expanded = ParameterSpace("phase0", cfg["grid"]).expand(base_minutes)
    configs, dropped = [], 0
    for c in expanded.configs:
        if c.entry_method == "limit" and c.entry_buffer_ticks == 0 and c.stop_method == "or_pct" and abs(c.stop_param - 0.5) < 1e-12:
            dropped += 1
            continue
        configs.append(c)
    info = {"raw": expanded.n_raw, "invalid": expanded.n_invalid, "canonical_duplicates": expanded.n_duplicates,
            "equivalent_limit_or_pct50_removed": dropped, "evaluated": len(configs), "invalid_examples": expanded.invalid_examples[:3]}
    return configs, info


def split_sessions(n: int, fractions: dict) -> tuple[int, int, int]:
    n_train = int(round(n * fractions["train"]))
    n_val = int(round(n * fractions["validation"]))
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(f"{n} sessions cannot be split {fractions}")
    return n_train, n_val, n_test


def _log(event: str, **kw) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    entry = {"ts_utc": datetime.now(timezone.utc).isoformat(), "family": FAMILY, "event": event, **kw}
    with open(RESULTS_DIR / "phase0_log.jsonl", "a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


# ----------------------------------------------------------------------------------------------- data
def availability(symbol: str, probe_file: Path) -> dict:
    """Sessions available per interval, from the measured probe (see yahoo.probe_depth)."""
    probe = json.loads(probe_file.read_text()) if probe_file.exists() else {}
    out = {}
    for interval in ("1m", "2m", "5m", "15m"):
        key = f"{symbol}_{interval}"
        if key not in probe:
            probe[key] = yahoo.probe_depth(symbol, interval, 7 if interval == "1m" else 30, 70 if interval == "1m" else 400)
        if probe[key]["earliest_bar"] and "earliest_bar_refined" not in probe[key]:
            probe[key]["earliest_bar_refined"] = yahoo.refine_earliest(symbol, interval, probe[key]["earliest_bar"])
        first, last = probe[key].get("earliest_bar_refined") or probe[key]["earliest_bar"], probe[key]["latest_bar"]
        if first:
            sessions = TradingCalendar("XNYS").sessions(pd.Timestamp(first[:10]), pd.Timestamp(last[:10]))
            out[interval] = {"first": first, "last": last, "sessions": int(len(sessions))}
    probe_file.write_text(json.dumps(probe, indent=2))
    return out


def choose_interval(avail: dict, min_sessions: int = 35) -> str:
    """Finest interval with at least ``min_sessions`` sessions; else the one with the most history."""
    for interval in ("1m", "2m", "5m", "15m"):
        if interval in avail and avail[interval]["sessions"] >= min_sessions:
            return interval
    return max(avail, key=lambda k: (avail[k]["sessions"], -yahoo.INTERVAL_MINUTES[k]))


def acquire(symbol: str, interval: str, avail: dict, refresh: bool = False) -> dict:
    out_dir = yahoo.DATA_DIR / "phase0"
    prov_file = out_dir / f"{symbol}_{interval}_provenance.json"
    if prov_file.exists() and not refresh:
        return json.loads(prov_file.read_text())
    first = pd.Timestamp(avail[interval]["first"][:10])
    last = pd.Timestamp(avail[interval]["last"][:10]) + pd.Timedelta(days=1)
    chunk = 7 if interval == "1m" else 30
    return yahoo.download(symbol, interval, str(first.date()), str(last.date()), chunk_days=chunk)


# ----------------------------------------------------------------------------------------------- DQ
def phase0_dq(symbol: str, path: Path, instrument: Instrument, bar_minutes: int, one_min_path: Path | None) -> tuple[object, dict]:
    ds = build_dataset(path, instrument, allow_errors=True)
    rdq = dq_evaluate(ds)
    prep = ds.prep
    et = ds.bars.index.tz_convert("America/New_York")
    offsets = sorted({str(x) for x in (et.tz_localize(None) - ds.bars.index.tz_localize(None)).unique()})
    rows = []
    for d in range(prep.n_days):
        lo, hi = int(prep.day_start[d]), int(prep.day_end[d])
        tod = prep.tod[lo:hi]
        close = int(prep.close_min[d])
        rows.append({"session": str(prep.dates[d]), "bars": hi - lo, "expected": (close - 570) // bar_minutes,
                     "first_bar": f"{tod[0] // 60:02d}:{tod[0] % 60:02d}", "last_bar": f"{tod[-1] // 60:02d}:{tod[-1] % 60:02d}",
                     "early_close": bool(prep.early_close[d])})
    sessions = pd.DataFrame(rows)
    misaligned_open = sessions[sessions["first_bar"] != "09:30"]
    incomplete = sessions[sessions["bars"] < sessions["expected"]]
    extra = sessions[sessions["bars"] > sessions["expected"]]
    cal = TradingCalendar("XNYS").sessions(prep.dates[0], prep.dates[-1])
    missing_days = sorted(set(cal.index.date) - {pd.Timestamp(d).date() for d in prep.dates})
    agg_check = None
    if one_min_path is not None and one_min_path.exists() and bar_minutes > 1:
        one = pd.read_parquet(one_min_path).set_index("timestamp")
        one.index = pd.DatetimeIndex(one.index)
        coarse = ds.bars
        common_start = max(one.index.min(), coarse.index.min())
        f = one[one.index >= common_start]
        agg = f.resample(f"{bar_minutes}min", label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        joined = coarse.join(agg, rsuffix="_1m", how="inner")
        if len(joined):
            diff = {c: float((joined[c] - joined[f"{c}_1m"]).abs().max()) for c in ("open", "high", "low", "close")}
            share_exact = float(np.mean(np.all(np.isclose(joined[["open", "high", "low", "close"]].to_numpy(),
                                                          joined[["open_1m", "high_1m", "low_1m", "close_1m"]].to_numpy(), atol=0.005), axis=1)))
            agg_check = {"bars_compared": int(len(joined)), "max_abs_diff": diff, "share_ohlc_identical_within_half_cent": share_exact,
                         "volume_ratio_median": float((joined["volume"] / joined["volume_1m"].replace(0, np.nan)).median())}
    verdict = rdq.verdict
    reasons = list(rdq.reasons)
    if len(misaligned_open) or not ds.report.ok:
        verdict = "STOP"
        reasons.append(f"{len(misaligned_open)} sessions whose first bar is not 09:30 ET")
    if missing_days:
        verdict = "STOP" if len(missing_days) > 2 else ("REVIEW" if verdict != "STOP" else verdict)
        reasons.append(f"missing XNYS sessions: {missing_days}")
    if len(incomplete):
        reasons.append(f"{len(incomplete)} sessions with missing bars: {incomplete[['session', 'bars', 'expected']].to_dict('records')[:5]}")
        if verdict == "PASS":
            verdict = "REVIEW"
    if agg_check and agg_check["share_ohlc_identical_within_half_cent"] < 0.95:
        reasons.append(f"{bar_minutes}m bars disagree with aggregated 1m bars in {1 - agg_check['share_ohlc_identical_within_half_cent']:.1%} of cases")
        if verdict == "PASS":
            verdict = "REVIEW"
    summary = {
        "verdict": verdict,
        "reasons": reasons,
        "sessions": int(prep.n_days),
        "first_session": str(prep.dates[0]),
        "last_session": str(prep.dates[-1]),
        "rows": int(len(ds.bars)),
        "utc_offsets_seen": offsets,
        "dst_transition_in_sample": len(offsets) > 1,
        "sessions_first_bar_not_0930": int(len(misaligned_open)),
        "sessions_incomplete": int(len(incomplete)),
        "sessions_with_extra_bars": int(len(extra)),
        "missing_calendar_sessions": [str(d) for d in missing_days],
        "early_close_sessions": sessions.loc[sessions["early_close"], "session"].tolist(),
        "duplicate_or_invalid": [f"[{i.severity}] {i.code} n={i.count}" for i in ds.report.issues],
        "modal_peak_volume_minute_et": rdq.summary.get("modal_peak_minute_et"),
        "one_minute_cross_check": agg_check,
    }
    return ds, {"summary": summary, "sessions": sessions, "monthly_timing": rdq.monthly_timing}


# ----------------------------------------------------------------------------------------------- analysis helpers
def phase_frame(configs: list[StrategyParams], cube, sessions: dict[str, int]) -> pd.DataFrame:
    frame = pd.DataFrame([config_frame_row(c) for c in configs])
    for k, phase in enumerate(cube.months):
        m = metrics_from_stats(cube.aggregate(k, k + 1), sessions[phase])
        traded = m["n_trades"].to_numpy() > 0
        for col in ("n_trades", "exp_r", "t_stat_r", "sharpe", "net_pnl", "profit_factor", "win_rate"):
            values = m[col].to_numpy().astype(float)
            # configurations with no trade in a phase have no expectancy: NaN, so they cannot drag medians to 0
            frame[f"{phase}_{col}"] = values if col in ("n_trades", "net_pnl") else np.where(traded, values, np.nan)
    return frame


def remove_best(r: np.ndarray, k: int) -> float:
    if len(r) == 0:
        return float("nan")
    return float(np.sort(r)[: max(len(r) - k, 0)].sum()) if len(r) > k else float("nan")


def split_result(prep, params, instrument, friction, shares, start, end) -> dict:
    inst = replace(instrument, friction_per_side=float(friction), slippage_ticks=0.0)
    prep_f = replace(prep, instrument=inst)
    res = run_backtest(prep_f, params, ExecutionParams(), SizingParams(mode="fixed_contracts", contracts=shares), start=start, end=end)
    t = res.trades
    m = res.metrics
    r = t["r_multiple"].to_numpy() if len(t) else np.zeros(0)
    cum = np.cumsum(r)
    dd_r = float((cum - np.maximum.accumulate(np.concatenate(([0.0], cum)))[1:]).min()) if len(r) else 0.0
    out = {
        "friction_usd_per_share": friction,
        "trades": int(m["n_trades"]),
        "win_rate": m["win_rate"],
        "exp_r": m["avg_r"],
        "t_stat_r": m["t_stat_r"],
        "profit_factor": m["profit_factor"],
        "net_pnl": m["net_pnl"],
        "gross_pnl": m["gross_pnl"],
        "friction_cost": m["total_cost"],
        "sharpe_daily": m["sharpe"],
        "max_dd_r": dd_r,
        "max_dd_usd": float((res.daily["net_equity"] - res.daily["net_equity"].cummax()).min()) if len(res.daily) else 0.0,
        "avg_winner": m["avg_winner"],
        "avg_loser": m["avg_loser"],
        "largest_winner": m["best_trade"],
        "largest_loser": m["worst_trade"],
        "long_trades": res.metrics_long["n_trades"],
        "long_exp_r": res.metrics_long["avg_r"],
        "long_net_pnl": res.metrics_long["net_pnl"],
        "short_trades": res.metrics_short["n_trades"],
        "short_exp_r": res.metrics_short["avg_r"],
        "short_net_pnl": res.metrics_short["net_pnl"],
        "sum_r": float(r.sum()),
    }
    for k in (1, 3, 5, 10):
        out[f"sum_r_without_best_{k}"] = remove_best(r, k)
    out["_trades"] = t
    out["_prep"] = prep_f
    return out


TRADE_LOG_COLUMNS = ["split", "session_date", "symbol", "direction", "or_high", "or_low", "or_width", "signal_time", "entry_time",
                     "entry_price", "stop_price", "target_price", "exit_time", "exit_price", "exit_reason", "r_multiple", "qty",
                     "gross_pnl", "estimated_friction", "net_pnl", "risk_ticks", "ambiguous_exit", "intrabar_entry"]


def select_without_test(prep_full, configs, nbrs, n_train: int, n_val: int, sel_cfg: SelectionConfig, execution: ExecutionParams,
                        n_workers: int = 1) -> dict:
    """Validation-based, neighbourhood-robust selection on a copy of the data that ENDS before the final test.

    The final-test sessions are removed from memory (``subset_before``) before any configuration is
    simulated, so no statistic of the test period can influence the choice.
    """
    test_start = prep_full.dates[n_train + n_val]
    dev_prep = subset_before(prep_full, test_start)
    if dev_prep.n_days != n_train + n_val or not dev_prep.dates[-1] < test_start:
        raise AssertionError("development data overlaps the final test")
    groups = np.array([0] * n_train + [1] * n_val)
    dev_cube = grouped_stats(dev_prep, configs, execution, groups, ["train", "val"], n_workers=n_workers)
    pick = select_window(dev_cube.view(0, 2), Window(0, (0, 1), (1, 2), (2, 3)), nbrs, np.ones(len(configs), bool), sel_cfg)
    train_m = metrics_from_stats(dev_cube.aggregate(0, 1), n_train)
    dev_m = metrics_from_stats(dev_cube.aggregate(0, 2), n_train + n_val)
    score_train = objective(train_m, sel_cfg.min_trades_train, sel_cfg)
    ok = train_m["n_trades"].to_numpy() >= sel_cfg.min_trades_train
    return {
        "pick": pick,
        "dev_prep": dev_prep,
        "train_metrics": train_m,
        "dev_metrics": dev_m,
        "best_in_sample": int(np.argmax(np.where(ok, score_train, -np.inf))) if ok.any() else -1,
        "best_dev_pnl": int(np.argmax(dev_m["net_pnl"].to_numpy())),
    }


# ----------------------------------------------------------------------------------------------- main
def run_free_test(symbol: str, *, n_workers: int = 4, refresh: bool = False, interval: str = "auto", n_charts_each: int = 10,
                  seed: int = 20260926) -> dict:
    cfg = load_config()
    out = RESULTS_DIR / symbol
    out.mkdir(parents=True, exist_ok=True)
    data_dir = yahoo.DATA_DIR / "phase0"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. data
    avail = availability(symbol, data_dir / "yahoo_probe.json")
    chosen = choose_interval(avail) if interval == "auto" else interval
    prov = acquire(symbol, chosen, avail, refresh)
    one_min = acquire(symbol, "1m", avail, refresh) if chosen != "1m" and "1m" in avail else None
    _log("data", symbol=symbol, interval=chosen, availability=avail, provenance=prov)
    bar_minutes = yahoo.INTERVAL_MINUTES[chosen]
    headline = float(cfg["headline_friction_usd_per_share"])
    instrument = etf_instrument(symbol, cfg, headline)

    # 2. DQ
    data_path = yahoo.DATA_DIR.parent / prov["processed_file"] if not Path(prov["processed_file"]).is_absolute() else Path(prov["processed_file"])
    one_path = (yahoo.DATA_DIR.parent / one_min["processed_file"]) if one_min else None
    ds, dq = phase0_dq(symbol, data_path, instrument, bar_minutes, one_path)
    dq["sessions"].to_csv(out / "dq_sessions.csv", index=False)
    (out / "dq_summary.json").write_text(json.dumps(dq["summary"], indent=2, default=str))
    _log("dq", symbol=symbol, verdict=dq["summary"]["verdict"], reasons=dq["summary"]["reasons"])
    if dq["summary"]["verdict"] == "STOP":
        raise RuntimeError(f"{symbol}: Phase-0 DQ verdict STOP: {dq['summary']['reasons']}")

    # 3. split
    prep_full = ds.prep
    n = prep_full.n_days
    n_train, n_val, n_test = split_sessions(n, cfg["split"])
    d = prep_full.dates
    split = {"train": [str(d[0]), str(d[n_train - 1])], "validation": [str(d[n_train]), str(d[n_train + n_val - 1])],
             "final_test": [str(d[n_train + n_val]), str(d[-1])], "sessions": {"train": n_train, "validation": n_val, "final_test": n_test}}
    test_start = d[n_train + n_val]

    # 4. selection WITHOUT the final test in memory
    configs, space_info = build_space(cfg, bar_minutes)
    s = cfg["selection"]
    sel_cfg = SelectionConfig(n_finalists=s["n_finalists"], min_trades_train=s["min_trades_train"], min_trades_val=s["min_trades_val"],
                              neighbour_weight=s["neighbour_weight"], validation_weight=s["validation_weight"])
    headline_exec = ExecutionParams()
    nbrs = neighbour_index(configs)
    chosen_sel = select_without_test(prep_full, configs, nbrs, n_train, n_val, sel_cfg, headline_exec, n_workers)
    pick, dev_m, train_m = chosen_sel["pick"], chosen_sel["dev_metrics"], chosen_sel["train_metrics"]
    best_is, best_dev_pnl = chosen_sel["best_in_sample"], chosen_sel["best_dev_pnl"]
    if pick["selected"] < 0:
        raise RuntimeError(f"{symbol}: no configuration met the minimum trade counts ({pick['reason']})")
    bias = selection_bias_summary(train_m["sharpe"].to_numpy())
    selected = configs[pick["selected"]]
    finalists = pick["finalists"].assign(config_key=[configs[i].key() for i in pick["finalists"]["config_index"]],
                                         **{k: [config_frame_row(configs[i])[k] for i in pick["finalists"]["config_index"]]
                                            for k in ("range_minutes", "entry_tf", "entry_method", "confirmation", "stop", "target_r", "cutoff", "direction")})
    frozen = {
        "family": FAMILY, "label": LABEL, "symbol": symbol, "interval": chosen, "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "split": split, "headline_friction_usd_per_share": headline, "selection_rule": asdict(sel_cfg), "space": space_info,
        "selected": {"config_key": selected.key(), **selected.to_dict(), "plateau_class_train": pick["plateau_class"],
                     "train_metrics": pick["train_metrics"], "val_metrics": pick["val_metrics"]},
        "best_in_sample_train_score": {"config_key": configs[best_is].key(), **configs[best_is].to_dict()},
        "best_in_sample_train_plus_val_pnl": {"config_key": configs[best_dev_pnl].key(), **configs[best_dev_pnl].to_dict()},
        "selection_bias_train": bias,
        "final_test_seen": False,
    }
    frozen_path = out / "selection_frozen.json"
    frozen_path.write_text(json.dumps(frozen, indent=2, default=str))
    finalists.to_csv(out / "finalists.csv", index=False)
    frozen_sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    _log("selection_frozen", symbol=symbol, sha256=frozen_sha, selected=selected.key(), test_period=split["final_test"])

    # 5. ---- final test unlocked (after freeze) ----
    phase_groups = np.array([0] * n_train + [1] * n_val + [2] * n_test)
    full_cube = grouped_stats(prep_full, configs, headline_exec, phase_groups, list(PHASES), n_workers=n_workers)
    sessions_by_phase = {"train": n_train, "val": n_val, "test": n_test}
    pf = phase_frame(configs, full_cube, sessions_by_phase)
    pf.to_parquet(out / "phase_metrics_all_configs.parquet")
    _log("final_test_unlocked", symbol=symbol, after_frozen_sha256=frozen_sha)

    # descriptive robustness on each phase
    for phase in PHASES:
        med, mn, cnt = neighbour_stats(pf[f"{phase}_exp_r"].to_numpy(), nbrs)
        pf[f"{phase}_nbr_median_exp_r"] = med
        pf[f"{phase}_plateau"] = classify_plateau(pf[f"{phase}_exp_r"].to_numpy(), med, mn)
    i_sel = pick["selected"]
    nb_idx = [i_sel] + [j for j in nbrs[i_sel] if j >= 0]
    cols = ["config_key", "range_minutes", "entry_tf", "entry_method", "confirmation", "stop", "target_r", "cutoff", "direction"] + \
           [f"{p}_{m}" for p in PHASES for m in ("n_trades", "exp_r")]
    neighbourhood = pf.iloc[nb_idx][cols].assign(role=["SELECTED"] + ["neighbour"] * (len(nb_idx) - 1))
    neighbourhood.to_csv(out / "selected_neighbourhood.csv", index=False)

    # heatmaps per phase (median across hidden parameters)
    heat = {}
    for x, y, title in HEATMAP_PAIRS:
        for phase in PHASES:
            tbl = heatmap_table(pf, y, x, f"{phase}_exp_r", "median")
            tbl.to_csv(out / f"heatmap_{x}_x_{y}_{phase}.csv")
            heatmap_png(tbl, f"{symbol} {phase.upper()} median expectancy (R) | {title} | friction ${headline}/sh\n{LABEL}",
                        out / "heatmaps" / f"{x}_x_{y}_{phase}.png")
            heat[f"{title}|{phase}"] = tbl

    # frozen candidate: every split x friction level
    fr_rows, headline_by_split = [], {}
    bounds = {"train": (d[0], d[n_train - 1]), "val": (d[n_train], d[n_train + n_val - 1]), "test": (test_start, d[-1])}
    for fr in cfg["friction_grid_usd_per_share"]:
        for phase, (a, b) in bounds.items():
            r = split_result(prep_full, selected, instrument, float(fr), cfg["shares"], a, b)
            if abs(fr - headline) < 1e-12:
                headline_by_split[phase] = r
            fr_rows.append({"split": phase, **{k: v for k, v in r.items() if not k.startswith("_")}})
    friction_table = pd.DataFrame(fr_rows)
    friction_table.to_csv(out / "selected_friction_sensitivity.csv", index=False)

    # comparison configurations on the final test (reported, not emphasised)
    comparisons = {}
    for name, idx in (("best_in_sample_train_score", best_is), ("best_in_sample_train_plus_val_pnl", best_dev_pnl)):
        comparisons[name] = {phase: {k: v for k, v in split_result(prep_full, configs[idx], instrument, headline, cfg["shares"], a, b).items()
                                     if not k.startswith("_")} for phase, (a, b) in bounds.items()}

    # trade log (all splits, headline friction)
    logs = []
    for phase, r in headline_by_split.items():
        t = r["_trades"]
        if len(t):
            logs.append(t.assign(split=phase, symbol=symbol, estimated_friction=t["total_cost"]))
    trades = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(columns=TRADE_LOG_COLUMNS)
    trades[[c for c in TRADE_LOG_COLUMNS if c in trades.columns]].to_csv(out / "trade_log_selected.csv", index=False)

    # controls & baselines per split
    exit_min = headline_by_split["test"]["_prep"].session_exit_min
    controls = {}
    for phase, (a, b) in bounds.items():
        t = headline_by_split[phase]["_trades"]
        oc = both_direction_outcomes(prep_full, t, selected.target_r, 0.0, exit_min, cost_ticks_round_trip=2 * headline / instrument.tick_size) if len(t) else pd.DataFrame()
        d_lo, d_hi = prep_full.day_range(a, b)
        controls[phase] = {"random_direction": random_direction_control(oc, seed=seed),
                           "buy_and_hold": buy_and_hold(prep_full, d_lo, d_hi, cfg["shares"]),
                           "sim_reproduces_engine_max_abs_r_diff": float((oc["actual_r_sim"] - t["r_multiple"].to_numpy()).abs().max()) if len(oc) else None}

    # charts: 10 winners + 10 losers (all trades if fewer than 20)
    rng = np.random.default_rng(seed)
    chart_dir = out / "trade_charts"
    for old in chart_dir.glob("*.png"):
        old.unlink()
    if len(trades) <= 2 * n_charts_each:
        pick_rows = trades
    else:
        win = trades[trades["net_pnl"] > 0]
        lose = trades[trades["net_pnl"] <= 0]
        pick_rows = pd.concat([win.iloc[rng.choice(len(win), min(n_charts_each, len(win)), replace=False)],
                               lose.iloc[rng.choice(len(lose), min(n_charts_each, len(lose)), replace=False)]])
    charted = []
    for _, t in pick_rows.sort_values("entry_time").iterrows():
        tag = "WIN" if t["net_pnl"] > 0 else "LOSS"
        p = chart_dir / f"{pd.Timestamp(t['session_date']).date()}_{t['split']}_{tag}.png"
        chart_trade(prep_full, t, selected, symbol, f"SELECTED {t['split']} {tag}", p)
        charted.append(p.name)
    # Supplementary engine-verification set: if the selected candidate has fewer than 10 winners or losers,
    # add trades of other configurations chosen on DEVELOPMENT data only (best dev P&L per entry method),
    # so every entry type is verified on real bars and at least 10 winners and 10 losers are charted.
    n_win, n_loss = int((trades["net_pnl"] > 0).sum()), int((trades["net_pnl"] <= 0).sum())
    verification = []
    if n_win < n_charts_each or n_loss < n_charts_each:
        dev_pnl = dev_m["net_pnl"].to_numpy()
        for method in ("stop", "limit", "market"):
            mask = np.array([c.entry_method == method and (method != "market" or c.direction == "both") for c in configs])
            if not mask.any():
                continue
            j = int(np.argmax(np.where(mask, dev_pnl, -np.inf)))
            vt = run_backtest(replace(prep_full, instrument=instrument), configs[j], ExecutionParams(),
                              SizingParams(mode="fixed_contracts", contracts=cfg["shares"]), start=d[0], end=d[n_train + n_val - 1]).trades
            verification.append((configs[j], vt))
        need = {"WIN": max(0, n_charts_each - n_win), "LOSS": max(0, n_charts_each - n_loss)}
        k = 0
        while any(need.values()) and k < 50:
            for cfg_j, vt in verification:
                for tag in ("WIN", "LOSS"):
                    pool = vt[(vt["net_pnl"] > 0) if tag == "WIN" else (vt["net_pnl"] <= 0)]
                    if need[tag] > 0 and k < len(pool):
                        t = pool.iloc[rng.permutation(len(pool))[k]]
                        name = f"verify_{cfg_j.entry_method}_{pd.Timestamp(t['session_date']).date()}_{tag}.png"
                        if not (chart_dir / name).exists():
                            chart_trade(prep_full, t, cfg_j, symbol, f"VERIFICATION ({cfg_j.entry_method} entry, dev data) {tag}", chart_dir / name)
                            charted.append(name)
                            need[tag] -= 1
            k += 1

    # resolution cross-check of the frozen candidate on 1-minute bars (overlapping dates only)
    xcheck = None
    if one_path is not None and one_path.exists():
        ds1 = build_dataset(one_path, instrument, allow_errors=True)
        overlap_start = max(pd.Timestamp(ds1.prep.dates[0]), pd.Timestamp(test_start))
        r1 = split_result(ds1.prep, selected, instrument, headline, cfg["shares"], overlap_start, d[-1])
        r5 = split_result(prep_full, selected, instrument, headline, cfg["shares"], overlap_start, d[-1])
        t1, t5 = r1["_trades"], r5["_trades"]
        xcheck = {"period": [str(overlap_start.date()), str(d[-1])], "trades_1m": len(t1), "trades_5m": len(t5),
                  "exp_r_1m": r1["exp_r"], "exp_r_5m": r5["exp_r"],
                  "same_direction_days": int(len(set(zip(t1.session_date, t1.direction)) & set(zip(t5.session_date, t5.direction)))) if len(t1) and len(t5) else 0}

    # grid-wide descriptive answers (median across all configurations)
    def group_median(col, by):
        return pf.groupby(by)[col].median().to_dict()

    answers = {
        "range_minutes": {p: group_median(f"{p}_exp_r", "range_minutes") for p in PHASES},
        "sub2R_vs_2R_plus": {p: {"R<2": float(pf.loc[pf.target_r < 2, f"{p}_exp_r"].median()), "R>=2": float(pf.loc[pf.target_r >= 2, f"{p}_exp_r"].median())} for p in PHASES},
        "direction": {p: group_median(f"{p}_exp_r", "direction") for p in PHASES},
        "entry_method": {p: group_median(f"{p}_exp_r", "entry_method") for p in PHASES},
        "cutoff": {p: group_median(f"{p}_exp_r", "cutoff") for p in PHASES},
        "share_configs_positive": {p: float((pf[f"{p}_exp_r"].dropna() > 0).mean()) for p in PHASES},
        "configs_with_trades": {p: int(pf[f"{p}_exp_r"].notna().sum()) for p in PHASES},
        "median_config_exp_r": {p: float(pf[f"{p}_exp_r"].median()) for p in PHASES},
        "rank_corr_train_vs_test": float(pf["train_exp_r"].rank().corr(pf["test_exp_r"].rank())),
        "rank_corr_val_vs_test": float(pf["val_exp_r"].rank().corr(pf["test_exp_r"].rank())),
        "selected_plateau": {p: str(pf.iloc[i_sel][f"{p}_plateau"]) for p in PHASES},
        "selected_neighbour_median_exp_r": {p: float(pf.iloc[i_sel][f"{p}_nbr_median_exp_r"]) for p in PHASES},
    }
    ht = headline_by_split["test"]
    verdict = classify(ht, friction_table, controls["test"], answers, headline_by_split["val"])
    summary = {
        "family": FAMILY, "label": LABEL, "symbol": symbol, "interval": chosen, "availability": avail, "provenance": prov,
        "dq": dq["summary"], "split": split, "space": space_info, "selection_frozen_sha256": frozen_sha, "frozen": frozen,
        "headline": {p: {k: v for k, v in r.items() if not k.startswith("_")} for p, r in headline_by_split.items()},
        "comparisons": comparisons, "controls": controls, "answers": answers, "resolution_cross_check_1m": xcheck,
        "charts": charted, "verdict": verdict,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    from .report import write_symbol_report

    write_symbol_report(summary, friction_table, neighbourhood, finalists, out)
    _log("report", symbol=symbol, verdict=verdict["category"])
    return summary


def classify(test: dict, friction: pd.DataFrame, ctrl: dict, answers: dict, val: dict) -> dict:
    """Pre-stated rule mapping evidence to PROMISING / MIXED / NO PRELIMINARY EVIDENCE."""
    reasons = []
    exp = test["exp_r"]
    t_test = friction[(friction.split == "test")].set_index("friction_usd_per_share")
    survives_003 = bool(t_test.loc[t_test.index >= 0.03 - 1e-12, "exp_r"].min() > 0) if len(t_test) else False
    p_rand = ctrl["random_direction"].get("p_value_random_ge_strategy", 1.0)
    plateau_test = answers["selected_plateau"]["test"] == "plateau"
    if test["trades"] == 0 or exp <= 0:
        cat = "NO PRELIMINARY EVIDENCE"
        reasons.append(f"final-test expectancy {exp:+.3f} R over {test['trades']} trades is not positive at the headline friction")
    else:
        checks = {"validation expectancy positive": val["exp_r"] > 0, "survives $0.03/share": survives_003,
                  "beats random-direction control (p<0.10)": p_rand < 0.10, "neighbours form a plateau in the final test": plateau_test,
                  "final-test t-stat >= 2": test["t_stat_r"] >= 2}
        failed = [k for k, ok in checks.items() if not ok]
        cat = "PROMISING" if not failed else "MIXED"
        reasons.append(f"final-test expectancy {exp:+.3f} R over {test['trades']} trades; failed checks: {failed or 'none'}")
    return {"category": cat, "reasons": reasons, "rule": "NO PRELIMINARY EVIDENCE if final-test expectancy <= 0; PROMISING only if "
            "validation > 0, survives $0.03/share, beats random-direction control at p<0.10, plateau in final test and t>=2; else MIXED"}

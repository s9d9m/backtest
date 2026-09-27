"""Session state shared by all pages: navigation, loaded data, the CURRENT STRATEGY and its research stages.

The current strategy = strategy parameters + execution assumptions + position sizing. Every page works on it;
results computed for it (robustness, stress, Monte Carlo, validation) are stored under a fingerprint of the
strategy AND the data, so editing anything marks them as "not run for the current strategy" instead of silently
showing stale results.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from ..engine.backtest import run_backtest
from ..engine.data_quality import CleaningPolicy
from ..engine.execution import resolve_costs
from ..engine.instruments import load_instruments
from ..engine.params import ExecutionParams, SizingParams, StrategyParams, load_strategy_config
from ..engine.pipeline import DataQualityError, build_dataset
from ..research import pipeline_state as ps
from ..research.lockbox import subset_before

PAGES = {
    "overview": ("Overview", ":material/dashboard:"),
    "strategy": ("Strategy", ":material/tune:"),
    "backtest": ("Backtest", ":material/show_chart:"),
    "optimize": ("Optimize", ":material/grid_view:"),
    "validate": ("Validate", ":material/verified:"),
    "stress": ("Stress test", ":material/bolt:"),
    "results": ("Results", ":material/summarize:"),
    "trades": ("Trade explorer", ":material/list_alt:"),
    "data": ("Data", ":material/database:"),
    "settings": ("Settings & research", ":material/settings:"),
}
YAHOO = "Free Yahoo (SPY/QQQ)"
STOP_LABELS = {"or_opposite": "Opposite side of range", "or_mid": "Range midpoint", "or_pct": "% of range width",
               "atr": "Multiple of daily ATR", "fixed_ticks": "Fixed ticks"}
ENTRY_LABELS = {"market": "Market (close confirmed)", "limit": "Limit (retest of the range)", "stop": "Stop order (intrabar breakout)"}
DIRECTION_LABELS = {"both": "Long & short", "long": "Long only", "short": "Short only"}
SIZING_SHORT = {"pct_equity": "risk % of equity (compounding)", "fixed_risk": "risk % of starting equity", "fixed_contracts": "fixed size"}


def S():
    return st.session_state


# ---------------------------------------------------------------------------------------------- navigation
def go(page: str) -> None:
    """Button callback: switch page (callbacks run before widgets are rebuilt)."""
    S()["nav"] = page


def nav_button(label: str, page: str, key: str, primary: bool = False, **kw) -> bool:
    return st.button(label, key=key, on_click=go, args=(page,), type="primary" if primary else "secondary", **kw)


# ---------------------------------------------------------------------------------------------- data
def dataset():
    return S().get("dataset")


def instrument():
    return S().get("instrument")


def is_proxy(inst=None) -> bool:
    inst = inst or instrument()
    return inst is not None and getattr(inst, "asset_class", "future") == "etf"


def is_synthetic(ds=None) -> bool:
    ds = ds or dataset()
    return ds is not None and str(getattr(ds.loaded, "source", "")).startswith("synthetic")


def etf_instruments() -> dict:
    from ..phase0.pipeline import etf_instrument, load_config

    cfg = load_config()
    headline = float(cfg["headline_friction_usd_per_share"])
    return {sym: etf_instrument(sym, cfg, headline) for sym in cfg["instruments"]}


def all_instruments() -> dict:
    return {**load_instruments(), **etf_instruments()}


def yahoo_saved_copy(symbol: str) -> Path | None:
    from ..data_sources import yahoo

    for p in (yahoo.DATA_DIR / "phase0" / f"{symbol}_5m.parquet", yahoo.DATA_DIR / "dashboard_yahoo" / "phase0" / f"{symbol}_5m.parquet"):
        if p.exists():
            return p
    return None


def yahoo_path(symbol: str, refresh: bool) -> tuple[Path, str]:
    """Free 5-minute Yahoo bars: the saved copy if present, otherwise the last ~60 days are downloaded."""
    from ..data_sources import yahoo

    saved = yahoo_saved_copy(symbol)
    if saved is not None and not refresh:
        return saved, f"Yahoo {symbol} 5m (saved copy)"
    dash_dir = yahoo.DATA_DIR / "dashboard_yahoo"
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    prov = yahoo.download(symbol, "5m", str((today - pd.Timedelta(days=58)).date()), str((today + pd.Timedelta(days=1)).date()),
                          chunk_days=29, data_dir=dash_dir)
    return dash_dir / "phase0" / f"{symbol}_5m.parquet", (f"Yahoo {symbol} 5m downloaded {prov['download_utc'][:16]} UTC "
                                                         f"({prov['returned_first_bar'][:10]}..{prov['returned_last_bar'][:10]})")


def load_dataset(kind: str, inst, *, symbol: str | None = None, refresh: bool = False, path: str | None = None, upload=None,
                 tz: str | None = None, convention: str = "start", start: str = "2021-01-04", end: str = "2021-12-30", seed: int = 7,
                 trend: float = 0.0, policy: CleaningPolicy | None = None, allow_errors: bool = False, min_or_coverage: float = 0.8,
                 exclude_early_close: bool = False) -> str:
    """Load, DQ-check and prepare a dataset into the session. Returns a short status message. Raises on failure."""
    from ..engine.synthetic import generate_bars
    from ..research.lockbox import apply_lockbox

    policy = policy or CleaningPolicy()
    common = dict(min_or_coverage=min_or_coverage, exclude_early_close=exclude_early_close)
    try:
        if kind == YAHOO:
            p, label = yahoo_path(symbol or inst.symbol, refresh)
            ds = build_dataset(p, inst, policy=policy, allow_errors=allow_errors, **common)
            ds.loaded.source = label
        elif kind == "Synthetic":
            frame = generate_bars(inst, start, end, seed=int(seed), trend_strength=trend)
            ds = build_dataset(frame, inst, **common)
            ds.loaded.source = f"synthetic(seed={int(seed)}, trend={trend}, {start}..{end})"
        else:
            if upload is not None:
                tmp = Path(tempfile.gettempdir()) / f"orb_upload_{upload.name}"
                tmp.write_bytes(upload.getvalue())
                path = str(tmp)
            if not path:
                raise ValueError("Choose a file first.")
            ds = build_dataset(path, inst, source_tz=tz, timestamp_convention=convention, policy=policy, allow_errors=allow_errors, **common)
    except DataQualityError as exc:
        S()["dq_block"] = exc.report
        raise
    ds.prep, lock = apply_lockbox(inst.symbol, ds.prep)
    ds.prep_full = ds.prep
    previous = instrument()
    S()["dataset"] = ds
    S()["instrument"] = inst
    S().pop("dq_block", None)
    for k in ("split", "cur_bt", "grid", "grid_stability", "candidates", "blind", "validation_cmp"):
        S().pop(k, None)
    if previous is not None and previous.symbol != inst.symbol:
        # execution overrides belong to a market; a new market starts from its own defaults
        strat = strategy()
        S()["strategy"] = {**strat, "execution": ExecutionParams(fill_model=strat["execution"].fill_model,
                                                                 ambiguity=strat["execution"].ambiguity)}
        _bump()
        S()["strategy_changed_note"] = f"Execution costs reset to {inst.symbol} defaults."
    restore_split(ds)
    msg = f"{ds.prep_full.n_days:,} tradable sessions"
    if lock and lock.get("lockbox_start"):
        msg += f"; lockbox withholds sessions from {lock['lockbox_start']}"
    return msg


def restore_split(ds) -> None:
    split = ps.load_split(ds.prep_full.data_hash)
    if not split:
        return
    dates = pd.DatetimeIndex(ds.prep_full.dates.astype("datetime64[ns]"))
    if len(dates) and pd.Timestamp(split["holdout_start"]) <= dates[-1]:
        ds.prep = subset_before(ds.prep_full, split["holdout_start"])
        S()["split"] = split


def autoload() -> None:
    """On first open, load a locally saved free SPY copy if one exists (no download, no network)."""
    if dataset() is not None or S().get("autoload_done"):
        return
    S()["autoload_done"] = True
    etfs = etf_instruments()
    for sym in ("SPY", "QQQ"):
        if yahoo_saved_copy(sym) is not None:
            try:
                load_dataset(YAHOO, etfs[sym], symbol=sym)
                S()["autoloaded"] = sym
            except Exception:  # never block the app on autoload
                pass
            return


# ---------------------------------------------------------------------------------------------- current strategy
def default_strategy() -> dict:
    cfg = load_strategy_config()
    return {"params": cfg["strategy"], "execution": ExecutionParams(),
            "sizing": SizingParams(mode="pct_equity", risk_pct=0.01, risk_dollars=400.0, starting_equity=40_000.0, max_contracts=1e6),
            "source": "default reference strategy", "name": "Reference 09:30 ORB"}


def strategy() -> dict:
    if "strategy" not in S():
        S()["strategy"] = default_strategy()
    return S()["strategy"]


def _bump() -> None:
    """Strategy changed from outside the Strategy form: rebuild the form's widgets from the new values."""
    S()["strat_ver"] = S().get("strat_ver", 0) + 1


def set_params(params: StrategyParams, source: str, name: str | None = None) -> None:
    strat = strategy()
    S()["strategy"] = {**strat, "params": params, "source": source, "name": name or f"Strategy {params.key()[:6]}"}
    _bump()


def set_strategy(params: StrategyParams, execution: ExecutionParams, sizing: SizingParams, source: str, name: str | None = None,
                 from_form: bool = False) -> None:
    S()["strategy"] = {"params": params, "execution": execution, "sizing": sizing, "source": source,
                       "name": name or strategy().get("name", "Strategy")}
    if not from_form:
        _bump()


def use_strategy(params: StrategyParams, source: str, page: str | None = None, name: str | None = None) -> None:
    """Callback for 'Use this strategy' / 'Backtest' / 'Validate' / 'Stress test' actions."""
    set_params(params, source, name)
    S()["flash"] = f"Current strategy set from {source}."
    if page:
        go(page)


def fingerprint(extra: str = "") -> str:
    """Identity of (data, strategy, execution, sizing): results are only shown for the exact same combination."""
    strat = strategy()
    ds = dataset()
    payload = {"data": ds.prep_full.data_hash if ds is not None else None, "n": ds.prep.n_days if ds is not None else 0,
               "sym": instrument().symbol if instrument() is not None else None, "params": strat["params"].to_dict(),
               "execution": strat["execution"].to_dict(), "sizing": strat["sizing"].to_dict(), "extra": extra}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def store(kind: str, value) -> None:
    S().setdefault("results", {}).setdefault(kind, {})[fingerprint()] = value


def fetch(kind: str):
    return S().get("results", {}).get(kind, {}).get(fingerprint())


def stop_text(p: StrategyParams) -> str:
    if p.stop_method == "or_opposite":
        return "Opposite-side stop"
    if p.stop_method == "or_mid":
        return "Midpoint stop"
    if p.stop_method == "or_pct":
        return f"{p.stop_param:.0%} of range stop"
    if p.stop_method == "atr":
        return f"{p.stop_param:g}× ATR stop"
    return f"{p.stop_param:g}-tick stop"


def confirmation_text(p: StrategyParams) -> str:
    from ..optimization.parameter_space import confirmation_label

    lab = confirmation_label(p.confirm_ticks, p.confirm_or_frac)
    return {"close": "close beyond range", "1tick": "close 1 tick beyond", "2tick": "close 2 ticks beyond", "5pct": "close 5% of range beyond",
            "10pct": "close 10% of range beyond"}.get(lab, lab)


def strategy_chips(p: StrategyParams | None = None, symbol: str | None = None) -> list[str]:
    p = p or strategy()["params"]
    symbol = symbol or (instrument().symbol if instrument() is not None else "no market")
    entry_tf = "intrabar" if p.entry_method == "stop" else f"{p.entry_tf}m entry"
    out = [symbol, f"{p.orb_start} ORB", f"{p.range_minutes}m range", entry_tf, p.entry_method.capitalize(), stop_text(p),
           f"{p.target_r:g}R target" if p.target_r > 0 else "no target", DIRECTION_LABELS[p.direction], f"last entry {p.cutoff}"]
    return out


def strategy_one_liner(p: StrategyParams | None = None, symbol: str | None = None) -> str:
    return " · ".join(strategy_chips(p, symbol))


def params_text(p: StrategyParams) -> str:
    """Strategy description without the market (pure function of the parameters)."""
    return " · ".join(strategy_chips(p, "-")[1:])


def portfolio_text(sizing: SizingParams | None = None, inst=None) -> str:
    sizing = sizing or strategy()["sizing"]
    inst = inst or instrument()
    unit = inst.unit if inst is not None else "unit"
    if sizing.mode == "fixed_contracts":
        return f"${sizing.starting_equity:,.0f} starting equity · fixed {sizing.contracts:g} {unit}{'s' if sizing.contracts != 1 else ''}/trade"
    return f"${sizing.starting_equity:,.0f} starting equity · {sizing.risk_pct * 100:g}% risk/trade ({SIZING_SHORT[sizing.mode]})"


def execution_text(execution: ExecutionParams | None = None, inst=None) -> str:
    execution = execution or strategy()["execution"]
    inst = inst or instrument()
    if inst is None:
        return "market defaults"
    c = resolve_costs(inst, execution)
    u = inst.unit
    parts = []
    if c.commission_per_side:
        parts.append(f"${c.commission_per_side:g}/{u}/side commission")
    if c.exchange_fee_per_side:
        parts.append(f"${c.exchange_fee_per_side:g}/{u}/side fees")
    if c.friction_per_side:
        parts.append(f"${c.friction_per_side:g}/{u}/side friction")
    parts.append(f"{c.slippage_ticks:g} tick slippage")
    parts.append(f"{execution.fill_model} fills")
    return " · ".join(parts)


# ---------------------------------------------------------------------------------------------- backtests
def default_stage_sizing(inst) -> SizingParams:
    """Fixed size for comparing stages: 100 shares for ETFs, 1 contract for futures (R figures do not depend on it)."""
    return SizingParams(mode="fixed_contracts", contracts=100 if is_proxy(inst) else 1, starting_equity=40_000, max_contracts=1e9,
                        max_leverage=0)


def current_backtest(start=None, end=None):
    """Backtest of the current strategy on the development data (holdout withheld). Cached per inputs."""
    ds = dataset()
    if ds is None:
        return None
    strat = strategy()
    key = (fingerprint(), str(start), str(end))
    cache = S().get("cur_bt")
    if cache and cache[0] == key:
        return cache[1]
    try:
        strat["params"].validate(ds.prep.base_minutes)
        strat["sizing"].validate()
    except ValueError as exc:
        S()["cur_bt_error"] = str(exc)
        return None
    S().pop("cur_bt_error", None)
    res = run_backtest(ds.prep, strat["params"], strat["execution"], strat["sizing"], start, end)
    bench = run_backtest(ds.prep, strat["params"], replace(strat["execution"], frictionless=True), strat["sizing"], start, end)
    S()["cur_bt"] = (key, {"res": res, "bench": bench, "period": (start, end)})
    return S()["cur_bt"][1]


# ---------------------------------------------------------------------------------------------- candidates (saved strategies)
def _candidate_file():
    ds = dataset()
    return None if ds is None else ps.data_dir(ds.prep_full.data_hash) / "candidates.json"


def candidates() -> dict:
    if "candidates" not in S():
        f = _candidate_file()
        S()["candidates"] = json.loads(f.read_text()) if f and f.exists() else {}
    return S()["candidates"]


def save_candidate(name: str, params: StrategyParams, source: str) -> str:
    store_ = candidates()
    base, k = name, 2
    while name in store_ and store_[name]["params"] != params.to_dict():
        name = f"{base} ({k})"
        k += 1
    store_[name] = {"params": params.to_dict(), "source": source, "key": params.key(),
                    "added_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    f = _candidate_file()
    if f is not None:
        f.write_text(json.dumps(store_, indent=2, default=str))
    return name


# ---------------------------------------------------------------------------------------------- research stages
def frozen_match():
    """Frozen record whose parameters equal the current strategy's, if any."""
    ds = dataset()
    if ds is None:
        return None
    key = strategy()["params"].key()
    match = [f for f in ps.list_frozen(ds.prep_full.data_hash) if StrategyParams(**f["strategy"]).key() == key]
    return match[-1] if match else None


def blind_result():
    ds = dataset()
    rec = frozen_match()
    if ds is None or rec is None:
        return None
    tests = [t for t in ps.read_blind_tests(ds.prep_full.data_hash) if t["sha256"] == rec["sha256"]]
    return tests[0] if tests else None


def latest_wfo_summary() -> tuple[str, dict] | None:
    from .. import jobs

    ds = dataset()
    if ds is None:
        return None
    for j in jobs.list_jobs(kind="wfo"):
        try:
            spec = json.loads((j / "spec.json").read_text())
        except OSError:
            continue
        if spec.get("data", {}).get("data_hash") != ds.prep_full.data_hash or jobs.read_status(j).get("state") != "done":
            continue
        for f in sorted((j / "result").glob("*/*/summary.json")):
            return f"{j.name} · {f.parent.parent.name}/{f.parent.name}", json.loads(f.read_text())
    return None


def stage_list() -> list[dict]:
    """Research stages of the current strategy, for the status badges. Never green just because a backtest looked good."""
    out = []
    bt = current_backtest()
    if bt is None:
        out.append({"name": "In-sample", "status": "Not run", "detail": "load data", "tone": "none"})
    else:
        m = bt["res"].metrics
        out.append({"name": "In-sample", "status": "Backtested", "tone": "info",
                    "detail": f"{rmult(m['avg_r'])} on {m['n_trades']} trades (development data) · not evidence"})
    val = fetch("validation")
    if val is None:
        out.append({"name": "Validation", "status": "Not run", "detail": "Validate → holdout test", "tone": "none"})
    else:
        v = val["validation"]
        ok = v["avg_r"] > 0
        out.append({"name": "Validation", "status": "Positive" if ok else "Not positive", "tone": "warn" if ok else "bad",
                    "detail": f"{rmult(v['avg_r'])} on {v['n_trades']} trades · still a selection step"})
    blind = blind_result()
    if blind is None:
        out.append({"name": "Blind OOS", "status": "Not run", "detail": "freeze, then test once", "tone": "none"})
    else:
        m = blind["metrics"]
        is_blind = blind["status"] == "BLIND"
        pos = (m.get("avg_r") or 0) > 0
        tone = ("good" if pos and m.get("n_trades", 0) >= 30 else "warn" if pos else "bad") if is_blind else "warn"
        out.append({"name": "Blind OOS", "status": ("Positive" if pos else "Not positive") + ("" if is_blind else " (not blind)"),
                    "tone": tone, "detail": f"{rmult(m.get('avg_r'))} on {m.get('n_trades')} trades"})
    w = latest_wfo_summary()
    if w is None:
        out.append({"name": "Walk-forward", "status": "Not run", "detail": "Validate → walk-forward", "tone": "none"})
    else:
        s = w[1].get("stitched_always_trade", {})
        n, e = int(s.get("n_trades") or 0), s.get("avg_r") or 0.0
        out.append({"name": "Walk-forward", "status": "OOS positive" if e > 0 else "OOS not positive",
                    "tone": ("good" if n >= 30 else "warn") if e > 0 else "bad", "detail": f"{rmult(e)} on {n} OOS trades (method)"})
    rob = fetch("robustness")
    if rob is None:
        out.append({"name": "Robustness", "status": "Not run", "detail": "Stress test → parameters", "tone": "none"})
    else:
        label, tone = robustness_label(rob["verdict"])
        out.append({"name": "Robustness", "status": label, "tone": tone, "detail": "parameter neighbourhood"})
    stress = fetch("stress")
    if stress is None:
        out.append({"name": "Execution", "status": "Not run", "detail": "Stress test → execution", "tone": "none"})
    else:
        v = stress[1]["verdict"]
        out.append({"name": "Execution", "status": v.title(), "tone": {"ROBUST": "good", "FRAGILE": "bad"}.get(v, "bad"),
                    "detail": "slippage, costs, fills"})
    mc = fetch("mc")
    if mc is None:
        out.append({"name": "Monte Carlo", "status": "Not run", "detail": "Stress test → Monte Carlo", "tone": "none"})
    else:
        p = mc["result"].probabilities["p_loss"]
        out.append({"name": "Monte Carlo", "status": f"{p:.0%} chance of loss", "tone": "good" if p < 0.1 else "warn" if p < 0.3 else "bad",
                    "detail": "path risk, not evidence"})
    return out


def robustness_label(verdict: str) -> tuple[str, str]:
    if verdict.startswith("PLATEAU"):
        return "Broad plateau", "good"
    if verdict.startswith("SPIKE"):
        return "Spike / fragile", "bad"
    if verdict.startswith("NOT POSITIVE"):
        return "Not positive", "bad"
    if verdict.startswith("MIXED"):
        return "Moderate", "warn"
    return verdict.title(), "none"


def rmult(x) -> str:
    from .theme import rmult as _r

    return _r(x)

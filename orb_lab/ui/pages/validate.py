"""VALIDATE: out-of-sample testing. Blind holdout (one shot) and walk-forward (rolling, stitched blind OOS)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from ... import jobs
from ...engine.backtest import run_backtest
from ...engine.params import ExecutionParams, SizingParams, StrategyParams, load_strategy_config
from ...optimization.parameter_space import load_search_spaces, params_from_row
from ...optimization.walk_forward import STRUCTURES, WFOStructure
from ...reports.experiment import RUNS_DIR
from ...research import pipeline_state as ps
from ...research.lockbox import subset_before
from .. import charts as C
from .. import logic as L
from .. import state as S
from .. import theme as T
from .. import widgets as W

MONTH_PRESETS = {k: v for k, v in STRUCTURES.items() if k != "model_C_48_12_12"}
KEYS = ("n_trades", "avg_r", "t_stat_r", "profit_factor", "net_pnl", "max_dd", "win_rate", "sharpe", "top5_share", "pnl_ex_top5", "n_long",
        "n_short", "pnl_long", "pnl_short", "total_cost", "cost_per_trade", "ambiguous_exit_pct", "years", "total_return", "end_equity")
EVIDENCE = ("Only **out-of-sample** results count as evidence: the one-shot blind holdout and the stitched blind-OOS pieces of the "
            "walk-forward. Training and validation numbers were seen while choosing, so they are shown small and grey.")


def metrics_for(prep, params, execution=None, sizing=None, start=None, end=None) -> tuple[dict, pd.DataFrame]:
    res = run_backtest(prep, params, execution or ExecutionParams(), sizing or S.default_stage_sizing(prep.instrument), start, end)
    return {k: res.metrics.get(k) for k in KEYS}, res.trades


def render() -> None:
    T.page_header("Evidence", "Validate", "Test the strategy on data it has never seen. This is the only step that can produce evidence.")
    W.strategy_header()
    T.flow([("TRAIN", "train"), ("VALIDATION", "val"), ("FREEZE", "freeze"), ("BLIND OOS", "oos"), ("ROLL FORWARD", "roll")])
    st.info(EVIDENCE, icon=":material/verified:")
    if W.need_data():
        return
    tab_h, tab_w = st.tabs(["Blind holdout test", "Walk-forward"])
    with tab_h:
        _holdout()
    with tab_w:
        _walk_forward()


# ---------------------------------------------------------------------------------------------- holdout
def _holdout() -> None:
    ds = S.dataset()
    split = S.S().get("split")
    st.markdown("#### 1 · Reserve a blind holdout")
    tests = ps.read_blind_tests(ds.prep_full.data_hash)
    if split:
        T.kpi_grid([
            T.kpi("Train", f"{split['n_train']} sessions", f"{split['train_start']} – {split['train_end']}", help_term="In-sample"),
            T.kpi("Validation", f"{split['n_val']} sessions", f"{split['val_start']} – {split['val_end']}", tone="warn", help_term="Validation"),
            T.kpi("Blind holdout", f"{split['n_holdout']} sessions", f"{split['holdout_start']} – {split['holdout_end']} · hidden", tone="info",
                  help_term="Blind holdout"),
        ], min_width=200)
    with st.expander("Change the split" if split else "Choose the split", expanded=split is None, icon=":material/call_split:"):
        c = st.columns([1, 1, 2])
        tr = c[0].slider("Train share", 0.3, 0.8, 0.6, 0.05, key="ps_train")
        va = c[1].slider("Validation share", 0.1, 0.4, 0.2, 0.05, key="ps_val")
        try:
            prop = ps.make_split(ds.prep_full.dates, tr, va)
        except ValueError as exc:
            st.error(str(exc))
            return
        c[2].caption(f"Train {prop.train_start} – {prop.train_end} ({prop.n_train}) · validation {prop.val_start} – {prop.val_end} "
                     f"({prop.n_val}) · blind holdout {prop.holdout_start} – {prop.holdout_end} ({prop.n_holdout})")
        if tests:
            st.warning("The holdout has already been used. A new split is recorded, but any further holdout test is labelled NOT BLIND.")
        if st.button("Save split and withhold the holdout" if not split else "Replace split", key="ps_save", type="primary"):
            saved = ps.save_split(ds.prep_full.data_hash, prop)
            ds.prep = subset_before(ds.prep_full, saved["holdout_start"])
            S.S()["split"] = saved
            for k in ("cur_bt", "grid", "grid_stability", "validation_cmp"):
                S.S().pop(k, None)
            st.rerun()
    if not split:
        T.note("Reserve the holdout <b>before</b> optimising. Every other page will then only see the train and validation sessions.", "warn")
        return

    strat = S.strategy()
    st.markdown("#### 2 · Check the strategy on the validation period")
    val = S.fetch("validation")
    if st.button("Run validation check", key="ps_validate_cur", icon=":material/play_arrow:"):
        tr_m, _ = metrics_for(ds.prep, strat["params"], strat["execution"], None, split["train_start"], split["train_end"])
        va_m, _ = metrics_for(ds.prep, strat["params"], strat["execution"], None, split["val_start"], split["val_end"])
        val = {"train": tr_m, "validation": va_m}
        S.store("validation", val)
    if val:
        t, v = val["train"], val["validation"]
        T.kpi_grid([
            T.kpi("Train expectancy", T.rmult(t["avg_r"]), f"{t['n_trades']} trades · NOT evidence", help_term="In-sample"),
            T.kpi("Validation expectancy", T.rmult(v["avg_r"]), f"{v['n_trades']} trades · t {T.ratio(v['t_stat_r'])}",
                  tone="good" if v["avg_r"] > 0 else "bad", help_term="Validation"),
            T.kpi("Validation profit factor", T.ratio(v["profit_factor"]), "gross win ÷ gross loss", tone="good" if (v["profit_factor"] or 0) > 1 else "bad"),
            T.kpi("Change train → validation", T.rmult((v["avg_r"] or 0) - (t["avg_r"] or 0)), "expectancy difference",
                  tone="bad" if (v["avg_r"] or 0) < (t["avg_r"] or 0) else "none"),
        ], min_width=180)
        if v["n_trades"] < 30:
            st.warning(f"Only {v['n_trades']} validation trades: this check can only rule out obvious failures.")
    with st.expander("Compare several optimizer results on the validation period", icon=":material/compare_arrows:"):
        _compare_candidates(ds, split)

    st.markdown("#### 3 · Freeze the strategy")
    frozen = S.frozen_match()
    if frozen:
        T.note(f"The current strategy is frozen as <b>{frozen['name']}</b> · SHA-256 <code>{frozen['sha256'][:16]}…</code> · "
               f"{frozen['frozen_utc']} · hash verified: {'yes' if ps.verify_frozen(frozen) else 'NO'}", "good")
    else:
        st.caption("Freezing writes the strategy, execution and sizing to a read-only file named by its fingerprint. Only frozen "
                   "strategies can be tested on the holdout, so nothing can be tweaked afterwards.")
        name = st.text_input("Name", value=strat.get("name", "candidate"), key="ps_freeze_name")
        if st.button("FREEZE (writes an immutable file with a SHA-256 hash)", key="ps_freeze", type="primary", icon=":material/lock:"):
            dev, _ = metrics_for(ds.prep, strat["params"], strat["execution"], None, split["train_start"], split["val_end"])
            rec = ps.freeze_candidate(ds.prep_full.data_hash, name, strat["params"].to_dict(), strat["execution"].to_dict(),
                                      strat["sizing"].to_dict(), S.instrument().to_dict(), evidence={"development_metrics": dev},
                                      development_end=split["val_end"])
            st.success(f"Frozen {rec['name']} · SHA-256 {rec['sha256'][:16]}…")
            st.rerun()

    st.markdown("#### 4 · Run the blind holdout test (once)")
    if tests:
        st.warning(f"The holdout was already used {len(tests)} time(s) (first: {tests[0]['candidate']}, {tests[0]['status']}). Any further "
                   "test is labelled NOT BLIND and does not count as evidence.")
    frozen = S.frozen_match()
    if frozen is None:
        st.caption("Freeze the current strategy first.")
    else:
        ok = st.checkbox("I will not change the strategy after seeing the result.", key="ps_blind_ok")
        if st.button("RUN BLIND HOLDOUT TEST", type="primary", disabled=not ok, key="ps_blind_run", icon=":material/visibility:"):
            if not ps.verify_frozen(frozen):
                st.error("The frozen file failed hash verification; refusing to run.")
                return
            m, trades = metrics_for(ds.prep_full, StrategyParams(**frozen["strategy"]), ExecutionParams(**frozen["execution"]),
                                    SizingParams(**frozen["sizing"]), split["holdout_start"], split["holdout_end"])
            entry = ps.record_blind_test(ds.prep_full.data_hash, frozen, (split["holdout_start"], split["holdout_end"]), m)
            S.S()["blind"] = {**entry, "trades": trades}
    blind = S.blind_result()
    if blind:
        m = blind["metrics"]
        is_blind = blind["status"] == "BLIND"
        (st.success if is_blind else st.warning)(f"**{blind['status']}** test of {blind['candidate']} "
                                                 f"({blind['holdout_start']} – {blind['holdout_end']})")
        T.kpi_grid([
            T.kpi("Holdout trades", T.count(m["n_trades"]), "small sample" if m["n_trades"] < 30 else "", tone="warn" if m["n_trades"] < 30 else "none"),
            T.kpi("Holdout expectancy", T.rmult(m["avg_r"]), f"t {T.ratio(m['t_stat_r'])}", tone="good" if m["avg_r"] > 0 else "bad", big=True,
                  help_term="OOS"),
            T.kpi("Holdout profit factor", T.ratio(m["profit_factor"]), "", tone="good" if (m["profit_factor"] or 0) > 1 else "bad"),
            T.kpi("Holdout net P&L", T.usd(m["net_pnl"], signed=True), "with the frozen sizing"),
            T.kpi("Holdout max drawdown", T.pct(m["max_dd"]), ""),
        ], min_width=170)
        W.metrics_explained(m, "What do the holdout numbers mean?")
        if S.S().get("blind") is not None:
            S.nav_button("Inspect holdout trades in the Trade Explorer", "trades", key="blind_to_trades", icon=":material/list_alt:")


def _compare_candidates(ds, split) -> None:
    grid = S.S().get("grid")
    options: dict[str, StrategyParams] = {"Current strategy": S.strategy()["params"]}
    if grid is not None and not grid.results.empty:
        for _, row in grid.results.sort_values("rank_composite").head(10).iterrows():
            options[f"Optimizer #{int(row['rank_composite'])}"] = params_from_row(row.to_dict())
    for name, rec in S.candidates().items():
        options.setdefault(name, StrategyParams(**rec["params"]))
    if len(options) == 1:
        st.caption("Run the optimizer (on the train period) or save strategies to compare them here.")
    if st.button("Evaluate on train and validation", key="ps_validate"):
        ex = S.strategy()["execution"]
        rows = []
        for name, p in options.items():
            tr, _ = metrics_for(ds.prep, p, ex, None, split["train_start"], split["train_end"])
            va, _ = metrics_for(ds.prep, p, ex, None, split["val_start"], split["val_end"])
            rows.append({"Candidate": name, "Strategy": S.params_text(p), "Train trades": tr["n_trades"],
                         "Train R": tr["avg_r"], "Val trades": va["n_trades"], "Val R": va["avg_r"], "Val t": va["t_stat_r"],
                         "Val PF": va["profit_factor"]})
        S.S()["validation_cmp"] = {"table": pd.DataFrame(rows), "options": {k: v.to_dict() for k, v in options.items()}}
    cmp_ = S.S().get("validation_cmp")
    if cmp_:
        st.dataframe(cmp_["table"], hide_index=True, width="stretch", column_config={
            "Train R": st.column_config.NumberColumn(format="%+.3f"), "Val R": st.column_config.NumberColumn(format="%+.3f"),
            "Val t": st.column_config.NumberColumn(format="%.2f"), "Val PF": st.column_config.NumberColumn(format="%.2f"),
            "Strategy": st.column_config.TextColumn(width="large")})
        st.caption("Picking the best validation row is itself a selection, so validation numbers are not final evidence.")
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        name = c1.selectbox("Carry forward", list(cmp_["options"]), key="ps_val_pick")
        c2.button("Use this strategy", key="ps_val_use", on_click=S.use_strategy,
                  args=(StrategyParams(**cmp_["options"][name]), f"validation comparison ({name})"), width="stretch")


# ---------------------------------------------------------------------------------------------- walk-forward
def _run_dirs() -> list[Path]:
    runs = [j / "result" for j in jobs.list_jobs(kind="wfo") if (j / "result" / "wfo_overview.csv").exists()]
    runs += sorted(RUNS_DIR.glob("wfo-*"), reverse=True)
    return runs


def _walk_forward() -> None:
    st.markdown("Walk-forward repeatedly **chooses settings on past data only**, **freezes** them, and trades them on the **next unseen "
                "period**, then rolls forward. The unseen pieces are stitched into one blind out-of-sample curve — that curve is the result.")
    runs = _run_dirs()
    with st.expander("Set up a new walk-forward run", expanded=not runs, icon=":material/add_circle:"):
        _launch_form()
    _jobs_panel()
    if runs:
        _browse(runs)


def _launch_form() -> None:
    ds, inst = S.dataset(), S.instrument()
    prep = ds.prep
    n_sessions = prep.n_days
    months = len({str(pd.Timestamp(d).to_period("M")) for d in prep.dates})
    st.caption(f"Development data: {n_sessions} sessions over {months} calendar months ({prep.dates[0]} – {prep.dates[-1]}). "
               "The blind holdout, if reserved, is never used here.")
    cfg = load_strategy_config()
    spaces = load_search_spaces(base=cfg["strategy"])
    unit_default = 0 if months >= 18 else 1
    with st.form("wfo_form", border=False):
        unit = st.radio("Window unit", ["Calendar months (real futures data, needs ≥ 18 months)",
                                        "Trading sessions (short samples, e.g. the free ~60-day data)"], index=unit_default, horizontal=True)
        c = st.columns(2, gap="large")
        with c[0]:
            chosen = st.multiselect("Month structures (train / validation / OOS = roll)", list(MONTH_PRESETS), default=["primary_12_3_3"],
                                    format_func=lambda k: f"{'Primary' if k.startswith('primary') else 'Sensitivity'} {MONTH_PRESETS[k].train_months}/"
                                                          f"{MONTH_PRESETS[k].val_months}/{MONTH_PRESETS[k].oos_months}/{MONTH_PRESETS[k].roll_months}")
            custom_m = st.checkbox("Add a custom month structure")
            cc = st.columns(3)
            cm_tr = cc[0].number_input("Train months", 1, 120, 12)
            cm_va = cc[1].number_input("Validation months", 1, 60, 3)
            cm_oo = cc[2].number_input("OOS = roll months", 1, 60, 3)
        with c[1]:
            d_tr = max(5, (n_sessions * 45 // 100) // 5 * 5)
            cc = st.columns(3)
            s_tr = cc[0].number_input("Train sessions", 2, 5000, min(d_tr, 250))
            s_va = cc[1].number_input("Validation sessions", 1, 2000, max(5, min(60, d_tr // 3 // 5 * 5)))
            s_oo = cc[2].number_input("OOS = roll sessions", 1, 2000, max(5, min(60, d_tr // 3 // 5 * 5)))
            preset = st.selectbox("Parameter space searched in each fold", list(spaces), index=list(spaces).index("primary_930_reduced"))
            families = st.multiselect("Entry families (separate run each)", ["all", "market", "limit", "stop"], default=["all"])
        c = st.columns(5)
        min_tr = c[0].number_input("Min trades in train", 1, 1000, 40 if unit_default == 0 else 8)
        min_va = c[1].number_input("Min trades in validation", 1, 500, 10 if unit_default == 0 else 3)
        n_fin = c[2].number_input("Finalists per fold", 1, 500, 25)
        equity = c[3].number_input("Starting equity ($)", 1000.0, 1e9, float(S.strategy()["sizing"].starting_equity), 1000.0)
        risk = c[4].selectbox("Risk per trade (compounding curve)", [0.25, 0.5, 1.0, 2.0], index=2, format_func=lambda v: f"{v}%")
        c = st.columns(3)
        start = c[0].date_input("First session to use", pd.Timestamp(prep.dates[0]), min_value=pd.Timestamp(prep.dates[0]),
                                max_value=pd.Timestamp(prep.dates[-1]))
        end = c[1].date_input("Last session", pd.Timestamp(prep.dates[-1]), min_value=pd.Timestamp(prep.dates[0]), max_value=pd.Timestamp(prep.dates[-1]))
        workers = c[2].number_input("Worker processes", 1, os.cpu_count() or 1, max(1, (os.cpu_count() or 2) - 1))
        st.caption(f"Costs: the current strategy's execution assumptions ({S.execution_text()}).".replace("$", "\\$"))
        submitted = st.form_submit_button("LAUNCH WALK-FORWARD (runs in the background)", type="primary", icon=":material/rocket_launch:")
    if not submitted:
        return
    if unit.startswith("Calendar"):
        structs = [MONTH_PRESETS[k] for k in chosen]
        if custom_m:
            structs.append(WFOStructure(f"custom_{cm_tr}_{cm_va}_{cm_oo}", int(cm_tr), int(cm_va), int(cm_oo), int(cm_oo)))
    else:
        structs = [WFOStructure(f"sessions_{s_tr}_{s_va}_{s_oo}", int(s_tr), int(s_va), int(s_oo), int(s_oo), "sessions")]
    if not structs or not families:
        st.error("Choose at least one structure and one entry family.")
        return
    space = spaces[preset]
    n_cfg = len(space.expand(prep.base_minutes).configs)
    need = max((s.train_months + s.val_months + s.oos_months) for s in structs)
    have = months if structs[0].unit == "months" else n_sessions
    if need > have:
        st.error(f"Not enough history: one fold needs {need} {structs[0].unit}, the data has {have}. "
                 + ("Switch the window unit to trading sessions for short samples." if structs[0].unit == "months" else "Use shorter windows."))
        return
    spec = {"structures": [asdict(s) for s in structs], "space": {"name": space.name, "grid": space.grid},
            "execution": S.strategy()["execution"].to_dict(),
            "selection": {"min_trades_train": int(min_tr), "min_trades_val": int(min_va), "n_finalists": int(n_fin)}, "families": families,
            "start": str(start), "end": str(end), "n_workers": int(workers), "starting_equity": float(equity), "risk_pct": float(risk) / 100,
            "data": {**ds.describe(), "data_hash": prep.data_hash, "source": ds.loaded.source, "symbol": inst.symbol, "proxy": S.is_proxy(inst),
                     "synthetic": S.is_synthetic(ds)}, "n_configs": n_cfg}
    job_dir = jobs.submit("wfo", spec, prep)
    st.success(f"Launched: {n_cfg:,} configurations × {len(structs)} structure(s) × {len(families)} family(ies). Progress appears below; "
               "you can use other pages or close the browser meanwhile.")
    S.S()["wfo_job"] = str(job_dir)


def _jobs_panel() -> None:
    all_jobs = jobs.list_jobs(kind="wfo")[:6]
    if not all_jobs:
        return
    running = any(jobs.read_status(j).get("state") in ("queued", "starting", "running") for j in all_jobs)

    @st.fragment(run_every=2 if running else None)
    def panel():
        with st.container(border=True):
            st.markdown("**Walk-forward runs**")
            for j in all_jobs:
                s = jobs.read_status(j)
                try:
                    spec = json.loads((j / "spec.json").read_text())
                except OSError:
                    continue
                state = s.get("state", "?")
                c = st.columns([3, 4, 1])
                c[0].markdown(f"{spec['data'].get('symbol', '?')} · {', '.join(x['name'] for x in spec['structures'])}  \n"
                              f"<span style='color:{T.MUTED};font-size:0.8rem'>{j.name}</span>", unsafe_allow_html=True)
                total = max(int(s.get("total") or 0), 1)
                frac = min(int(s.get("done") or 0) / total, 1.0) if state == "running" else (1.0 if state == "done" else 0.0)
                c[1].progress(frac, text=f"{state} · {s.get('stage', '')} · {s.get('elapsed_s', 0)}s")
                if state in ("running", "starting", "queued"):
                    if c[2].button("Stop", key=f"stop_{j.name}"):
                        jobs.stop(j)
                        st.rerun()
                elif state in ("stopped", "failed", "interrupted"):
                    if c[2].button("Resume", key=f"resume_{j.name}"):
                        jobs.resume(j)
                        st.rerun()
                if state == "failed":
                    st.error(s.get("error", "failed"))
    panel()


def _browse(runs: list[Path]) -> None:
    labels = {str(r): (r.parent.name + " (dashboard)" if r.name == "result" else r.name + " (command line)") for r in runs}
    c1, c2, c3 = st.columns([2, 1, 1])
    run = Path(c1.selectbox("Walk-forward run", [str(r) for r in runs], format_func=labels.get, key="wfo_run"))
    overview = pd.read_csv(run / "wfo_overview.csv") if (run / "wfo_overview.csv").exists() else pd.DataFrame()
    if overview.empty:
        st.info("This run produced no folds (not enough history for the structure).")
        return
    structure = c2.selectbox("Structure", sorted(overview["structure"].unique()), key="wfo_struct")
    family = c3.selectbox("Entry family", sorted(overview[overview["structure"] == structure]["family"].unique()), key="wfo_fam")
    manifest = json.loads((run / "manifest.json").read_text()) if (run / "manifest.json").exists() else {}
    data = manifest.get("data", {})
    spec_file = run.parent / "spec.json"
    spec = json.loads(spec_file.read_text()) if spec_file.exists() else {}
    if data.get("proxy"):
        st.warning("Free proxy data (ETF). Not futures validation.")
    if data.get("synthetic") or "synthetic" in str(data.get("source", "")):
        st.warning("Synthetic data. Software test only; not market evidence.")
    d = run / structure / family
    summary = json.loads((d / "summary.json").read_text())
    windows = pd.read_csv(d / "windows.csv")
    daily = pd.read_csv(d / "stitched_oos_daily.csv", index_col=0, parse_dates=True)
    s = summary.get("stitched_always_trade", {})
    pctq = summary.get("stitched_pct_equity_1pct", {})
    equity0 = float(spec.get("starting_equity", 100_000.0))
    risk_pct = float(spec.get("risk_pct", 0.01))
    n_oos = int(s.get("n_trades") or 0)
    fc = L.fold_consistency(windows)
    deg = L.degradation(summary)

    T.section("Stitched blind-OOS performance (the evidence)")
    T.kpi_grid([
        T.kpi("OOS return", T.pct((pctq.get("net_pnl") or 0) / equity0, signed=True), f"compounding at {risk_pct * 100:g}% risk/trade",
              tone="good" if (pctq.get("net_pnl") or 0) > 0 else "bad", big=True, help_term="OOS"),
        T.kpi("OOS expectancy", T.rmult(s.get("avg_r")), f"t {T.ratio(s.get('t_stat_r'))}", tone="good" if (s.get("avg_r") or 0) > 0 else "bad",
              big=True, help_term="Expectancy"),
        T.kpi("OOS profit factor", T.ratio(s.get("profit_factor")), "", tone="good" if (s.get("profit_factor") or 0) > 1 else "bad"),
        T.kpi("OOS max drawdown", T.pct(pctq.get("max_dd", s.get("max_dd"))), "compounding curve"),
        T.kpi("OOS trades", T.count(n_oos), "too few to conclude" if n_oos < 30 else "", tone="warn" if n_oos < 30 else "none"),
        T.kpi("Profitable folds", f"{fc['positive']} of {fc['folds']}", "fold-by-fold consistency",
              tone="good" if fc["folds"] and fc["share"] >= 0.6 else "warn"),
    ], min_width=160)
    flags = summary.get("overfit_flags", [])
    if flags:
        st.error("Overfitting warnings: " + "; ".join(flags), icon=":material/report:")
    if summary.get("execution_sensitive"):
        st.error("Execution-sensitive: stitched OOS expectancy turns non-positive within +2 ticks of slippage.")
    oos_file = d / "oos_trades.csv"
    if oos_file.exists():
        ot = pd.read_csv(oos_file, parse_dates=["exit_time"]).sort_values("exit_time")
        if "sized_out" in ot:
            ot = ot[~ot["sized_out"].astype(bool)]
        if len(ot):
            st.plotly_chart(C.cum_r_time(ot, "Stitched blind-OOS result: cumulative R (size-independent, net of costs)"), width="stretch")
    elif len(daily):
        st.plotly_chart(C.equity(daily, "Stitched blind-OOS equity (1 unit, net of costs)"), width="stretch")
    c1, c2 = st.columns(2, gap="medium")
    traded = windows[windows["selected"] >= 0] if "selected" in windows else windows
    if len(traded):
        c1.plotly_chart(C.bars([f"Fold {int(w) + 1}" for w in traded["window"]], traded["oos_exp_r"].to_numpy(),
                               "Blind-OOS expectancy by fold"), width="stretch")
    c2.plotly_chart(C.bars(["Training", "Validation", "Blind OOS"], [deg["train"] or 0, deg["validation"] or 0, deg["oos"] or 0],
                           "Degradation: median expectancy by phase", muted=[True, True, False]), width="stretch")
    if deg["drop_share"] == deg["drop_share"]:
        st.caption(f"Out of sample kept {max(0.0, 1 - deg['drop_share']):.0%} of the training expectancy (grey bars were seen while choosing).")
    with st.expander("Folds: layout and the settings chosen in each (frozen before its OOS period)", icon=":material/view_timeline:"):
        st.plotly_chart(C.timeline(windows), width="stretch")
        cols = [c for c in ("window", "train_start", "train_end", "val_start", "val_end", "oos_start", "oos_end", "param_range_minutes",
                            "param_entry_tf", "param_entry_method", "param_stop", "param_target_r", "param_cutoff", "param_direction",
                            "train_exp_r", "val_exp_r", "oos_trades", "oos_exp_r", "frozen_sha256") if c in windows]
        st.dataframe(windows[cols], hide_index=True, width="stretch", column_config={
            "train_exp_r": st.column_config.NumberColumn("train R (not evidence)", format="%+.3f"),
            "val_exp_r": st.column_config.NumberColumn("val R (not evidence)", format="%+.3f"),
            "oos_exp_r": st.column_config.NumberColumn("OOS R", format="%+.3f")})
        stab = summary.get("parameter_stability", {})
        if stab:
            st.plotly_chart(C.bars(list(stab), list(stab.values()), "Parameter stability (share of folds choosing the most common value)",
                                   fmt=".0%", suffix="", text=[f"{v:.0%}" for v in stab.values()]), width="stretch")
        if len(traded):
            last = traded.iloc[-1]
            row = {k[len("param_"):]: v for k, v in last.items() if str(k).startswith("param_")}
            try:
                p = params_from_row(row)
                st.caption(f"Most recent fold's frozen choice: {S.strategy_one_liner(p)}")
                st.button("Use the latest fold's choice as the current strategy", key="wfo_use", on_click=S.use_strategy,
                          args=(p, f"walk-forward latest fold ({structure}/{family})"))
            except Exception:
                pass
    with st.expander("Advanced: slippage sensitivity, all structures, raw summary", icon=":material/table:"):
        st.dataframe(pd.read_csv(d / "slippage_sensitivity.csv"), hide_index=True, width="stretch")
        st.dataframe(overview, hide_index=True, width="stretch")
        st.json(summary, expanded=False)

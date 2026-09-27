"""WALK-FORWARD tab: configure and launch walk-forward runs in the background, watch progress, stop/resume,
and inspect results (GUI jobs and ``cli wfo`` runs alike)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import jobs
from ..engine.params import load_strategy_config
from ..optimization.heatmaps import AGGREGATIONS, heatmap_table
from ..optimization.parameter_space import ParameterSpace, load_search_spaces, params_from_row
from ..optimization.walk_forward import STRUCTURES, WFOStructure
from ..reports import plots
from ..reports.experiment import RUNS_DIR
from . import common

PHASE_PAIRS = [
    ("range_minutes", "target_r", "Range x R"),
    ("range_minutes", "entry_tf", "Range x Entry TF"),
    ("target_r", "cutoff", "R x Cutoff"),
    ("stop", "target_r", "Stop x R"),
    ("confirmation", "target_r", "Confirmation x R"),
    ("range_minutes", "stop", "Range x Stop"),
    ("orb_start", "range_minutes", "Start time x Range"),
]
MONTH_PRESETS = {k: v for k, v in STRUCTURES.items() if k != "model_C_48_12_12"}
NOT_EVIDENCE = ("**Only the stitched blind out-of-sample (OOS) segments are evidence.** Training and validation numbers are what the "
                "selection saw; they are shown only to measure how much performance degrades out of sample.")


def _window_bounds(w: pd.Series, phase: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if f"{phase}_start" in w and pd.notna(w.get(f"{phase}_start")):
        return pd.Timestamp(w[f"{phase}_start"]), pd.Timestamp(w[f"{phase}_end"]) + pd.Timedelta(days=1)
    a, b = str(w[f"{phase}_months"]).split("..")
    return pd.Period(a, "M").start_time, pd.Period(b, "M").end_time


def window_timeline(windows: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = {"train": "#9aa5b1", "val": "#f2b134", "oos": "#2a6fdb"}
    names = {"train": "train (not evidence)", "val": "validation (not evidence)", "oos": "blind OOS (evidence)"}
    for _, w in windows.iterrows():
        for phase in ("train", "val", "oos"):
            start, end = _window_bounds(w, phase)
            fig.add_trace(go.Bar(x=[(end - start).total_seconds() * 1000], base=[start], y=[f"W{int(w['window']):02d}"], orientation="h",
                                 marker_color=colors[phase], name=names[phase], showlegend=bool(w["window"] == 0),
                                 hovertemplate=f"{phase}: {start.date()}..{(end - pd.Timedelta(days=1)).date()}<extra></extra>"))
    fig.update_layout(barmode="overlay", title="Walk-forward windows (grey train, amber validation, blue blind OOS)",
                      height=max(300, 18 * len(windows) + 120), xaxis_type="date", **{k: v for k, v in plots.LAYOUT.items() if k != "hovermode"})
    fig.update_yaxes(autorange="reversed")
    return fig


# ------------------------------------------------------------------------------------------ launch
def _launch_form() -> None:
    ds = common.dataset()
    inst = common.instrument()
    if ds is None:
        st.info("Load data in the sidebar to configure a new walk-forward run. Existing runs are listed below.")
        return
    prep = ds.prep
    n_sessions = prep.n_days
    months = len({str(pd.Timestamp(d).to_period("M")) for d in prep.dates})
    common.dev_notice()
    st.caption(f"Loaded development data: {n_sessions} sessions over {months} calendar months "
               f"({prep.dates[0]} .. {prep.dates[-1]}).")
    cfg = load_strategy_config()
    spaces = load_search_spaces(base=cfg["strategy"])
    unit_default = 0 if months >= 18 else 1
    with st.form("wfo_form"):
        st.markdown("**1. Window structure** (train → validation → blind OOS, then roll forward by the OOS length)")
        unit = st.radio("Window unit", ["Calendar months (real futures data, needs ≥ 18 months)",
                                        "Trading sessions (short samples, e.g. the free ~60-day data)"], index=unit_default, horizontal=True)
        c = st.columns(2)
        chosen = c[0].multiselect("Month structures (train/validation/OOS/roll)", list(MONTH_PRESETS), default=["primary_12_3_3"],
                                  format_func=lambda k: f"{k.split('_', 1)[0]} {MONTH_PRESETS[k].train_months}/{MONTH_PRESETS[k].val_months}/"
                                                        f"{MONTH_PRESETS[k].oos_months}/{MONTH_PRESETS[k].roll_months}")
        with c[1]:
            custom_m = st.checkbox("Add a custom month structure")
            cc = st.columns(3)
            cm_tr = cc[0].number_input("Train months", 1, 120, 12)
            cm_va = cc[1].number_input("Validation months", 1, 60, 3)
            cm_oo = cc[2].number_input("OOS = roll months", 1, 60, 3)
        c = st.columns(3)
        d_tr = max(5, (n_sessions * 45 // 100) // 5 * 5)
        s_tr = c[0].number_input("Train sessions", 2, 5000, min(d_tr, 250))
        s_va = c[1].number_input("Validation sessions", 1, 2000, max(5, min(60, d_tr // 3 // 5 * 5)))
        s_oo = c[2].number_input("OOS = roll sessions", 1, 2000, max(5, min(60, d_tr // 3 // 5 * 5)))
        st.markdown("**2. Parameter space** (every configuration is simulated once; selection per window uses only its train/validation data)")
        c = st.columns(2)
        preset = c[0].selectbox("Parameter space", list(spaces), index=list(spaces).index("primary_930_reduced"))
        families = c[1].multiselect("Entry families (separate walk-forward per family)", ["all", "market", "limit", "stop"], default=["all"])
        st.markdown("**3. Costs and execution**")
        execution = common.execution_form(inst, "wfo")
        st.markdown("**4. Selection rules and sizing of the stitched curve**")
        c = st.columns(5)
        min_tr = c[0].number_input("Min trades in train", 1, 1000, 40 if unit_default == 0 else 8)
        min_va = c[1].number_input("Min trades in validation", 1, 500, 10 if unit_default == 0 else 3)
        n_fin = c[2].number_input("Finalists per window", 1, 500, 25)
        equity = c[3].number_input("Starting equity ($)", 1000.0, 1e9, 40_000.0, 1000.0)
        risk = c[4].selectbox("Risk per trade (compounding curve)", [0.25, 0.5, 1.0, 2.0], index=2, format_func=lambda v: f"{v} %")
        c = st.columns(3)
        start = c[0].date_input("First session windows may use", pd.Timestamp(prep.dates[0]), min_value=pd.Timestamp(prep.dates[0]),
                                max_value=pd.Timestamp(prep.dates[-1]))
        end = c[1].date_input("Last session", pd.Timestamp(prep.dates[-1]), min_value=pd.Timestamp(prep.dates[0]),
                              max_value=pd.Timestamp(prep.dates[-1]))
        workers = c[2].number_input("Worker processes", 1, os.cpu_count() or 1, max(1, (os.cpu_count() or 2) - 1))
        submitted = st.form_submit_button("LAUNCH WALK-FORWARD (runs in the background)", type="primary")
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
        st.error(f"Not enough history: one window needs {need} {structs[0].unit}, the data has {have}. "
                 + ("Switch the window unit to trading sessions for short samples." if structs[0].unit == "months" else "Use shorter windows."))
        return
    spec = {"structures": [asdict(s) for s in structs], "space": {"name": space.name, "grid": space.grid}, "execution": execution.to_dict(),
            "selection": {"min_trades_train": int(min_tr), "min_trades_val": int(min_va), "n_finalists": int(n_fin)}, "families": families,
            "start": str(start), "end": str(end), "n_workers": int(workers), "starting_equity": float(equity), "risk_pct": float(risk) / 100,
            "data": {**ds.describe(), "data_hash": prep.data_hash, "source": ds.loaded.source, "symbol": inst.symbol, "proxy": common.is_proxy(inst), "synthetic": common.is_synthetic(ds)},
            "n_configs": n_cfg}
    job_dir = jobs.submit("wfo", spec, prep)
    common.state()["wfo_job"] = str(job_dir)
    st.success(f"Launched {job_dir.name}: {n_cfg:,} configurations x {len(structs)} structure(s) x {len(families)} family(ies). "
               "Progress is shown below; you can keep using other tabs or close the browser.")


# ------------------------------------------------------------------------------------------ jobs
def _jobs_panel() -> None:
    all_jobs = jobs.list_jobs(kind="wfo")[:10]
    if not all_jobs:
        return
    running = any(jobs.read_status(j).get("state") in ("queued", "starting", "running") for j in all_jobs)

    @st.fragment(run_every=2 if running else None)
    def panel():
        for j in all_jobs:
            s = jobs.read_status(j)
            spec = json.loads((j / "spec.json").read_text())
            state = s.get("state", "?")
            c = st.columns([4, 3, 1, 1])
            label = f"**{j.name}** · {spec['data'].get('symbol', '?')} · {', '.join(x['name'] for x in spec['structures'])}"
            c[0].markdown(label)
            total = max(int(s.get("total") or 0), 1)
            frac = min(int(s.get("done") or 0) / total, 1.0) if state == "running" else (1.0 if state == "done" else 0.0)
            c[1].progress(frac, text=f"{state}: {s.get('stage', '')} ({s.get('done', 0)}/{s.get('total', 0)}) · {s.get('elapsed_s', 0)}s")
            if state in ("running", "starting", "queued"):
                if c[2].button("Stop", key=f"stop_{j.name}"):
                    jobs.stop(j)
                    st.rerun()
            elif state in ("stopped", "failed", "interrupted"):
                if c[3].button("Resume", key=f"resume_{j.name}"):
                    jobs.resume(j)
                    st.rerun()
            if state == "failed":
                st.error(s.get("error", "failed"))
    panel()


# ------------------------------------------------------------------------------------------ browse
def _run_dirs() -> list[Path]:
    runs = [j / "result" for j in jobs.list_jobs(kind="wfo") if (j / "result" / "wfo_overview.csv").exists()]
    runs += sorted(RUNS_DIR.glob("wfo-*"), reverse=True)
    return runs


def _browse() -> None:
    runs = _run_dirs()
    if not runs:
        st.info("No finished walk-forward runs yet.")
        return
    labels = {str(r): (r.parent.name + " (dashboard)" if r.name == "result" else r.name + " (command line)") for r in runs}
    run = Path(st.selectbox("Walk-forward run", [str(r) for r in runs], format_func=labels.get, key="wfo_run"))
    manifest = json.loads((run / "manifest.json").read_text()) if (run / "manifest.json").exists() else {}
    research = "research_mode=True" in manifest.get("notes", [])
    data = manifest.get("data", {})
    if data.get("proxy"):
        st.warning("FREE PROXY DATA (ETF). NOT FUTURES VALIDATION.")
    if data.get("synthetic") or "synthetic" in str(data.get("source", "")):
        st.warning("SYNTHETIC DATA. Software test only; not market evidence.")
    (st.success if research else st.info)(
        "Research-mode run (DQ-gated, lockbox withheld, registered)." if research else
        "Exploratory run (not registered in the research log).")
    overview = pd.read_csv(run / "wfo_overview.csv") if (run / "wfo_overview.csv").exists() else pd.DataFrame()
    if overview.empty:
        st.info("This run produced no windows (not enough history for the structure).")
        return
    st.subheader("All structures and entry families (stitched blind OOS)")
    st.dataframe(overview, width="stretch", hide_index=True)
    c = st.columns(2)
    structure = c[0].selectbox("WFO structure", sorted(overview["structure"].unique()), key="wfo_struct")
    family = c[1].selectbox("Entry family", sorted(overview[overview["structure"] == structure]["family"].unique()), key="wfo_fam")
    d = run / structure / family
    summary = json.loads((d / "summary.json").read_text())
    windows = pd.read_csv(d / "windows.csv")
    daily = pd.read_csv(d / "stitched_oos_daily.csv", index_col=0, parse_dates=True)
    st.info(NOT_EVIDENCE)
    flags = summary.get("overfit_flags", [])
    if flags:
        st.error("Overfitting flags: " + "; ".join(flags))
    if summary.get("execution_sensitive"):
        st.error("EXECUTION-SENSITIVE: stitched OOS expectancy turns non-positive within +2 ticks of slippage.")
    s = summary.get("stitched_always_trade", {})
    st.markdown("#### Stitched blind OOS (the evidence)")
    cols = st.columns(6)
    cols[0].metric("OOS trades", f"{int(s.get('n_trades') or 0):,}")
    cols[1].metric("OOS net P&L (1 unit)", f"${s.get('net_pnl') or 0:,.2f}")
    cols[2].metric("OOS expectancy", f"{s.get('avg_r') or 0:+.3f} R", help=f"t = {s.get('t_stat_r') or 0:.2f}")
    cols[3].metric("OOS profit factor", f"{s.get('profit_factor') or 0:.2f}")
    cols[4].metric("OOS max drawdown", f"{s.get('max_dd') or 0:.1%}")
    cols[5].metric("OOS windows profitable", f"{summary.get('pct_windows_oos_profitable', 0):.0%}")
    n_oos = int(s.get("n_trades") or 0)
    if n_oos < 30:
        st.warning(f"Only {n_oos} OOS trades: far too few to conclude anything.")
    st.plotly_chart(window_timeline(windows), width="stretch")
    if len(daily):
        st.plotly_chart(plots.equity_and_drawdown(daily, title="Stitched blind-OOS equity (1 unit, net of costs)"), width="stretch")
    comp = pd.DataFrame({
        "phase": ["training (selected) — NOT evidence", "validation (selected) — NOT evidence", "blind OOS (selected) — EVIDENCE",
                  "best raw training config: training", "best raw training config: OOS"],
        "median expectancy R": [summary.get("median_train_exp_r"), summary.get("median_val_exp_r"), summary.get("median_oos_exp_r"),
                                summary.get("median_best_train_exp_r"), summary.get("median_best_train_oos_exp_r")]})
    st.subheader("Degradation from in-sample to out-of-sample")
    st.dataframe(comp, hide_index=True)
    st.subheader("Per-window selections (frozen before each OOS window; frozen_sha256 = hash of the parameters)")
    st.dataframe(windows, width="stretch", hide_index=True)
    traded = windows[windows["selected"] >= 0] if "selected" in windows else windows.iloc[0:0]
    if len(traded):
        last = traded.iloc[-1]
        row = {k[len("param_"):]: v for k, v in last.items() if str(k).startswith("param_")}
        try:
            params = params_from_row(row)
            st.caption(f"Most recent window's frozen selection: {common.describe_params(params)}")
            common.send_to_buttons(f"WFO {structure}/{family} latest {params.key()}", params,
                                   f"WALK-FORWARD latest selection ({structure}/{family})", key="wfo_to_cand")
        except Exception:  # older runs may lack columns
            pass
    st.subheader("Slippage sensitivity of the frozen selections")
    st.dataframe(pd.read_csv(d / "slippage_sensitivity.csv"), hide_index=True)
    st.subheader("Parameter stability (share of windows choosing the most common value)")
    st.dataframe(pd.Series(summary.get("parameter_stability", {}), name="share").to_frame(), width="content")
    pm_file = run / "phase_metrics.parquet"
    if pm_file.exists():
        st.subheader("Phase heatmaps (descriptive; median across hidden parameters)")
        pm = pd.read_parquet(pm_file)
        c = st.columns(4)
        pairs = [p for p in PHASE_PAIRS if pm[p[0]].nunique() > 1 and pm[p[1]].nunique() > 1]
        if not pairs:
            return
        label = c[0].selectbox("Pair", [p[2] for p in pairs], key="wfo_pair")
        y, x, _ = next(p for p in pairs if p[2] == label)
        metric = c[1].selectbox("Metric", ["exp_r", "sharpe"], key="wfo_metric")
        agg = c[2].selectbox("Aggregation", list(AGGREGATIONS), format_func=lambda k: AGGREGATIONS[k], key="wfo_agg")
        fam = c[3].selectbox("Entry method", ["(all)"] + sorted(pm["entry_method"].unique()), key="wfo_em")
        filters = {} if fam == "(all)" else {"entry_method": fam}
        cols = st.columns(3)
        for col, phase in zip(cols, ("train", "val", "oos")):
            tbl = heatmap_table(pm, x, y, f"{phase}_{metric}", agg, filters)
            col.plotly_chart(plots.heatmap(tbl, f"{phase.upper()} {metric}", metric), width="stretch")


def tab_walk_forward() -> None:
    st.markdown("Walk-forward testing repeatedly **chooses parameters on past data only** (train → validation), **freezes** them, "
                "and trades them on the **next, unseen period** (blind OOS). The OOS pieces are stitched into one curve.")
    st.info(NOT_EVIDENCE)
    with st.expander("Configure and launch a new walk-forward run", expanded=common.dataset() is not None):
        _launch_form()
    _jobs_panel()
    st.divider()
    _browse()

"""PIPELINE tab (guided research workflow) and REPORT tab (final report).

DATA -> BACKTEST -> OPTIMIZE -> VALIDATE -> FREEZE -> BLIND OOS / WALK-FORWARD -> ROBUSTNESS -> MONTE CARLO -> REPORT
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from .. import jobs
from ..engine.backtest import run_backtest
from ..engine.params import ExecutionParams, SizingParams, StrategyParams
from ..optimization.parameter_space import params_from_row
from ..reports.final_report import build_markdown, classify
from ..research import pipeline_state as ps
from ..research.lockbox import subset_before
from . import common

STAGES = ["DATA", "SPLIT", "BACKTEST / OPTIMIZE", "VALIDATE", "FREEZE", "WALK-FORWARD", "ROBUSTNESS + STRESS", "MONTE CARLO",
          "BLIND HOLDOUT", "REPORT"]
KEYS = ("n_trades", "avg_r", "t_stat_r", "profit_factor", "net_pnl", "max_dd", "win_rate", "sharpe", "top5_share", "pnl_ex_top5", "n_long",
        "n_short", "pnl_long", "pnl_short", "total_cost", "cost_per_trade", "ambiguous_exit_pct", "years")


def restore_split(ds) -> None:
    """Re-apply a saved train/validation/holdout split when the same data is loaded again."""
    split = ps.load_split(ds.prep_full.data_hash)
    if not split:
        return
    dates = pd.DatetimeIndex(ds.prep_full.dates.astype("datetime64[ns]"))
    if len(dates) and pd.Timestamp(split["holdout_start"]) <= dates[-1]:
        ds.prep = subset_before(ds.prep_full, split["holdout_start"])
        common.state()["split"] = split


def default_sizing(inst) -> SizingParams:
    """Fixed size for stage comparisons: 100 shares for ETFs (1 share makes $ figures meaningless), 1 contract for futures."""
    return SizingParams(mode="fixed_contracts", contracts=100 if common.is_proxy(inst) else 1, starting_equity=40_000, max_contracts=1e9,
                        max_leverage=0)


def _metrics(prep, params: StrategyParams, execution=None, sizing=None, start=None, end=None) -> tuple[dict, pd.DataFrame]:
    res = run_backtest(prep, params, execution or ExecutionParams(), sizing or default_sizing(prep.instrument), start, end)
    return {k: res.metrics.get(k) for k in KEYS}, res.trades


def _status_rows(ds) -> list[tuple[str, str, str]]:
    s = common.state()
    split = s.get("split")
    grid = s.get("grid")
    rows = [("DATA", "done" if ds is not None and ds.report.ok else "todo",
             f"{ds.prep_full.n_days} sessions" if ds is not None else "load data in the sidebar")]
    rows.append(("SPLIT", "done" if split else "todo", f"holdout {split['holdout_start']}..{split['holdout_end']}" if split else "reserve a blind holdout"))
    if grid is not None and split:
        leak = pd.Timestamp(s.get("grid_period", ("", split["val_end"]))[1]) > pd.Timestamp(split["train_end"])
        rows.append(("BACKTEST / OPTIMIZE", "warn" if leak else "done",
                     "optimizer period includes validation data" if leak else "optimized on train only"))
    else:
        rows.append(("BACKTEST / OPTIMIZE", "done" if grid is not None or s.get("result") is not None else "todo", "BACKTEST and OPTIMIZATION tabs"))
    rows.append(("VALIDATE", "done" if s.get("validated") else "todo", "compare candidates on the validation period"))
    frozen = ps.list_frozen(ds.prep_full.data_hash) if ds is not None else []
    rows.append(("FREEZE", "done" if frozen else "todo", f"{len(frozen)} frozen candidate(s)"))
    wfo_jobs = [j for j in jobs.list_jobs(kind="wfo") if ds is not None and
                json.loads((j / "spec.json").read_text()).get("data", {}).get("data_hash") == ds.prep_full.data_hash]
    rows.append(("WALK-FORWARD", "done" if any(jobs.read_status(j).get("state") == "done" for j in wfo_jobs) else "todo",
                 f"{len(wfo_jobs)} run(s) on this data"))
    rows.append(("ROBUSTNESS + STRESS", "done" if s.get("robustness") and s.get("stress") else ("warn" if s.get("robustness") or s.get("stress") else "todo"),
                 "ROBUSTNESS tab + stress test"))
    rows.append(("MONTE CARLO", "done" if s.get("mc") else "todo", "MONTE CARLO tab"))
    tests = ps.read_blind_tests(ds.prep_full.data_hash) if ds is not None else []
    rows.append(("BLIND HOLDOUT", "done" if tests else "todo", tests[0]["status"] + f" test of {tests[0]['candidate']}" if tests else "one shot"))
    rows.append(("REPORT", "todo", "REPORT tab"))
    return rows


def tab_pipeline() -> None:
    st.markdown("### Research pipeline")
    st.markdown(" → ".join(STAGES))
    st.caption("Work left to right. The blind holdout is reserved BEFORE optimisation and is never shown to the development tabs; "
               "only a frozen candidate can be tested on it, and only the first test counts as blind.")
    ds = common.dataset()
    if ds is None:
        st.info("Start by loading data in the left sidebar (the free Yahoo SPY/QQQ source needs no account). "
                "The PHASE-0 RESULTS tab shows the finished free experiment.")
        _databento_panel()
        return
    icons = {"done": "✅", "todo": "⬜", "warn": "⚠️"}
    st.dataframe(pd.DataFrame([{"stage": a, "status": icons[b], "detail": c} for a, b, c in _status_rows(ds)]), hide_index=True,
                 width="stretch")
    _split_step(ds)
    if common.state().get("split"):
        _validate_step(ds)
        _freeze_step(ds)
        _stress_step(ds)
        _blind_step(ds)
    _databento_panel()


def _split_step(ds) -> None:
    st.subheader("SPLIT: reserve a blind holdout")
    split = common.state().get("split")
    tests = ps.read_blind_tests(ds.prep_full.data_hash)
    c = st.columns(3)
    tr = c[0].slider("Train share", 0.3, 0.8, 0.6, 0.05, key="ps_train")
    va = c[1].slider("Validation share", 0.1, 0.4, 0.2, 0.05, key="ps_val")
    try:
        proposal = ps.make_split(ds.prep_full.dates, tr, va)
        c[2].caption(f"Train {proposal.train_start}..{proposal.train_end} ({proposal.n_train}) · validation {proposal.val_start}.."
                     f"{proposal.val_end} ({proposal.n_val}) · blind holdout {proposal.holdout_start}..{proposal.holdout_end} ({proposal.n_holdout})")
    except ValueError as exc:
        st.error(str(exc))
        return
    if tests:
        st.warning("The holdout has already been used for a test. A new split is recorded, but any later holdout test is labelled NOT BLIND.")
    if st.button("Save split and withhold the holdout" if not split else "Replace split", key="ps_save"):
        saved = ps.save_split(ds.prep_full.data_hash, proposal)
        ds.prep = subset_before(ds.prep_full, saved["holdout_start"])
        common.state()["split"] = saved
        for k in ("result", "grid", "grid_stability", "robustness", "mc", "stress", "validated"):
            common.state().pop(k, None)
        st.rerun()
    if split:
        st.success(f"Development tabs now see {ds.prep.n_days} sessions (to {split['val_end']}). Blind holdout "
                   f"{split['holdout_start']}..{split['holdout_end']} ({split['n_holdout']} sessions) is withheld.")


def _validate_step(ds) -> None:
    st.subheader("VALIDATE: compare candidates on the validation period")
    split = common.state()["split"]
    st.caption("Candidates chosen on TRAIN are evaluated once on VALIDATION. Choosing the best validation result is itself a selection, "
               "so validation numbers are NOT evidence; they only filter candidates before the blind test.")
    s = common.state()
    grid = s.get("grid")
    options: dict[str, StrategyParams] = {}
    if grid is not None and not grid.results.empty:
        top = grid.results.sort_values("rank_composite").head(int(st.number_input("Top-N optimizer rows", 1, 100, 10, key="ps_topn")))
        for _, row in top.iterrows():
            p = params_from_row(row.to_dict())
            options[f"optimizer #{int(row['rank_composite'])} {p.key()}"] = p
    for name, rec in common.candidates().items():
        options.setdefault(name, StrategyParams(**rec["params"]))
    if not options:
        st.info("Run OPTIMIZATION (on the train period) or save a candidate first.")
        return
    execution = s.get("grid_execution") or ExecutionParams()
    if st.button("Evaluate on train and validation", key="ps_validate"):
        rows = []
        for name, p in options.items():
            tr, _ = _metrics(ds.prep, p, execution, None, split["train_start"], split["train_end"])
            va, _ = _metrics(ds.prep, p, execution, None, split["val_start"], split["val_end"])
            rows.append({"candidate": name, "describe": common.describe_params(p), "train trades": tr["n_trades"], "train exp R": tr["avg_r"],
                         "val trades": va["n_trades"], "val exp R": va["avg_r"], "val t": va["t_stat_r"], "val PF": va["profit_factor"]})
        s["validated"] = {"table": pd.DataFrame(rows), "options": {k: v.to_dict() for k, v in options.items()}}
    val = s.get("validated")
    if val:
        st.dataframe(val["table"].style.format({"train exp R": "{:+.3f}", "val exp R": "{:+.3f}", "val t": "{:.2f}", "val PF": "{:.2f}"}),
                     hide_index=True, width="stretch")
        name = st.selectbox("Candidate to carry forward", list(val["options"]), key="ps_val_pick")
        if st.button("Use as candidate", key="ps_val_use"):
            stored = common.add_candidate(name, StrategyParams(**val["options"][name]), "VALIDATE (chosen on validation)")
            st.success(f"Candidate {stored} saved.")


def _freeze_step(ds) -> None:
    st.subheader("FREEZE: lock the candidate before any out-of-sample test")
    pick = common.candidate_picker("Candidate to freeze", key="ps_freeze_cand")
    if pick is None:
        st.info("Choose a candidate in VALIDATE (or any tab) first.")
        return
    name, params = pick
    inst = common.instrument()
    with st.expander("Execution and sizing to freeze with it", expanded=False):
        execution = common.execution_form(inst, "frz")
        sizing = common.sizing_form(inst, "frz")
    if st.button("FREEZE (writes an immutable file with a SHA-256 hash)", key="ps_freeze", type="primary"):
        split = common.state()["split"]
        dev, _ = _metrics(ds.prep, params, execution, None, split["train_start"], split["val_end"])
        rec = ps.freeze_candidate(ds.prep_full.data_hash, name, params.to_dict(), execution.to_dict(), sizing.to_dict(), inst.to_dict(),
                                  evidence={"development_metrics_1unit": dev}, development_end=split["val_end"])
        st.success(f"Frozen {rec['name']} · SHA-256 {rec['sha256'][:16]}… · {rec['path']}")
    frozen = ps.list_frozen(ds.prep_full.data_hash)
    if frozen:
        st.dataframe(pd.DataFrame([{"name": f["name"], "sha256": f["sha256"][:16] + "…", "frozen (UTC)": f["frozen_utc"],
                                    "hash verified": f["verified"], "parameters": common.describe_params(StrategyParams(**f["strategy"]))}
                                   for f in frozen]), hide_index=True, width="stretch")


def _stress_step(ds) -> None:
    st.subheader("ROBUSTNESS + STRESS")
    st.caption("Use the ROBUSTNESS tab for parameter neighbourhoods. The execution stress test for the active candidate can be run here.")
    pick = common.candidate_picker("Candidate", key="ps_stress_cand")
    if pick and st.button("Run execution stress test on development data", key="ps_stress"):
        from ..optimization.stress import stress_test

        common.state()["stress"] = stress_test(ds.prep, pick[1], ExecutionParams())
        common.state()["stress_key"] = pick[1].key()
    if common.state().get("stress"):
        from .app_main import show_stress

        show_stress(*common.state()["stress"])


def _blind_step(ds) -> None:
    st.subheader("BLIND HOLDOUT: one-shot test of a frozen candidate")
    split = common.state()["split"]
    frozen = ps.list_frozen(ds.prep_full.data_hash)
    tests = ps.read_blind_tests(ds.prep_full.data_hash)
    if not frozen:
        st.info("Freeze a candidate first. Only frozen candidates can be tested on the holdout.")
        return
    if tests:
        st.warning(f"The holdout was already used {len(tests)} time(s) (first: {tests[0]['candidate']}, {tests[0]['status']}). "
                   "Any further test is labelled NOT BLIND.")
    names = {f"{f['name']} · {f['sha256'][:12]}": f for f in frozen}
    choice = st.selectbox("Frozen candidate", list(names), key="ps_blind_pick")
    rec = names[choice]
    ok = st.checkbox("I will not change the candidate after seeing the result.", key="ps_blind_ok")
    if st.button("RUN BLIND HOLDOUT TEST", type="primary", disabled=not ok, key="ps_blind_run"):
        if not ps.verify_frozen(rec):
            st.error("Frozen file failed hash verification; refusing to run.")
            return
        params = StrategyParams(**rec["strategy"])
        execution = ExecutionParams(**rec["execution"])
        sizing = SizingParams(**rec["sizing"])
        m, trades = _metrics(ds.prep_full, params, execution, sizing, split["holdout_start"], split["holdout_end"])
        entry = ps.record_blind_test(ds.prep_full.data_hash, rec, (split["holdout_start"], split["holdout_end"]), m)
        common.state()["blind"] = {**entry, "trades": trades}
    blind = common.state().get("blind")
    if blind:
        (st.success if blind["status"] == "BLIND" else st.warning)(f"{blind['status']} test of {blind['candidate']} "
                                                                  f"({blind['holdout_start']}..{blind['holdout_end']})")
        m = blind["metrics"]
        c = st.columns(5)
        c[0].metric("Trades", m["n_trades"])
        c[1].metric("Expectancy", f"{m['avg_r']:+.3f} R")
        c[2].metric("t-stat", f"{m['t_stat_r']:.2f}")
        c[3].metric("Profit factor", f"{m['profit_factor']:.2f}")
        c[4].metric("Net P&L", f"${m['net_pnl']:,.2f}")
        common.metrics_explained(m, "What do the holdout numbers mean?")
        st.dataframe(common.trades_frame(blind["trades"]), hide_index=True, width="stretch")


def _databento_panel() -> None:
    with st.expander("Real futures data (Databento): prepared, NOT active", expanded=False):
        st.markdown(
            "The futures workflow is built and tested with mocked downloads, but **no paid data has been requested and no key is "
            "configured**. When you decide to buy data:\n\n"
            "- Dataset **GLBX.MDP3** (CME Globex), schema **OHLCV-1m**, individual outright contracts of **ES, NQ, GC, 6E** "
            "(~2015-2026); micro specs MES/MNQ/MGC/M6E can reuse the same prices.\n"
            "- A causal front-month series is built (roll on the previous session's volume, never back-adjusted); sessions, DST, "
            "holidays and early closes use the exchange calendar; tick sizes/values and multipliers come from `config/instruments.yaml`.\n"
            "- The first data-quality report seals a 12-month **lockbox** that no tab can see until a one-shot final test.\n"
            "- Commands (terminal, because they involve a paid key): `python -m orb_lab.cli fetch --instrument ES --start 2015-06-01 "
            "--end 2026-09-01 --estimate-only` (free cost estimate), then the same without `--estimate-only` and `--budget`; then "
            "`dq-report`. After that the file loads here with **Data source = File path**, and every tab works unchanged.\n\n"
            "See DATA_SOURCES.md for costs and details.")


# ------------------------------------------------------------------------------------------ REPORT
def _wfo_summaries(data_hash: str) -> dict[str, dict]:
    out = {}
    for j in jobs.list_jobs(kind="wfo"):
        spec = json.loads((j / "spec.json").read_text())
        if spec.get("data", {}).get("data_hash") != data_hash or jobs.read_status(j).get("state") != "done":
            continue
        for f in (j / "result").glob("*/*/summary.json"):
            out[f"{j.name} · {f.parent.parent.name}/{f.parent.name}"] = json.loads(f.read_text())
    return out


def tab_report() -> None:
    ds = common.dataset()
    if ds is None:
        st.info("Load data first. The REPORT combines every pipeline stage for one candidate.")
        return
    inst = common.instrument()
    s = common.state()
    st.markdown("The report summarises one candidate across every stage. **Only out-of-sample results (the first blind holdout test and "
                "the stitched walk-forward OOS) count as evidence.** Verdicts: PROMISING · MIXED · NO PRELIMINARY EVIDENCE · "
                "INSUFFICIENT EVIDENCE.")
    frozen = ps.list_frozen(ds.prep_full.data_hash)
    cands = {f"frozen: {f['name']} · {f['sha256'][:12]}": ("frozen", f) for f in frozen}
    cands.update({f"candidate: {n}": ("cand", r) for n, r in common.candidates().items()})
    if not cands:
        st.info("No candidate yet. Save one with 'Use as candidate' (BACKTEST, OPTIMIZATION, WALK-FORWARD) or freeze one in PIPELINE.")
        return
    label = st.selectbox("Candidate", list(cands), key="rep_cand")
    kind, rec = cands[label]
    params = StrategyParams(**(rec["strategy"] if kind == "frozen" else rec["params"]))
    if kind == "cand":  # the same parameters may have been frozen: use that record (hash, costs, blind test)
        match = [f for f in frozen if StrategyParams(**f["strategy"]).key() == params.key()]
        if match:
            kind, rec = "frozen", match[-1]
            st.caption(f"This candidate was frozen as {rec['name']} ({rec['sha256'][:12]}…); using the frozen record.")
    execution = ExecutionParams(**rec["execution"]) if kind == "frozen" else ExecutionParams()
    sizing = SizingParams(**rec["sizing"]) if kind == "frozen" else default_sizing(inst)
    st.caption(f"Stage metrics use {'the frozen sizing' if kind == 'frozen' else f'a fixed {sizing.contracts:g} {inst.unit}(s)'}; "
               "expectancy in R does not depend on size.")
    wfos = _wfo_summaries(ds.prep_full.data_hash)
    wfo_pick = st.selectbox("Walk-forward result to include", ["(none)"] + list(wfos), index=1 if wfos else 0, key="rep_wfo")
    split = s.get("split")
    stages: dict = {"candidate": {"name": rec.get("name", label), "sha256": rec.get("sha256", "not frozen"), "params": params.to_dict(),
                                  "costs": inst.cost_description()},
                    "data": {"symbol": inst.symbol, "unit": inst.unit, "proxy": common.is_proxy(inst), "synthetic": common.is_synthetic(ds),
                             "sessions": ds.prep_full.n_days, "first": str(ds.prep_full.dates[0]), "last": str(ds.prep_full.dates[-1]),
                             "source": ds.loaded.source, "hash": ds.prep_full.data_hash}}
    if split:
        stages["split"] = split
        stages["train"], _ = _metrics(ds.prep, params, execution, sizing, split["train_start"], split["train_end"])
        stages["validation"], _ = _metrics(ds.prep, params, execution, sizing, split["val_start"], split["val_end"])
        stages["concentration"], _ = _metrics(ds.prep, params, execution, sizing, split["train_start"], split["val_end"])
        tests = [t for t in ps.read_blind_tests(ds.prep_full.data_hash) if t["sha256"] == rec.get("sha256")]
        if tests:
            stages["blind"] = tests[0]
    else:
        stages["train"], _ = _metrics(ds.prep, params, execution, sizing)
        stages["concentration"] = stages["train"]
    if wfo_pick != "(none)":
        stages["wfo"] = wfos[wfo_pick]
    rob = s.get("robustness")
    if rob and common.candidates().get(rob["candidate"], {}).get("key") == params.key():
        stages["robustness"] = {"verdict": rob["verdict"], "lines": [f"{r.label}: {r.classification} (candidate {r.candidate_exp_r:+.3f} R, "
                                                                    f"neighbours median {r.neighbour_median_exp_r:+.3f} R)"
                                                                    for r in rob["summary"].itertuples()]}
    if s.get("stress") and s.get("stress_key") == params.key():
        stages["stress"] = s["stress"][1]
    if s.get("mc"):
        stages["mc"] = s["mc"]["result"].to_dict()
    if s.get("grid") is not None:
        stages["grid"] = s["grid"].selection_bias
    if common.is_proxy(inst):
        stages["limitations"] = ["ETF proxy data from Yahoo (~60 days of 5-minute bars): far too short for conclusions about futures."]
    verdict, reasons = classify(stages)
    (st.success if verdict == "PROMISING" else st.warning if verdict == "MIXED" else st.error)(f"Verdict: **{verdict}**")
    md = build_markdown(stages)
    st.markdown(md)
    c = st.columns(2)
    c[0].download_button("Download report (Markdown)", md.encode(), file_name=f"orb_report_{params.key()}.md")
    if c[1].button("Save report to runs/pipeline/"):
        path = ps.data_dir(ds.prep_full.data_hash) / f"report_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}_{params.key()}.md"
        path.write_text(md)
        st.success(f"Saved {path}")
    with st.expander("Data provenance"):
        st.json(ds.describe(), expanded=False)

"""SETTINGS & RESEARCH: market specifications, frozen strategies and hashes, blind-test ledger, research registry,
software diagnostics, the Phase-0 archive and the test suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from ... import __version__
from ...engine.params import StrategyParams
from ...research import diagnostics as dg
from ...research import pipeline_state as ps
from .. import state as S
from .. import theme as T

ROOT = Path(__file__).resolve().parents[3]


def render() -> None:
    T.page_header("Admin", "Settings & research", "Specifications, research records and software checks. You rarely need this page.")
    tabs = st.tabs(["Market specifications", "Frozen strategies & holdout", "Research registry", "Diagnostics", "Phase-0 archive", "About"])
    with tabs[0]:
        _markets()
    with tabs[1]:
        _frozen()
    with tabs[2]:
        _registry()
    with tabs[3]:
        _diagnostics()
    with tabs[4]:
        from ..phase0_tab import tab_phase0

        tab_phase0()
    with tabs[5]:
        st.markdown(f"**ORB Lab v{__version__}** — research platform, not a signal service.")
        st.caption(f"Project folder: {ROOT}")


def _markets() -> None:
    st.markdown("Default costs per market. Every $ figure is **per contract (futures) or per share (ETF), per side**. Adjust them for a "
                "specific strategy on the Strategy page; edit `config/instruments.yaml` to change the defaults.")
    rows = []
    for sym, i in S.all_instruments().items():
        rows.append({"Market": sym, "Name": i.name, "Type": "ETF (shares)" if i.asset_class == "etf" else "Futures (contracts)",
                     "Tick size": i.tick_size, "Tick value $": i.tick_value, "Multiplier": i.multiplier,
                     "Commission $/unit/side": i.commission_per_side, "Fees $/unit/side": i.exchange_fee_per_side,
                     "Friction $/unit/side": i.friction_per_side, "Slippage ticks/side": i.slippage_ticks,
                     "Max value ÷ equity": f"{i.max_notional_leverage:g}×" if i.max_notional_leverage else "no cap"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _frozen() -> None:
    ds = S.dataset()
    if ds is None:
        st.info("Load data to see its frozen strategies and holdout ledger.")
        return
    split = ps.load_split(ds.prep_full.data_hash)
    if split:
        st.markdown(f"**Split** · train {split['train_start']} – {split['train_end']} · validation {split['val_start']} – {split['val_end']} · "
                    f"holdout {split['holdout_start']} – {split['holdout_end']}")
    frozen = ps.list_frozen(ds.prep_full.data_hash)
    st.markdown("**Frozen strategies**")
    if frozen:
        st.dataframe(pd.DataFrame([{"Name": f["name"], "SHA-256": f["sha256"], "Frozen (UTC)": f["frozen_utc"], "Hash verified": f["verified"],
                                    "Strategy": S.params_text(StrategyParams(**f["strategy"])),
                                    "File": f["path"]} for f in frozen]), hide_index=True, width="stretch")
    else:
        st.caption("None yet.")
    st.markdown("**Blind holdout ledger** (the first test is BLIND; every later one is NOT BLIND)")
    tests = ps.read_blind_tests(ds.prep_full.data_hash)
    if tests:
        st.dataframe(pd.DataFrame([{"UTC": t["utc"], "Status": t["status"], "Strategy": t["candidate"], "SHA-256": t["sha256"][:16],
                                    "Trades": t["metrics"].get("n_trades"), "Expectancy R": t["metrics"].get("avg_r")} for t in tests]),
                     hide_index=True, width="stretch")
    else:
        st.caption("No holdout tests yet.")
    st.caption(f"Stored in {ps.data_dir(ds.prep_full.data_hash)}")


def _registry() -> None:
    from ...research.registry import hypotheses, read_events

    st.markdown("Pre-registered hypotheses and experiments (append-only). Research-mode walk-forward runs are logged here automatically.")
    hyps = hypotheses()
    if hyps:
        st.dataframe(pd.DataFrame([{"ID": k, "Status": h.get("status"), "Statement": h.get("statement"), "Data seen": h.get("data_seen")}
                                   for k, h in hyps.items()]), hide_index=True, width="stretch",
                     column_config={"Statement": st.column_config.TextColumn(width="large")})
    else:
        st.caption("No hypotheses registered.")
    events = read_events()
    with st.expander(f"All {len(events)} registry events"):
        st.json(events, expanded=False)


def _show(res: dict) -> None:
    ok = res["passed"]
    st.markdown(f"{'✅ PASS' if ok else '❌ FAIL'} · **{res['check']}**")
    st.caption("Expected: " + res["expected"])
    if len(res.get("table", [])):
        st.dataframe(res["table"], hide_index=True, width="stretch")


def _diagnostics() -> None:
    st.error("Software checks with known answers. Nothing here is evidence about the strategy, and none of it feeds other pages.")
    out = S.S().setdefault("diagnostics", {})
    c = st.columns(4)
    years = c[0].selectbox("Synthetic years", [1, 2, 4], index=1)
    if c[1].button("Null control (random walk)"):
        with st.spinner("Random-walk data..."):
            out["null"] = dg.null_control(years=years)
    if c[2].button("Planted-edge check"):
        with st.spinner("Trend-planted data..."):
            out["edge"] = dg.planted_edge(years=years)
    ds = S.dataset()
    if ds is not None:
        p = S.strategy()["params"]
        c = st.columns(3)
        if c[0].button("Lookahead truncation test"):
            out["lookahead"] = dg.lookahead_truncation(ds.prep, p)
        if c[1].button("Random-direction control"):
            with st.spinner("Simulating both directions of every trade..."):
                out["random"] = dg.random_direction(ds.prep, p)
        if c[2].button("Cost-unit check"):
            out["costs"] = dg.cost_units(S.instrument())
    for k in ("null", "edge", "lookahead", "random", "costs"):
        if k in out:
            _show(out[k])
    st.markdown("**Automated test suite** (about 3–6 minutes)")
    if st.button("Run all tests"):
        with st.spinner("pytest running..."):
            proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True,
                                  timeout=3600)
        out["pytest"] = (proc.returncode, proc.stdout[-4000:] + proc.stderr[-2000:])
    if "pytest" in out:
        code, text = out["pytest"]
        (st.success if code == 0 else st.error)("All tests passed." if code == 0 else f"Tests failed (exit code {code}).")
        st.code(text)

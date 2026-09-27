"""DIAGNOSTICS tab: software sanity checks with known answers. Never mixed with real results."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

from ..engine.params import StrategyParams
from ..research import diagnostics as dg
from . import common


def _show(res: dict) -> None:
    icon = "✅ PASS" if res["passed"] else "❌ FAIL"
    st.markdown(f"**{icon} · {res['check']}**")
    st.caption("Expected: " + res["expected"])
    if len(res.get("table", [])):
        st.dataframe(res["table"], hide_index=True, width="stretch")


def tab_diagnostics() -> None:
    st.error("**DIAGNOSTICS: synthetic and controlled tests of the SOFTWARE.** Nothing on this page is evidence about ORB, and none of "
             "these numbers feed into any other tab or report.")
    out = common.state().setdefault("diagnostics", {})
    st.subheader("Engine checks on synthetic data (known answers)")
    c = st.columns(3)
    years = c[0].selectbox("Synthetic years", [1, 2, 4], index=1, help="More years = stronger checks, slower (≈ 10-40 s).")
    if c[1].button("Run null control"):
        with st.spinner("Random-walk data..."):
            out["null"] = dg.null_control(years=years)
    if c[2].button("Run planted-edge check"):
        with st.spinner("Trend-planted data..."):
            out["edge"] = dg.planted_edge(years=years)
    for k in ("null", "edge"):
        if k in out:
            _show(out[k])
    st.subheader("Checks on the loaded data")
    ds = common.dataset()
    inst = common.instrument()
    if ds is None:
        st.info("Load data to run the lookahead, random-direction and cost-unit checks.")
    else:
        pick = common.candidate_picker("Configuration to check (defaults to the reference if there is no candidate)", key="dg_cand")
        params = pick[1] if pick else StrategyParams(range_minutes=15 if ds.prep.base_minutes <= 5 else 30,
                                                     entry_tf=max(5, ds.prep.base_minutes))
        c = st.columns(3)
        if c[0].button("Lookahead truncation test"):
            out["lookahead"] = dg.lookahead_truncation(ds.prep, params)
        if c[1].button("Random-direction control"):
            with st.spinner("Simulating both directions of every trade..."):
                out["random"] = dg.random_direction(ds.prep, params)
        if c[2].button("Cost-unit check"):
            out["costs"] = dg.cost_units(inst)
        for k in ("lookahead", "random", "costs"):
            if k in out:
                _show(out[k])
    st.subheader("Automated test suite")
    st.caption("Runs the project's full automated tests (about 2-5 minutes). Includes mutation-checked lookahead and walk-forward leakage tests.")
    if st.button("Run all tests"):
        root = Path(__file__).resolve().parents[2]
        with st.spinner("pytest running..."):
            proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=root, capture_output=True, text=True,
                                  timeout=3600)
        out["pytest"] = (proc.returncode, proc.stdout[-4000:] + proc.stderr[-2000:])
    if "pytest" in out:
        code, text = out["pytest"]
        (st.success if code == 0 else st.error)("All tests passed." if code == 0 else f"Tests failed (exit code {code}).")
        st.code(text)

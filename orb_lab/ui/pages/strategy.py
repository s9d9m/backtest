"""STRATEGY: define the current strategy (grouped, plain-language controls; rarely changed ones under Advanced)."""

from __future__ import annotations

import streamlit as st

from ...engine.params import StrategyParams
from ...research import pipeline_state as ps
from .. import state as S
from .. import theme as T
from .. import widgets as W


def render() -> None:
    T.page_header("Build", "Strategy", "Define the opening-range breakout, how trades are sized and what execution costs to assume. "
                  "Every other page uses the strategy saved here.")
    W.strategy_header(show_edit=False)
    note = S.S().pop("strategy_changed_note", None)
    if note:
        st.info(note)
    if S.dataset() is None:
        T.note("No data is loaded, so entry timeframes are not yet limited to the bar size of your data. Load data on the Data page.", "warn")
    W.strategy_editor()

    T.section("Saved strategies")
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    options = {}
    for name, rec in S.candidates().items():
        options[f"{name} · {rec['source']}"] = StrategyParams(**rec["params"])
    ds = S.dataset()
    if ds is not None:
        for f in ps.list_frozen(ds.prep_full.data_hash):
            options[f"Frozen: {f['name']} · {f['sha256'][:10]}"] = StrategyParams(**f["strategy"])
    if options:
        labels = {n: f"{n} — {S.params_text(p)}" for n, p in options.items()}
        pick = c1.selectbox("Load a saved or frozen strategy", list(options), key="sp_load", format_func=labels.get)
        c2.button("Load", key="sp_load_btn", on_click=S.use_strategy, args=(options[pick], f"saved strategy '{pick}'"), width="stretch")
    else:
        c1.caption("Strategies you keep from the optimizer or freeze for validation appear here.")
    c1, c2, c3 = st.columns([1.2, 1.2, 2])
    name = c1.text_input("Save the current strategy as", value="", placeholder="e.g. QQQ 15m midpoint", key="sp_name")
    if c2.button("Save to list", key="sp_save", disabled=ds is None or not name.strip()):
        stored = S.save_candidate(name.strip(), S.strategy()["params"], "saved on the Strategy page")
        st.success(f"Saved as '{stored}'.")
    with c3:
        st.button("Reset to the reference strategy", key="sp_reset", on_click=_reset)
    with st.expander("All current settings", icon=":material/list:"):
        st.dataframe(W.all_settings_table(), hide_index=True, width="stretch")


def _reset() -> None:
    d = S.default_strategy()
    S.set_strategy(d["params"], d["execution"], d["sizing"], d["source"], d["name"])

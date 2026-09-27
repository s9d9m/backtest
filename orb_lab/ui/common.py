"""Widgets and state shared by the dashboard tabs."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ..engine.params import AMBIGUITY_MODES, FILL_MODELS, ExecutionParams, SizingParams, StrategyParams
from ..reports import plots
from ..reports.metric_help import HEADLINE, cautions, metric_table

RISK_PRESETS = {"0.25 %": 0.0025, "0.5 %": 0.005, "1 %": 0.01, "2 %": 0.02, "custom": None}
SIZING_LABELS = {
    "pct_equity": "Risk % of current equity (compounding)",
    "fixed_risk": "Risk % of starting equity (fixed $ risk, no compounding)",
    "fixed_contracts": "Fixed quantity",
}


def state():
    return st.session_state


def dataset():
    return state().get("dataset")


def instrument():
    return state().get("instrument")


def unit_of(inst) -> str:
    return getattr(inst, "unit", "contract")


def is_proxy(inst) -> bool:
    return inst is not None and getattr(inst, "asset_class", "future") == "etf"


def is_synthetic(ds) -> bool:
    return ds is not None and str(getattr(ds.loaded, "source", "")).startswith("synthetic")


# ------------------------------------------------------------------------------------------ costs
def execution_form(inst, prefix: str = "bt") -> ExecutionParams:
    """Cost and fill inputs. Every $ amount is per unit (contract or share) per side; units always shown."""
    u = unit_of(inst)
    st.caption(f"All \\$ costs below are **per {u}, per side** (charged on every fill: entry and exit). "
               f"1 tick = {inst.tick_size:g} in price = \\${inst.tick_value:g} per {u}.")
    c = st.columns(4)
    slip = c[0].number_input("Slippage (ticks per side)", 0.0, 20.0, float(inst.slippage_ticks), 0.5, key=f"{prefix}_slip",
                             help="Ticks lost on every market or stop fill (entry and stop-loss exits). Limit orders and targets get "
                                  "no slippage but must trade through the price by one tick under the conservative fill model.")
    comm = c[1].number_input(f"Commission ($ per {u} per side)", 0.0, 50.0, float(inst.commission_per_side), 0.01, format="%.4f",
                             key=f"{prefix}_comm", help=f"Broker commission for ONE {u} on ONE fill. Not per order.")
    fee = c[2].number_input(f"Exchange + regulatory fees ($ per {u} per side)", 0.0, 50.0, float(inst.exchange_fee_per_side), 0.01,
                            format="%.4f", key=f"{prefix}_fee")
    fric = c[3].number_input(f"Modelled bid/ask friction ($ per {u} per side)", 0.0, 50.0, float(getattr(inst, "friction_per_side", 0.0)),
                             0.01, format="%.4f", key=f"{prefix}_fric",
                             help="Extra adverse price on every fill (any order type), e.g. half the bid/ask spread. "
                                  "For ETFs the default $0.02/share is modelled here.")
    c = st.columns(4)
    fill = c[0].selectbox("Limit fill model", FILL_MODELS, index=FILL_MODELS.index("conservative"), key=f"{prefix}_fill",
                          help="conservative: a limit/target fills only if price trades THROUGH it by one tick; standard: a touch fills.")
    amb = c[1].selectbox("Same-bar ambiguity", AMBIGUITY_MODES, index=AMBIGUITY_MODES.index("conservative"), key=f"{prefix}_amb",
                         help="When stop and target are both inside one bar: optimistic = target first, pessimistic = stop first, "
                              "conservative = stop first and no target on the entry bar.")
    fixed_rt = 2 * (comm + fee + fric)
    slip_rt = 2 * slip * inst.tick_value
    c[2].metric(f"Fixed costs per round trip (1 {u})", f"${fixed_rt:,.4g}")
    c[3].metric(f"+ slippage if both fills slip (1 {u})", f"${slip_rt:,.4g}",
                help="Upper bound: a trade entered with a market/stop order and stopped out pays slippage twice; target exits pay none.")
    return ExecutionParams(slippage_ticks=slip, commission_per_side=comm, exchange_fee_per_side=fee, friction_per_side=fric,
                           fill_model=fill, ambiguity=amb)


# ------------------------------------------------------------------------------------------ sizing
def sizing_form(inst, prefix: str = "bt") -> SizingParams:
    u = unit_of(inst)
    etf = is_proxy(inst)
    c = st.columns(4)
    mode = c[0].selectbox("Position sizing", list(SIZING_LABELS), format_func=SIZING_LABELS.get, key=f"{prefix}_mode",
                          help="Risk-based sizing: quantity = floor(risk budget / (entry-to-stop distance x $ per point)). "
                               f"Always whole {u}s.")
    equity = c[1].number_input("Starting equity ($)", 1000.0, 1e9, 40_000.0, 1000.0, key=f"{prefix}_equity")
    preset = c[2].selectbox("Risk per trade", list(RISK_PRESETS), index=2, key=f"{prefix}_riskp", disabled=mode == "fixed_contracts")
    risk_pct = RISK_PRESETS[preset]
    if risk_pct is None:
        risk_pct = c[3].number_input("Custom risk % per trade", 0.01, 10.0, 0.75, 0.05, key=f"{prefix}_riskc",
                                     disabled=mode == "fixed_contracts") / 100
        qty = 100 if etf else 1
    else:
        qty = c[3].number_input(f"Fixed quantity ({u}s)", 1, 1_000_000, 100 if etf else 1, key=f"{prefix}_qty",
                                disabled=mode != "fixed_contracts")
    c = st.columns(4)
    max_units = c[0].number_input(f"Max {u}s per trade", 1, 10_000_000, 100_000 if etf else 100, key=f"{prefix}_max")
    lev_default = float(getattr(inst, "max_notional_leverage", 0.0))
    lev = c[1].number_input("Max notional / equity (0 = no cap)", 0.0, 100.0, lev_default, 0.5, key=f"{prefix}_lev",
                            help="Caps position value (price x multiplier x quantity) at this multiple of equity, e.g. 4 for "
                                 "intraday stock buying power. Trades reduced by the cap are flagged in the trade log.")
    if mode != "fixed_contracts":
        c[2].metric("Risk budget on first trade", f"${equity * risk_pct:,.0f}")
    c[3].caption(f"Sizing unit: **{u}s** ({'ETF: shares' if etf else 'futures: whole contracts'}). "
                 f"{'Tip: micro contracts (MES, MNQ, MGC, M6E) risk 1/10 of the full contract.' if not etf else ''}")
    return SizingParams(mode=mode, contracts=float(qty), risk_dollars=equity * risk_pct, risk_pct=risk_pct, starting_equity=equity,
                        max_contracts=float(max_units), max_leverage=float(lev))


# ------------------------------------------------------------------------------------------ candidates
def _candidate_file():
    ds = dataset()
    if ds is None:
        return None
    from ..research.pipeline_state import data_dir

    return data_dir(ds.prep.data_hash) / "candidates.json"


def candidates() -> dict:
    if "candidates" not in state():
        f = _candidate_file()
        state()["candidates"] = json.loads(f.read_text()) if f and f.exists() else {}
    return state()["candidates"]


def add_candidate(name: str, params: StrategyParams, source: str, extra: dict | None = None) -> str:
    store = candidates()
    base, k = name, 2
    while name in store and store[name]["params"] != params.to_dict():
        name = f"{base} ({k})"
        k += 1
    store[name] = {"params": params.to_dict(), "source": source, "key": params.key(),
                   "added_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(extra or {})}
    state()["active_candidate"] = name
    f = _candidate_file()
    if f is not None:
        f.write_text(json.dumps(store, indent=2, default=str))
    return name


def candidate_picker(label: str = "Candidate", key: str = "cand_pick") -> tuple[str, StrategyParams] | None:
    store = candidates()
    if not store:
        return None
    names = list(store)
    active = state().get("active_candidate")
    idx = names.index(active) if active in names else len(names) - 1
    name = st.selectbox(label, names, index=idx, key=key,
                        format_func=lambda n: f"{n} · {store[n]['source']} · {describe_params(StrategyParams(**store[n]['params']))}")
    state()["active_candidate"] = name
    return name, StrategyParams(**store[name]["params"])


def describe_params(p: StrategyParams) -> str:
    stop = p.stop_method if p.stop_method in ("or_opposite", "or_mid") else f"{p.stop_method} {p.stop_param:g}"
    tf = "intrabar" if p.entry_method == "stop" else f"{p.entry_tf}m"
    return (f"{p.orb_start}+{p.range_minutes}m, {p.entry_method} ({tf}), stop {stop}, {p.target_r:g}R, cutoff {p.cutoff}, {p.direction}")


def send_to_buttons(name: str, params: StrategyParams, source: str, key: str) -> None:
    """'Use as candidate' button: stores the configuration for the other tabs."""
    if st.button(f"Use as candidate → Robustness / Monte Carlo / Walk-forward / Pipeline", key=key):
        stored = add_candidate(name, params, source)
        st.success(f"Saved as candidate **{stored}**. Open ROBUSTNESS, MONTE CARLO or PIPELINE and it is preselected.")


# ------------------------------------------------------------------------------------------ metrics
def metrics_explained(m: dict, title: str = "What do these numbers mean?") -> None:
    for w in cautions(m):
        st.warning(w)
    with st.expander(title, expanded=False):
        st.dataframe(metric_table(m, HEADLINE), hide_index=True, width="stretch")


def equity_risk_figure(res) -> go.Figure:
    """Equity, drawdown, cumulative R, trade-by-trade P&L and position size/risk per trade."""
    t = res.trades
    t = t[~t["sized_out"]] if len(t) else t
    fig = make_subplots(rows=5, cols=1, shared_xaxes=False, vertical_spacing=0.05, row_heights=[0.26, 0.14, 0.2, 0.2, 0.2],
                        subplot_titles=("Equity (net, $)", "Drawdown", "Cumulative R (trade by trade)", "Net P&L per trade ($)",
                                        f"Position size ({res.diagnostics.get('unit', 'unit')}s, bars) and $ at risk (line)"))
    d = res.daily
    fig.add_trace(go.Scatter(x=d.index, y=d["net_equity"], name="equity", line=dict(color=plots.NET)), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d["net_drawdown"], name="drawdown", fill="tozeroy", line=dict(color="#d62728")), 2, 1)
    if len(t):
        n = np.arange(1, len(t) + 1)
        fig.add_trace(go.Scatter(x=n, y=t["r_multiple"].cumsum(), name="cum R", line=dict(color="#2ca02c")), 3, 1)
        fig.add_trace(go.Bar(x=n, y=t["net_pnl"], name="P&L", marker_color=np.where(t["net_pnl"] >= 0, "#2a6fdb", "#d62728")), 4, 1)
        fig.add_trace(go.Bar(x=n, y=t["qty"], name="quantity", marker_color="#9467bd"), 5, 1)
        fig.add_trace(go.Scatter(x=n, y=t["risk_dollars"], name="$ at risk", line=dict(color="#ff7f0e")), 5, 1)
    fig.update_xaxes(title_text="trade #", row=5, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_layout(height=1100, showlegend=False, **{k: v for k, v in plots.LAYOUT.items() if k not in ("hovermode",)})
    return fig


def dev_notice() -> None:
    """Explain which sessions the development tools can see."""
    ds = dataset()
    if ds is None:
        return
    split = state().get("split")
    if split:
        st.caption(f"Development data only: sessions up to {split['val_end']}. The blind holdout ({split['holdout_start']}..{split['holdout_end']}) "
                   "is withheld from this tab (see PIPELINE).")


def trades_frame(trades: pd.DataFrame) -> pd.DataFrame:
    cols = ["trade_id", "session_date", "weekday", "direction", "signal_time", "entry_time", "entry_price", "stop_price", "target_price",
            "exit_time", "exit_price", "exit_reason", "qty", "risk_ticks", "risk_per_contract", "risk_dollars", "notional", "size_capped",
            "gross_pnl", "commission", "fees", "friction", "slippage_cost", "total_cost", "net_pnl", "r_multiple", "mfe_r", "mae_r",
            "holding_minutes", "or_high", "or_low", "or_width", "atr_daily", "or_atr", "ambiguous_exit", "intrabar_entry",
            "breakeven_moved", "sized_out", "equity_before", "equity_after"]
    return trades[[c for c in cols if c in trades.columns]]

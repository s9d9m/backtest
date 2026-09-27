"""Reusable dashboard widgets: current-strategy header, strategy editor, metric explanations, empty states."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from ..engine.execution import resolve_costs
from ..engine.params import AMBIGUITY_MODES, FILL_MODELS, REENTRY_POLICIES, ExecutionParams, SizingParams, StrategyParams
from ..optimization.parameter_space import CONFIRMATION_PRESETS, parse_stop, stop_label
from ..reports.metric_help import HEADLINE, cautions, metric_table
from . import state as S
from . import theme as T

ORB_STARTS = ["09:00", "09:15", "09:30", "09:45", "10:00"]
RANGES = [5, 10, 15, 20, 30, 45, 60]
ENTRY_TFS = [1, 2, 3, 5, 10, 15]
TARGETS = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
CUTOFFS = ["10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "14:00", "15:00"]
TIME_EXITS = ["session", "10:30", "11:00", "11:30", "12:00", "13:00", "14:00", "15:00", "15:45", "15:55"]
STOPS = ["or_opposite", "or_mid", "or_pct:0.25", "or_pct:0.33", "or_pct:0.5", "or_pct:0.67", "or_pct:0.75", "or_pct:1",
         "atr:0.5", "atr:0.75", "atr:1", "atr:1.25", "atr:1.5", "atr:2", "fixed_ticks"]
CONF_LABELS = {"close": "Close beyond the range", "1tick": "Close 1 tick beyond", "2tick": "Close 2 ticks beyond",
               "5pct": "Close 5% of range width beyond", "10pct": "Close 10% of range width beyond"}
RISK_CHOICES = [0.25, 0.5, 1.0, 2.0]


def stop_choice_label(c: str) -> str:
    if c == "or_opposite":
        return "Opposite side of the range"
    if c == "or_mid":
        return "Range midpoint"
    if c == "fixed_ticks":
        return "Fixed number of ticks"
    m, p = parse_stop(c)
    return f"{p:.0%} of range width" if m == "or_pct" else f"{p:g}× daily ATR"


# ---------------------------------------------------------------------------------------------- header
def strategy_header(show_edit: bool = True, compact: bool = False) -> None:
    """Compact card: which strategy, market, account and execution assumptions are being analysed."""
    strat = S.strategy()
    p = strat["params"]
    inst = S.instrument()
    chips = T.chips(S.strategy_chips(p), first_is_market=True)
    conf = S.confirmation_text(p)
    port = html.escape(S.portfolio_text())
    exe = html.escape(S.execution_text())
    left, right = st.columns([6, 1], vertical_alignment="center")
    with left:
        st.markdown(f'<div class="orb-card" style="margin-bottom:6px"><div class="ttl">Current strategy · {html.escape(strat.get("name", ""))}</div>'
                    f'<div class="body">{chips}<div style="margin-top:6px;font-size:0.86rem;color:{T.INK_2}">'
                    f'<b>Account:</b> {port} &nbsp;·&nbsp; <b>Execution:</b> {exe} &nbsp;·&nbsp; <b>Signal:</b> {html.escape(conf)}'
                    f'{"" if inst is not None else " &nbsp;·&nbsp; <b>No market data loaded</b>"}</div></div></div>', unsafe_allow_html=True)
    with right:
        if show_edit:
            S.nav_button("Edit strategy", "strategy", key=f"edit_{st.session_state.get('nav', 'x')}", icon=":material/edit:",
                         width="stretch")
    note = S.S().pop("flash", None)
    if note:
        st.toast(note)


def need_data() -> bool:
    """Empty state when no data is loaded. Returns True if the page should stop."""
    if S.dataset() is not None:
        return False
    T.note("No market data is loaded yet. Research pages need price data to run.", "warn")
    c1, c2, _ = st.columns([1.3, 1, 2])
    if c1.button("Load free SPY data (no account)", type="primary", key=f"quick_load_{st.session_state.get('nav', '')}"):
        with st.spinner("Loading free Yahoo SPY 5-minute bars..."):
            try:
                msg = S.load_dataset(S.YAHOO, S.etf_instruments()["SPY"], symbol="SPY")
                st.success(f"Loaded SPY: {msg}.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not load data: {exc}")
    S.nav_button("Open the Data page", "data", key=f"to_data_{st.session_state.get('nav', '')}")
    return True


# ---------------------------------------------------------------------------------------------- metric help
def metrics_explained(m: dict, title: str = "What do these numbers mean?") -> None:
    ws = cautions(m)
    if ws:
        st.warning("**Read these numbers with care**\n" + "\n".join(f"- {w}" for w in ws), icon=":material/warning:")
    with st.expander(title, expanded=False, icon=":material/help:"):
        st.dataframe(metric_table(m, HEADLINE), hide_index=True, width="stretch")


def headline_cards(m: dict, start_equity: float, unit: str = "") -> None:
    end = m.get("end_equity", start_equity + m.get("net_pnl", 0))
    ret = m.get("total_return", 0.0)
    exp = m.get("avg_r", 0.0)
    n = int(m.get("n_trades", 0))
    few = n < 30
    T.kpi_grid([
        T.kpi("Starting balance", T.usd(start_equity), "account at the start"),
        T.kpi("Ending balance", T.usd(end), f"{T.pct(ret, signed=True)} net return", tone="good" if ret > 0 else "bad" if ret < 0 else "none",
              help_term="Net return"),
        T.kpi("Total P&L", T.usd(m.get("net_pnl"), signed=True), f"after {T.usd(m.get('total_cost'))} costs", tone="good" if m.get("net_pnl", 0) > 0 else "bad"),
        T.kpi("Trades", T.count(n), "small sample" if few else f"{m.get('n_long', 0)} long · {m.get('n_short', 0)} short",
              tone="warn" if few else "none"),
        T.kpi("Win rate", T.pct(m.get("win_rate"), 0), f"payoff {T.ratio(m.get('payoff_ratio'))}"),
        T.kpi("Expectancy", T.rmult(exp), f"t-stat {T.ratio(m.get('t_stat_r'))}", tone="good" if exp > 0 else "bad"),
        T.kpi("Profit factor", T.ratio(m.get("profit_factor")), "gross win ÷ gross loss", tone="good" if (m.get("profit_factor") or 0) > 1 else "bad"),
        T.kpi("Max drawdown", T.pct(m.get("max_dd")), "largest peak-to-trough fall", tone="bad" if (m.get("max_dd") or 0) < -0.2 else "none"),
    ], min_width=150)


# ---------------------------------------------------------------------------------------------- strategy editor
def strategy_editor() -> None:
    """The Strategy page form. Saving replaces the current strategy."""
    strat = S.strategy()
    p: StrategyParams = strat["params"]
    ex: ExecutionParams = strat["execution"]
    sz: SizingParams = strat["sizing"]
    inst = S.instrument()
    ds = S.dataset()
    base = ds.prep.base_minutes if ds is not None else 1
    unit = inst.unit if inst is not None else "contract"
    v = st.session_state.get("strat_ver", 0)
    k = lambda name: f"sf_{name}_{v}"  # noqa: E731
    ranges = [r for r in RANGES if r % base == 0]
    tfs = [t for t in ENTRY_TFS if t % base == 0]

    def idx(options, value, default=0):
        return options.index(value) if value in options else default

    with st.form("strategy_form", border=False):
        left, right = st.columns(2, gap="large")
        with left:
            with st.container(border=True):
                st.markdown("**Opening range**")
                c = st.columns(2)
                orb_start = c[0].selectbox("Range starts at (ET)", ORB_STARTS, index=idx(ORB_STARTS, p.orb_start, 2), key=k("start"),
                                           help="The opening range covers the first minutes after this time.")
                rng = c[1].selectbox("Range length", ranges, index=idx(ranges, p.range_minutes), key=k("range"),
                                     format_func=lambda x: f"{x} minutes", help=f"Only multiples of the loaded {base}-minute bars.")
            with st.container(border=True):
                st.markdown("**Entry**")
                c = st.columns(2)
                em = c[0].selectbox("Entry type", list(S.ENTRY_LABELS), index=idx(list(S.ENTRY_LABELS), p.entry_method), key=k("em"),
                                    format_func=S.ENTRY_LABELS.get,
                                    help="Market: a bar closes beyond the range, enter at the next bar's open. Limit: after the close, "
                                         "wait for price to come back to the range edge. Stop: enter the moment price trades through the range.")
                tf = c[1].selectbox("Entry timeframe (signal bars)", tfs, index=idx(tfs, p.entry_tf), key=k("tf"),
                                    format_func=lambda x: f"{x}-minute bars", disabled=em == "stop",
                                    help="Bar size whose close confirms the breakout. Not used by stop entries.")
                conf_now = next((n for n, (t, f) in CONFIRMATION_PRESETS.items() if t == p.confirm_ticks and f == p.confirm_or_frac), "close")
                conf = st.selectbox("Breakout confirmation", list(CONFIRMATION_PRESETS), index=idx(list(CONFIRMATION_PRESETS), conf_now),
                                    key=k("conf"), format_func=lambda c_: CONF_LABELS.get(c_, c_),
                                    help="How far beyond the range the signal bar must close (or price must trade, for stop entries).")
            with st.container(border=True):
                st.markdown("**Stop and target**")
                c = st.columns(2)
                cur_stop = "fixed_ticks" if p.stop_method == "fixed_ticks" else stop_label(p.stop_method, p.stop_param)
                stops = STOPS if cur_stop in STOPS else STOPS + [cur_stop]
                stop_c = c[0].selectbox("Stop placement", stops, index=idx(stops, cur_stop), key=k("stop"), format_func=stop_choice_label,
                                        help="Where the protective stop goes. The entry-to-stop distance defines 1R.")
                target = c[1].selectbox("Profit target", TARGETS if p.target_r in TARGETS else TARGETS + [p.target_r],
                                        index=idx(TARGETS if p.target_r in TARGETS else TARGETS + [p.target_r], p.target_r), key=k("r"),
                                        format_func=lambda x: "No target (exit on time)" if x == 0 else f"{x:g}R",
                                        help="Target distance in multiples of the risk (1R = entry-to-stop).")
        with right:
            with st.container(border=True):
                st.markdown("**Trade management & direction**")
                c = st.columns(2)
                cutoff = c[0].selectbox("Last entry time (ET)", CUTOFFS, index=idx(CUTOFFS, p.cutoff, 2), key=k("cut"),
                                        help="No new trades after this time.")
                te_now = p.time_exit or "session"
                texit = c[1].selectbox("Exit open trades at (ET)", TIME_EXITS, index=idx(TIME_EXITS, te_now), key=k("texit"),
                                       format_func=lambda x: "End of session (15:55)" if x == "session" else x)
                c = st.columns(2)
                direction = c[0].selectbox("Direction", list(S.DIRECTION_LABELS), index=idx(list(S.DIRECTION_LABELS), p.direction),
                                           key=k("dir"), format_func=S.DIRECTION_LABELS.get)
                mt = c[1].selectbox("Max trades per day", [1, 2, 0], index=idx([1, 2, 0], p.max_trades), key=k("mt"),
                                    format_func=lambda x: "Unlimited" if x == 0 else str(x))
                be = st.select_slider("Move stop to break-even after", [0.0, 0.5, 0.75, 1.0, 1.25, 1.5], value=p.breakeven_r if p.breakeven_r in
                                      [0.0, 0.5, 0.75, 1.0, 1.25, 1.5] else 0.0, key=k("be"),
                                      format_func=lambda x: "never" if x == 0 else f"+{x:g}R")
            with st.container(border=True):
                st.markdown("**Position sizing**")
                c = st.columns(2)
                modes = list(S.SIZING_SHORT)
                mode = c[0].selectbox("Method", modes, index=idx(modes, sz.mode), key=k("mode"),
                                      format_func=lambda m: {"pct_equity": "Risk % of current equity (compounding)",
                                                             "fixed_risk": "Risk % of starting equity (fixed $)",
                                                             "fixed_contracts": f"Fixed number of {unit}s"}[m],
                                      help=f"Risk-based sizing: {unit}s = risk budget ÷ (entry-to-stop distance × $ per point), rounded down.")
                equity = c[1].number_input("Starting equity ($)", 1000.0, 1e9, float(sz.starting_equity), 1000.0, key=k("equity"), format="%.0f")
                c = st.columns(2)
                risk_now = round(sz.risk_pct * 100, 4)
                labels = [f"{x:g}%" for x in RISK_CHOICES] + ["Custom"]
                cur_label = f"{risk_now:g}%" if risk_now in RISK_CHOICES else "Custom"
                risk_sel = c[0].selectbox("Risk per trade", labels, index=labels.index(cur_label), key=k("risk"),
                                          disabled=mode == "fixed_contracts", help="Share of equity lost if the stop is hit.")
                custom = c[1].number_input("Custom risk % (when 'Custom')", 0.01, 10.0, float(risk_now), 0.05, key=k("riskc"),
                                           disabled=mode == "fixed_contracts")
                qty = st.number_input(f"Fixed quantity ({unit}s)", 1, 10_000_000, int(sz.contracts), key=k("qty"),
                                      disabled=mode != "fixed_contracts")
            with st.container(border=True):
                st.markdown("**Execution assumptions**")
                d = resolve_costs(inst, ExecutionParams()) if inst is not None else None
                cur = resolve_costs(inst, ex) if inst is not None else None
                st.caption((f"All $ costs are per {unit}, per side (every fill). Market defaults"
                            + (f": {inst.cost_description()}" if inst is not None else ".")).replace("$", "\\$"))
                c = st.columns(2)
                slip = c[0].number_input("Slippage (ticks per side)", 0.0, 20.0, float(cur.slippage_ticks if cur else 1.0), 0.5, key=k("slip"),
                                         help="Ticks lost on every market or stop fill. Limit fills need a trade-through instead.")
                comm = c[1].number_input(f"Commission ($ per {unit} per side)", 0.0, 50.0, float(cur.commission_per_side if cur else 0.0), 0.01,
                                         format="%.4f", key=k("comm"), help=f"Broker commission for ONE {unit} on ONE fill. Never per order.")
                c = st.columns(2)
                fee = c[0].number_input(f"Exchange + regulatory fees ($ per {unit} per side)", 0.0, 50.0,
                                        float(cur.exchange_fee_per_side if cur else 0.0), 0.01, format="%.4f", key=k("fee"))
                fric = c[1].number_input(f"Bid/ask friction ($ per {unit} per side)", 0.0, 50.0, float(cur.friction_per_side if cur else 0.0), 0.01,
                                         format="%.4f", key=k("fric"), help="Extra adverse price on every fill, e.g. half the spread.")
        with st.expander("Advanced settings", icon=":material/settings:"):
            c = st.columns(4)
            fixed = c[0].number_input("Fixed stop distance (ticks)", 1, 1000, int(p.stop_param) if p.stop_method == "fixed_ticks" else 8,
                                      key=k("fixed"), help="Used only when the stop placement is 'Fixed number of ticks'.")
            c = st.columns(4)
            buffer = c[0].selectbox("Order buffer (ticks)", [0, 1, 2, 3, 4], index=idx([0, 1, 2, 3, 4], p.entry_buffer_ticks), key=k("buf"),
                                    help="Limit/stop orders placed this many ticks beyond the range edge.")
            atr_p = c[1].selectbox("ATR period (sessions)", [7, 10, 14, 20], index=idx([7, 10, 14, 20], p.atr_period or 14, 2), key=k("atr"))
            filters = ["none", "0.25-0.50", "0.50-0.75", "0.75-1.00", "1.00-1.25", "1.25-1.50"]
            f_now = "none" if not p.or_atr_max else f"{p.or_atr_min:.2f}-{p.or_atr_max:.2f}"
            orf = c[2].selectbox("Only trade if range ÷ ATR is", filters, index=idx(filters, f_now), key=k("orf"))
            reentry = c[3].selectbox("Re-entry after a loss", REENTRY_POLICIES, index=idx(list(REENTRY_POLICIES), p.reentry), key=k("re"),
                                     disabled=mt == 1)
            c = st.columns(4)
            fill = c[0].selectbox("Limit fill model", FILL_MODELS, index=idx(list(FILL_MODELS), ex.fill_model, 1), key=k("fill"),
                                  help="conservative = price must trade through a limit/target by one tick.")
            amb = c[1].selectbox("Stop & target in the same bar", AMBIGUITY_MODES, index=idx(list(AMBIGUITY_MODES), ex.ambiguity, 2), key=k("amb"),
                                 help="Bars cannot show which came first. conservative = stop first, and no target on the entry bar.")
            lev_default = float(sz.max_leverage if sz.max_leverage is not None else (inst.max_notional_leverage if inst is not None else 0.0))
            lev = c[2].number_input("Max position value ÷ equity (0 = off)", 0.0, 100.0, lev_default, 0.5, key=k("lev"))
            max_units = c[3].number_input(f"Max {unit}s per trade", 1, 100_000_000, int(min(sz.max_contracts, 100_000_000)), key=k("maxu"))
        saved = st.form_submit_button("Save strategy", type="primary", icon=":material/save:")
    if not saved:
        return
    stop_method, stop_param = ("fixed_ticks", float(fixed)) if stop_c == "fixed_ticks" else parse_stop(stop_c)
    lo, hi = (0.0, 0.0) if orf == "none" else tuple(float(x) for x in orf.split("-"))
    ticks, frac = CONFIRMATION_PRESETS[conf]
    params = StrategyParams(orb_start=orb_start, range_minutes=int(rng), entry_tf=int(tf), entry_method=em, confirm_ticks=ticks,
                            confirm_or_frac=frac, entry_buffer_ticks=int(buffer), stop_method=stop_method, stop_param=float(stop_param),
                            atr_period=int(atr_p), target_r=float(target), cutoff=cutoff, time_exit=None if texit == "session" else texit,
                            direction=direction, max_trades=int(mt), reentry=reentry if mt != 1 else "any", breakeven_r=float(be),
                            or_atr_min=lo, or_atr_max=hi)
    risk_pct = (float(custom) if risk_sel == "Custom" else float(risk_sel.rstrip("%"))) / 100
    sizing = SizingParams(mode=mode, contracts=float(qty), risk_dollars=equity * risk_pct, risk_pct=risk_pct, starting_equity=float(equity),
                          max_contracts=float(max_units), max_leverage=float(lev))

    def over(value, default):  # store only real overrides of the market default
        return None if d is not None and abs(value - default) < 1e-12 else float(value)

    execution = ExecutionParams(
        slippage_ticks=over(slip, d.slippage_ticks) if d else slip, commission_per_side=over(comm, d.commission_per_side) if d else comm,
        exchange_fee_per_side=over(fee, d.exchange_fee_per_side) if d else fee, friction_per_side=over(fric, d.friction_per_side) if d else fric,
        fill_model=fill, ambiguity=amb)
    try:
        params.validate(base)
        sizing.validate()
        execution.validate()
    except ValueError as exc:
        st.error(f"Not saved: {exc}")
        return
    S.set_strategy(params, execution, sizing, "edited on the Strategy page", from_form=True)
    S.S()["flash"] = "Strategy saved. Every page now uses it."
    st.rerun()


def all_settings_table() -> pd.DataFrame:
    strat = S.strategy()
    p = strat["params"]
    rows = [("Market", S.instrument().symbol if S.instrument() else "—"), ("Range start", p.orb_start), ("Range length", f"{p.range_minutes} min"),
            ("Entry type", S.ENTRY_LABELS[p.entry_method]), ("Entry timeframe", "intrabar" if p.entry_method == "stop" else f"{p.entry_tf} min"),
            ("Confirmation", S.confirmation_text(p)), ("Stop", S.stop_text(p)), ("Target", f"{p.target_r:g}R" if p.target_r else "none"),
            ("Direction", S.DIRECTION_LABELS[p.direction]), ("Last entry", p.cutoff), ("Time exit", p.time_exit or "end of session"),
            ("Max trades/day", "unlimited" if p.max_trades == 0 else str(p.max_trades)),
            ("Break-even", "off" if not p.breakeven_r else f"+{p.breakeven_r:g}R"), ("Sizing", S.portfolio_text()),
            ("Execution", S.execution_text())]
    return pd.DataFrame(rows, columns=["setting", "value"])


def sized_out_warning(res) -> None:
    """Explain skipped signals (stop too wide for the risk budget) wherever results are summarised."""
    t = res.trades
    if not len(t) or not t["sized_out"].any():
        return
    n, total = int(t["sized_out"].sum()), len(t)
    unit = res.diagnostics.get("unit", "contract")
    tip_ = (" Micro contracts (MES, MNQ, MGC, M6E) risk one tenth as much per contract." if unit == "contract" else "")
    st.warning(f"**{n} of {total} signals were skipped**: one {unit} would have risked more than the risk budget "
               f"({S.portfolio_text()}). Raise the risk per trade, use fixed sizing, or trade a smaller instrument.{tip_}",
               icon=":material/block:")

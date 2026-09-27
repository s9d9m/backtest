"""RESULTS: one-page research summary of the current strategy across every stage, with a conservative verdict."""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from ...reports.final_report import build_markdown, classify
from ...research import pipeline_state as ps
from .. import state as S
from .. import theme as T
from .. import widgets as W


def build_stages(for_overview: bool = False, wfo_pick: tuple[str, dict] | None = None) -> dict:
    """Collect everything known about the current strategy into the structure used by the report and its verdict."""
    ds, inst = S.dataset(), S.instrument()
    strat = S.strategy()
    p = strat["params"]
    frozen = S.frozen_match()
    stages: dict = {"candidate": {"name": frozen["name"] if frozen else strat.get("name", "current strategy"),
                                  "sha256": frozen["sha256"] if frozen else "not frozen", "params": p.to_dict(),
                                  "costs": inst.cost_description() if inst else ""},
                    "data": {"symbol": inst.symbol if inst else "—", "unit": inst.unit if inst else "", "proxy": S.is_proxy(inst),
                             "synthetic": S.is_synthetic(ds), "sessions": ds.prep_full.n_days if ds else 0,
                             "first": str(ds.prep_full.dates[0]) if ds else "", "last": str(ds.prep_full.dates[-1]) if ds else "",
                             "source": ds.loaded.source if ds else "", "hash": ds.prep_full.data_hash if ds else ""}}
    if ds is None:
        return stages
    split = S.S().get("split")
    bt = S.current_backtest()
    if bt is not None:
        m = bt["res"].metrics
        stages["concentration"] = m
        if split is None:
            stages["train"] = m
    if split:
        stages["split"] = split
        val = S.fetch("validation")
        if val:
            stages["train"], stages["validation"] = val["train"], val["validation"]
        elif bt is not None:
            stages["train"] = bt["res"].metrics
    blind = S.blind_result()
    if blind:
        stages["blind"] = blind
    w = wfo_pick if wfo_pick is not None else S.latest_wfo_summary()
    if w:
        stages["wfo"] = w[1]
    rob = S.fetch("robustness")
    if rob:
        stages["robustness"] = {"verdict": rob["verdict"], "lines": [f"{r.label}: {r.classification} (setting {T.rmult(r.candidate_exp_r)}, "
                                                                    f"neighbours {T.rmult(r.neighbour_median_exp_r)})"
                                                                    for r in rob["summary"].itertuples()]}
    stress = S.fetch("stress")
    if stress:
        stages["stress"] = stress[1]
    mc = S.fetch("mc")
    if mc:
        stages["mc"] = mc["result"].to_dict()
    if S.S().get("grid") is not None:
        stages["grid"] = S.S()["grid"].selection_bias
    if S.is_proxy(inst):
        stages["limitations"] = ["ETF proxy data from Yahoo (~60 days of 5-minute bars): far too short for conclusions about futures."]
    return stages


def _stage_card(title: str, m: dict | None, evidence: bool, note: str = "") -> str:
    if not m:
        return T.kpi(title, "Not run", note or "", tone="none")
    tone = ("good" if (m.get("avg_r") or 0) > 0 else "bad") if evidence else "none"
    sub = f"{int(m.get('n_trades') or 0)} trades · PF {T.ratio(m.get('profit_factor'))} · t {T.ratio(m.get('t_stat_r'))}"
    return T.kpi(title, T.rmult(m.get("avg_r")), sub + ("" if evidence else " · not evidence"), tone=tone)


def render() -> None:
    T.page_header("Evidence", "Results", "Everything known about the current strategy on one page, and what it adds up to.")
    W.strategy_header()
    if W.need_data():
        return
    wfo = S.latest_wfo_summary()
    stages = build_stages(wfo_pick=wfo)
    verdict, reasons = classify(stages)
    T.verdict_banner("Research verdict", verdict, reasons)
    st.caption("PROMISING needs positive, statistically notable out-of-sample results on ≥ 100 trades with no fragility flags. "
               "MIXED = something positive but weak or fragile. NO PRELIMINARY EVIDENCE = out-of-sample results are not positive. "
               "INSUFFICIENT EVIDENCE = no or too little out-of-sample testing. In-sample results never change the verdict.")
    if stages["data"]["proxy"]:
        st.warning("Free proxy ETF data — not futures validation.")
    if stages["data"]["synthetic"]:
        st.warning("Synthetic data — not market evidence.")

    T.section("Performance by stage")
    wfo_m = stages.get("wfo", {}).get("stitched_always_trade") if stages.get("wfo") else None
    blind = stages.get("blind")
    T.kpi_grid([
        _stage_card("In-sample (train)" if stages.get("split") else "In-sample", stages.get("train"), False, "Backtest"),
        _stage_card("Validation", stages.get("validation"), False, "Validate → holdout, step 2"),
        _stage_card("Blind holdout" + ("" if not blind or blind["status"] == "BLIND" else " (not blind)"), blind and blind.get("metrics"),
                    bool(blind and blind["status"] == "BLIND"), "Validate → holdout, step 4"),
        _stage_card("Walk-forward OOS", wfo_m, True, "Validate → walk-forward"),
    ], min_width=200)

    T.section("Stress tests")
    rob, stress, mc = stages.get("robustness"), stages.get("stress"), stages.get("mc")
    rob_label, rob_tone = S.robustness_label(rob["verdict"]) if rob else ("Not run", "none")
    st_v = stress["verdict"] if stress else "Not run"
    cards = [T.kpi("Parameter robustness", rob_label, "neighbouring settings", tone=rob_tone, help_term="Plateau"),
             T.kpi("Execution sensitivity", st_v.title() if stress else "Not run",
                   ("breaks under " + ", ".join(stress["breaks_under"])) if stress and stress["breaks_under"] else "slippage, costs, fills",
                   tone={"ROBUST": "good", "FRAGILE": "bad", "NOT POSITIVE": "bad"}.get(st_v, "none"))]
    if mc:
        prob = mc["probabilities"]
        cards += [T.kpi("Monte Carlo: chance of loss", T.pct(prob["p_loss"], 0), "path risk, not evidence", help_term="Monte Carlo",
                        tone="good" if prob["p_loss"] < 0.1 else "warn" if prob["p_loss"] < 0.3 else "bad"),
                  T.kpi("Monte Carlo: bad-case drawdown", T.pct(next((r["max_drawdown"] for r in mc["percentiles"] if r["index"] == "5th"), None)),
                        "5th percentile of simulations")]
    else:
        cards.append(T.kpi("Monte Carlo", "Not run", "Stress test → Monte Carlo"))
    T.kpi_grid(cards, min_width=200)

    conc = stages.get("concentration")
    if conc:
        T.section("Trades, sides and costs (development data)")
        T.kpi_grid([
            T.kpi("Top-5 share", T.pct(conc.get("top5_share"), 0), f"without top 5: {T.usd(conc.get('pnl_ex_top5'), signed=True)}",
                  tone="warn" if (conc.get("top5_share") or 0) > 0.5 else "none"),
            T.kpi("Long", f"{conc.get('n_long', 0)} trades", f"{T.usd(conc.get('pnl_long'), signed=True)} · {T.rmult(conc.get('expectancy_r_long'))}"),
            T.kpi("Short", f"{conc.get('n_short', 0)} trades", f"{T.usd(conc.get('pnl_short'), signed=True)} · {T.rmult(conc.get('expectancy_r_short'))}"),
            T.kpi("Total costs", T.usd(conc.get("total_cost")), f"{T.usd(conc.get('cost_per_trade'), dp=2)} per trade", help_term="Cost per trade"),
            T.kpi("Ambiguous exits", T.pct(conc.get("ambiguous_exit_pct"), 0), "stop and target in the same bar",
                  tone="warn" if (conc.get("ambiguous_exit_pct") or 0) > 0.1 else "none"),
        ], min_width=180)

    T.section("Strategy and assumptions")
    T.card("Strategy", T.chips(S.strategy_chips(), first_is_market=True))
    T.card("Portfolio & execution", f"{S.portfolio_text()}<br>{S.execution_text()}".replace("$", "&#36;"))
    grid = stages.get("grid")
    if grid:
        T.note(f"Optimisation context: {grid['n_configs']:,} configurations searched (≈ {grid['n_effective']:,.0f} independent). The best "
               f"in-sample Sharpe was {grid['best_sharpe']:.2f} vs ≈ {grid['expected_max_sharpe_under_null']:.2f} expected from luck alone.", "warn")

    md = build_markdown(stages)
    c1, c2, _ = st.columns([1, 1, 2])
    c1.download_button("Download report (Markdown)", md.encode(), file_name=f"orb_report_{S.strategy()['params'].key()}.md",
                       icon=":material/download:")
    if c2.button("Save report to the research folder", icon=":material/save:"):
        path = ps.data_dir(S.dataset().prep_full.data_hash) / f"report_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.md"
        path.write_text(md)
        st.success(f"Saved {path}")
    with st.expander("Full written report", icon=":material/article:"):
        st.markdown(md)

"""Final research report for one candidate, combining every pipeline stage.

Only out-of-sample results (the first blind holdout test and the stitched walk-forward OOS) count as
evidence. In-sample optimisation and validation results are shown for comparison only. The verdict
language is deliberately conservative:

* ``PROMISING``: positive OOS expectancy with |t| >= 2 on at least 100 OOS trades, no fragility flags
  (robustness spike, execution fragility, WFO overfitting majority). Still not proof.
* ``NO PRELIMINARY EVIDENCE``: out-of-sample expectancy is not positive.
* ``MIXED``: anything in between (positive but small / insignificant / fragile).
* ``INSUFFICIENT EVIDENCE``: no out-of-sample test, or fewer than 30 OOS trades.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

VERDICTS = ("PROMISING", "MIXED", "NO PRELIMINARY EVIDENCE", "INSUFFICIENT EVIDENCE")


def _num(x, default=0.0) -> float:
    try:
        v = float(x)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


def classify(stages: dict) -> tuple[str, list[str]]:
    reasons: list[str] = []
    oos = []
    blind = stages.get("blind")
    if blind and blind.get("status") == "BLIND":
        m = blind["metrics"]
        oos.append(("blind holdout", _num(m.get("n_trades")), _num(m.get("avg_r")), _num(m.get("t_stat_r"))))
    elif blind:
        reasons.append("The holdout had already been used before this test, so it is not blind and is not counted as evidence.")
    wfo = stages.get("wfo")
    if wfo:
        s = wfo.get("stitched_always_trade", {})
        oos.append(("stitched walk-forward OOS", _num(s.get("n_trades")), _num(s.get("avg_r")), _num(s.get("t_stat_r"))))
    if not oos:
        return "INSUFFICIENT EVIDENCE", reasons + ["No out-of-sample test has been run (blind holdout or walk-forward)."]
    total = sum(o[1] for o in oos)
    for name, n, exp, t in oos:
        reasons.append(f"{name}: {int(n)} trades, expectancy {exp:+.3f} R, t = {t:.2f}.")
    if total < 30:
        return "INSUFFICIENT EVIDENCE", reasons + [f"Only {int(total)} out-of-sample trades in total (< 30)."]
    if all(o[2] <= 0 for o in oos):
        return "NO PRELIMINARY EVIDENCE", reasons + ["Out-of-sample expectancy is not positive."]
    fragile = []
    rob = stages.get("robustness")
    if rob and str(rob.get("verdict", "")).startswith("SPIKE"):
        fragile.append("parameter neighbourhood is a spike")
    stress = stages.get("stress")
    if stress and stress.get("verdict") in ("FRAGILE", "NOT POSITIVE"):
        fragile.append(f"execution stress verdict {stress['verdict']}")
    if wfo and len(wfo.get("overfit_flags", [])) >= 2:
        fragile.append("walk-forward overfitting flags")
    if any(o[2] <= 0 for o in oos):
        fragile.append("out-of-sample sources disagree in sign")
    strong = all(o[2] > 0 and o[3] >= 2 for o in oos) and total >= 100
    if strong and not fragile:
        return "PROMISING", reasons + ["Positive and statistically notable out of sample; no fragility flags. This is preliminary, not proof."]
    if fragile:
        reasons.append("Caveats: " + "; ".join(fragile) + ".")
    if not strong:
        reasons.append("Out-of-sample result is not strong enough (needs t >= 2 on >= 100 trades in every OOS source).")
    return "MIXED", reasons


def _row(label, m: dict | None) -> str:
    if not m:
        return f"| {label} | — | — | — | — | — | — |"
    return (f"| {label} | {int(_num(m.get('n_trades')))} | {_num(m.get('avg_r')):+.3f} | {_num(m.get('t_stat_r')):.2f} | "
            f"{_num(m.get('profit_factor')):.2f} | {_num(m.get('net_pnl')):,.0f} | {_num(m.get('max_dd')):.1%} |")


def build_markdown(stages: dict) -> str:
    verdict, reasons = classify(stages)
    c = stages.get("candidate", {})
    d = stages.get("data", {})
    lines = [f"# ORB Lab research report: {c.get('name', 'candidate')}", "",
             f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.", ""]
    if d.get("proxy"):
        lines += ["> **FREE PROXY EXPERIMENT — NOT FUTURES VALIDATION.** ETF data; nothing here validates ES, NQ, GC or 6E.", ""]
    if d.get("synthetic"):
        lines += ["> **SYNTHETIC DATA — NOT MARKET EVIDENCE.**", ""]
    lines += [f"## Verdict: **{verdict}**", ""] + [f"- {r}" for r in reasons] + [""]
    lines += ["## Data", f"- Instrument: {d.get('symbol')} ({d.get('unit', '')}), {d.get('sessions', '?')} sessions, "
              f"{d.get('first', '?')} to {d.get('last', '?')}", f"- Source: {d.get('source', '?')}", f"- Data SHA-256: `{d.get('hash', '?')}`"]
    sp = stages.get("split")
    if sp:
        lines.append(f"- Split: train {sp['train_start']}..{sp['train_end']} ({sp['n_train']}), validation {sp['val_start']}..{sp['val_end']} "
                     f"({sp['n_val']}), blind holdout {sp['holdout_start']}..{sp['holdout_end']} ({sp['n_holdout']})")
    lines += ["", "## Candidate", f"- Frozen SHA-256: `{c.get('sha256', 'not frozen')}`", f"- Parameters: `{c.get('params', {})}`",
              f"- Costs: {c.get('costs', '')}", ""]
    lines += ["## Results by stage", "", "| Stage | Trades | Exp. R | t | PF | Net $ | Max DD |", "|---|---|---|---|---|---|---|",
              _row("In-sample (train) — NOT evidence", stages.get("train")),
              _row("Validation — NOT evidence (used for choosing)", stages.get("validation"))]
    blind = stages.get("blind")
    lines.append(_row(f"Holdout ({blind['status']})" if blind else "Blind holdout — not run", blind and blind.get("metrics")))
    wfo = stages.get("wfo")
    lines.append(_row("Stitched walk-forward OOS", wfo and wfo.get("stitched_always_trade")))
    lines.append("")
    grid = stages.get("grid")
    if grid:
        lines += ["## Optimisation context (multiple testing)",
                  f"- Configurations searched: {grid.get('n_configs')} (effective ≈ {_num(grid.get('n_effective')):,.0f})",
                  f"- Best in-sample Sharpe {_num(grid.get('best_sharpe')):.2f} vs ≈ {_num(grid.get('expected_max_sharpe_under_null')):.2f} "
                  "expected from the best of that many no-edge configurations.", ""]
    if wfo:
        lines += ["## Walk-forward", f"- Structure: {wfo.get('structure', {})}", f"- Windows: {wfo.get('windows')}, profitable OOS windows: "
                  f"{_num(wfo.get('pct_windows_oos_profitable')):.0%}", f"- Parameter stability: {wfo.get('parameter_stability', {})}",
                  f"- Overfitting flags: {'; '.join(wfo.get('overfit_flags', [])) or 'none'}",
                  f"- Execution-sensitive: {wfo.get('execution_sensitive')}", ""]
    rob = stages.get("robustness")
    if rob:
        lines += ["## Parameter robustness", f"- Verdict: **{rob.get('verdict')}**"] + [f"- {x}" for x in rob.get("lines", [])] + [""]
    stress = stages.get("stress")
    if stress:
        lines += ["## Execution stress", f"- Verdict: **{stress.get('verdict')}**",
                  f"- Breaks under: {', '.join(stress.get('breaks_under', [])) or 'nothing tested'}", ""]
    mc = stages.get("mc")
    if mc:
        p = mc.get("probabilities", {})
        h = mc.get("historical", {})
        lines += ["## Monte Carlo (simulated from historical trade outcomes; not new evidence)",
                  f"- Method: {mc.get('config', {}).get('method')}, {mc.get('config', {}).get('n_sims')} simulations, seed {mc.get('config', {}).get('seed')}",
                  f"- Historical max drawdown {_num(h.get('max_drawdown')):.1%}; probability a simulated path is worse: {_num(p.get('p_dd_worse_than_historical')):.0%}",
                  f"- Probability of a net loss: {_num(p.get('p_loss')):.0%}; drawdown worse than 20 %: {_num(p.get('p_dd_worse_than_20pct')):.0%}", ""]
    conc = stages.get("concentration")
    if conc:
        lines += ["## Concentration and sides", f"- Top-5 trade share of profit: {_num(conc.get('top5_share'), float('nan')):.0%}; "
                  f"P&L without top 5: ${_num(conc.get('pnl_ex_top5')):,.0f}",
                  f"- Long: {conc.get('n_long')} trades, ${_num(conc.get('pnl_long')):,.0f}; short: {conc.get('n_short')} trades, "
                  f"${_num(conc.get('pnl_short')):,.0f}", f"- Costs: ${_num(conc.get('total_cost')):,.0f} total, "
                  f"${_num(conc.get('cost_per_trade')):,.2f} per trade", ""]
    lines += ["## Limitations", "- OHLC bars cannot show the order of prices inside a bar; same-bar stop/target ambiguity is resolved by assumption.",
              "- Costs and slippage are modelled, not measured from real fills.",
              "- Past performance, including out-of-sample, does not guarantee future results; regimes change."]
    lines += [f"- {x}" for x in stages.get("limitations", [])]
    return "\n".join(lines) + "\n"

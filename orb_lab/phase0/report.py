"""Markdown reports for PHASE0_FREE_PROXY."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..data_sources.yahoo import PHASE0_LABEL

KEY_ROWS = [
    ("trades", "Trades", "{:.0f}"), ("win_rate", "Win rate", "{:.1%}"), ("exp_r", "Expectancy (R)", "{:+.3f}"),
    ("t_stat_r", "t-stat of mean R", "{:+.2f}"), ("profit_factor", "Profit factor", "{:.2f}"), ("net_pnl", "Net P&L (100 sh)", "${:,.2f}"),
    ("gross_pnl", "Gross P&L (100 sh)", "${:,.2f}"), ("friction_cost", "Friction paid", "${:,.2f}"), ("sharpe_daily", "Sharpe (daily, annualised)", "{:+.2f}"),
    ("max_dd_r", "Max drawdown (R)", "{:+.2f}"), ("max_dd_usd", "Max drawdown ($, 100 sh)", "${:,.2f}"), ("avg_winner", "Average winner", "${:,.2f}"),
    ("avg_loser", "Average loser", "${:,.2f}"), ("largest_winner", "Largest winner", "${:,.2f}"), ("largest_loser", "Largest loser", "${:,.2f}"),
    ("long_trades", "Long trades", "{:.0f}"), ("long_exp_r", "Long expectancy (R)", "{:+.3f}"), ("long_net_pnl", "Long net P&L", "${:,.2f}"),
    ("short_trades", "Short trades", "{:.0f}"), ("short_exp_r", "Short expectancy (R)", "{:+.3f}"), ("short_net_pnl", "Short net P&L", "${:,.2f}"),
]


def _fmt(v, f):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "n/a"
        return f.format(v)
    except (ValueError, TypeError):
        return str(v)


def split_table(headline: dict) -> str:
    rows = []
    for key, label, f in KEY_ROWS:
        vals = {}
        for ph in ("train", "val", "test"):
            v = headline[ph].get(key)
            if key == "t_stat_r" and (headline[ph].get("trades", 0) < 10 or abs(v or 0) > 20):
                vals[ph] = "not meaningful (n<10 or ~zero variance)"
            elif key == "profit_factor" and v == float("inf"):
                vals[ph] = "inf (no losing trades)"
            else:
                vals[ph] = _fmt(v, f)
        rows.append({"metric": label, **vals})
    return pd.DataFrame(rows).rename(columns={"train": "TRAIN", "val": "VALIDATION", "test": "FINAL TEST (unseen)"}).to_markdown(index=False)


def _dict_table(d: dict, value_fmt="{:+.3f}") -> str:
    frame = pd.DataFrame(d)
    return frame.map(lambda v: _fmt(v, value_fmt)).to_markdown()


def write_symbol_report(summary: dict, friction: pd.DataFrame, neighbourhood: pd.DataFrame, finalists: pd.DataFrame, out: Path) -> Path:
    s = summary
    fz = s["frozen"]
    sel = fz["selected"]
    prov = s["provenance"]
    dq = s["dq"]
    ans = s["answers"]
    ctrl = s["controls"]
    lines = [
        f"# PHASE0_FREE_PROXY — {s['symbol']}",
        f"**{PHASE0_LABEL}.** {s['symbol']} is an ETF proxy. Nothing here validates any futures strategy.",
        "",
        f"## Conclusion: **{s['verdict']['category']}**",
        *[f"- {r}" for r in s["verdict"]["reasons"]],
        f"- Rule (stated before looking at the test): {s['verdict']['rule']}",
        "",
        "## Data used",
        f"- Source: {prov['source']}",
        f"- Interval: **{s['interval']}** (chosen by measured availability: " + ", ".join(
            f"{k}: {v['sessions']} sessions from {v['first'][:10]}" for k, v in s["availability"].items()) + ")",
        f"- Returned: {prov['returned_first_bar']} to {prov['returned_last_bar']}; {prov['rows_raw']:,} bars; source timezone {prov['source_timezone']}; "
        f"{prov['prices']}",
        f"- Downloaded {prov['download_utc']}; processed file SHA-256 `{prov['processed_sha256'][:16]}…`",
        "",
        "## Data quality",
        f"- Phase-0 DQ verdict: **{dq['verdict']}**",
        f"- Sessions {dq['sessions']} ({dq['first_session']} … {dq['last_session']}); UTC offsets seen {dq['utc_offsets_seen']} "
        f"(DST transition in sample: {dq['dst_transition_in_sample']})",
        f"- Sessions whose first bar is not 09:30 ET: {dq['sessions_first_bar_not_0930']}; incomplete sessions: {dq['sessions_incomplete']}; "
        f"missing XNYS sessions: {dq['missing_calendar_sessions'] or 'none'}; early closes: {dq['early_close_sessions'] or 'none'}",
        f"- Modal peak-volume minute: {dq['modal_peak_volume_minute_et']} (570 = 09:30 ET)",
        f"- Engine DQ issues: {dq['duplicate_or_invalid'] or 'none'}",
        f"- 1-minute cross-check: {json.dumps(dq['one_minute_cross_check'])}",
        *[f"- note: {r}" for r in dq["reasons"]],
        "",
        "## Sample design (chronological, no shuffling)",
        f"- TRAIN: {s['split']['train'][0]} … {s['split']['train'][1]} ({s['split']['sessions']['train']} sessions)",
        f"- VALIDATION: {s['split']['validation'][0]} … {s['split']['validation'][1]} ({s['split']['sessions']['validation']} sessions)",
        f"- FINAL TEST: {s['split']['final_test'][0]} … {s['split']['final_test'][1]} ({s['split']['sessions']['final_test']} sessions); "
        "not in memory during selection. The selection was frozen to `selection_frozen.json` "
        f"(SHA-256 `{s['selection_frozen_sha256'][:16]}…`) before the test was evaluated.",
        f"- Configurations evaluated: {s['space']['evaluated']:,} (raw {s['space']['raw']:,}; invalid {s['space']['invalid']:,}; "
        f"canonical duplicates {s['space']['canonical_duplicates']:,}; limit-entry 50 %-width stops identical to midpoint stops removed "
        f"{s['space']['equivalent_limit_or_pct50_removed']:,})",
        f"- Friction for selection and headline: ${fz['headline_friction_usd_per_share']:.2f}/share charged on EVERY fill (entry and exit, "
        "market, stop and limit alike), i.e. a fixed round-trip cost per share; limit fills must also trade through by $0.01. "
        "$0.00 is a reference only.",
        f"- Multiple testing (train Sharpe across configs): best {fz['selection_bias_train']['best_sharpe']:.2f} vs "
        f"{fz['selection_bias_train']['expected_max_sharpe_under_null']:.2f} expected from selection alone under the null.",
        "",
        "## Best in-sample configuration (reported, not emphasised)",
        f"- Best training score: `{fz['best_in_sample_train_score']['config_key']}` "
        + _cfg(fz["best_in_sample_train_score"]),
        f"- Highest train+validation net P&L: `{fz['best_in_sample_train_plus_val_pnl']['config_key']}` " + _cfg(fz["best_in_sample_train_plus_val_pnl"]),
        "",
        _comparison_table(s["comparisons"]),
        "",
        "## Validation-selected configuration",
        f"`{sel['config_key']}` " + _cfg(sel),
        f"- Why: highest selection score = ½·validation score + ½·(½·training score + ½·median training score of its immediate parameter "
        f"neighbours) among the 25 finalists ranked by that neighbourhood-robust training score. Training neighbourhood class: **{sel['plateau_class_train']}**.",
        f"- Training: {sel['train_metrics']['n_trades']:.0f} trades, {sel['train_metrics']['exp_r']:+.3f} R; "
        f"Validation: {sel['val_metrics']['n_trades']:.0f} trades, {sel['val_metrics']['exp_r']:+.3f} R.",
        "",
        "Top finalists:",
        finalists.sort_values("selection_score", ascending=False).head(8)[
            ["range_minutes", "entry_tf", "entry_method", "confirmation", "stop", "target_r", "cutoff", "direction", "train_trades", "train_exp_r",
             "neighbour_median_train_score", "val_trades", "val_exp_r", "selection_score"]].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Unseen final-test result (the result that matters)",
        split_table(s["headline"]),
        "",
        "## Controls and baselines (same dates)",
        _controls_table(ctrl),
        "",
        "## Robustness: neighbourhood of the selected configuration (expectancy R per phase)",
        neighbourhood.to_markdown(index=False, floatfmt=".3f"),
        "",
        f"Selected configuration neighbourhood class by phase: {ans['selected_plateau']}; neighbour median expectancy: "
        + ", ".join(f"{k} {v:+.3f}" for k, v in ans["selected_neighbour_median_exp_r"].items()),
        f"Rank correlation of all configurations' expectancy: train→test {ans['rank_corr_train_vs_test']:+.3f}, validation→test {ans['rank_corr_val_vs_test']:+.3f}.",
        "Heatmaps: `heatmaps/*.png` (median expectancy across hidden parameters, per phase).",
        "",
        "## Execution sensitivity (selected configuration)",
        friction[["split", "friction_usd_per_share", "trades", "exp_r", "t_stat_r", "profit_factor", "net_pnl", "friction_cost"]].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Concentration (sum of R after removing the best trades)",
        pd.DataFrame({ph: {k: s["headline"][ph].get(k) for k in ("sum_r", "sum_r_without_best_1", "sum_r_without_best_3", "sum_r_without_best_5",
                                                                  "sum_r_without_best_10")} for ph in ("train", "val", "test")}).to_markdown(floatfmt="+.2f"),
        "(n/a = fewer trades than removed)",
        "",
        "## Grid-wide descriptive evidence (median expectancy R across all configurations sharing the value that traded in that phase)",
        f"Configurations with at least one trade: {ans['configs_with_trades']}. Descriptive only; never used for selection.",
        "### Opening-range duration", _dict_table(ans["range_minutes"]),
        "### Targets below 2R vs 2R and above", _dict_table(ans["sub2R_vs_2R_plus"]),
        "### Direction", _dict_table(ans["direction"]),
        "### Entry method", _dict_table(ans["entry_method"]),
        "### Entry cutoff", _dict_table(ans["cutoff"]),
        f"### Share of configurations with positive expectancy: " + ", ".join(f"{k} {v:.1%}" for k, v in ans["share_configs_positive"].items()),
        "",
        "## Resolution cross-check (frozen candidate on 1-minute bars, overlapping final-test dates)",
        f"{json.dumps(s['resolution_cross_check_1m'])}",
        "",
        *_run_history(out),
        "## Trade verification",
        f"Complete trade log: `trade_log_selected.csv`. Charts ({len(s['charts'])}): `trade_charts/`.",
    ]
    path = out / "phase0_report.md"
    path.write_text("\n".join(lines))
    return path


def _run_history(out: Path) -> list[str]:
    runs = sorted(p for p in out.iterdir() if p.is_dir() and p.name.startswith("run") and (p / "summary.json").exists())
    if not runs:
        return []
    lines = ["## Run history (disclosure)"]
    for r in runs:
        s = json.loads((r / "summary.json").read_text())
        t = s["headline"]["test"]
        lines.append(
            f"- `{r.name}`: selected `{s['frozen']['selected']['config_key']}` {_cfg(s['frozen']['selected'])}; final test {t['trades']} trades, "
            f"{t['exp_r']:+.3f} R; verdict {s['verdict']['category']}. Superseded because the engine's path-slippage model moves stop and target "
            "with the fill, so slippage never reduced the P&L of trades that still reached their target (net P&L was identical at $0.00, $0.01 "
            "and $0.02). This did not match the specified per-share adverse-execution cost. The cost model was changed to a fixed per-share "
            "charge on every fill. The change was made after run 1's report (including final-test rows) had been produced; this is disclosed "
            "here, and both runs are in `phase0_results/phase0_log.jsonl`."
        )
    lines.append("")
    return lines


def _cfg(c: dict) -> str:
    stop = c["stop_method"] + (f":{c['stop_param']:g}" if c["stop_method"] not in ("or_mid", "or_opposite") else "")
    conf = f"close+{c['confirm_ticks']:g}t+{c['confirm_or_frac']:g}W"
    return (f"(range {c['range_minutes']}m, entry TF {c['entry_tf'] or 'intrabar'}, {c['entry_method']}, confirm {conf}, stop {stop}, "
            f"{c['target_r']}R, cutoff {c['cutoff']}, {c['direction']})")


def _comparison_table(comp: dict) -> str:
    rows = []
    for name, by_phase in comp.items():
        for ph in ("train", "val", "test"):
            r = by_phase[ph]
            rows.append({"configuration": name, "phase": ph, "trades": r["trades"], "exp_r": r["exp_r"], "net_pnl": r["net_pnl"]})
    return "In-sample winners on every phase (degradation check):\n\n" + pd.DataFrame(rows).to_markdown(index=False, floatfmt="+.3f")


def _controls_table(ctrl: dict) -> str:
    rows = []
    for ph in ("train", "val", "test"):
        c = ctrl[ph]
        rd, bh = c["random_direction"], c["buy_and_hold"]
        rows.append({"phase": ph, "strategy mean R": rd.get("strategy_mean_r"), "random-direction median": rd.get("random_mean_r_median"),
                     "random 5%": rd.get("random_mean_r_p05"), "random 95%": rd.get("random_mean_r_p95"),
                     "p(random >= strategy)": rd.get("p_value_random_ge_strategy"), "buy&hold %": bh.get("buy_hold_return_pct"),
                     "buy&hold $ (100 sh)": bh.get("buy_hold_pnl_per_shares"), "intraday-long $ (100 sh)": bh.get("intraday_long_total_per_shares"),
                     "sim check |dR|": c.get("sim_reproduces_engine_max_abs_r_diff")})
    return pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f")


def write_combined(summaries: dict[str, dict], out: Path) -> Path:
    rows = []
    for sym, s in summaries.items():
        t = s["headline"]["test"]
        v = s["headline"]["val"]
        sel = s["frozen"]["selected"]
        rows.append({"symbol": sym, "verdict": s["verdict"]["category"], "selected": _cfg(sel), "val trades": v["trades"], "val exp R": v["exp_r"],
                     "test trades": t["trades"], "test exp R": t["exp_r"], "test t": t["t_stat_r"], "test PF": t["profit_factor"],
                     "test net $ (100 sh)": t["net_pnl"],
                     "random-dir p": s["controls"]["test"]["random_direction"].get("p_value_random_ge_strategy"),
                     "median config test exp R": s["answers"]["median_config_exp_r"]["test"],
                     "share configs positive (test)": s["answers"]["share_configs_positive"]["test"]})
    frame = pd.DataFrame(rows)
    path = out / "PHASE0_COMPARISON.md"
    path.write_text(f"# PHASE0_FREE_PROXY — SPY vs QQQ (auto-generated)\n\n**{PHASE0_LABEL}.**\n\n" + frame.to_markdown(index=False, floatfmt=".3f"))
    return path

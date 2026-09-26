"""Two-parameter views of grid-search results.

A heatmap cell aggregates every configuration sharing that (x, y) pair. The default aggregation is the
**median** across the remaining parameters, which shows whether a region is broadly good. ``max`` shows
the best configuration per cell and is labelled as optimistic because it silently re-optimises the
hidden dimensions.
"""

from __future__ import annotations

import pandas as pd

AGGREGATIONS = {
    "median": "Median across other parameters (robust view)",
    "mean": "Mean across other parameters",
    "p25": "25th percentile across other parameters (downside view)",
    "max": "Best across other parameters (OPTIMISTIC: re-optimises hidden dimensions)",
    "share_positive": "Share of configurations with a positive value",
}

STANDARD_PAIRS = [
    ("range_minutes", "target_r", "Range x R"),
    ("range_minutes", "entry_tf", "Range x Entry TF"),
    ("target_r", "cutoff", "R x Cutoff"),
    ("stop", "target_r", "Stop x R"),
    ("orb_start", "range_minutes", "Start time x Range"),
]


def heatmap_table(results: pd.DataFrame, x: str, y: str, metric: str, agg: str = "median", filters: dict | None = None) -> pd.DataFrame:
    frame = results
    for key, value in (filters or {}).items():
        frame = frame[frame[key] == value]
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby([y, x], observed=True)[metric]
    if agg == "p25":
        values = grouped.quantile(0.25)
    elif agg == "share_positive":
        values = grouped.apply(lambda s: float((s > 0).mean()))
    else:
        values = grouped.agg(agg)
    table = values.unstack(x)
    table.index.name = y
    table.columns.name = x
    return table


def varying_parameters(results: pd.DataFrame, candidates: list[str] | None = None) -> list[str]:
    candidates = candidates or [
        "orb_start", "range_minutes", "entry_tf", "entry_method", "confirmation", "entry_buffer_ticks", "stop",
        "target_r", "cutoff", "direction", "max_trades", "breakeven_r", "time_exit",
    ]
    return [c for c in candidates if c in results.columns and results[c].nunique(dropna=False) > 1]

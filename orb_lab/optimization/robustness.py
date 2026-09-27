"""Parameter neighbourhoods: plateau vs. spike.

Two configurations are *immediate neighbours* when they differ in exactly one parameter and that
parameter moves one step:

* numeric axes (range, entry TF, R, cutoff, ORB start): adjacent values present in the searched space;
* stop: or_pct:0.50 - or_pct:0.67 - or_pct:0.75 - or_pct:1.00, or_mid - or_pct:0.50,
  or_mid - or_opposite, or_opposite - or_pct:1.00 (all measured in units of OR width);
* confirmation: close - 1tick - 2tick, close - 5pct - 10pct.

All other parameters (entry method, direction, ...) must match exactly.

A configuration whose neighbours perform dramatically worse is an optimisation spike, not a plateau.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine.params import StrategyParams
from .parameter_space import CONFIRMATION_PRESETS, confirmation_label, parse_stop, stop_label

NUMERIC_AXES = ("orb_start", "range_minutes", "entry_tf", "target_r", "cutoff")
STOP_EDGES = [
    ("or_pct:0.5", "or_pct:0.67"),
    ("or_pct:0.67", "or_pct:0.75"),
    ("or_pct:0.75", "or_pct:1"),
    ("or_mid", "or_pct:0.5"),
    ("or_mid", "or_opposite"),
    ("or_opposite", "or_pct:1"),
]
CONFIRM_EDGES = [("close", "1tick"), ("1tick", "2tick"), ("close", "5pct"), ("5pct", "10pct")]


def _adjacency(edges):
    adj: dict[str, set[str]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


STOP_ADJ = _adjacency(STOP_EDGES)
CONFIRM_ADJ = _adjacency(CONFIRM_EDGES)


def _key(params: StrategyParams) -> tuple:
    return tuple(sorted(params.canonical().to_dict().items(), key=lambda kv: kv[0]))


def neighbour_index(configs: list[StrategyParams], max_neighbours: int = 16) -> np.ndarray:
    """(n_configs, max_neighbours) int64 array of neighbour indices, -1 padded."""
    lookup = {_key(c): i for i, c in enumerate(configs)}
    values = {axis: sorted({getattr(c, axis) for c in configs}, key=lambda v: (str(type(v)), v)) for axis in NUMERIC_AXES}
    out = np.full((len(configs), max_neighbours), -1, dtype=np.int64)
    for i, cfg in enumerate(configs):
        found: list[int] = []
        base = cfg.to_dict()
        for axis in NUMERIC_AXES:
            vals = values[axis]
            if len(vals) < 2:
                continue
            pos = vals.index(getattr(cfg, axis))
            for step in (-1, 1):
                if 0 <= pos + step < len(vals):
                    cand = StrategyParams(**{**base, axis: vals[pos + step]})
                    j = lookup.get(_key(cand))
                    if j is not None and j != i:
                        found.append(j)
        for label in STOP_ADJ.get(stop_label(cfg.stop_method, cfg.stop_param), ()):
            method, param = parse_stop(label)
            j = lookup.get(_key(StrategyParams(**{**base, "stop_method": method, "stop_param": param})))
            if j is not None:
                found.append(j)
        for label in CONFIRM_ADJ.get(confirmation_label(cfg.confirm_ticks, cfg.confirm_or_frac), ()):
            ticks, frac = CONFIRMATION_PRESETS[label]
            j = lookup.get(_key(StrategyParams(**{**base, "confirm_ticks": ticks, "confirm_or_frac": frac})))
            if j is not None:
                found.append(j)
        found = sorted(set(found))[:max_neighbours]
        out[i, : len(found)] = found
    return out


def neighbour_stats(values: np.ndarray, nbrs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median, minimum and count of ``values`` over each configuration's neighbours (NaN if none)."""
    gathered = np.where(nbrs >= 0, values[np.clip(nbrs, 0, None)], np.nan)
    count = (nbrs >= 0).sum(axis=1)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(gathered, axis=1)
            mn = np.nanmin(gathered, axis=1)
    return med, mn, count


def classify_plateau(score: np.ndarray, nbr_median: np.ndarray, nbr_min: np.ndarray) -> np.ndarray:
    """``plateau`` if neighbours keep most of the score, ``spike`` if they collapse, ``isolated`` if none."""
    out = np.full(score.shape, "mixed", dtype=object)
    out[np.isnan(nbr_median)] = "isolated"
    pos = score > 0
    plateau = pos & (nbr_median >= 0.5 * score) & (nbr_min > 0)
    spike = pos & ((nbr_median <= 0.25 * score) | (nbr_median <= 0))
    out[plateau] = "plateau"
    out[spike & ~np.isnan(nbr_median)] = "spike"
    out[~pos & ~np.isnan(nbr_median)] = "not_positive"
    return out


def neighbour_table(configs: list[StrategyParams], nbrs: np.ndarray, i: int, metrics: pd.DataFrame) -> pd.DataFrame:
    """Human-readable neighbourhood of configuration ``i`` with the supplied per-config metrics."""
    idx = [i] + [j for j in nbrs[i] if j >= 0]
    rows = []
    for j in idx:
        c = configs[j]
        rows.append({"role": "candidate" if j == i else "neighbour", "range": c.range_minutes, "entry_tf": c.entry_tf,
                     "entry": c.entry_method, "confirm": confirmation_label(c.confirm_ticks, c.confirm_or_frac),
                     "stop": stop_label(c.stop_method, c.stop_param), "R": c.target_r, "cutoff": c.cutoff,
                     "direction": c.direction, **metrics.iloc[j].to_dict()})
    return pd.DataFrame(rows)


def grid_stability(results: pd.DataFrame, configs: list[StrategyParams], score_col: str = "avg_r") -> pd.DataFrame:
    """Neighbourhood stability of every grid-search row: neighbour median/min of ``score_col`` and a class.

    Returns a frame indexed like ``results`` with ``nbr_median``, ``nbr_min``, ``n_neighbours`` and ``stability``.
    """
    keys = [c.key() for c in configs]
    pos = {k: i for i, k in enumerate(keys)}
    values = np.full(len(configs), np.nan)
    for k, v in zip(results["config_key"], results[score_col]):
        if k in pos:
            values[pos[k]] = v
    nbrs = neighbour_index(configs)
    med, mn, cnt = neighbour_stats(values, nbrs)
    cls = classify_plateau(values, med, mn)
    idx = [pos.get(k, -1) for k in results["config_key"]]
    pick = lambda arr: [arr[i] if i >= 0 else np.nan for i in idx]  # noqa: E731
    return pd.DataFrame({"nbr_median": pick(med), "nbr_min": pick(mn), "n_neighbours": pick(cnt), "stability": pick(cls)},
                        index=results.index)

"""Causal front-month construction from individual-contract bars.

Rule (DATA_SOURCES.md): for session *d* trade the outright contract with the highest total volume in
session *d-1*. Rolls only move forward in expiry. The first session (no prior information) is
dropped. Each output session contains exactly one contract, at its raw traded prices (no
back-adjustment).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..engine.sessions import assign_sessions

MONTH_CODES = "FGHJKMNQUVXZ"


def outright_pattern(root: str) -> re.Pattern:
    return re.compile(rf"^{re.escape(root)}([{MONTH_CODES}])(\d{{1,2}})$")


def is_outright(symbol: str, root: str) -> bool:
    return bool(outright_pattern(root).match(str(symbol)))


def expiry_key(symbol: str, root: str, ref_year: int) -> int:
    """Sortable expiry (year*12 + month index) for an outright, resolving 1-digit years near ``ref_year``."""
    m = outright_pattern(root).match(str(symbol))
    if not m:
        raise ValueError(f"{symbol!r} is not an outright {root} contract")
    month = MONTH_CODES.index(m.group(1))
    digits = m.group(2)
    if len(digits) == 2:
        year = 2000 + int(digits)
    else:
        # one-digit year: the smallest year >= ref_year - 1 ending in that digit
        base = ref_year - 1
        year = base + ((int(digits) - base) % 10)
    return year * 12 + month


@dataclass
class RollResult:
    bars: pd.DataFrame  # UTC bar-start index, OHLCV + contract
    rolls: pd.DataFrame  # session_date, from_contract, to_contract, prev_session_volume_from, prev_session_volume_to
    dropped_spread_rows: int
    dropped_first_session: str | None
    sessions: int


def build_front_month(bars: pd.DataFrame, root: str, session_open: str = "18:00", tz: str = "America/New_York") -> RollResult:
    """``bars``: UTC-indexed rows for many contracts with a ``contract`` (raw symbol) column."""
    if "contract" not in bars.columns:
        raise ValueError("per-contract bars need a 'contract' column")
    mask = bars["contract"].astype(str).map(lambda s: is_outright(s, root)).to_numpy()
    dropped_spreads = int((~mask).sum())
    data = bars[mask].copy()
    if data.empty:
        raise ValueError(f"no outright {root} contracts in input")
    session_date, _ = assign_sessions(data.index, session_open, tz)
    data["_session"] = pd.to_datetime(session_date)
    pairs = data[["contract"]].assign(_year=data["_session"].dt.year).drop_duplicates()
    keys = {(c, y): expiry_key(c, root, int(y)) for c, y in zip(pairs["contract"], pairs["_year"])}
    data["_expiry"] = [keys[(c, y)] for c, y in zip(data["contract"], data["_session"].dt.year)]

    vol = data.groupby(["_session", "contract"])["volume"].sum().rename("vol").reset_index()
    expiry_of = data.groupby(["_session", "contract"])["_expiry"].first().reset_index()
    vol = vol.merge(expiry_of, on=["_session", "contract"])
    sessions = sorted(vol["_session"].unique())
    by_session = {sess: grp for sess, grp in vol.groupby("_session")}

    chosen: dict[pd.Timestamp, str] = {}
    roll_rows = []
    current, current_exp = None, -1
    prev_leader = None
    for k, sess in enumerate(sessions):
        if k == 0:
            prev_leader = by_session[sess].sort_values(["vol", "_expiry"], ascending=[False, True]).iloc[0]
            continue
        leader = prev_leader
        cand, cand_exp = leader["contract"], int(leader["_expiry"])
        if current is None or cand_exp > current_exp:
            if current is not None:
                prev_tbl = by_session[sessions[k - 1]].set_index("contract")["vol"]
                roll_rows.append(
                    {
                        "session_date": sess,
                        "from_contract": current,
                        "to_contract": cand,
                        "prev_session_volume_from": float(prev_tbl.get(current, 0.0)),
                        "prev_session_volume_to": float(prev_tbl.get(cand, 0.0)),
                    }
                )
            current, current_exp = cand, cand_exp
        chosen[sess] = current
        prev_leader = by_session[sess].sort_values(["vol", "_expiry"], ascending=[False, True]).iloc[0]

    pick = pd.Series(chosen, name="_chosen")
    data = data.join(pick, on="_session")
    out = data[data["contract"] == data["_chosen"]].drop(columns=["_session", "_expiry", "_chosen"])
    out = out.sort_index()
    return RollResult(
        bars=out,
        rolls=pd.DataFrame(roll_rows, columns=["session_date", "from_contract", "to_contract", "prev_session_volume_from", "prev_session_volume_to"]),
        dropped_spread_rows=dropped_spreads,
        dropped_first_session=str(pd.Timestamp(sessions[0]).date()) if sessions else None,
        sessions=len(chosen),
    )

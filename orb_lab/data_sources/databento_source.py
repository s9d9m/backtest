"""Databento (CME Globex MDP 3.0, dataset GLBX.MDP3) 1-minute bar integration.

* ``parent`` mode (default) requests every contract under ``ROOT.FUT``. Spreads are then removed and
  the front month is built causally with :func:`futures_roll.build_front_month`.
* ``continuous`` mode requests ``ROOT.v.0`` (volume lead, ranked on the previous day's volume, raw
  prices). It is cheaper, but the roll takes effect at Databento's daily boundary, which can fall
  inside a Globex session. Such sessions are then detected and excluded as intraday contract changes.

Raw vendor files are cached per calendar year under ``data/raw/databento/<ROOT>/`` so interrupted
downloads resume without paying twice. The API key is read from ``DATABENTO_API_KEY``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .futures_roll import build_front_month

DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1m"
DATASET_START = pd.Timestamp("2010-06-06")
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class DatabentoUnavailable(RuntimeError):
    pass


def get_client(key: str | None = None):
    try:
        import databento as db
    except ImportError as exc:  # pragma: no cover - dependency present in requirements
        raise DatabentoUnavailable("pip install databento") from exc
    key = key or os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise DatabentoUnavailable(
            "DATABENTO_API_KEY is not set. Create a key at databento.com and add it as an environment secret "
            "(see DATA_SOURCES.md)."
        )
    return db.Historical(key)


def request_symbols(root: str, mode: str) -> tuple[str, str]:
    if mode == "parent":
        return f"{root}.FUT", "parent"
    if mode == "continuous":
        return f"{root}.v.0", "continuous"
    raise ValueError("mode must be 'parent' or 'continuous'")


def year_chunks(start: str, end: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if s < DATASET_START:
        s = DATASET_START
    chunks = []
    cur = s
    while cur < e:
        nxt = min(pd.Timestamp(year=cur.year + 1, month=1, day=1), e)
        chunks.append((cur, nxt))
        cur = nxt
    return chunks


def estimate_cost(root: str, start: str, end: str, mode: str = "parent", client=None) -> dict:
    client = client or get_client()
    symbols, stype = request_symbols(root, mode)
    per_year = {}
    for a, b in year_chunks(start, end):
        per_year[str(a.year)] = float(
            client.metadata.get_cost(dataset=DATASET, symbols=symbols, schema=SCHEMA, stype_in=stype, start=a.isoformat(), end=b.isoformat())
        )
    return {"root": root, "mode": mode, "symbols": symbols, "per_year_usd": per_year, "total_usd": sum(per_year.values())}


def _to_frame(store) -> pd.DataFrame:
    frame = store.to_df(price_type="float", pretty_ts=True, map_symbols=True)
    if "ts_event" in frame.columns:
        frame = frame.set_index("ts_event")
    frame.index = pd.DatetimeIndex(frame.index)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    frame.index.name = "ts"
    cols = ["open", "high", "low", "close", "volume", "symbol", "instrument_id"]
    frame = frame[[c for c in cols if c in frame.columns]].rename(columns={"symbol": "contract"})
    return frame


@dataclass
class FetchResult:
    root: str
    mode: str
    front: pd.DataFrame
    rolls: pd.DataFrame
    provenance: dict = field(default_factory=dict)


def fetch(
    root: str,
    start: str,
    end: str,
    mode: str = "parent",
    client=None,
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
    session_open: str = "18:00",
) -> FetchResult:
    """Download (or reuse cached) raw bars, build the front-month series and write outputs."""
    raw_dir = raw_dir or DATA_DIR / "raw" / "databento" / root
    out_dir = out_dir or DATA_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols, stype = request_symbols(root, mode)
    frames, raw_files = [], []
    for a, b in year_chunks(start, end):
        cache = raw_dir / f"{mode}_{a:%Y%m%d}_{b:%Y%m%d}.parquet"
        if not cache.exists():
            client = client or get_client()
            store = client.timeseries.get_range(
                dataset=DATASET, symbols=symbols, schema=SCHEMA, stype_in=stype, start=a.isoformat(), end=b.isoformat()
            )
            part = _to_frame(store)
            tmp = cache.with_suffix(".tmp")
            part.to_parquet(tmp)
            os.replace(tmp, cache)
        frames.append(pd.read_parquet(cache))
        raw_files.append({"file": str(cache.relative_to(DATA_DIR.parent)) if cache.is_relative_to(DATA_DIR.parent) else str(cache),
                          "sha256": hashlib.sha256(cache.read_bytes()).hexdigest(), "rows": int(len(frames[-1]))})
    raw = pd.concat(frames).sort_index()
    if mode == "parent":
        result = build_front_month(raw, root, session_open=session_open)
        front, rolls = result.bars, result.rolls
        construction = {
            "type": "front month from individual contracts, causal previous-session-volume roll, forward-only, raw prices (no back-adjustment)",
            "dropped_spread_rows": result.dropped_spread_rows,
            "dropped_first_session": result.dropped_first_session,
            "sessions": result.sessions,
        }
    else:
        front = raw.copy()
        if "instrument_id" in front.columns:
            front["contract"] = front["instrument_id"].astype(str)
        rolls = pd.DataFrame()
        construction = {"type": "Databento continuous ROOT.v.0 (previous-day volume lead, raw prices); roll may fall inside a session -> such sessions are excluded"}
    front = front[["open", "high", "low", "close", "volume", "contract"]]
    out_file = out_dir / f"{root}_databento_front_1m.parquet"
    front.reset_index().rename(columns={"ts": "timestamp"}).to_parquet(out_file, index=False)
    roll_file = out_dir / f"{root}_databento_rolls.csv"
    rolls.to_csv(roll_file, index=False)
    provenance = {
        "vendor": "Databento",
        "dataset": DATASET,
        "schema": SCHEMA,
        "request_symbols": symbols,
        "stype_in": stype,
        "mode": mode,
        "start": start,
        "end": end,
        "timestamp_convention": "ts_event = bar open, UTC",
        "construction": construction,
        "raw_files": raw_files,
        "output_file": str(out_file),
        "output_sha256": hashlib.sha256(out_file.read_bytes()).hexdigest(),
        "output_rows": int(len(front)),
        "rolls": int(len(rolls)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / f"{root}_databento_provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    return FetchResult(root=root, mode=mode, front=front, rolls=rolls, provenance=provenance)

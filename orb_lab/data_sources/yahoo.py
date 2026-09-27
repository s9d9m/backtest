"""Yahoo Finance (via ``yfinance``) free intraday provider, PHASE0_FREE_PROXY only.

FREE PROXY EXPERIMENT — NOT FUTURES VALIDATION. SPY/QQQ are ETF proxies for the equity-index futures
part of the research. They are never labelled or treated as ES/NQ.

Nothing here assumes Yahoo's limits. :func:`probe_depth` measures them by requesting successive
chunks further back in time until Yahoo refuses or returns nothing, and records the exact responses.

Layout (``data/`` is git-ignored):

* ``data/raw/yahoo/<SYM>/<interval>_<start>_<end>.parquet``: raw yfinance output per request chunk
  (unmodified; index in America/New_York). Cached, so re-runs do not hit Yahoo again.
* ``data/phase0/<SYM>_<interval>.parquet``: processed canonical bars (UTC bar-start timestamp,
  unadjusted OHLCV).
* ``data/phase0/<SYM>_<interval>_provenance.json``: symbol, source, download time, requested and
  returned periods, interval, rows, timezone, SHA-256 of every raw and processed file.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PHASE0_LABEL = "FREE PROXY EXPERIMENT — NOT FUTURES VALIDATION"
EXPERIMENT_FAMILY = "PHASE0_FREE_PROXY"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
INTERVAL_MINUTES = {"1m": 1, "2m": 2, "5m": 5, "15m": 15}


class YahooUnavailable(RuntimeError):
    pass


def _ticker(symbol: str):
    import yfinance as yf

    return yf.Ticker(symbol)


def fetch_chunk(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp, prepost: bool = False, ticker=None) -> pd.DataFrame:
    """One raw request. Raises on Yahoo errors (``raise_errors=True``) so refusals are recorded, not hidden."""
    t = ticker or _ticker(symbol)
    frame = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval=interval, prepost=prepost,
                      auto_adjust=False, actions=False, raise_errors=True)
    return frame


def probe_depth(symbol: str, interval: str, chunk_days: int, max_lookback_days: int, today: pd.Timestamp | None = None,
                ticker=None) -> dict:
    """Walk back in ``chunk_days`` steps and record what Yahoo returns for each chunk."""
    today = (today or pd.Timestamp.now(tz="America/New_York")).normalize().tz_localize(None)
    end = today + pd.Timedelta(days=1)
    rows = []
    earliest, latest, refusals_in_a_row = None, None, 0
    while (today - end).days < max_lookback_days:
        start = end - pd.Timedelta(days=chunk_days)
        rec = {"request_start": str(start.date()), "request_end": str(end.date())}
        try:
            f = fetch_chunk(symbol, interval, start, end, ticker=ticker)
            rec.update({"rows": int(len(f)), "first": str(f.index.min()) if len(f) else None, "last": str(f.index.max()) if len(f) else None,
                        "error": None})
            if len(f):
                earliest = f.index.min() if earliest is None else min(earliest, f.index.min())
                latest = f.index.max() if latest is None else max(latest, f.index.max())
                refusals_in_a_row = 0
            else:
                refusals_in_a_row += 1
        except Exception as exc:  # Yahoo refusal messages are the measurement
            rec.update({"rows": 0, "first": None, "last": None, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            refusals_in_a_row += 1
        rows.append(rec)
        if refusals_in_a_row >= 2:
            break
        end = start
    return {"symbol": symbol, "interval": interval, "earliest_bar": str(earliest) if earliest is not None else None,
            "latest_bar": str(latest) if latest is not None else None, "chunks": rows}


def refine_earliest(symbol: str, interval: str, coarse_first: str, max_days_back: int = 20, ticker=None) -> str:
    """Step back one day at a time from the coarse earliest bar until Yahoo refuses the range."""
    earliest = pd.Timestamp(coarse_first)
    day = earliest.tz_localize(None).normalize()
    for k in range(1, max_days_back + 1):
        start = day - pd.Timedelta(days=k)
        try:
            f = fetch_chunk(symbol, interval, start, start + pd.Timedelta(days=2), ticker=ticker)
        except Exception as exc:
            if "within the last" in str(exc):
                break
            continue
        if len(f) and f.index.min() < earliest:
            earliest = f.index.min()
    return str(earliest)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(symbol: str, interval: str, start: str, end: str, chunk_days: int, ticker=None, data_dir: Path | None = None,
             yahoo_symbol: str | None = None, decimals: int = 4) -> dict:
    """Download [start, end) in chunks (cached raw files), then build the canonical processed file.

    ``symbol`` names the files; ``yahoo_symbol`` is what Yahoo is asked for (e.g. ``6E`` -> ``6E=F``).
    ``decimals`` must resolve the instrument's tick (4 for $0.01 stocks, 6 for 6E's 0.00005).
    """
    if ticker is None and yahoo_symbol:
        ticker = _ticker(yahoo_symbol)
    data_dir = data_dir or DATA_DIR
    raw_dir = data_dir / "raw" / "yahoo" / symbol
    out_dir = data_dir / "phase0"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    frames, raw_files, errors = [], [], []
    cur = s
    while cur < e:
        nxt = min(cur + pd.Timedelta(days=chunk_days), e)
        cache = raw_dir / f"{interval}_{cur:%Y%m%d}_{nxt:%Y%m%d}.parquet"
        if not cache.exists():
            try:
                f = fetch_chunk(symbol, interval, cur, nxt, ticker=ticker)
            except Exception as exc:
                errors.append({"chunk": f"{cur.date()}..{nxt.date()}", "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
                cur = nxt
                continue
            tmp = cache.with_suffix(".tmp")
            f.to_parquet(tmp)
            os.replace(tmp, cache)
        f = pd.read_parquet(cache)
        if len(f):
            frames.append(f)
            raw_files.append({"file": cache.name, "rows": int(len(f)), "sha256": _sha(cache)})
        cur = nxt
    if not frames:
        raise YahooUnavailable(f"{symbol} {interval}: Yahoo returned no bars for {start}..{end}; errors: {errors[:3]}")
    raw = pd.concat(frames)
    tzname = str(raw.index.tz)
    # Rounding only removes float representation noise (e.g. 766.590027 -> 766.5900 at 4 decimals); genuine
    # sub-tick prints are kept. 6E needs 6 decimals for its 0.00005 tick.
    processed = pd.DataFrame(
        {"timestamp": raw.index.tz_convert("UTC"), **{c.lower(): raw[c].to_numpy().round(decimals) for c in ("Open", "High", "Low", "Close")},
         "volume": raw["Volume"].to_numpy()}
    )
    out = out_dir / f"{symbol}_{interval}.parquet"
    processed.to_parquet(out, index=False)
    import yfinance

    prov = {
        "experiment_family": EXPERIMENT_FAMILY,
        "label": PHASE0_LABEL,
        "symbol": symbol,
        "yahoo_symbol": yahoo_symbol or symbol,
        "source": f"Yahoo Finance via yfinance {yfinance.__version__} (free, no key, no account)",
        "download_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": start,
        "requested_end": end,
        "returned_first_bar": str(raw.index.min()),
        "returned_last_bar": str(raw.index.max()),
        "interval": interval,
        "rows_raw": int(len(raw)),
        "rows_unique_timestamps": int(raw.index.nunique()),
        "source_timezone": tzname,
        "prices": "unadjusted (auto_adjust=False); regular trading hours only (prepost=False)",
        "timestamp_convention": "bar start",
        "raw_files": raw_files,
        "chunk_errors": errors,
        "processed_file": str(out.relative_to(data_dir.parent)) if out.is_relative_to(data_dir.parent) else str(out),
        "processed_sha256": _sha(out),
    }
    (out_dir / f"{symbol}_{interval}_provenance.json").write_text(json.dumps(prov, indent=2))
    return prov

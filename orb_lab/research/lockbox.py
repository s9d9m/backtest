"""Final lockbox: the most recent N months of each instrument, withheld from all development.

Life cycle (recorded in ``research/lockbox.json`` and the registry):

1. **define/seal**: the first time an instrument's real data passes through the research pipeline,
   its lockbox start is fixed at ``last_session - months``. It is immutable afterwards; adding newer
   data later does not move it.
2. **development**: :func:`apply_lockbox` returns a dataset truncated before the lockbox start. No
   strategy result for lockbox sessions can be produced through the normal pipeline.
3. **unlock**: only with a frozen candidate file (its SHA-256 is recorded). It can be done once.
4. **single run**: :func:`run_lockbox_test` evaluates the frozen candidate exactly once and stores
   the result. Later calls return the stored result instead of re-running.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..engine.session_data import PreparedData
from .registry import log_event, research_dir


class LockboxError(RuntimeError):
    pass


def _ledger_path() -> Path:
    return research_dir() / "lockbox.json"


def load_ledger() -> dict:
    path = _ledger_path()
    return json.loads(path.read_text()) if path.exists() else {}


def _save_ledger(ledger: dict) -> None:
    _ledger_path().write_text(json.dumps(ledger, indent=2, default=str))


def define_lockbox(symbol: str, prep: PreparedData, months: int = 12, min_history_years: float = 5.0) -> dict:
    """Seal the lockbox for ``symbol`` if not already defined. Returns the ledger entry."""
    ledger = load_ledger()
    if symbol in ledger:
        return ledger[symbol]
    if prep.n_days == 0:
        raise LockboxError("no sessions")
    first, last = pd.Timestamp(prep.dates[0]), pd.Timestamp(prep.dates[-1])
    start = (last - pd.DateOffset(months=months)) + pd.Timedelta(days=1)
    dev_years = (start - first).days / 365.25
    entry = {
        "symbol": symbol,
        "lockbox_months": months,
        "lockbox_start": str(start.date()),
        "data_first_session": str(first.date()),
        "data_last_session_at_definition": str(last.date()),
        "development_years": round(dev_years, 2),
        "data_hash_at_definition": prep.data_hash,
        "defined_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sealed" if dev_years >= min_history_years else "not_used_insufficient_history",
        "unlocked_utc": None,
        "frozen_candidate_sha256": None,
        "result_file": None,
    }
    if entry["status"] != "sealed":
        entry["lockbox_start"] = None
    ledger[symbol] = entry
    _save_ledger(ledger)
    log_event("lockbox_define", **entry)
    return entry


def subset_before(prep: PreparedData, cutoff_date) -> PreparedData:
    """A copy of ``prep`` containing only sessions strictly before ``cutoff_date``."""
    cut = np.datetime64(pd.Timestamp(cutoff_date).date(), "D")
    n = int(np.searchsorted(prep.dates, cut, "left"))
    end_bar = int(prep.day_end[n - 1]) if n > 0 else 0
    daily = prep.daily.iloc[:n].copy()
    return PreparedData(
        instrument=prep.instrument,
        base_minutes=prep.base_minutes,
        dates=prep.dates[:n].copy(),
        close_min=prep.close_min[:n].copy(),
        early_close=prep.early_close[:n].copy(),
        session_exit_min=prep.session_exit_min[:n].copy(),
        day_start=prep.day_start[:n].copy(),
        day_end=prep.day_end[:n].copy(),
        tod=prep.tod[:end_bar].copy(),
        open=prep.open[:end_bar].copy(),
        high=prep.high[:end_bar].copy(),
        low=prep.low[:end_bar].copy(),
        close=prep.close[:end_bar].copy(),
        volume=prep.volume[:end_bar].copy(),
        daily=daily,
        excluded=prep.excluded[pd.to_datetime(prep.excluded["session_date"]) < pd.Timestamp(cutoff_date)].copy(),
        data_hash=prep.data_hash,
        min_or_coverage=prep.min_or_coverage,
    )


def apply_lockbox(symbol: str, prep: PreparedData) -> tuple[PreparedData, dict | None]:
    """Development view: remove sealed lockbox sessions. Raises if the lockbox was already consumed."""
    entry = load_ledger().get(symbol)
    if entry is None or entry["status"] == "not_used_insufficient_history":
        return prep, entry
    return subset_before(prep, entry["lockbox_start"]), entry


def unlock(symbol: str, frozen_candidate_file: str | Path) -> dict:
    ledger = load_ledger()
    entry = ledger.get(symbol)
    if entry is None or entry["status"] != "sealed":
        raise LockboxError(f"{symbol}: lockbox is not in the sealed state ({entry and entry['status']})")
    path = Path(frozen_candidate_file)
    if not path.exists():
        raise LockboxError("a frozen candidate file (parameters / selection rule) is required to unlock")
    entry["frozen_candidate_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    entry["frozen_candidate_file"] = str(path)
    entry["status"] = "unlocked"
    entry["unlocked_utc"] = datetime.now(timezone.utc).isoformat()
    ledger[symbol] = entry
    _save_ledger(ledger)
    log_event("lockbox_unlock", symbol=symbol, frozen_candidate_sha256=entry["frozen_candidate_sha256"])
    return entry


def run_lockbox_test(symbol: str, full_prep: PreparedData, evaluate) -> dict:
    """Run ``evaluate(prep, start_date)`` on the lockbox exactly once; afterwards return the stored result."""
    ledger = load_ledger()
    entry = ledger.get(symbol)
    if entry is None or entry["status"] not in ("unlocked", "consumed"):
        raise LockboxError("unlock the lockbox with a frozen candidate first")
    if entry["status"] == "consumed":
        return json.loads(Path(entry["result_file"]).read_text())
    frozen = Path(entry["frozen_candidate_file"])
    if hashlib.sha256(frozen.read_bytes()).hexdigest() != entry["frozen_candidate_sha256"]:
        raise LockboxError("frozen candidate file changed after unlocking")
    result = evaluate(full_prep, entry["lockbox_start"])
    out = research_dir() / f"lockbox_result_{symbol}.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    entry["status"] = "consumed"
    entry["result_file"] = str(out)
    ledger[symbol] = entry
    _save_ledger(ledger)
    log_event("lockbox_consumed", symbol=symbol, result_file=str(out))
    return result

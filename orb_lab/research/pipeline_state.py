"""Research pipeline state for the dashboard: split, frozen candidates and the blind-holdout ledger.

DATA -> BACKTEST -> OPTIMIZE -> VALIDATE -> FREEZE -> BLIND OOS / WALK-FORWARD -> ROBUSTNESS -> MONTE CARLO -> REPORT

* **Split** (per dataset hash): sessions are divided chronologically into train / validation / blind
  holdout. Once a split is saved, the dashboard's development tools only receive sessions *before* the
  holdout, so the holdout cannot influence optimisation, validation or candidate choice. The split is
  saved to disk and re-applied when the same data is loaded again; changing it after the holdout was
  used is recorded.
* **Freeze**: a candidate's parameters, execution and sizing assumptions are written to an immutable
  JSON file whose name contains the SHA-256 of its content.
* **Blind test**: a frozen candidate is evaluated once on the holdout. Every run is appended to a
  ledger. The first blind test per dataset is labelled ``BLIND``; any later one is labelled
  ``NOT BLIND (holdout already used)`` because the researcher has seen holdout results.

This is complementary to the futures lockbox (``lockbox.py``), which is applied before any of this.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..reports.experiment import RUNS_DIR


def _root() -> Path:
    return Path(os.environ.get("ORB_PIPELINE_DIR", RUNS_DIR / "pipeline"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def data_dir(data_hash: str) -> Path:
    d = _root() / str(data_hash)[:16]
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass(frozen=True)
class Split:
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    holdout_start: str
    holdout_end: str
    n_train: int
    n_val: int
    n_holdout: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def make_split(dates, train: float = 0.6, val: float = 0.2) -> Split:
    """Chronological split of session dates. Every part gets at least one session."""
    dates = pd.DatetimeIndex(np.asarray(dates, dtype="datetime64[D]").astype("datetime64[ns]"))
    n = len(dates)
    if n < 3:
        raise ValueError("need at least 3 sessions to split into train / validation / holdout")
    if not (0 < train < 1 and 0 < val < 1 and train + val < 1):
        raise ValueError("train and validation shares must be positive and sum to less than 1")
    n_tr = max(1, int(round(n * train)))
    n_va = max(1, int(round(n * val)))
    n_tr = min(n_tr, n - 2)
    n_va = min(n_va, n - n_tr - 1)
    d = [str(x.date()) for x in dates]
    return Split(d[0], d[n_tr - 1], d[n_tr], d[n_tr + n_va - 1], d[n_tr + n_va], d[-1], n_tr, n_va, n - n_tr - n_va)


def save_split(data_hash: str, split: Split) -> dict:
    f = data_dir(data_hash) / "split.json"
    history = []
    if f.exists():
        old = json.loads(f.read_text())
        history = old.get("history", []) + [{k: v for k, v in old.items() if k != "history"}]
    ledger = read_blind_tests(data_hash)
    payload = {**split.to_dict(), "saved_utc": _now(), "holdout_already_used": bool(ledger), "history": history}
    f.write_text(json.dumps(payload, indent=2))
    return payload


def load_split(data_hash: str) -> dict | None:
    f = data_dir(data_hash) / "split.json"
    return json.loads(f.read_text()) if f.exists() else None


def params_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def freeze_candidate(data_hash: str, name: str, strategy: dict, execution: dict, sizing: dict, instrument: dict, evidence: dict | None = None,
                     development_end: str | None = None) -> dict:
    """Write an immutable frozen-candidate file. Returns the record (with ``sha256`` and ``path``)."""
    core = {"strategy": strategy, "execution": execution, "sizing": sizing, "instrument": instrument}
    digest = params_hash(core)
    record = {"name": name, "sha256": digest, "frozen_utc": _now(), "data_hash": data_hash, "development_end": development_end,
              **core, "evidence_at_freeze": evidence or {}}
    d = data_dir(data_hash) / "frozen"
    d.mkdir(exist_ok=True)
    path = d / f"{digest[:16]}.json"
    if not path.exists():  # identical content -> same file; never overwritten
        path.write_text(json.dumps(record, indent=2, default=str))
        path.chmod(0o444)
    record = json.loads(path.read_text())
    record["path"] = str(path)
    return record


def list_frozen(data_hash: str) -> list[dict]:
    d = data_dir(data_hash) / "frozen"
    out = []
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        rec = json.loads(f.read_text())
        rec["path"] = str(f)
        rec["verified"] = params_hash({k: rec[k] for k in ("strategy", "execution", "sizing", "instrument")}) == rec["sha256"]
        out.append(rec)
    return sorted(out, key=lambda r: r["frozen_utc"])


def read_blind_tests(data_hash: str) -> list[dict]:
    f = data_dir(data_hash) / "blind_tests.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


def record_blind_test(data_hash: str, frozen: dict, holdout: tuple[str, str], metrics: dict) -> dict:
    """Append a blind-test result. The first test on this dataset is BLIND; later ones are not."""
    previous = read_blind_tests(data_hash)
    status = "BLIND" if not previous else "NOT BLIND (holdout already used)"
    entry = {"utc": _now(), "status": status, "candidate": frozen["name"], "sha256": frozen["sha256"], "holdout_start": holdout[0],
             "holdout_end": holdout[1], "previous_tests": len(previous), "metrics": metrics}
    with open(data_dir(data_hash) / "blind_tests.jsonl", "a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    return entry


def verify_frozen(record: dict) -> bool:
    return params_hash({k: record[k] for k in ("strategy", "execution", "sizing", "instrument")}) == record["sha256"]

"""Append-only research registry (anti-cherry-picking record).

Every experiment, parameter space, walk-forward configuration, hypothesis and lockbox event is
appended to ``research/registry.jsonl`` with a UTC timestamp. Entries are never edited. A
hypothesis's status changes by appending a new event that references it.

Hypothesis statuses:

* ``PRE_REGISTERED``: stated before the data it will be tested on was examined.
* ``EXPLORATORY``: discovered after looking at OOS (or any) results. It **cannot** be called
  confirmed until it passes on subsequent untouched data.
* ``CONFIRMED_OOS``: passed a test on data that was untouched when the hypothesis was registered.
* ``REJECTED``: failed its test.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
HYPOTHESIS_STATUSES = ("PRE_REGISTERED", "EXPLORATORY", "CONFIRMED_OOS", "REJECTED")


def research_dir() -> Path:
    path = Path(os.environ.get("ORB_RESEARCH_DIR", REPO_ROOT / "research"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _registry_path() -> Path:
    return research_dir() / "registry.jsonl"


def log_event(kind: str, **details: Any) -> dict:
    entry = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "event_id": uuid.uuid4().hex[:12],
        "kind": kind,
        **details,
    }
    with open(_registry_path(), "a") as handle:
        handle.write(json.dumps(entry, default=str, sort_keys=True) + "\n")
    return entry


def read_events(kind: str | None = None) -> list[dict]:
    path = _registry_path()
    if not path.exists():
        return []
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [e for e in events if kind is None or e["kind"] == kind]


def add_hypothesis(statement: str, *, status: str, source: str, data_seen: str, test_plan: str = "") -> dict:
    """Register a hypothesis. ``data_seen`` states which data had been examined when it was formed."""
    if status not in ("PRE_REGISTERED", "EXPLORATORY"):
        raise ValueError("new hypotheses are PRE_REGISTERED or EXPLORATORY")
    return log_event(
        "hypothesis",
        hypothesis_id=f"H-{uuid.uuid4().hex[:8]}",
        statement=statement,
        status=status,
        source=source,
        data_seen=data_seen,
        test_plan=test_plan,
    )


def update_hypothesis(hypothesis_id: str, status: str, evidence: str, test_data: str) -> dict:
    if status not in HYPOTHESIS_STATUSES:
        raise ValueError(f"status must be one of {HYPOTHESIS_STATUSES}")
    current = hypotheses().get(hypothesis_id)
    if current is None:
        raise KeyError(hypothesis_id)
    if status == "CONFIRMED_OOS" and not test_data:
        raise ValueError("confirmation requires naming the untouched test data")
    return log_event("hypothesis_update", hypothesis_id=hypothesis_id, status=status, evidence=evidence, test_data=test_data)


def hypotheses() -> dict[str, dict]:
    """Current state of every hypothesis (latest status wins; full history kept in the log)."""
    state: dict[str, dict] = {}
    for e in read_events():
        if e["kind"] == "hypothesis":
            state[e["hypothesis_id"]] = {**e, "history": [e["status"]]}
        elif e["kind"] == "hypothesis_update" and e["hypothesis_id"] in state:
            state[e["hypothesis_id"]]["status"] = e["status"]
            state[e["hypothesis_id"]]["history"].append(e["status"])
    return state

"""Research-mode gates that enforce the agreed order of work on real data.

1. A data-quality report must exist for the exact data file (by SHA-256) with verdict PASS or
   REVIEW before any optimisation may run. STOP blocks the instrument.
2. The DQ step seals the 12-month lockbox (if at least 5 years of development data remain).
3. Every optimisation dataset is the lockbox-truncated development view.
4. Every run is appended to the registry.

Synthetic or exploratory runs may bypass the gates only with ``research=False``. Such runs are
never written to the registry and must not be quoted as evidence.
"""

from __future__ import annotations

from pathlib import Path

from ..engine.instruments import Instrument
from ..engine.pipeline import Dataset, build_dataset
from ..reports.dq_report import evaluate, write_report
from .lockbox import apply_lockbox, define_lockbox
from .registry import log_event, read_events, research_dir


class ResearchGateError(RuntimeError):
    pass


def run_dq_stage(path: str | Path, instrument: Instrument, *, source_tz=None, convention="start", lockbox_months: int = 12,
                 provenance: dict | None = None) -> tuple[Dataset, object, Path, dict | None]:
    ds = build_dataset(path, instrument, source_tz=source_tz, timestamp_convention=convention, allow_errors=True)
    rdq = evaluate(ds)
    report = write_report(rdq, ds, research_dir() / "data_quality", provenance)
    lock = None
    if rdq.verdict != "STOP":
        lock = define_lockbox(instrument.symbol, ds.prep, months=lockbox_months)
    log_event("dq_report", symbol=instrument.symbol, data_hash=ds.loaded.file_hash, source=str(path), verdict=rdq.verdict,
              reasons=rdq.reasons[:20], report=str(report), lockbox_start=lock and lock.get("lockbox_start"))
    return ds, rdq, report, lock


def research_dataset(path: str | Path, instrument: Instrument, *, source_tz=None, convention="start") -> tuple[Dataset, dict]:
    """Development dataset for optimisation: DQ-gated and lockbox-truncated."""
    ds = build_dataset(path, instrument, source_tz=source_tz, timestamp_convention=convention)
    reports = [e for e in read_events("dq_report") if e["symbol"] == instrument.symbol and e["data_hash"] == ds.loaded.file_hash]
    if not reports:
        raise ResearchGateError(f"{instrument.symbol}: no data-quality report for this exact file. Run `cli dq-report` first.")
    if reports[-1]["verdict"] == "STOP":
        raise ResearchGateError(f"{instrument.symbol}: data-quality verdict STOP. Research on this instrument is halted: {reports[-1]['reasons'][:3]}")
    dev_prep, lock = apply_lockbox(instrument.symbol, ds.prep)
    if lock is None:
        raise ResearchGateError(f"{instrument.symbol}: lockbox not defined (the DQ stage defines it)")
    ds.prep = dev_prep
    return ds, lock

"""Experiment manifests: everything needed to rerun a result exactly.

Each run gets a unique ``experiment_id`` and a directory under ``runs/`` containing ``manifest.json``
(parameters, instrument spec, data file + hash, costs, software versions, seed, WFO structure,
objective) plus result files.
"""

from __future__ import annotations

import json
import platform
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .. import __version__

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


def new_experiment_id(kind: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{kind}-{stamp}-{uuid.uuid4().hex[:6]}"


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "pandas", "numba", "pyarrow", "exchange_calendars", "plotly", "streamlit"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "?")
        except Exception:
            versions[name] = "not installed"
    return versions


@dataclass
class ExperimentManifest:
    experiment_id: str
    kind: str  # backtest | grid | wfo | ...
    created_utc: str
    software_version: str
    versions: dict[str, str]
    data: dict[str, Any]
    instrument: dict[str, Any]
    strategy: dict[str, Any] | None = None
    search_space: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None
    sizing: dict[str, Any] | None = None
    costs: dict[str, Any] | None = None
    date_range: dict[str, Any] | None = None
    wfo: dict[str, Any] | None = None
    objective: dict[str, Any] | None = None
    random_seed: int | None = None
    n_configs: int | None = None
    notes: list[str] = field(default_factory=list)

    @classmethod
    def create(cls, kind: str, **kwargs) -> "ExperimentManifest":
        return cls(
            experiment_id=new_experiment_id(kind),
            kind=kind,
            created_utc=datetime.now(timezone.utc).isoformat(),
            software_version=__version__,
            versions=package_versions(),
            **kwargs,
        )

    def run_dir(self, root: Path | None = None) -> Path:
        path = (root or RUNS_DIR) / self.experiment_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, root: Path | None = None) -> Path:
        path = self.run_dir(root) / "manifest.json"
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentManifest":
        path = Path(path)
        if path.is_dir():
            path = path / "manifest.json"
        return cls(**json.loads(path.read_text()))


def save_backtest(result, dataset, instrument, root: Path | None = None, notes: list[str] | None = None) -> Path:
    """Persist a single backtest (manifest + trade log + daily equity + metrics)."""
    manifest = ExperimentManifest.create(
        "backtest",
        data=dataset.describe(),
        instrument=instrument.to_dict(),
        strategy=result.params.to_dict(),
        execution=result.execution.to_dict(),
        sizing=result.sizing.to_dict(),
        costs=asdict(result.costs),
        date_range={
            "start": str(result.daily.index.min().date()) if len(result.daily) else None,
            "end": str(result.daily.index.max().date()) if len(result.daily) else None,
        },
        notes=notes or [],
    )
    run_dir = manifest.run_dir(root)
    manifest.save(root)
    result.trades.to_csv(run_dir / "trades.csv", index=False)
    result.daily.to_csv(run_dir / "daily.csv")
    metrics = {"net": result.metrics, "gross": result.metrics_gross, "long": result.metrics_long, "short": result.metrics_short,
               "diagnostics": result.diagnostics}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return run_dir


def rerun_backtest(manifest_path: str | Path, data_path: str | Path | None = None):
    """Reproduce a saved backtest. Refuses to run if the data file hash differs from the manifest."""
    from ..engine.backtest import run_backtest
    from ..engine.data_quality import CleaningPolicy
    from ..engine.instruments import Instrument
    from ..engine.params import ExecutionParams, SizingParams, StrategyParams
    from ..engine.pipeline import build_dataset

    manifest = ExperimentManifest.load(manifest_path)
    data = manifest.data
    path = Path(data_path or data["source"])
    instrument = Instrument(**manifest.instrument)
    dataset = build_dataset(
        path,
        instrument,
        source_tz=data.get("source_tz"),
        timestamp_convention=data.get("timestamp_convention", "start"),
        policy=CleaningPolicy(**data.get("cleaning_policy", {})),
        allow_errors=bool(data.get("allow_errors", False)),
    )
    if dataset.loaded.file_hash != data["file_hash"]:
        raise ValueError(f"Data hash mismatch: manifest {data['file_hash'][:12]} vs file {dataset.loaded.file_hash[:12]}")
    dr = manifest.date_range or {}
    return run_backtest(
        dataset.prep,
        StrategyParams.from_dict(manifest.strategy),
        ExecutionParams.from_dict(manifest.execution),
        SizingParams.from_dict(manifest.sizing),
        start=dr.get("start"),
        end=dr.get("end"),
    )


def metrics_table(metrics: dict, gross: dict | None = None) -> pd.DataFrame:
    rows = [
        ("Total return", "total_return", "pct"), ("CAGR", "cagr", "pct"), ("Annualized volatility", "ann_vol", "pct"),
        ("Sharpe", "sharpe", "f2"), ("Sortino", "sortino", "f2"), ("Calmar", "calmar", "f2"),
        ("Max drawdown", "max_dd", "pct"), ("Max DD duration (sessions)", "max_dd_duration_days", "int"),
        ("Profit factor", "profit_factor", "f2"), ("Expectancy ($/trade)", "expectancy", "usd"),
        ("Win rate", "win_rate", "pct"), ("Loss rate", "loss_rate", "pct"),
        ("Average winner", "avg_winner", "usd"), ("Average loser", "avg_loser", "usd"),
        ("Average R/trade", "avg_r", "f3"), ("Median R/trade", "median_r", "f3"), ("t-stat (mean R)", "t_stat_r", "f2"),
        ("Payoff ratio", "payoff_ratio", "f2"), ("Number of trades", "n_trades", "int"), ("Trades/year", "trades_per_year", "f1"),
        ("Longest winning streak", "longest_win_streak", "int"), ("Longest losing streak", "longest_loss_streak", "int"),
        ("Best trade", "best_trade", "usd"), ("Worst trade", "worst_trade", "usd"),
        ("Average holding (min)", "avg_hold_min", "f1"), ("Median holding (min)", "median_hold_min", "f1"),
        ("Commission paid", "commission_paid", "usd"), ("Estimated slippage cost", "slippage_cost", "usd"),
        ("Net P&L", "net_pnl", "usd"), ("P&L excluding top-5 trades", "pnl_ex_top5", "usd"),
        ("Top-5 trades share of profit", "top5_share", "pct"), ("Largest single-year share of profit", "max_year_share", "pct"),
        ("Share of years profitable", "pct_years_profitable", "pct"),
    ]

    def fmt(value, kind):
        if value is None or (isinstance(value, float) and not pd.notna(value)):
            return "n/a"
        if kind == "pct":
            return f"{value * 100:,.2f}%"
        if kind == "usd":
            return f"${value:,.2f}"
        if kind == "int":
            return f"{int(value):,}"
        return f"{value:,.{int(kind[1])}f}"

    data = {"Metric": [r[0] for r in rows], "Net": [fmt(metrics.get(r[1]), r[2]) for r in rows]}
    if gross is not None:
        data["Gross (before costs)"] = [fmt(gross.get(r[1]), r[2]) for r in rows]
    return pd.DataFrame(data)

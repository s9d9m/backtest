"""Instrument specifications loaded from ``config/instruments.yaml``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@dataclass(frozen=True)
class Instrument:
    symbol: str
    name: str
    tick_size: float
    tick_value: float
    multiplier: float
    commission_per_side: float
    exchange_fee_per_side: float
    slippage_ticks: float
    timezone: str = "America/New_York"
    calendar: str = "XNYS"
    session_open: str = "18:00"
    session_close: str = "17:00"
    session_exit: str = "15:55"
    early_close_exit_buffer_min: int = 5
    analysis_window_start: str = "08:00"
    min_qty: float = 1
    qty_step: float = 1
    currency: str = "USD"

    def __post_init__(self) -> None:
        if self.tick_size <= 0 or self.tick_value <= 0 or self.multiplier <= 0:
            raise ValueError(f"{self.symbol}: tick_size, tick_value and multiplier must be positive")
        implied = self.tick_size * self.multiplier
        if abs(implied - self.tick_value) > 1e-6 * max(1.0, self.tick_value):
            raise ValueError(
                f"{self.symbol}: tick_value {self.tick_value} != tick_size*multiplier {implied}; check the spec"
            )
        if self.min_qty <= 0 or self.qty_step <= 0:
            raise ValueError(f"{self.symbol}: min_qty and qty_step must be positive")

    @property
    def point_value(self) -> float:
        return self.multiplier

    @property
    def round_trip_fixed_cost(self) -> float:
        """Commission + exchange fees for one contract, both sides."""
        return 2.0 * (self.commission_per_side + self.exchange_fee_per_side)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_instruments(path: str | Path | None = None) -> dict[str, Instrument]:
    path = Path(path) if path else CONFIG_DIR / "instruments.yaml"
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    defaults = raw.get("defaults", {}) or {}
    allowed = {f.name for f in fields(Instrument)}
    out: dict[str, Instrument] = {}
    for symbol, spec in (raw.get("instruments") or {}).items():
        merged = {**defaults, **(spec or {}), "symbol": str(symbol)}
        unknown = set(merged) - allowed
        if unknown:
            raise ValueError(f"Unknown instrument keys for {symbol}: {sorted(unknown)}")
        out[str(symbol)] = Instrument(**merged)
    return out


def get_instrument(symbol: str, path: str | Path | None = None) -> Instrument:
    instruments = load_instruments(path)
    if symbol not in instruments:
        raise KeyError(f"Instrument {symbol!r} not configured. Available: {sorted(instruments)}")
    return instruments[symbol]

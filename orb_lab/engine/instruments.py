"""Instrument specifications loaded from ``config/instruments.yaml``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
ASSET_CLASSES = ("future", "etf")
UNITS = {"future": "contract", "etf": "share"}


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
    asset_class: str = "future"          # "future" (sized in contracts) | "etf" (sized in shares)
    friction_per_side: float = 0.0       # modelled bid/ask friction, $ per unit (contract/share) per fill
    max_notional_leverage: float = 0.0   # position notional <= equity x this (0 = no cap)

    def __post_init__(self) -> None:
        if self.asset_class not in ASSET_CLASSES:
            raise ValueError(f"{self.symbol}: asset_class must be one of {ASSET_CLASSES}")
        if min(self.commission_per_side, self.exchange_fee_per_side, self.friction_per_side, self.slippage_ticks) < 0:
            raise ValueError(f"{self.symbol}: costs must be >= 0")
        if self.max_notional_leverage < 0:
            raise ValueError(f"{self.symbol}: max_notional_leverage must be >= 0")
        if self.asset_class == "future" and (self.qty_step != int(self.qty_step) or self.min_qty < 1):
            raise ValueError(f"{self.symbol}: futures trade in whole contracts (min_qty >= 1, integer qty_step)")
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
    def unit(self) -> str:
        """Unit of quantity: 'contract' for futures, 'share' for ETFs. Never interchangeable."""
        return UNITS[self.asset_class]

    @property
    def fixed_cost_per_side(self) -> float:
        """$ per unit per fill: commission + exchange/regulatory fees + modelled friction."""
        return self.commission_per_side + self.exchange_fee_per_side + self.friction_per_side

    @property
    def point_value(self) -> float:
        return self.multiplier

    @property
    def round_trip_fixed_cost(self) -> float:
        """Commission + exchange fees + friction for one unit (contract or share), both sides, excluding slippage."""
        return 2.0 * self.fixed_cost_per_side

    def cost_description(self) -> str:
        """Human-readable cost schedule with explicit units."""
        u = self.unit
        return (f"commission ${self.commission_per_side:g}/{u}/side · exchange+regulatory fees ${self.exchange_fee_per_side:g}/{u}/side · "
                f"modelled friction ${self.friction_per_side:g}/{u}/side · slippage {self.slippage_ticks:g} tick(s)/side on market & stop fills "
                f"(1 tick = {self.tick_size:g} price = ${self.tick_value:g}/{u})")

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

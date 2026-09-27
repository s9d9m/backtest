"""Typed parameter objects for strategies, execution assumptions and position sizing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml

from .sessions import hhmm_to_minutes

ENTRY_METHODS = ("market", "limit", "stop")
STOP_METHODS = ("or_opposite", "or_mid", "or_pct", "atr", "fixed_ticks")
DIRECTIONS = ("both", "long", "short")
REENTRY_POLICIES = ("any", "opposite_after_loss", "same_after_loss", "either_after_loss")
FILL_MODELS = ("standard", "conservative")
AMBIGUITY_MODES = ("optimistic", "pessimistic", "conservative")
SIZING_MODES = ("fixed_contracts", "fixed_risk", "pct_equity")

#: Hard cap on trades per session in "unlimited" mode (max_trades = 0).
UNLIMITED_TRADES_CAP = 32


class ParamError(ValueError):
    pass


@dataclass(frozen=True)
class StrategyParams:
    """Parameters of the basic-breakout ORB family.

    Semantics (full detail in RESEARCH_NOTES.md):

    * The opening range covers bars whose start time is in ``[orb_start, orb_start + range_minutes)``.
    * ``entry_method="market"``: an entry-timeframe bar must *close* beyond ``boundary + offset``
      (offset = ``confirm_ticks`` ticks + ``confirm_or_frac`` * OR width); fill at the open of the next
      base bar plus slippage.
    * ``entry_method="limit"``: same close confirmation, then a limit order at ``boundary +
      entry_buffer_ticks`` (in the breakout direction) waits for a retrace.
    * ``entry_method="stop"``: an intrabar stop order rests at the first tick strictly beyond
      ``boundary + offset``, plus ``entry_buffer_ticks``. Intrabar confirmation; ``entry_tf`` unused.
    """

    orb_start: str = "09:30"
    range_minutes: int = 15
    entry_tf: int = 5
    entry_method: str = "market"
    confirm_ticks: float = 0.0
    confirm_or_frac: float = 0.0
    entry_buffer_ticks: int = 0
    stop_method: str = "or_opposite"
    stop_param: float = 0.0
    atr_period: int = 14
    target_r: float = 1.0
    cutoff: str = "11:00"
    time_exit: str | None = None
    direction: str = "both"
    max_trades: int = 1
    reentry: str = "any"
    breakeven_r: float = 0.0
    or_atr_min: float = 0.0
    or_atr_max: float = 0.0

    # ----------------------------------------------------------------- helpers
    @property
    def orb_start_min(self) -> int:
        return hhmm_to_minutes(self.orb_start)

    @property
    def range_end_min(self) -> int:
        return self.orb_start_min + int(self.range_minutes)

    @property
    def cutoff_min(self) -> int:
        return hhmm_to_minutes(self.cutoff)

    @property
    def time_exit_min(self) -> int | None:
        return None if self.time_exit in (None, "", "session") else hhmm_to_minutes(self.time_exit)

    @property
    def needs_atr(self) -> bool:
        return self.stop_method == "atr" or self.or_atr_min > 0 or self.or_atr_max > 0

    def validate(self, base_minutes: int = 1) -> None:
        if self.entry_method not in ENTRY_METHODS:
            raise ParamError(f"entry_method must be one of {ENTRY_METHODS}")
        if self.stop_method not in STOP_METHODS:
            raise ParamError(f"stop_method must be one of {STOP_METHODS}")
        if self.direction not in DIRECTIONS:
            raise ParamError(f"direction must be one of {DIRECTIONS}")
        if self.reentry not in REENTRY_POLICIES:
            raise ParamError(f"reentry must be one of {REENTRY_POLICIES}")
        if self.range_minutes <= 0 or self.range_minutes % base_minutes:
            raise ParamError(f"range_minutes must be a positive multiple of the {base_minutes}-min base bar")
        if self.entry_method != "stop" and (self.entry_tf <= 0 or self.entry_tf % base_minutes):
            raise ParamError(f"entry_tf must be a positive multiple of the {base_minutes}-min base bar")
        if self.orb_start_min % base_minutes:
            raise ParamError("orb_start must align with the base bar grid")
        if self.cutoff_min <= self.range_end_min:
            raise ParamError(
                f"cutoff {self.cutoff} must be after the opening range ends "
                f"({self.orb_start} + {self.range_minutes}m)"
            )
        if self.time_exit_min is not None and self.time_exit_min <= self.range_end_min:
            raise ParamError("time_exit must be after the opening range ends")
        if self.stop_method in ("or_pct", "atr", "fixed_ticks") and self.stop_param <= 0:
            raise ParamError(f"stop_method {self.stop_method} requires stop_param > 0")
        if self.needs_atr and self.atr_period < 1:
            raise ParamError("atr_period must be >= 1")
        if self.target_r < 0 or self.breakeven_r < 0:
            raise ParamError("target_r and breakeven_r must be >= 0")
        if self.confirm_ticks < 0 or self.confirm_or_frac < 0 or self.entry_buffer_ticks < 0:
            raise ParamError("confirmation offsets and buffers must be >= 0")
        if self.max_trades < 0:
            raise ParamError("max_trades must be >= 0 (0 = unlimited)")
        if self.or_atr_max and self.or_atr_max <= self.or_atr_min:
            raise ParamError("or_atr_max must exceed or_atr_min")

    def canonical(self) -> "StrategyParams":
        """Zero out parameters that cannot influence results, so duplicates can be detected."""
        updates: dict[str, Any] = {}
        if self.stop_method in ("or_opposite", "or_mid"):
            updates["stop_param"] = 0.0
        if not self.needs_atr:
            updates["atr_period"] = 0
        if self.entry_method == "market":
            updates["entry_buffer_ticks"] = 0
        if self.entry_method == "stop":
            updates["entry_tf"] = 0
        if self.max_trades == 1:
            updates["reentry"] = "any"
        if self.time_exit in ("", "session"):
            updates["time_exit"] = None
        updates["confirm_ticks"] = float(self.confirm_ticks)
        updates["confirm_or_frac"] = float(self.confirm_or_frac)
        updates["stop_param"] = float(updates.get("stop_param", self.stop_param))
        updates["target_r"] = float(self.target_r)
        return replace(self, **updates)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def key(self) -> str:
        """Stable 12-hex identifier of the canonical parameter set."""
        payload = json.dumps(self.canonical().to_dict(), sort_keys=True, default=str)
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyParams":
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ParamError(f"Unknown strategy parameters: {sorted(unknown)}")
        return cls(**data)


@dataclass(frozen=True)
class ExecutionParams:
    """Transaction-cost and fill assumptions. ``None`` means "use the instrument default"."""

    slippage_ticks: float | None = None
    commission_per_side: float | None = None
    exchange_fee_per_side: float | None = None
    friction_per_side: float | None = None
    fill_model: str = "conservative"
    ambiguity: str = "conservative"
    max_fill_delay_minutes: int = 5
    frictionless: bool = False

    def validate(self) -> None:
        if self.fill_model not in FILL_MODELS:
            raise ParamError(f"fill_model must be one of {FILL_MODELS}")
        if self.ambiguity not in AMBIGUITY_MODES:
            raise ParamError(f"ambiguity must be one of {AMBIGUITY_MODES}")
        for name in ("slippage_ticks", "commission_per_side", "exchange_fee_per_side", "friction_per_side"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ParamError(f"{name} must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionParams":
        return cls(**data)


@dataclass(frozen=True)
class SizingParams:
    mode: str = "fixed_contracts"
    contracts: float = 1
    risk_dollars: float = 1000.0
    risk_pct: float = 0.01
    starting_equity: float = 100_000.0
    max_contracts: float = 1000
    max_leverage: float | None = None  # notional cap (x equity); None = instrument default, 0 = off

    def validate(self) -> None:
        if self.mode not in SIZING_MODES:
            raise ParamError(f"sizing mode must be one of {SIZING_MODES}")
        if self.starting_equity <= 0:
            raise ParamError("starting_equity must be positive")
        if self.mode == "fixed_contracts" and self.contracts <= 0:
            raise ParamError("contracts must be positive")
        if self.mode == "fixed_risk" and self.risk_dollars <= 0:
            raise ParamError("risk_dollars must be positive")
        if self.mode == "pct_equity" and not (0 < self.risk_pct < 1):
            raise ParamError("risk_pct must be in (0, 1)")
        if self.max_contracts <= 0:
            raise ParamError("max_contracts (max units) must be positive")
        if self.max_leverage is not None and self.max_leverage < 0:
            raise ParamError("max_leverage must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SizingParams":
        return cls(**data)


@dataclass(frozen=True)
class DataParams:
    min_or_coverage: float = 0.8
    exclude_early_close: bool = False


def load_strategy_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config/strategy.yaml`` into typed objects."""
    from .instruments import CONFIG_DIR

    path = Path(path) if path else CONFIG_DIR / "strategy.yaml"
    with open(path) as handle:
        raw = yaml.safe_load(handle) or {}
    return {
        "strategy": StrategyParams.from_dict(raw.get("strategy", {})),
        "execution": ExecutionParams.from_dict(raw.get("execution", {})),
        "sizing": SizingParams.from_dict(raw.get("sizing", {})),
        "data": DataParams(**raw.get("data", {})),
        "objective": raw.get("objective", {}),
    }

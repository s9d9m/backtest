"""Parameter spaces: YAML definition -> deterministic, canonical, de-duplicated list of configurations."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from ..engine.params import ParamError, StrategyParams

CONFIRMATION_PRESETS: dict[str, tuple[float, float]] = {
    "close": (0.0, 0.0),
    "1tick": (1.0, 0.0),
    "2tick": (2.0, 0.0),
    "5pct": (0.0, 0.05),
    "10pct": (0.0, 0.10),
}


def parse_stop(label: str) -> tuple[str, float]:
    """``"or_pct:0.5"`` -> ("or_pct", 0.5); ``"or_mid"`` -> ("or_mid", 0.0)."""
    text = str(label).strip()
    if ":" in text:
        method, value = text.split(":", 1)
        return method.strip(), float(value)
    return text, 0.0


def stop_label(method: str, param: float) -> str:
    if method in ("or_opposite", "or_mid"):
        return method
    return f"{method}:{param:g}"


def confirmation_label(ticks: float, frac: float) -> str:
    for name, (t, f) in CONFIRMATION_PRESETS.items():
        if abs(t - ticks) < 1e-12 and abs(f - frac) < 1e-12:
            return name
    parts = []
    if ticks:
        parts.append(f"{ticks:g}tick")
    if frac:
        parts.append(f"{frac * 100:g}pct")
    return "+".join(parts) or "close"


def _expand_value(key: str, value: Any) -> dict[str, Any]:
    if key == "stop":
        method, param = parse_stop(value)
        return {"stop_method": method, "stop_param": param}
    if key == "confirmation":
        if value not in CONFIRMATION_PRESETS:
            raise ParamError(f"Unknown confirmation preset {value!r}; choose from {list(CONFIRMATION_PRESETS)}")
        ticks, frac = CONFIRMATION_PRESETS[value]
        return {"confirm_ticks": ticks, "confirm_or_frac": frac}
    return {key: value}


@dataclass
class ExpandedSpace:
    configs: list[StrategyParams]
    n_raw: int
    n_invalid: int
    n_duplicates: int
    invalid_examples: list[str] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        keys = "|".join(c.key() for c in self.configs)
        return hashlib.sha1(keys.encode()).hexdigest()[:16]


@dataclass
class ParameterSpace:
    name: str
    grid: dict[str, list]
    base: StrategyParams = field(default_factory=StrategyParams)
    description: str = ""

    def raw_size(self) -> int:
        size = 1
        for values in self.grid.values():
            size *= max(len(values), 1)
        return size

    def expand(self, base_minutes: int = 1) -> ExpandedSpace:
        keys = list(self.grid)
        seen: set[str] = set()
        configs: list[StrategyParams] = []
        n_raw = n_invalid = n_dup = 0
        examples: list[str] = []
        base = self.base.to_dict()
        for combo in itertools.product(*(self.grid[k] for k in keys)):
            n_raw += 1
            values = dict(base)
            for key, value in zip(keys, combo):
                values.update(_expand_value(key, value))
            try:
                params = StrategyParams.from_dict(values)
                params.validate(base_minutes)
            except (ParamError, ValueError) as exc:
                n_invalid += 1
                if len(examples) < 5:
                    examples.append(f"{dict(zip(keys, combo))}: {exc}")
                continue
            canonical = params.canonical()
            key = canonical.key()
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            configs.append(canonical)
        return ExpandedSpace(configs, n_raw, n_invalid, n_dup, examples)

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description, "grid": self.grid, "base": self.base.to_dict()}


def load_search_spaces(path: str | Path | None = None, base: StrategyParams | None = None) -> dict[str, ParameterSpace]:
    from ..engine.instruments import CONFIG_DIR

    path = Path(path) if path else CONFIG_DIR / "search_spaces.yaml"
    with open(path) as handle:
        raw = yaml.safe_load(handle) or {}
    out = {}
    for name, spec in raw.items():
        out[name] = ParameterSpace(name=name, grid=spec["grid"], base=base or StrategyParams(), description=str(spec.get("description", "")).strip())
    return out


def config_frame_row(params: StrategyParams) -> dict[str, Any]:
    """Flat, human-readable parameter columns for results tables."""
    row = params.to_dict()
    row["stop"] = stop_label(params.stop_method, params.stop_param)
    row["confirmation"] = confirmation_label(params.confirm_ticks, params.confirm_or_frac)
    row["config_key"] = params.key()
    return row


def params_from_row(row: dict[str, Any]) -> StrategyParams:
    fields_ = StrategyParams().to_dict().keys()
    data = {k: row[k] for k in fields_ if k in row}
    for k in ("range_minutes", "entry_tf", "entry_buffer_ticks", "atr_period", "max_trades"):
        if k in data:
            data[k] = int(data[k])
    if data.get("time_exit") in ("", "None") or (isinstance(data.get("time_exit"), float)):
        data["time_exit"] = None
    return replace(StrategyParams(), **data)


def space_to_json(space: ParameterSpace) -> str:
    return json.dumps(space.to_dict(), sort_keys=True, default=str)

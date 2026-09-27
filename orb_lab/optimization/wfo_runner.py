"""Run a full walk-forward experiment: cube -> neighbours -> every WFO structure x entry family."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..engine.params import ExecutionParams
from ..engine.session_data import PreparedData
from ..reports.experiment import RUNS_DIR, ExperimentManifest
from .parameter_space import ParameterSpace
from .robustness import neighbour_index
from .stats_cube import build_cube, build_session_cube
from .walk_forward import STRUCTURES, SelectionConfig, WFOResult, WFOStructure, phase_metrics, run_wfo

FAMILIES = {
    "all": lambda c: True,
    "market": lambda c: c.entry_method == "market",
    "limit": lambda c: c.entry_method == "limit",
    "stop": lambda c: c.entry_method == "stop",
}


def run_experiment(
    prep: PreparedData,
    space: ParameterSpace,
    execution: ExecutionParams,
    *,
    structures: list[str | WFOStructure],
    families: list[str],
    n_workers: int = 1,
    selection: SelectionConfig | None = None,
    research: bool = False,
    data_description: dict | None = None,
    lockbox: dict | None = None,
    progress: Callable[[str, int, int, float], None] | None = None,
    root: Path | None = None,
    start=None,
    run_dir: Path | None = None,
    starting_equity: float = 100_000.0,
    risk_pct: float = 0.01,
    slippage_grid: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0),
) -> tuple[Path, dict[tuple[str, str], WFOResult]]:
    """Run every structure x family. ``structures`` may be preset names or :class:`WFOStructure` objects.

    Structures in ``sessions`` units (for short samples) are evaluated on a cube of session blocks whose
    size is the greatest common divisor of their lengths; ``months`` structures use the monthly cube.
    Sessions before ``start`` are only used as indicator history. ``progress(stage, done, total, elapsed)``
    may raise to stop the run; the monthly cube is checkpointed and resumes on the next call.
    """
    root = root or RUNS_DIR
    structs = [STRUCTURES[s] if isinstance(s, str) else s for s in structures]
    for st_ in structs:
        st_.validate()
    units = {st_.unit for st_ in structs}
    if len(units) > 1:
        raise ValueError("mixing month- and session-based structures in one run is not supported")
    unit = units.pop()
    expanded = space.expand(prep.base_minutes)
    configs = expanded.configs
    if not configs:
        raise ValueError("the parameter space produced no valid configurations")
    manifest = ExperimentManifest.create(
        "wfo",
        data=data_description or {"data_hash": prep.data_hash},
        instrument=prep.instrument.to_dict(),
        search_space=space.to_dict(),
        execution=execution.to_dict(),
        sizing={"mode": "fixed_contracts", "contracts": 1},
        date_range={"development_first": str(prep.dates[0]), "development_last": str(prep.dates[-1]),
                    "lockbox_start": lockbox and lockbox.get("lockbox_start")},
        wfo={"structures": {s.name: asdict(s) for s in structs}, "families": families,
             "selection": asdict(selection or SelectionConfig()), "start": str(start) if start is not None else None},
        n_configs=len(configs),
        notes=[f"research_mode={research}", f"space_fingerprint={expanded.fingerprint}",
               f"raw={expanded.n_raw} invalid={expanded.n_invalid} duplicates={expanded.n_duplicates}"],
    )
    if run_dir is None:
        run_dir = manifest.run_dir(root)
        manifest.save(root)
    else:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2, default=str))
    cube_progress = (lambda d, t, e: progress("simulating every configuration once", d, t, e)) if progress else None
    if unit == "sessions":
        block = int(np.gcd.reduce([s.train_months for s in structs] + [s.val_months for s in structs] + [s.oos_months for s in structs]))
        cube = build_session_cube(prep, configs, execution, block, start=start, n_workers=n_workers, progress=cube_progress)
        structs = [WFOStructure(s.name, s.train_months // block, s.val_months // block, s.oos_months // block, s.roll_months // block, "sessions")
                   for s in structs]
        min_first = 1
        nbrs = neighbour_index(configs)
    else:
        cube = build_cube(prep, configs, execution, root / "cubes", n_workers=n_workers, progress=cube_progress)
        min_first = 0
        if start is not None:
            first_month = str(pd.Timestamp(start).to_period("M"))
            min_first = next((k for k, m in enumerate(cube.months) if m >= first_month), len(cube.months))
        nbr_file = cube.path / "neighbours.npy"
        if nbr_file.exists():
            nbrs = np.load(nbr_file)
        else:
            nbrs = neighbour_index(configs)
            np.save(nbr_file, nbrs)
    results: dict[tuple[str, str], WFOResult] = {}
    rows = []
    total_runs = len(structs) * len(families)
    for struct in structs:
        s_name = struct.name
        for fam in families:
            mask = np.array([FAMILIES[fam](c) for c in configs])
            if not mask.any():
                continue
            out = run_dir / s_name / fam
            stage = f"walk-forward {s_name} / {fam}: window"
            res = run_wfo(prep, cube, nbrs, struct, execution, selection=selection, eligible=mask,
                          family_label=fam, run_dir=out, min_first=min_first, starting_equity=starting_equity, risk_pct=risk_pct,
                          slippage_grid=slippage_grid,
                          progress=(lambda d, t, _s=stage: progress(_s, d, t, 0.0)) if progress else None)
            results[(s_name, fam)] = res
            st = res.summary.get("stitched_always_trade", {})
            rows.append({"structure": s_name, "family": fam, "windows": res.summary["windows"], "oos_first": res.summary["oos_first"],
                         "oos_last": res.summary["oos_last"], **{f"oos_{k}": v for k, v in st.items()},
                         "pct_windows_profitable": res.summary.get("pct_windows_oos_profitable"),
                         "execution_sensitive": res.summary.get("execution_sensitive"),
                         "overfit_flags": "; ".join(res.summary.get("overfit_flags", []))})
            if progress:
                progress("structures finished", len(rows), total_runs, 0.0)
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "wfo_overview.csv", index=False)
    try:
        phase_metrics(cube, structs[0], min_first).to_parquet(run_dir / "phase_metrics.parquet")
    except ValueError:
        pass
    (run_dir / "cube.json").write_text(json.dumps({"cube_path": str(cube.path), "key": cube.key, "months": cube.months, "unit": unit,
                                                   "bounds": cube.bounds}, indent=2))
    if research:
        from ..research.registry import log_event

        log_event("experiment_wfo", experiment_id=manifest.experiment_id, symbol=prep.instrument.symbol, data_hash=prep.data_hash,
                  space=space.to_dict(), space_fingerprint=expanded.fingerprint, n_configs=len(configs), structures=structures,
                  families=families, selection=asdict(selection or SelectionConfig()), execution=execution.to_dict(),
                  development_last=str(prep.dates[-1]), run_dir=str(run_dir))
    return run_dir, results

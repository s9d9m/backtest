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
from .stats_cube import build_cube
from .walk_forward import STRUCTURES, SelectionConfig, WFOResult, phase_metrics, run_wfo

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
    structures: list[str],
    families: list[str],
    n_workers: int = 1,
    selection: SelectionConfig | None = None,
    research: bool = False,
    data_description: dict | None = None,
    lockbox: dict | None = None,
    progress: Callable[[str, int, int, float], None] | None = None,
    root: Path | None = None,
) -> tuple[Path, dict[tuple[str, str], WFOResult]]:
    root = root or RUNS_DIR
    expanded = space.expand(prep.base_minutes)
    configs = expanded.configs
    manifest = ExperimentManifest.create(
        "wfo",
        data=data_description or {"data_hash": prep.data_hash},
        instrument=prep.instrument.to_dict(),
        search_space=space.to_dict(),
        execution=execution.to_dict(),
        sizing={"mode": "fixed_contracts", "contracts": 1},
        date_range={"development_first": str(prep.dates[0]), "development_last": str(prep.dates[-1]),
                    "lockbox_start": lockbox and lockbox.get("lockbox_start")},
        wfo={"structures": {s: asdict(STRUCTURES[s]) for s in structures}, "families": families,
             "selection": asdict(selection or SelectionConfig())},
        n_configs=len(configs),
        notes=[f"research_mode={research}", f"space_fingerprint={expanded.fingerprint}",
               f"raw={expanded.n_raw} invalid={expanded.n_invalid} duplicates={expanded.n_duplicates}"],
    )
    run_dir = manifest.run_dir(root)
    manifest.save(root)
    cube = build_cube(prep, configs, execution, root / "cubes", n_workers=n_workers,
                      progress=(lambda d, t, e: progress("cube", d, t, e)) if progress else None)
    nbr_file = cube.path / "neighbours.npy"
    if nbr_file.exists():
        nbrs = np.load(nbr_file)
    else:
        nbrs = neighbour_index(configs)
        np.save(nbr_file, nbrs)
    results: dict[tuple[str, str], WFOResult] = {}
    rows = []
    for s_name in structures:
        for fam in families:
            mask = np.array([FAMILIES[fam](c) for c in configs])
            if not mask.any():
                continue
            out = run_dir / s_name / fam
            res = run_wfo(prep, cube, nbrs, STRUCTURES[s_name], execution, selection=selection, eligible=mask,
                          family_label=fam, run_dir=out)
            results[(s_name, fam)] = res
            st = res.summary.get("stitched_always_trade", {})
            rows.append({"structure": s_name, "family": fam, "windows": res.summary["windows"], "oos_first": res.summary["oos_first"],
                         "oos_last": res.summary["oos_last"], **{f"oos_{k}": v for k, v in st.items()},
                         "pct_windows_profitable": res.summary.get("pct_windows_oos_profitable"),
                         "execution_sensitive": res.summary.get("execution_sensitive"),
                         "overfit_flags": "; ".join(res.summary.get("overfit_flags", []))})
            if progress:
                progress(f"wfo {s_name}/{fam}", len(rows), len(structures) * len(families), 0.0)
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "wfo_overview.csv", index=False)
    phase_metrics(cube, STRUCTURES[structures[0]]).to_parquet(run_dir / "phase_metrics.parquet")
    (run_dir / "cube.json").write_text(json.dumps({"cube_path": str(cube.path), "key": cube.key, "months": cube.months}, indent=2))
    if research:
        from ..research.registry import log_event

        log_event("experiment_wfo", experiment_id=manifest.experiment_id, symbol=prep.instrument.symbol, data_hash=prep.data_hash,
                  space=space.to_dict(), space_fingerprint=expanded.fingerprint, n_configs=len(configs), structures=structures,
                  families=families, selection=asdict(selection or SelectionConfig()), execution=execution.to_dict(),
                  development_last=str(prep.dates[-1]), run_dir=str(run_dir))
    return run_dir, results

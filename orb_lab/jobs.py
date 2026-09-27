"""Background jobs for long runs started from the dashboard (walk-forward).

A job is a directory ``runs/jobs/<job-id>/`` holding:

* ``spec.json``   what to run (JSON, human readable);
* ``prep.pkl``    the prepared dataset exactly as loaded in the dashboard (lockbox already withheld);
* ``status.json`` state, stage, progress and timing, rewritten as the job advances;
* ``log.txt``     stdout/stderr of the worker process;
* ``result/``     the walk-forward run directory (same layout as ``cli wfo`` runs).

The worker runs as a separate process (``python -m orb_lab.jobs <job_dir>``), so closing the browser tab
does not stop it. ``stop`` writes a ``STOP`` file that the worker checks at every progress update, and
terminates the process if it does not react. ``resume`` restarts a stopped or failed job with the same
spec; the monthly statistics cube is checkpointed, so finished work is reused.
"""

from __future__ import annotations

import json
import os
import pickle
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .reports.experiment import RUNS_DIR

JOBS_DIR = RUNS_DIR / "jobs"
TERMINAL = ("done", "failed", "stopped")


class StopRequested(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


def create_job(kind: str, spec: dict, prep, root: Path | None = None) -> Path:
    root = Path(root or JOBS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    job_id = f"{kind}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex()}"
    job_dir = root / job_id
    job_dir.mkdir()
    _write_json(job_dir / "spec.json", {"kind": kind, "job_id": job_id, "created_utc": _now(), **spec})
    with open(job_dir / "prep.pkl", "wb") as fh:
        pickle.dump(prep, fh, protocol=pickle.HIGHEST_PROTOCOL)
    _write_json(job_dir / "status.json", {"state": "queued", "stage": "waiting to start", "done": 0, "total": 0, "updated_utc": _now()})
    return job_dir


def launch(job_dir: Path) -> int:
    """Start the worker process detached from the dashboard. Returns its PID."""
    job_dir = Path(job_dir)
    (job_dir / "STOP").unlink(missing_ok=True)
    log = open(job_dir / "log.txt", "a")
    proc = subprocess.Popen([sys.executable, "-m", "orb_lab.jobs", str(job_dir)], stdout=log, stderr=subprocess.STDOUT,
                            cwd=str(Path(__file__).resolve().parents[1]), start_new_session=True)
    status = read_status(job_dir)
    status.update({"state": "starting", "pid": proc.pid, "updated_utc": _now()})
    _write_json(job_dir / "status.json", status)
    return proc.pid


def submit(kind: str, spec: dict, prep, root: Path | None = None) -> Path:
    job_dir = create_job(kind, spec, prep, root)
    launch(job_dir)
    return job_dir


def read_status(job_dir: Path) -> dict:
    f = Path(job_dir) / "status.json"
    try:
        status = json.loads(f.read_text())
    except (OSError, json.JSONDecodeError):
        return {"state": "unknown"}
    if status.get("state") in ("running", "starting") and status.get("pid") and not _alive(status["pid"]):
        # the process vanished without writing a terminal state (killed, machine restarted)
        status["state"] = "interrupted"
    return status


def _alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    try:  # a zombie child of this process counts as dead
        with open(f"/proc/{int(pid)}/stat") as fh:
            return fh.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def stop(job_dir: Path, grace_seconds: float = 5.0) -> None:
    job_dir = Path(job_dir)
    (job_dir / "STOP").write_text(_now())
    status = read_status(job_dir)
    pid = status.get("pid")
    if not pid:
        return
    deadline = time.time() + grace_seconds
    while time.time() < deadline and _alive(pid):
        time.sleep(0.2)
    if _alive(pid):
        try:
            os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        status = read_status(job_dir)
        status.update({"state": "stopped", "stage": "terminated by user", "updated_utc": _now()})
        _write_json(job_dir / "status.json", status)


def resume(job_dir: Path) -> int:
    return launch(job_dir)


def list_jobs(root: Path | None = None, kind: str | None = None) -> list[Path]:
    root = Path(root or JOBS_DIR)
    if not root.exists():
        return []
    jobs = [d for d in root.iterdir() if (d / "spec.json").exists()]
    if kind:
        jobs = [d for d in jobs if d.name.startswith(kind + "-")]
    return sorted(jobs, key=lambda d: d.name, reverse=True)


# ----------------------------------------------------------------------------------------------- worker
def _run_wfo_job(job_dir: Path, spec: dict, prep, report) -> dict:
    from .engine.params import ExecutionParams
    from .optimization.parameter_space import ParameterSpace
    from .optimization.walk_forward import SelectionConfig, WFOStructure
    from .optimization.wfo_runner import run_experiment
    from .research.lockbox import subset_before

    import pandas as pd

    if spec.get("end"):
        prep = subset_before(prep, pd.Timestamp(spec["end"]) + pd.Timedelta(days=1))
    structures = [WFOStructure(**s) for s in spec["structures"]]
    space = ParameterSpace(spec["space"]["name"], spec["space"]["grid"])
    run_dir, results = run_experiment(
        prep, space, ExecutionParams(**spec["execution"]), structures=structures, families=spec.get("families", ["all"]),
        n_workers=int(spec.get("n_workers", 1)), selection=SelectionConfig(**spec.get("selection", {})), research=False,
        data_description=spec.get("data"), progress=report, start=spec.get("start"), run_dir=job_dir / "result",
        starting_equity=float(spec.get("starting_equity", 100_000.0)), risk_pct=float(spec.get("risk_pct", 0.01)),
        slippage_grid=tuple(spec.get("slippage_grid", (0.5, 1.0, 1.5, 2.0, 3.0))),
    )
    return {"result_dir": str(run_dir), "runs": len(results)}


RUNNERS = {"wfo": _run_wfo_job}


def run_job(job_dir: Path) -> dict:
    job_dir = Path(job_dir)
    spec = json.loads((job_dir / "spec.json").read_text())
    status = read_status(job_dir)
    t0 = time.time()
    status.update({"state": "running", "pid": os.getpid(), "started_utc": _now(), "stage": "loading data", "error": None})
    _write_json(job_dir / "status.json", status)

    def report(stage: str, done: int, total: int, elapsed: float = 0.0) -> None:
        if (job_dir / "STOP").exists():
            raise StopRequested("stopped by user")
        status.update({"stage": stage, "done": int(done), "total": int(total), "elapsed_s": round(time.time() - t0, 1), "updated_utc": _now()})
        _write_json(job_dir / "status.json", status)

    try:
        with open(job_dir / "prep.pkl", "rb") as fh:
            prep = pickle.load(fh)
        out = RUNNERS[spec["kind"]](job_dir, spec, prep, report)
        status.update({"state": "done", "stage": "finished", **out})
    except StopRequested:
        status.update({"state": "stopped", "stage": "stopped by user (resume reuses finished work)"})
    except Exception as exc:  # recorded for the dashboard
        status.update({"state": "failed", "stage": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    status.update({"elapsed_s": round(time.time() - t0, 1), "updated_utc": _now(), "finished_utc": _now()})
    _write_json(job_dir / "status.json", status)
    return status


if __name__ == "__main__":
    final = run_job(Path(sys.argv[1]))
    print(json.dumps({k: final.get(k) for k in ("state", "stage", "error")}))
    sys.exit(0 if final["state"] in ("done", "stopped") else 1)

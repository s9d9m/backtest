"""Monte Carlo analysis of a trade sequence.

Everything here works on a list of historical trade outcomes in **R multiples** (net of costs). A
simulated path is a re-ordering or re-sampling of those outcomes; it contains no new market
information. Simulations answer "how bad could the path have been with these trade outcomes?", not
"will the strategy work?". Historical and simulated numbers are always reported side by side.

Methods
-------
``reshuffle``        random permutation of the historical trades (same trades, different order). Under
                     fixed-risk sizing the ending equity is identical; only drawdowns and streaks change.
``bootstrap``        i.i.d. resampling with replacement (same number of trades).
``block_bootstrap``  resampling of consecutive blocks (keeps short-range dependence such as clustered
                     losing days).
``missed_trades``    each trade is skipped with probability ``skip_prob`` (execution failures, days away).

Every method can be combined with ``extra_cost_r`` (additional cost per trade, in R) as a cost stress.

Sizing
------
``fixed_risk``   every trade risks ``risk_pct`` of the *starting* equity (no compounding): P&L = R x risk $.
``compounding``  every trade risks ``risk_pct`` of the *current* equity: equity *= 1 + R x risk_pct.

Results are reproducible for a given ``seed``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

METHODS = {
    "reshuffle": "Reshuffle order (same trades)",
    "bootstrap": "Bootstrap (resample with replacement)",
    "block_bootstrap": "Block bootstrap (resample runs of consecutive trades)",
    "missed_trades": "Randomly skip trades",
}
SIZING = {"fixed_risk": "Fixed $ risk (% of starting equity, no compounding)", "compounding": "Compounding (% of current equity)"}


@dataclass(frozen=True)
class MCConfig:
    method: str = "bootstrap"
    n_sims: int = 2000
    seed: int = 12345
    starting_equity: float = 40_000.0
    risk_pct: float = 0.01
    sizing: str = "fixed_risk"
    block_size: int = 5
    skip_prob: float = 0.1
    extra_cost_r: float = 0.0
    ruin_dd: float = 0.5  # drawdown treated as "ruin" for the probability table

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {list(METHODS)}")
        if self.sizing not in SIZING:
            raise ValueError(f"sizing must be one of {list(SIZING)}")
        if not (1 <= self.n_sims <= 200_000):
            raise ValueError("n_sims must be between 1 and 200,000")
        if not (0 < self.risk_pct < 1):
            raise ValueError("risk_pct must be in (0, 1)")
        if self.starting_equity <= 0:
            raise ValueError("starting_equity must be positive")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if not (0 <= self.skip_prob < 1):
            raise ValueError("skip_prob must be in [0, 1)")
        if self.extra_cost_r < 0:
            raise ValueError("extra_cost_r must be >= 0")


def resample(r: np.ndarray, cfg: MCConfig, rng: np.random.Generator) -> np.ndarray:
    """(n_sims, n_trades) matrix of simulated R outcomes (skipped trades are 0 R)."""
    n = len(r)
    if cfg.method == "reshuffle":
        idx = np.argsort(rng.random((cfg.n_sims, n)), axis=1)
        out = r[idx]
    elif cfg.method == "bootstrap":
        out = r[rng.integers(0, n, size=(cfg.n_sims, n))]
    elif cfg.method == "block_bootstrap":
        b = min(cfg.block_size, n)
        n_blocks = int(np.ceil(n / b))
        starts = rng.integers(0, n - b + 1, size=(cfg.n_sims, n_blocks))
        idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(cfg.n_sims, -1)[:, :n]
        out = r[idx]
    else:  # missed_trades
        keep = rng.random((cfg.n_sims, n)) >= cfg.skip_prob
        out = np.where(keep, r[None, :], 0.0)
    if cfg.extra_cost_r > 0:
        out = np.where(out != 0.0, out - cfg.extra_cost_r, out) if cfg.method == "missed_trades" else out - cfg.extra_cost_r
    return out


def equity_paths(r_matrix: np.ndarray, cfg: MCConfig) -> np.ndarray:
    """Equity after each trade, shape (n_sims, n_trades + 1) including the starting value."""
    r_matrix = np.atleast_2d(r_matrix)
    start = cfg.starting_equity
    if cfg.sizing == "fixed_risk":
        pnl = r_matrix * (start * cfg.risk_pct)
        eq = start + np.cumsum(pnl, axis=1)
    else:
        growth = np.clip(1.0 + r_matrix * cfg.risk_pct, 0.0, None)
        eq = start * np.cumprod(growth, axis=1)
    eq = np.maximum(eq, 0.0)
    return np.concatenate([np.full((len(eq), 1), start), eq], axis=1)


def max_drawdown(paths: np.ndarray) -> np.ndarray:
    """Maximum drawdown per path as a negative fraction of the running peak."""
    peak = np.maximum.accumulate(paths, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, paths / peak - 1.0, -1.0)
    return dd.min(axis=1)


def max_drawdown_r(r_matrix: np.ndarray) -> np.ndarray:
    cum = np.concatenate([np.zeros((len(r_matrix), 1)), np.cumsum(r_matrix, axis=1)], axis=1)
    return (cum - np.maximum.accumulate(cum, axis=1)).min(axis=1)


def longest_losing_streak(r_matrix: np.ndarray) -> np.ndarray:
    r_matrix = np.atleast_2d(r_matrix)
    cur = np.zeros(len(r_matrix), dtype=np.int64)
    best = np.zeros(len(r_matrix), dtype=np.int64)
    for k in range(r_matrix.shape[1]):
        losing = r_matrix[:, k] < 0
        cur = np.where(losing, cur + 1, np.where(r_matrix[:, k] > 0, 0, cur))
        best = np.maximum(best, cur)
    return best


def path_stats(r_matrix: np.ndarray, cfg: MCConfig) -> pd.DataFrame:
    paths = equity_paths(r_matrix, cfg)
    end = paths[:, -1]
    return pd.DataFrame(
        {
            "ending_equity": end,
            "total_return": end / cfg.starting_equity - 1.0,
            "max_drawdown": max_drawdown(paths),
            "max_drawdown_r": max_drawdown_r(r_matrix),
            "longest_losing_streak": longest_losing_streak(r_matrix),
            "sum_r": r_matrix.sum(axis=1),
        }
    )


@dataclass
class MCResult:
    config: MCConfig
    n_trades: int
    historical: dict
    sims: pd.DataFrame
    percentiles: pd.DataFrame
    probabilities: dict
    sample_paths: np.ndarray  # (k, n_trades + 1) equity paths for plotting
    historical_path: np.ndarray

    def to_dict(self) -> dict:
        return {"config": asdict(self.config), "n_trades": self.n_trades, "historical": self.historical,
                "percentiles": self.percentiles.reset_index().to_dict(orient="records"), "probabilities": self.probabilities}


def run_monte_carlo(r: np.ndarray, cfg: MCConfig, n_sample_paths: int = 100) -> MCResult:
    cfg.validate()
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        raise ValueError("Monte Carlo needs at least 2 trades")
    rng = np.random.default_rng(cfg.seed)
    sims_r = resample(r, cfg, rng)
    sims = path_stats(sims_r, cfg)
    hist = path_stats(r[None, :], cfg).iloc[0].to_dict()
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    pct = sims.quantile(qs)
    pct.index = [f"{int(q * 100)}th" for q in qs]
    probabilities = {
        "p_loss": float((sims["ending_equity"] < cfg.starting_equity).mean()),
        "p_dd_worse_than_10pct": float((sims["max_drawdown"] <= -0.10).mean()),
        "p_dd_worse_than_20pct": float((sims["max_drawdown"] <= -0.20).mean()),
        f"p_dd_worse_than_{int(cfg.ruin_dd * 100)}pct": float((sims["max_drawdown"] <= -cfg.ruin_dd).mean()),
        "p_dd_worse_than_historical": float((sims["max_drawdown"] < hist["max_drawdown"] - 1e-12).mean()),
        "p_streak_longer_than_historical": float((sims["longest_losing_streak"] > hist["longest_losing_streak"]).mean()),
    }
    paths = equity_paths(sims_r[: min(n_sample_paths, cfg.n_sims)], cfg)
    return MCResult(cfg, len(r), {k: float(v) for k, v in hist.items()}, sims, pct, probabilities, paths,
                    equity_paths(r[None, :], cfg)[0])


def trade_r(trades: pd.DataFrame) -> np.ndarray:
    """Net R multiples of executed trades, in chronological order."""
    if trades is None or len(trades) == 0:
        return np.zeros(0)
    t = trades
    if "sized_out" in t:
        t = t[~t["sized_out"].astype(bool)]
    if "entry_time" in t:
        t = t.sort_values("entry_time", kind="stable")
    return t["r_multiple"].to_numpy(dtype=float)

"""Multiple-testing context for optimization results.

When N configurations are tried, the best in-sample Sharpe ratio is biased upward even if no
configuration has any edge. Under the null (all true Sharpe = 0) and independent trials, the expected
maximum of N estimated Sharpe ratios with cross-sectional standard deviation ``sd`` is approximately
(Bailey & Lopez de Prado, 2014):

    E[max SR] ~= sd * ((1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N e)))      g = Euler-Mascheroni

Grid configurations are strongly correlated, so the *effective* number of independent trials is
smaller than N. We estimate it from the eigenvalues of the correlation matrix of daily P&L of a
sample of configurations (participation ratio). The full Deflated Sharpe Ratio and PBO/CSCV are
Milestone 5.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(n_trials: float, sharpe_std: float) -> float:
    if n_trials <= 1 or sharpe_std <= 0:
        return 0.0
    z = NormalDist().inv_cdf
    return float(sharpe_std * ((1 - EULER_GAMMA) * z(1 - 1.0 / n_trials) + EULER_GAMMA * z(1 - 1.0 / (n_trials * math.e))))


def effective_trials(returns_matrix: np.ndarray) -> float:
    """Participation-ratio estimate of independent trials from a (days x configs) P&L matrix."""
    if returns_matrix.ndim != 2 or returns_matrix.shape[1] < 2:
        return float(returns_matrix.shape[1] if returns_matrix.ndim == 2 else 1)
    std = returns_matrix.std(axis=0)
    keep = std > 0
    if keep.sum() < 2:
        return 1.0
    corr = np.corrcoef(returns_matrix[:, keep], rowvar=False)
    eig = np.clip(np.linalg.eigvalsh(np.nan_to_num(corr)), 0, None)
    if eig.sum() <= 0:
        return 1.0
    return float(eig.sum() ** 2 / np.sum(eig**2))


def selection_bias_summary(sharpes: np.ndarray, n_effective: float | None = None) -> dict:
    """Compare the best observed Sharpe with what pure selection among null trials would produce."""
    sharpes = np.asarray(sharpes, dtype=float)
    sharpes = sharpes[np.isfinite(sharpes)]
    n = len(sharpes)
    if n == 0:
        return {"n_configs": 0}
    sd = float(sharpes.std(ddof=1)) if n > 1 else 0.0
    n_eff = n_effective if n_effective else n
    null_max = expected_max_sharpe(n_eff, sd)
    best = float(sharpes.max())
    return {
        "n_configs": n,
        "n_effective": float(n_eff),
        "sharpe_cross_sectional_std": sd,
        "best_sharpe": best,
        "expected_max_sharpe_under_null": null_max,
        "best_exceeds_null_max": bool(best > null_max),
        "share_positive_sharpe": float(np.mean(sharpes > 0)),
    }

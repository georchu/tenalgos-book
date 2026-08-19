"""Performance statistics, including the ones that tell you the truth.

Every strategy in this book reports through `performance_report`, so the numbers
in every chapter are computed the same way and are comparable to each other.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

TRADING_DAYS = 252


# ----------------------------------------------------------------- basics ---
def ann_return(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return float(r.mean() * TRADING_DAYS)


def ann_vol(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(r: pd.Series, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio of a daily return series."""
    r = r.dropna() - rf / TRADING_DAYS
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sortino(r: pd.Series) -> float:
    r = r.dropna()
    downside = r[r < 0].std(ddof=1)
    if downside == 0 or np.isnan(downside):
        return np.nan
    return float(r.mean() / downside * np.sqrt(TRADING_DAYS))


def equity_curve(r: pd.Series) -> pd.Series:
    return (1.0 + r.fillna(0.0)).cumprod()


def drawdown(r: pd.Series) -> pd.Series:
    eq = equity_curve(r)
    return eq / eq.cummax() - 1.0


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown(r).min())


def calmar(r: pd.Series) -> float:
    mdd = abs(max_drawdown(r))
    return float(ann_return(r) / mdd) if mdd > 0 else np.nan


def hit_rate(r: pd.Series) -> float:
    r = r.dropna()
    return float((r > 0).mean()) if len(r) else np.nan


def turnover(weights: pd.DataFrame) -> pd.Series:
    """One-sided daily turnover: 0.5 * sum |w_t - w_{t-1}|."""
    return 0.5 * weights.diff().abs().sum(axis=1)


# ------------------------------------------------- the honest Sharpe ratio ---
def sharpe_std_error(sr: float, n: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Standard error of an annualised Sharpe, adjusted for higher moments.

    Lo (2002) with the Mertens correction. Using the naive sqrt(1/n) version on
    a fat-tailed, negatively skewed return stream understates the error badly —
    which is exactly the case for most of the strategies in Part II.
    """
    sr_d = sr / np.sqrt(TRADING_DAYS)
    var = (1 + 0.5 * sr_d**2 - skew * sr_d + (kurt - 3) / 4 * sr_d**2) / n
    return float(np.sqrt(max(var, 0.0)) * np.sqrt(TRADING_DAYS))


def deflated_sharpe(
    sr: float,
    n: int,
    n_trials: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    sr_benchmark: float | None = None,
) -> float:
    """Probability the true Sharpe exceeds zero, after multiple testing.

    Bailey & López de Prado (2014). `n_trials` is the number of strategy
    configurations you actually tried — including the ones you tried in your
    head and discarded. If you swept 6 lookbacks x 5 holding periods x 4
    universes, that is 120, not 1.

    Returns a probability in [0, 1]. Below ~0.95 the result should not be
    treated as a discovery.
    """
    if not np.isfinite(sr) or n < 2 or n_trials < 1:
        return np.nan

    se = sharpe_std_error(sr, n, skew, kurt)
    if se == 0 or not np.isfinite(se):
        return np.nan

    if sr_benchmark is None:
        # Expected maximum Sharpe produced by n_trials independent zero-alpha
        # trials. The bracket is in units of standard errors, so it is scaled
        # by the standard error of the Sharpe estimate to become a Sharpe.
        if n_trials <= 1:
            sr_benchmark = 0.0
        else:
            g = np.euler_gamma
            z1 = sps.norm.ppf(1.0 - 1.0 / n_trials)
            z2 = sps.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
            sr_benchmark = se * ((1 - g) * z1 + g * z2)

    return float(sps.norm.cdf((sr - sr_benchmark) / se))


def expected_max_sharpe(n_trials: int, n: int, sr_guess: float = 0.0) -> float:
    """The annualised Sharpe a *zero-alpha* strategy is expected to print after
    n_trials attempts. Print this next to every backtest: if your result is not
    comfortably above this bar, you have found nothing.
    """
    if n_trials <= 1:
        return 0.0
    g = np.euler_gamma
    z1 = sps.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sharpe_std_error(sr_guess, n) * ((1 - g) * z1 + g * z2))


# ----------------------------------------------------------------- report ---
def performance_report(
    r: pd.Series,
    weights: pd.DataFrame | None = None,
    n_trials: int = 1,
    name: str = "strategy",
) -> pd.Series:
    r = r.dropna()
    sr = sharpe(r)
    sk = float(sps.skew(r)) if len(r) > 3 else 0.0
    ku = float(sps.kurtosis(r, fisher=False)) if len(r) > 3 else 3.0

    out = {
        "ann_return": ann_return(r),
        "ann_vol": ann_vol(r),
        "sharpe": sr,
        "sharpe_se": sharpe_std_error(sr, len(r), sk, ku),
        "sortino": sortino(r),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar(r),
        "hit_rate": hit_rate(r),
        "skew": sk,
        "excess_kurtosis": ku - 3.0,
        "n_days": len(r),
        "n_trials_declared": n_trials,
        "deflated_sharpe_prob": deflated_sharpe(sr, len(r), n_trials, sk, ku),
    }
    if weights is not None:
        to = turnover(weights)
        out["daily_turnover"] = float(to.mean())
        out["ann_turnover"] = float(to.mean() * TRADING_DAYS)
        out["avg_gross"] = float(weights.abs().sum(axis=1).mean())
        out["avg_net"] = float(weights.sum(axis=1).mean())
        out["avg_names"] = float((weights.abs() > 1e-9).sum(axis=1).mean())
    return pd.Series(out, name=name)

"""The backtest engine.

One engine for all ten strategies, with one non-negotiable rule enforced in
code rather than left to the author's discipline:

    A signal computed from data up to and including the close of day t
    may only earn the return of day t+1.

`run_backtest` applies that shift itself. A strategy module returns *target
weights indexed by the day the signal was known*, and the engine does the
lagging. This removes the single most common source of fictitious alpha in
published backtests, because it makes the mistake impossible to make by
accident rather than merely discouraged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import CostModel
from .stats import performance_report, turnover


@dataclass
class BacktestResult:
    gross_returns: pd.Series
    net_returns: pd.Series
    costs: pd.Series
    weights: pd.DataFrame
    stats_gross: pd.Series
    stats_net: pd.Series
    meta: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        return pd.concat([self.stats_gross.rename("gross"),
                          self.stats_net.rename("net")], axis=1)


def run_backtest(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_model: CostModel | None = None,
    vol: pd.DataFrame | None = None,
    dollar_volume: pd.DataFrame | None = None,
    capital: float = 1e8,
    n_trials: int = 1,
    name: str = "strategy",
    execution_lag: int = 1,
) -> BacktestResult:
    """Run a long/short backtest.

    Parameters
    ----------
    weights
        Target weights **indexed by the date the signal was known**. The engine
        shifts them forward by `execution_lag` days before applying returns.
    returns
        Simple daily asset returns, same columns as `weights`.
    execution_lag
        1 means: signal known at the close of t, held through t+1. Set to 2 to
        model a slower operation and watch how much of the edge is timing.
    """
    weights = weights.reindex(columns=returns.columns).fillna(0.0)
    weights = weights.reindex(returns.index).ffill().fillna(0.0)

    held = weights.shift(execution_lag).fillna(0.0)

    gross = (held * returns.fillna(0.0)).sum(axis=1)

    traded = held.diff().abs().fillna(held.abs())
    cm = cost_model or CostModel()
    tc = cm.trade_cost(traded, vol=vol, dollar_volume=dollar_volume, capital=capital)
    cc = cm.carry_cost(held)
    costs = tc.add(cc, fill_value=0.0).reindex(gross.index).fillna(0.0)

    net = gross - costs

    return BacktestResult(
        gross_returns=gross,
        net_returns=net,
        costs=costs,
        weights=held,
        stats_gross=performance_report(gross, held, n_trials=n_trials, name=f"{name} (gross)"),
        stats_net=performance_report(net, held, n_trials=n_trials, name=f"{name} (net)"),
        meta={"capital": capital, "execution_lag": execution_lag, "n_trials": n_trials},
    )


# --------------------------------------------------------------- utilities ---
def information_coefficient(signal: pd.DataFrame, fwd_returns: pd.DataFrame) -> pd.Series:
    """Daily cross-sectional Spearman rank correlation between signal and
    next-period return. The cleanest single diagnostic of whether a signal
    contains information at all, independent of portfolio construction."""
    ic = {}
    for dt in signal.index:
        if dt not in fwd_returns.index:
            continue
        s = signal.loc[dt].dropna()
        f = fwd_returns.loc[dt].dropna()
        common = s.index.intersection(f.index)
        if len(common) < 20:
            continue
        ic[dt] = s[common].rank().corr(f[common].rank())
    return pd.Series(ic).sort_index()


def ic_summary(ic: pd.Series) -> pd.Series:
    ic = ic.dropna()
    n = len(ic)
    mean, sd = ic.mean(), ic.std(ddof=1)
    return pd.Series({
        "mean_ic": mean,
        "ic_std": sd,
        "ic_ir": mean / sd if sd else np.nan,
        "t_stat": mean / (sd / np.sqrt(n)) if sd and n else np.nan,
        "pct_positive": (ic > 0).mean(),
        "n_periods": n,
    })


def quantile_returns(signal: pd.DataFrame, fwd_returns: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    """Mean forward return by signal quantile — the monotonicity check.

    A signal whose quantile means are not monotone is telling you the
    relationship is not what you think it is, however good the top-minus-bottom
    spread looks.
    """
    rows = {}
    for dt in signal.index:
        if dt not in fwd_returns.index:
            continue
        s = signal.loc[dt].dropna()
        f = fwd_returns.loc[dt]
        common = s.index.intersection(f.dropna().index)
        if len(common) < q * 5:
            continue
        buckets = pd.qcut(s[common].rank(method="first"), q, labels=False)
        rows[dt] = f[common].groupby(buckets).mean()
    return pd.DataFrame(rows).T.sort_index()

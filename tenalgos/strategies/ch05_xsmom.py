"""Chapter 5 — Cross-Sectional Equity Momentum.

The construction, in order:

    1. raw signal   = cumulative return from t-252 to t-21   (the "12-1")
    2. skip month   = the most recent 21 days are excluded, because at that
                      horizon the effect reverses
    3. neutralise   = demean within industry, and regress out beta, so you are
                      paid for momentum rather than for accidentally owning
                      the tech sector in 1999
    4. scale        = divide by trailing volatility, so a 40-vol name and a
                      15-vol name contribute equal risk, not equal dollars
    5. size         = long the top decile, short the bottom, dollar-neutral,
                      then scale the whole book to a target volatility

Steps 3 and 4 are what separate this from the version in most textbooks, and
they are the difference between a strategy with a Sharpe near 0.5 and one that
survives 2009.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------- signal ---
def momentum_signal(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Cumulative return from t-lookback to t-skip, per asset.

    Uses only prices at or before t, so the value on row t is knowable at the
    close of t. The engine applies the one-day execution lag.
    """
    past = prices.shift(skip)
    base = prices.shift(lookback)
    return (past / base - 1.0)


def trailing_vol(returns: pd.DataFrame, window: int = 126, min_periods: int = 60) -> pd.DataFrame:
    return returns.rolling(window, min_periods=min_periods).std()


def trailing_beta(returns: pd.DataFrame, market: pd.Series, window: int = 252,
                  min_periods: int = 120) -> pd.DataFrame:
    """Rolling OLS beta to an equal-weighted market proxy."""
    cov = returns.rolling(window, min_periods=min_periods).cov(market)
    var = market.rolling(window, min_periods=min_periods).var()
    return cov.div(var, axis=0)


# --------------------------------------------------------- neutralisation ---
def zscore_cross_section(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    z = df.sub(mu, axis=0).div(sd, axis=0)
    return z.clip(-clip, clip)


def industry_neutralise(signal: pd.DataFrame, industry: pd.Series) -> pd.DataFrame:
    """Subtract the industry mean each day."""
    groups = industry.reindex(signal.columns)
    out = signal.copy()
    for code, members in groups.groupby(groups).groups.items():
        cols = list(members)
        block = signal[cols]
        out[cols] = block.sub(block.mean(axis=1), axis=0)
    return out


def beta_neutralise(signal: pd.DataFrame, beta: pd.DataFrame) -> pd.DataFrame:
    """Regress the signal on beta cross-sectionally and keep the residual."""
    out = pd.DataFrame(np.nan, index=signal.index, columns=signal.columns)
    sv, bv = signal.values, beta.reindex_like(signal).values
    for i in range(len(signal)):
        s, b = sv[i], bv[i]
        ok = np.isfinite(s) & np.isfinite(b)
        if ok.sum() < 30:
            continue
        x, y = b[ok], s[ok]
        x_ = np.column_stack([np.ones(ok.sum()), x])
        coef, *_ = np.linalg.lstsq(x_, y, rcond=None)
        resid = np.full_like(s, np.nan)
        resid[ok] = y - x_ @ coef
        out.iloc[i] = resid
    return out


# ------------------------------------------------------------ portfolio ---
def decile_weights(signal: pd.DataFrame, n_bins: int = 10, gross: float = 2.0) -> pd.DataFrame:
    """Long the top bin, short the bottom bin, dollar-neutral, gross = `gross`."""
    w = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for dt in signal.index:
        s = signal.loc[dt].dropna()
        if len(s) < n_bins * 5:
            continue
        ranks = s.rank(method="first")
        edge = len(s) / n_bins
        longs = ranks[ranks > len(s) - edge].index
        shorts = ranks[ranks <= edge].index
        if len(longs) == 0 or len(shorts) == 0:
            continue
        w.loc[dt, longs] = (gross / 2) / len(longs)
        w.loc[dt, shorts] = -(gross / 2) / len(shorts)
    return w


def volatility_target(weights: pd.DataFrame, portfolio_returns: pd.Series,
                      target_vol: float = 0.10, window: int = 63,
                      max_leverage: float = 3.0) -> pd.DataFrame:
    """Scale the whole book so realised vol tracks `target_vol`.

    Uses only trailing information (shifted by one day), so the scaling itself
    cannot leak. This is the fix for momentum crashes: after a volatile,
    correlated market the book is automatically smaller.
    """
    realised = portfolio_returns.rolling(window, min_periods=20).std() * np.sqrt(252)
    scale = (target_vol / realised.replace(0, np.nan)).shift(1)
    scale = scale.clip(upper=max_leverage).fillna(1.0)
    return weights.mul(scale, axis=0)


# ------------------------------------------------------------------ API ---
def build(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    industry: pd.Series | None = None,
    lookback: int = 252,
    skip: int = 21,
    n_bins: int = 10,
    vol_window: int = 126,
    neutralise_industry: bool = True,
    neutralise_beta: bool = True,
    rebalance: str = "M",
    gross: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (signal, target weights).

    `rebalance="M"` holds the month-end signal for the following month, which
    is what the original literature does and what keeps turnover survivable.
    Set rebalance="D" to see what daily rebalancing costs you.
    """
    sig = momentum_signal(prices, lookback, skip)

    vol = trailing_vol(returns, vol_window)
    sig = sig / vol.replace(0, np.nan)

    if neutralise_industry and industry is not None:
        sig = industry_neutralise(sig, industry)
    if neutralise_beta:
        market = returns.mean(axis=1)
        sig = beta_neutralise(sig, trailing_beta(returns, market))

    sig = zscore_cross_section(sig)

    if rebalance.upper().startswith("M"):
        month_ends = sig.resample("ME").last().index
        held = sig.reindex(sig.index)
        mask = pd.Series(False, index=sig.index)
        mask[sig.index.isin(month_ends)] = True
        held = sig.where(mask, np.nan).ffill()
        sig_used = held
    else:
        sig_used = sig

    w = decile_weights(sig_used, n_bins=n_bins, gross=gross)
    return sig, w

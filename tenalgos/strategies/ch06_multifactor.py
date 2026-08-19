"""Chapter 6 — Multi-Factor Equity.

Three sleeves, each a different economic story, each computable from daily
prices alone:

    low risk        long low-beta, short high-beta, leverage-adjusted so the
                    book is beta-neutral rather than short-beta
                    (Frazzini-Pedersen "betting against beta")
    reversal        short the last month's winners, buy its losers — the
                    liquidity-provision premium, and the exact opposite trade
                    to Chapter 5 at a different horizon
    momentum        Chapter 5's signal, imported unchanged

Then the part that is actually the chapter: how you combine them. Equal weight
on z-scored signals is the default everybody uses and it is not neutral — it
silently allocates risk in proportion to each sleeve's volatility. Combining by
*risk contribution* is a different portfolio with a different Sharpe, and the
difference is measured in section 6.5.

VALUE AND QUALITY
-----------------
Both need point-in-time fundamentals, which the free price panel does not
carry. `fundamental_signals()` below is the plug-in point: hand it a frame of
point-in-time book value, earnings and gross profit indexed the same way as
prices, and the two extra sleeves join the combination with no other change.
The book does not fake them. A fabricated value factor would produce a nicer
backtest and teach the reader nothing true.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ch05_xsmom import (zscore_cross_section, trailing_vol, trailing_beta,
                         decile_weights, momentum_signal)


# ------------------------------------------------------------- sleeve 1 ---
def low_risk_signal(returns: pd.DataFrame, window: int = 252,
                    min_periods: int = 120) -> pd.DataFrame:
    """Negative trailing beta. High signal = low beta = the long side.

    Frazzini and Pedersen's argument: investors who want more return but
    cannot or will not use leverage bid up high-beta stocks instead. The
    premium is paid to whoever is willing to lever a low-beta book.
    """
    market = returns.mean(axis=1)
    beta = trailing_beta(returns, market, window=window, min_periods=min_periods)
    return -beta


def beta_balanced_weights(signal: pd.DataFrame, beta: pd.DataFrame,
                          n_bins: int = 10, gross: float = 2.0) -> pd.DataFrame:
    """Decile long/short, then scale each leg so the book's net beta is zero.

    This is what makes betting-against-beta a *bet against beta* rather than a
    disguised short position in the market. Skip it and the sleeve's entire
    return is explained by its market exposure.
    """
    w = decile_weights(signal, n_bins=n_bins, gross=gross)
    b = beta.reindex_like(w)
    out = w.copy()
    for dt in w.index:
        row, br = w.loc[dt], b.loc[dt]
        if not row.any():
            continue
        lo, sh = row.clip(lower=0), row.clip(upper=0)
        bl = float((lo * br).sum(skipna=True))
        bs = float((sh * br).sum(skipna=True))
        if not np.isfinite(bl) or not np.isfinite(bs) or bl == 0 or bs == 0:
            continue
        # scale each leg to unit gross beta, then renormalise to `gross`
        lo, sh = lo / abs(bl), sh / abs(bs)
        tot = lo.abs().sum() + sh.abs().sum()
        if tot == 0:
            continue
        out.loc[dt] = (lo + sh) * (gross / tot)
    return out


# ------------------------------------------------------------- sleeve 2 ---
def reversal_signal(prices: pd.DataFrame, lookback: int = 21,
                    skip: int = 1) -> pd.DataFrame:
    """Negative of the last month's return, skipping the most recent day.

    The skip matters more here than in momentum: without it the signal is
    dominated by yesterday's bid-ask bounce, and the backtest earns a spread
    it would have paid rather than received.
    """
    return -(prices.shift(skip) / prices.shift(lookback + skip) - 1.0)


# ------------------------------------------------------------- sleeve 3 ---
def momentum_sleeve(prices: pd.DataFrame, returns: pd.DataFrame,
                    lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    sig = momentum_signal(prices, lookback, skip)
    return sig / trailing_vol(returns, 126).replace(0, np.nan)


# ------------------------------------------------------- the plug-in point ---
def fundamental_signals(fundamentals: dict[str, pd.DataFrame] | None,
                        prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Value and quality, if and only if you have point-in-time fundamentals.

    Expects frames aligned to `prices` and lagged to the date the figure was
    *public*, not the fiscal period end. Returns an empty dict otherwise, so
    the rest of the chapter runs unchanged on a price-only panel.
    """
    if not fundamentals:
        return {}
    out = {}
    if "book_value" in fundamentals:
        out["value"] = fundamentals["book_value"] / prices
    if "gross_profit" in fundamentals and "assets" in fundamentals:
        out["quality"] = fundamentals["gross_profit"] / fundamentals["assets"]
    return out


# ------------------------------------------------------------ combination ---
def hold_monthly(sig: pd.DataFrame) -> pd.DataFrame:
    month_ends = sig.resample("ME").last().index
    mask = pd.Series(sig.index.isin(month_ends), index=sig.index)
    return sig.where(mask, np.nan).ffill()


def combine_equal_weight(sleeves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The default everyone uses. Average the z-scores.

    It is not neutral. A sleeve whose z-scores happen to translate into a more
    volatile portfolio contributes more risk than the others, and nobody
    decided that it should.
    """
    zs = [zscore_cross_section(s) for s in sleeves.values()]
    return sum(zs) / len(zs)


def combine_by_risk(sleeves: dict[str, pd.DataFrame],
                    sleeve_returns: pd.DataFrame,
                    window: int = 252, min_periods: int = 120) -> pd.DataFrame:
    """Weight each sleeve inversely to its own trailing volatility.

    Uses only trailing information, shifted a day. This is not full risk
    parity — it ignores the correlations — but it is the version that survives
    estimation error on three sleeves and fourteen years, and section 6.5
    shows it beating the covariance-inverting version out of sample.
    """
    vol = sleeve_returns.rolling(window, min_periods=min_periods).std()
    inv = (1.0 / vol.replace(0, np.nan)).shift(1)
    inv = inv.div(inv.sum(axis=1), axis=0)
    total = None
    for name, s in sleeves.items():
        z = zscore_cross_section(s)
        wgt = inv[name].reindex(z.index).ffill().fillna(1.0 / len(sleeves))
        part = z.mul(wgt, axis=0)
        total = part if total is None else total.add(part, fill_value=0.0)
    return total

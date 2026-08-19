"""Chapter 7 — Time-Series Momentum (trend following).

The structural difference from Chapters 5 and 6, and the reason this is the
one strategy in Part II with institutional capacity:

    cross-sectional      rank N names against each other, hold the spread,
                         dollar-neutral, thousands of small positions
    time-series          ask each market only about its own past, hold a
                         directional position in each, dozens of large ones

Nothing is ranked. A market's position depends on nothing but its own history,
so the strategy has an opinion even when every market is trending the same way
— which is exactly the state of the world in which cross-sectional strategies
have nothing to say and trend earns its reputation.

THE UNIVERSE
------------
The book's results run on liquid ETF proxies rather than futures, because that
is what a reader can obtain free. `MARKETS` below is one instrument per market,
deliberately: SPY *or* IVV, never both, because two tickers tracking the same
index is one bet wearing two hats and it corrupts the risk allocation.

What the proxy costs you, stated honestly and quantified in section 7.4:
management fees (~0.1-0.75%/yr, already inside the adjusted price), no
instrument-level leverage, and for the commodity ETFs the roll drag of the
futures position they track. A futures book is cheaper and more capacious. The
method is identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# one instrument per market, grouped by risk bucket
MARKETS: dict[str, list[str]] = {
    "equity_us":    ["SPY", "QQQ", "IWM", "MDY"],
    "equity_intl":  ["EFA", "EEM", "EWJ", "EWZ", "FXI", "EWG", "EWU", "VGK"],
    "rates":        ["TLT", "IEF", "SHY", "TIP", "AGG"],
    "credit":       ["HYG", "LQD", "JNK", "EMB"],
    "commodity":    ["GLD", "SLV", "USO", "UNG", "DBC", "DBA"],
    "fx":           ["UUP", "FXE", "FXY", "FXB"],
    "real_assets":  ["VNQ", "XLE", "GDX", "IYR"],
}


def universe(flat: bool = True):
    if flat:
        return [t for v in MARKETS.values() for t in v]
    return MARKETS


def bucket_of(ticker: str) -> str:
    for k, v in MARKETS.items():
        if ticker in v:
            return k
    return "other"


# ---------------------------------------------------------------- signals ---
def ewma_crossover(prices: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """Normalised fast-minus-slow EWMA, the standard managed-futures signal.

    Dividing the raw crossover by trailing price volatility puts every market
    and every speed on the same scale, which is what lets you average them.
    """
    f = prices.ewm(span=fast, min_periods=fast).mean()
    s = prices.ewm(span=slow, min_periods=slow).mean()
    raw = (f - s) / prices
    return raw / raw.rolling(252, min_periods=120).std().replace(0, np.nan)


def tsmom_sign(prices: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Moskowitz-Ooi-Pedersen in its original form: the sign of the past return."""
    return np.sign(prices / prices.shift(lookback) - 1.0)


def breakout(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Where price sits in its own N-day range, mapped to [-1, +1]."""
    hi = prices.rolling(window, min_periods=window // 2).max()
    lo = prices.rolling(window, min_periods=window // 2).min()
    return (2 * (prices - lo) / (hi - lo).replace(0, np.nan) - 1.0)


def ensemble(prices: pd.DataFrame,
             pairs: tuple[tuple[int, int], ...] = ((8, 24), (16, 48), (32, 96),
                                                   (64, 192))) -> pd.DataFrame:
    """Average several speeds.

    Any single speed is a bet on a particular trend duration, and which
    duration pays changes decade by decade. Averaging four is not a hedge
    against being wrong; it is an admission that the horizon is not knowable,
    and it is worth roughly as much as the signal itself (section 7.5).
    """
    parts = [ewma_crossover(prices, f, s) for f, s in pairs]
    return sum(parts) / len(parts)


def squash(signal: pd.DataFrame, cap: float = 2.0) -> pd.DataFrame:
    """Clip the signal.

    Uncapped, a trend signal takes its largest position in the most extended
    market — which is precisely where reversals happen. The cap costs a little
    return in long trends and removes the worst single days.
    """
    return signal.clip(-cap, cap)


# -------------------------------------------------------------- portfolio ---
def risk_scaled_positions(signal: pd.DataFrame, returns: pd.DataFrame,
                          target_vol_per_market: float = 0.02,
                          vol_window: int = 63, min_periods: int = 40,
                          max_weight: float = 0.25) -> pd.DataFrame:
    """Convert a unit-free signal into weights of equal risk per market.

    Each market is sized so its *own* contribution to portfolio volatility is
    roughly `target_vol_per_market`, regardless of whether it is 6-vol bonds or
    45-vol natural gas. Without this the book is a natural-gas fund with a bond
    ticker attached.
    """
    vol = returns.rolling(vol_window, min_periods=min_periods).std() * np.sqrt(252)
    w = signal * (target_vol_per_market / vol.replace(0, np.nan))
    return w.clip(-max_weight, max_weight).fillna(0.0)


def bucket_normalise(weights: pd.DataFrame) -> pd.DataFrame:
    """Divide each market's weight by the number of live markets in its bucket.

    Eight international equity ETFs and four FX ETFs are not eight and four
    independent bets. Equalising risk across *buckets* rather than across
    tickers is the cheapest correlation control there is, and on this universe
    it does most of what a full covariance model would do.
    """
    out = weights.copy()
    for _, tickers in MARKETS.items():
        cols = [c for c in tickers if c in weights.columns]
        if not cols:
            continue
        live = (weights[cols] != 0).sum(axis=1).replace(0, np.nan)
        out[cols] = weights[cols].div(live, axis=0)
    return out.fillna(0.0)


def portfolio_vol_target(weights: pd.DataFrame, portfolio_returns: pd.Series,
                         target_vol: float = 0.10, window: int = 63,
                         max_leverage: float = 4.0) -> pd.DataFrame:
    realised = portfolio_returns.rolling(window, min_periods=20).std() * np.sqrt(252)
    scale = (target_vol / realised.replace(0, np.nan)).shift(1)
    return weights.mul(scale.clip(upper=max_leverage).fillna(1.0), axis=0)


# ------------------------------------------------------------------- API ---
def build(prices: pd.DataFrame, returns: pd.DataFrame,
          pairs=((8, 24), (16, 48), (32, 96), (64, 192)),
          cap: float = 2.0, target_vol_per_market: float = 0.02,
          rebalance: str = "W", normalise_buckets: bool = True):
    """Return (signal, target weights) for the trend book."""
    cols = [c for c in universe() if c in prices.columns]
    px, rt = prices[cols], returns[cols]

    sig = squash(ensemble(px, pairs), cap)

    if rebalance.upper().startswith("W"):
        marks = sig.resample("W-FRI").last().index
        sig_used = sig.where(pd.Series(sig.index.isin(marks), index=sig.index),
                             np.nan).ffill()
    elif rebalance.upper().startswith("M"):
        marks = sig.resample("ME").last().index
        sig_used = sig.where(pd.Series(sig.index.isin(marks), index=sig.index),
                             np.nan).ffill()
    else:
        sig_used = sig

    w = risk_scaled_positions(sig_used, rt, target_vol_per_market)
    if normalise_buckets:
        w = bucket_normalise(w)
    return sig, w

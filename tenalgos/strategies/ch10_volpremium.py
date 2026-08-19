"""Chapter 10 — The Volatility Risk Premium.

Implied volatility exceeds subsequently realised volatility, most of the time.
Whoever sells the difference is selling insurance, and gets paid for it right up
until the claim arrives.

MEASURING IT WITHOUT AN OPTIONS CHAIN
-------------------------------------
The textbook construction is a delta-hedged short straddle, which needs an
options chain nobody gives away. But the premium is directly observable in
instruments our ETF panel already contains, because a VIX-futures ETF *is* a
packaged short- or long-volatility position:

    VIXY   long front VIX futures  — the insurance BUYER. Its long-run decay
                                     is the premium being paid, measured.
    SVXY   inverse VIX futures     — the insurance SELLER, daily rebalanced.
    UVXY   2x long VIX futures     — the leveraged buyer.
    VIXM   mid-term VIX futures    — slower decay, shallower drawdown.

That is a narrower instrument set than an options desk has, and it is honest,
free, and contains the event that matters most.

THE TWO WAYS TO BE SHORT, AND WHY THEY DIFFER
---------------------------------------------
Shorting VIXY and holding SVXY are not the same trade, and the difference is
the whole risk-management lesson of the chapter:

    short VIXY at fixed notional   your loss is unbounded; a 100% rise in VIXY
                                   costs you 100% of the position, and you must
                                   post margin against it as it goes against you
    hold SVXY                      the ETF rebalances daily, so your exposure
                                   shrinks automatically as it loses. You cannot
                                   lose more than you put in — but daily
                                   rebalancing also guarantees decay in choppy
                                   markets.

DATA NOTE
---------
This module requires the panel loaded with `max_abs_return=None`. The default
+/-60% outlier mask is correct for equities and deletes the real -83% SVXY
move of 6 February 2018.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VOL_ETFS = {
    "VIXY": "long front VIX futures — the buyer",
    "UVXY": "2x long front VIX futures",
    "VIXM": "long mid-term VIX futures",
    "SVXY": "inverse VIX futures — the seller",
}


# --------------------------------------------------------- measuring the VRP ---
def realised_vol(returns: pd.Series, window: int = 21) -> pd.Series:
    """Close-to-close realised volatility, annualised."""
    return returns.rolling(window, min_periods=window // 2).std() * np.sqrt(252)


def premium_from_decay(vixy_returns: pd.Series, window: int = 252) -> pd.Series:
    """The variance risk premium, read off the insurance buyer's P&L.

    A long VIX-futures position earns the spot VIX move and pays the roll.
    Over any period in which volatility ends where it started, the entire
    return is the premium — paid by the holder, received by whoever is short.
    Rolling one-year decay is therefore a direct, tradeable estimate of the
    premium, with no model and no options chain.
    """
    return vixy_returns.rolling(window, min_periods=window // 2).sum()


# -------------------------------------------------------------- strategies ---
def short_vol_fixed_notional(vixy_returns: pd.Series, gross: float = 0.20
                             ) -> pd.Series:
    """Short a fixed notional of the long-vol ETF, rebalanced monthly.

    `gross` is deliberately small. A 20% position in an instrument that rose
    34% in two days in February 2018 is already a 7% portfolio loss, and that
    is the *conservative* version.
    """
    w = pd.Series(-gross, index=vixy_returns.index)
    return w


def vol_targeted_short(vixy_returns: pd.Series, target_vol: float = 0.10,
                       window: int = 63, max_gross: float = 0.35) -> pd.Series:
    """Size the short so the position's own volatility tracks a target.

    Same rule as Chapters 5, 6, 7 and 16, applied to the most volatile
    instrument in the book. The cap matters more here than anywhere else:
    trailing volatility is low precisely when the next claim is closest.
    """
    rv = realised_vol(vixy_returns, window)
    w = -(target_vol / rv.replace(0, np.nan)).shift(1)
    return w.clip(lower=-max_gross, upper=0.0).fillna(0.0)


def tail_hedged_short(vixy_returns: pd.Series, hedge_returns: pd.Series,
                      target_vol: float = 0.10, hedge_frac: float = 0.15,
                      window: int = 63) -> tuple[pd.Series, pd.Series]:
    """Vol-targeted short, plus a small long in a slower long-vol instrument.

    The hedge costs premium every day and pays on the days that matter. The
    question section 10.7 answers with numbers is whether it costs less than
    the disaster it prevents.
    """
    w_short = vol_targeted_short(vixy_returns, target_vol, window)
    w_hedge = (-w_short) * hedge_frac
    return w_short, w_hedge


def backtest_weights(weights: dict[str, pd.Series], returns: pd.DataFrame,
                     execution_lag: int = 1) -> pd.Series:
    """Apply weights to returns with the usual lag. Costs handled by caller."""
    total = None
    for tk, w in weights.items():
        r = returns[tk].fillna(0.0)
        part = w.reindex(r.index).ffill().fillna(0.0).shift(execution_lag) * r
        total = part if total is None else total.add(part, fill_value=0.0)
    return total.fillna(0.0)

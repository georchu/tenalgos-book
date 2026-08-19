"""Chapter 8 — Carry.

Carry is what you earn if nothing moves. Own the thing with the higher yield,
short the thing with the lower one, and collect the difference for as long as
prices stay where they are. Koijen, Moskowitz, Pedersen and Vrugt showed the
same trade works in every asset class anyone has measured.

MEASURING CARRY FROM PRICES ALONE
---------------------------------
For an ETF the cleanest price-only measure is its distribution yield, and both
series needed are already in the panel:

    adjclose   total return — price change plus reinvested distributions
    close      price only

    distribution_t = adjclose_return_t - close_return_t

Sum that over a year and you have a trailing yield per market, computed with no
vendor, no assumptions and no look-ahead. Median across our ETF universe is
1.72% a year, which is the right order of magnitude and the first sign the
measure is not nonsense.

WHAT THIS DOES NOT COVER, AND WHY THE CHAPTER SAYS SO
----------------------------------------------------
Two of the four classical carry markets are missing:

    commodity carry  is roll yield — the slope of the futures curve. An ETF
                     distributes nothing and buries the roll inside its price,
                     so the signal is unmeasurable from the panel.
    FX carry         is the interest-rate differential. A currency ETF earns
                     local money-market interest, but the large ones distribute
                     it irregularly and some accrue it into NAV, so the yield
                     estimate is too noisy to trade.

Carry here therefore runs on equities, rates, credit and real assets — four
sleeves, not six. The book does not fake the other two.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ch05_xsmom import zscore_cross_section, trailing_vol, decile_weights


# ---------------------------------------------------------------------------
# The tradeable carry universe, classified by hand and printed here so the
# classification is auditable rather than hidden in a vendor field.
#
# Two exclusions are deliberate and load-bearing:
#   * LEVERAGED AND INVERSE funds (TQQQ, SQQQ, FAZ, UVXY, TBT, SDOW, ...) are
#     excluded entirely. Their distributions are an artefact of daily
#     rebalancing, they decay against their own index, and no carry investor
#     holds them. Leaving them in produces a much better backtest and a
#     completely fictitious one.
#   * COMMODITY and FX funds are excluded because their carry is unmeasurable
#     from a distribution yield — see the module docstring.
# ---------------------------------------------------------------------------
GROUPS: dict[str, list[str]] = {
    # government and aggregate duration
    "rates": ["SHY", "SHV", "IEI", "IEF", "TLT", "TIP", "AGG", "BND", "BSV",
              "BIV", "MBB", "MUB", "BWX", "BIL", "MINT", "FLOT"],
    # corporate, high yield, EM and floating-rate credit
    "credit": ["LQD", "HYG", "JNK", "SJNK", "EMB", "PCY", "CWB", "BKLN",
               "IGIB", "IGSB", "VCIT", "VCSH", "PFF"],
    # broad and style US equity
    "equity_us": ["SPY", "IVV", "VOO", "VTI", "VT", "VV", "VO", "VB", "IWB",
                  "IWM", "IWR", "IWV", "IJH", "IJR", "MDY", "OEF", "RSP",
                  "SCHB", "SCHX", "VXF", "QQQ", "DIA", "IVE", "IVW", "IWD",
                  "IWF", "IWN", "IWO", "IWP", "IWS", "IJJ", "IJK", "IJS",
                  "IJT", "VBK", "VBR", "VOE", "VTV", "VUG", "SDY", "DVY",
                  "VYM", "VIG", "HDV", "SPLV", "USMV"],
    # international equity
    "equity_intl": ["EFA", "EEM", "IEMG", "VEA", "VEU", "VWO", "VGK", "VPL",
                    "AAXJ", "ACWI", "SCZ", "EZU", "IEV", "FEZ", "EWA", "EWC",
                    "EWG", "EWH", "EWI", "EWJ", "EWL", "EWM", "EWP", "EWS",
                    "EWT", "EWU", "EWW", "EWY", "EWZ", "EZA", "EPP", "ILF",
                    "MCHI", "FXI", "DXJ", "EPI", "PIN", "RSX"],
    # US equity sectors — same asset class, different exposure
    "sector": ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
               "IYE", "IYF", "IYM", "IYT", "IYW", "VDE", "VFH", "VGT", "VHT",
               "IBB", "XBI", "SMH", "SOXX", "OIH", "XOP", "XME", "XRT", "XHB",
               "ITB", "KBE", "KRE", "MOO", "IGE", "FDN"],
    # property and real assets that actually distribute
    "real": ["VNQ", "IYR", "ICF", "RWR", "RWX", "AMLP"],
}


def group_series(columns) -> "pd.Series":
    """Map tickers to groups; anything unclassified is dropped, not guessed."""
    m = {t: g for g, ts in GROUPS.items() for t in ts}
    s = pd.Series({c: m.get(c) for c in columns}, name="group")
    return s.dropna()


def universe() -> list[str]:
    return sorted({t for ts in GROUPS.values() for t in ts})


# ---------------------------------------------------------------- signal ---
def distribution_yield(adj_close: pd.DataFrame, close: pd.DataFrame,
                       window: int = 252, min_periods: int = 200) -> pd.DataFrame:
    """Trailing distribution yield, in annual return units.

    Uses only data up to and including t. Negative values are data errors
    rather than negative distributions, and are dropped rather than trusted.
    """
    tr = adj_close.pct_change(fill_method=None)
    pr = close.reindex_like(adj_close).pct_change(fill_method=None)
    dist = (tr - pr).where(lambda d: d.abs() < 0.25)      # drop adjustment glitches
    y = dist.rolling(window, min_periods=min_periods).sum()
    return y.where(y > -0.02)


def carry_signal(adj_close: pd.DataFrame, close: pd.DataFrame,
                 returns: pd.DataFrame, vol_window: int = 126,
                 risk_adjust: bool = True) -> pd.DataFrame:
    """Yield, divided by volatility so it is a *risk-adjusted* carry.

    This is the step that separates carry from reaching for yield. A 9% yielding
    fund at 30% volatility and a 3% yielding fund at 4% volatility offer very
    different compensation per unit of risk, and only one of them is a trade.
    """
    y = distribution_yield(adj_close, close)
    if not risk_adjust:
        return y
    vol = trailing_vol(returns, vol_window) * np.sqrt(252)
    return y / vol.replace(0, np.nan)


def group_neutralise(signal: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Demean within asset group.

    Without it the strategy is one bet — long credit, short government bonds —
    dressed as a diversified carry book. With it, you hold the high-carry names
    *within* each group, which is what the literature actually documents.
    """
    out = signal.copy()
    g = groups.reindex(signal.columns)
    for _, members in g.groupby(g).groups.items():
        cols = [c for c in members if c in signal.columns]
        if len(cols) < 4:
            continue
        blk = signal[cols]
        out[cols] = blk.sub(blk.mean(axis=1), axis=0)
    return out


# ------------------------------------------------------------------- API ---
def build(adj_close: pd.DataFrame, close: pd.DataFrame, returns: pd.DataFrame,
          groups: pd.Series | None = None, n_bins: int = 5,
          neutralise_groups: bool = True, rebalance: str = "M",
          risk_adjust: bool = True, gross: float = 2.0):
    """Return (signal, target weights) for the carry book."""
    sig = carry_signal(adj_close, close, returns, risk_adjust=risk_adjust)
    if neutralise_groups and groups is not None:
        sig = group_neutralise(sig, groups)
    sig = zscore_cross_section(sig)

    if rebalance.upper().startswith("M"):
        marks = sig.resample("ME").last().index
        used = sig.where(pd.Series(sig.index.isin(marks), index=sig.index),
                         np.nan).ffill()
    else:
        used = sig
    return sig, decile_weights(used, n_bins=n_bins, gross=gross)

"""Chapter 11 — Machine-Learning Alpha Ensemble.

Deliberately the last of the seven strategies, not the first.

WHAT ML IS AND IS NOT
---------------------
A gradient-boosted tree is a better *estimator*. It is not a new *edge*. It can
find that momentum works better in low-volatility names, or that reversal
reverses at a different speed after an earnings gap — interactions a linear
model cannot express. What it cannot do is invent a reason someone will pay
you, and every chapter in Part II opened with that reason for a purpose.

So the honest test of this chapter is not "does the model beat momentum". It is
"does the model add anything the six previous sleeves do not already contain",
and section 11.7 answers it.

THE FEATURES
------------
All price-derived, all computed from data at or before day t, all
cross-sectionally z-scored so the model sees relative position rather than
level. The z-scoring matters more than it looks: without it the model spends
its capacity learning that 2008 had big numbers in it.

THE LABEL
---------
Forward 21-day return, cross-sectionally demeaned and z-scored. Demeaning is
what makes this a relative-value model rather than a market-timing one, and it
is the difference between predicting "will this stock go up" (hard, dominated
by the market) and "will this stock beat its peers" (the actual question).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ch05_xsmom import zscore_cross_section, trailing_vol, trailing_beta


# ---------------------------------------------------------------- features ---
def make_features(prices: pd.DataFrame, returns: pd.DataFrame,
                  volume: pd.DataFrame | None = None,
                  dollar_volume: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Twelve price-derived features, each cross-sectionally z-scored.

    Every one of these uses only information available at the close of day t.
    The engine still applies its own execution lag on top, so there are two
    independent layers of protection against the mistake in Chapter 2.2.
    """
    f: dict[str, pd.DataFrame] = {}
    vol126 = trailing_vol(returns, 126).replace(0, np.nan)

    # --- trend, at four horizons -------------------------------------------
    for lb, skip, name in [(252, 21, "mom_12_1"), (126, 21, "mom_6_1"),
                           (63, 5, "mom_3_1"), (21, 1, "rev_1m")]:
        raw = prices.shift(skip) / prices.shift(lb) - 1.0
        f[name] = raw / vol126

    # short-horizon reversal carries the opposite sign by convention
    f["rev_1m"] = -f["rev_1m"]
    f["rev_1w"] = -(prices.shift(1) / prices.shift(6) - 1.0) / vol126

    # --- risk ---------------------------------------------------------------
    f["vol_126"] = vol126
    f["vol_ratio"] = (trailing_vol(returns, 21, 15) / vol126)  # vol regime shift
    market = returns.mean(axis=1)
    f["beta"] = trailing_beta(returns, market)
    resid = returns.sub(f["beta"].mul(market, axis=0))
    f["idio_vol"] = resid.rolling(126, min_periods=60).std()

    # --- shape --------------------------------------------------------------
    f["skew_126"] = returns.rolling(126, min_periods=60).skew()
    f["max_ret_21"] = returns.rolling(21, min_periods=15).max()   # lottery demand

    # --- liquidity ----------------------------------------------------------
    if dollar_volume is not None:
        f["log_dv"] = np.log(dollar_volume.replace(0, np.nan))
        # Amihud illiquidity: |return| per dollar traded
        f["amihud"] = (returns.abs() / dollar_volume.replace(0, np.nan)
                       ).rolling(21, min_periods=15).mean()

    return {k: zscore_cross_section(v) for k, v in f.items()}


def make_label(returns: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Forward `horizon`-day return, cross-sectionally demeaned and z-scored.

    NOTE the sign of the shift. This is a *forward* return and therefore
    contains the future by construction; it is the thing being predicted, and it
    must never appear in the feature set. The purged CV in `cv.py` exists to
    stop the OVERLAP between adjacent labels leaking across a fold boundary.
    """
    fwd = returns.shift(-1).rolling(horizon).sum().shift(-(horizon - 1))
    return zscore_cross_section(fwd)


# ------------------------------------------------------------------ panel ---
def to_long(features: dict[str, pd.DataFrame], label: pd.DataFrame | None = None,
            min_names: int = 100) -> pd.DataFrame:
    """Stack the wide frames into one long (date, ticker) table for the model."""
    parts = {k: v.stack(future_stack=True) for k, v in features.items()}
    X = pd.DataFrame(parts)
    if label is not None:
        X["y"] = label.stack(future_stack=True)
    X = X.dropna()
    counts = X.groupby(level=0).size()
    keep = counts[counts >= min_names].index
    return X[X.index.get_level_values(0).isin(keep)]


# ------------------------------------------------------------- prediction ---
def predictions_to_signal(pred: pd.Series, index: pd.Index,
                          columns: pd.Index) -> pd.DataFrame:
    """Unstack model predictions back to a dates x tickers signal frame."""
    s = pred.unstack()
    return zscore_cross_section(s.reindex(index=index, columns=columns))

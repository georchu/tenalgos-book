"""A synthetic equity market with known ground truth.

Why this module exists, and why it is Chapter 4 rather than an appendix:

You cannot validate a backtester on real data. On real data you do not know the
right answer, so a bug that inflates returns looks exactly like a discovery. The
only way to know your engine is correct is to run it on a market you built
yourself, where you injected the alpha and therefore know what the engine is
supposed to find.

`make_market` generates daily returns from a factor model:

    r[i,t] = beta[i] . f[t] + alpha_signal[i,t] * ic_strength + eps[i,t]

with an optional momentum effect: a slow-moving latent score whose lagged value
genuinely predicts the next period's idiosyncratic return. Set `mom_strength=0`
and the market has no cross-sectional momentum at all — which is the single most
useful test in the whole library, because a backtester with look-ahead
leakage will still report a Sharpe of 2 on that market.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def make_market(
    n_assets: int = 500,
    n_days: int = 252 * 20,
    n_factors: int = 4,
    mom_strength: float = 0.0006,
    mom_halflife: int = 90,
    factor_vol: float = 0.010,
    idio_vol: float = 0.018,
    vol_clustering: float = 0.94,
    survivorship: bool = True,
    seed: int = 20260101,
    start: str = "2006-01-03",
) -> dict:
    """Generate a synthetic equity panel with a known momentum effect.

    Parameters
    ----------
    mom_strength
        Daily expected excess return per unit of standardised momentum score.
        0.0006 means an asset one cross-sectional standard deviation above
        average earns 6bp/day more — about 15% a year at one sigma, which is
        the right order of magnitude for the real effect before costs and
        before the signal's own measurement error.
    survivorship
        If True, some assets are delisted over time and new ones list, so the
        panel has genuine entry and exit. Loaders that ignore this produce
        survivorship bias, which Chapter 2 measures using exactly this switch.

    Returns
    -------
    dict with keys:
        prices      (n_days x n_assets) DataFrame, adjusted close
        returns     (n_days x n_assets) DataFrame, simple daily returns
        mcap        (n_days x n_assets) DataFrame, market capitalisation in USD
        industry    Series, asset -> industry code
        alive       (n_days x n_assets) boolean DataFrame, tradeable mask
        truth       dict of the parameters actually used
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    assets = [f"SY{i:04d}" for i in range(n_assets)]

    # ---- factor structure -------------------------------------------------
    # One market factor with high loadings, plus style/sector factors.
    betas = np.column_stack([
        rng.normal(1.0, 0.30, n_assets),
        *[rng.normal(0.0, 0.55, n_assets) for _ in range(n_factors - 1)],
    ])

    # Factor returns with volatility clustering (GARCH-like), so that drawdowns
    # cluster the way they do in real markets.
    f = np.zeros((n_days, n_factors))
    sig = np.full(n_factors, factor_vol)
    for t in range(n_days):
        shock = rng.normal(0.0, 1.0, n_factors)
        f[t] = sig * shock
        sig = np.sqrt(vol_clustering * sig**2 + (1 - vol_clustering) * (factor_vol**2 + 0.55 * f[t] ** 2))

    # ---- the latent momentum score ---------------------------------------
    # A slow AR(1) score. Its value at t-1 predicts idiosyncratic return at t.
    phi = 0.5 ** (1.0 / mom_halflife)
    score = np.zeros((n_days, n_assets))
    s = rng.normal(0, 1, n_assets)
    for t in range(n_days):
        s = phi * s + np.sqrt(1 - phi**2) * rng.normal(0, 1, n_assets)
        score[t] = s

    # Standardise cross-sectionally so mom_strength has a stable interpretation.
    z = (score - score.mean(axis=1, keepdims=True)) / (score.std(axis=1, keepdims=True) + 1e-12)

    idio = rng.normal(0.0, idio_vol, (n_days, n_assets))
    common = f @ betas.T

    rets = common + idio
    # The predictive part: yesterday's score pays off today. Lagging by one day
    # is what makes this an honestly tradeable effect rather than a same-bar
    # look-ahead.
    rets[1:] += mom_strength * z[:-1]

    # ---- listings and delistings -----------------------------------------
    alive = np.ones((n_days, n_assets), dtype=bool)
    if survivorship:
        # ~3% of names per year leave; a matching number arrive late.
        n_dead = int(n_assets * 0.35)
        dead_idx = rng.choice(n_assets, n_dead, replace=False)
        for i in dead_idx:
            death = rng.integers(int(n_days * 0.15), n_days)
            alive[death:, i] = False
            # A delisting is usually not a good day.
            rets[death - 1, i] += rng.normal(-0.22, 0.18)
        late_idx = rng.choice(np.setdiff1d(np.arange(n_assets), dead_idx),
                              int(n_assets * 0.25), replace=False)
        for i in late_idx:
            birth = rng.integers(0, int(n_days * 0.6))
            alive[:birth, i] = False

    rets = np.where(alive, rets, np.nan)

    returns = pd.DataFrame(rets, index=dates, columns=assets)
    prices = 20.0 * np.exp(np.nancumsum(np.nan_to_num(rets), axis=0))
    prices = pd.DataFrame(prices, index=dates, columns=assets).where(alive)

    # Market cap: a wide, realistic power-law spread, drifting with price.
    base_cap = np.exp(rng.normal(np.log(2.5e9), 1.5, n_assets))
    mcap = pd.DataFrame(base_cap * (prices.values / prices.iloc[0].values), index=dates, columns=assets)

    industry = pd.Series(rng.integers(0, 11, n_assets), index=assets, name="industry")

    return {
        "prices": prices,
        "returns": returns,
        "mcap": mcap,
        "industry": industry,
        "alive": pd.DataFrame(alive, index=dates, columns=assets),
        "score": pd.DataFrame(z, index=dates, columns=assets),
        "truth": {
            "mom_strength": mom_strength,
            "mom_halflife": mom_halflife,
            "n_factors": n_factors,
            "seed": seed,
            "annualised_alpha_per_sd": mom_strength * TRADING_DAYS,
        },
    }

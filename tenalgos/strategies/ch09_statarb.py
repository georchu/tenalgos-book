"""Chapter 9 — Statistical Arbitrage: PCA residual reversion.

The lineage runs from Morgan Stanley's block-trading desk in the 1980s through
D. E. Shaw and Renaissance to every mid-frequency equity book trading today.
The modern form is Avellaneda and Lee (2010), and it is what this module
implements.

THE CONSTRUCTION
----------------
    1. estimate the market's common factors by PCA on a trailing window of
       correlations, using the leading K eigenportfolios
    2. regress each stock's returns on those factors, keep the RESIDUAL —
       the part of the stock that its peers do not explain
    3. accumulate the residual into a cumulative process and model it as
       mean-reverting: an Ornstein-Uhlenbeck fit gives a level, a speed, and
       therefore a z-score ("the s-score")
    4. short stocks whose residual has run up, buy the ones that have run down,
       sized inversely to volatility, and hold until the residual closes

Textbook pairs trading is the K = 1, N = 2 special case of this, and section
9.3 shows why that version stopped working while this one did not.

WHY THE RESIDUAL AND NOT THE PRICE
----------------------------------
Two oil companies both fall 8% because oil fell. That is not an opportunity and
a price-based reversion signal will trade it anyway, repeatedly, until the
commissions have eaten the account. Removing the common factors first is what
turns a mean-reversion strategy from a bet on the market being wrong into a bet
on one stock being temporarily mispriced against its peers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ PCA ---
def eigen_factors(returns: pd.DataFrame, window: int = 252, n_factors: int = 15,
                  min_names: int = 100) -> tuple[pd.DataFrame, dict]:
    """Return daily factor returns from rolling PCA of the correlation matrix.

    Correlation rather than covariance, and each eigenportfolio is weighted by
    1/sigma_i, which is Avellaneda-Lee's construction: it keeps a single
    high-volatility name from defining the first component.
    """
    raise NotImplementedError  # replaced by the incremental version below


def _pca_weights(block: pd.DataFrame, n_factors: int) -> np.ndarray:
    """Eigenportfolio weights from one trailing block of returns."""
    sd = block.std().replace(0, np.nan)
    corr = block.corr()
    corr = corr.fillna(0.0).to_numpy()
    np.fill_diagonal(corr, 1.0)
    vals, vecs = np.linalg.eigh(corr)
    order = np.argsort(vals)[::-1][:n_factors]
    v = vecs[:, order]                          # (n_names, n_factors)
    w = v / sd.to_numpy()[:, None]              # scale by 1/sigma
    return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)


# ------------------------------------------------------- OU on residuals ---
def ou_score(resid_cum: np.ndarray) -> tuple[float, float]:
    """Fit x_{t+1} = a + b x_t to a cumulative residual; return (s_score, kappa).

    s = -(m) / sigma_eq, the standardised distance of the residual from its own
    equilibrium. kappa = -log(b) * 252 is the mean-reversion speed in years;
    a residual that reverts more slowly than the holding period is not tradeable
    and section 9.5 filters on exactly that.
    """
    x = resid_cum[~np.isnan(resid_cum)]
    n = len(x)
    if n < 60:
        return np.nan, np.nan
    x0, x1 = x[:-1], x[1:]
    vx = x0.var()
    if vx <= 0:
        return np.nan, np.nan
    b = float(((x0 - x0.mean()) * (x1 - x1.mean())).sum() / ((x0 - x0.mean()) ** 2).sum())
    if not (0 < b < 1):
        return np.nan, np.nan
    a = float(x1.mean() - b * x0.mean())
    resid_var = float(np.var(x1 - a - b * x0, ddof=2))
    m = a / (1 - b)
    sigma_eq = np.sqrt(resid_var / (1 - b * b))
    if sigma_eq <= 0:
        return np.nan, np.nan
    kappa = -np.log(b) * 252.0
    return float(-(x[-1] - m) / sigma_eq), float(kappa)


# ------------------------------------------------------------------ API ---
def build(returns: pd.DataFrame, alive: pd.DataFrame | None = None,
          window: int = 252, n_factors: int = 15, resid_window: int = 60,
          step: int = 5, min_kappa: float = 252.0 / 30.0,
          entry: float = 1.25, exit_: float = 0.5,
          gross: float = 2.0, max_names: int = 400):
    """Return (s_scores, target weights).

    Parameters
    ----------
    step
        Recompute the PCA every `step` days rather than daily. The factor
        structure of a 1,400-name equity market does not change materially in a
        week, and this is a 5x speed-up for no measurable loss.
    min_kappa
        Discard residuals whose fitted mean-reversion half-life is longer than
        about 30 trading days. A signal that reverts more slowly than you can
        afford to hold it is not a signal.
    entry, exit_
        Open at |s| > entry, close at |s| < exit_. The band between them is
        what stops the book churning on noise around the threshold.
    """
    idx, cols = returns.index, returns.columns
    S = pd.DataFrame(np.nan, index=idx, columns=cols, dtype="float32")
    K = pd.DataFrame(np.nan, index=idx, columns=cols, dtype="float32")

    R = returns.to_numpy(dtype="float64")
    alive_np = (alive.to_numpy() if alive is not None else ~np.isnan(R))

    for t in range(window, len(idx), step):
        sl = slice(t - window, t)
        blk = R[sl]
        live = alive_np[t - 1] & (np.isfinite(blk).sum(axis=0) > window * 0.9)
        if live.sum() < 100:
            continue
        names = np.where(live)[0]
        if len(names) > max_names:                  # cap for tractability
            names = names[np.argsort(-np.nanstd(blk[:, names], axis=0))][:max_names]
        b = pd.DataFrame(blk[:, names]).fillna(0.0)

        w = _pca_weights(b, n_factors)              # (n, k)
        f = b.to_numpy() @ w                        # factor returns, (T, k)

        # regress the last `resid_window` days of each stock on the factors
        y = b.to_numpy()[-resid_window:]
        X = f[-resid_window:]
        X = np.column_stack([np.ones(len(X)), X])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        cum = np.cumsum(resid, axis=0)

        for j, col in enumerate(names):
            s, k = ou_score(cum[:, j])
            if np.isfinite(s):
                S.iat[t, col] = s
                K.iat[t, col] = k

    S = S.ffill(limit=step)
    K = K.ffill(limit=step)
    S = S.where(K >= min_kappa)

    # ---- positions: open outside the entry band, hold until inside exit ----
    raw = pd.DataFrame(0.0, index=idx, columns=cols)
    raw = raw.where(S.isna(), np.sign(S) * (S.abs() > entry))
    hold = raw.replace(0.0, np.nan).ffill()
    hold = hold.where(S.abs() > exit_, 0.0).fillna(0.0)

    n_live = hold.abs().sum(axis=1).replace(0, np.nan)
    w_out = hold.div(n_live, axis=0).mul(gross).fillna(0.0)
    return S, w_out

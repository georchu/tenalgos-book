"""Load the real equity panel produced by fetch.py.

Every strategy in Part II takes the same three objects — prices, returns and a
tradeability mask — so the same code runs against the synthetic market and
against real data with nothing changed but the loader.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DEFAULT_DIR = DATA_ROOT / "equities"


def panel_dir(name: str = "equities") -> Path:
    """Resolve a named panel, falling back to a flat data/ layout.

    Every distinct universe lives in its own directory — data/equities,
    data/etfs — because a panel is defined by its universe as much as by its
    dates. Ingesting a second universe into the same folder rebuilds one panel
    from two, and every backtest already run against it silently changes.
    """
    d = DATA_ROOT / name
    if (d / "close.parquet").exists():
        return d
    if (DATA_ROOT / "close.parquet").exists():
        return DATA_ROOT
    return d


def load_panel(
    path: str | Path | None = None,
    start: str | None = "2006-01-01",
    end: str | None = None,
    min_price: float = 5.0,
    min_dollar_volume: float = 2_000_000,
    min_history_days: int = 300,
    top_n_by_liquidity: int | None = 1500,
    max_stale_days: int = 5,
    min_names_per_day: int = 100,
    max_abs_return: float | None = 0.60,
) -> dict:
    """Load and screen the panel.

    The screen matters more than most people expect. A universe that includes
    $2 stocks trading $80,000 a day will produce a beautiful backtest that
    cannot be traded with real money, because the strategy's returns come
    entirely from names you could never get filled in. `min_price` and
    `min_dollar_volume` are the two numbers that make a backtest honest, and
    Chapter 5 shows what happens to the Sharpe when you relax them.

    Returns the same dict shape as `tenalgos.data.synthetic.make_market`.
    """
    path = panel_dir(path) if isinstance(path, str) and "/" not in path else Path(
        path or DEFAULT_DIR)
    if not (path / "close.parquet").exists():
        raise FileNotFoundError(
            f"no panel at {path}. Run fetch.py first — see DATA.md.")

    close = pd.read_parquet(path / "close.parquet")
    volume = pd.read_parquet(path / "volume.parquet")
    dv = pd.read_parquet(path / "dollar_volume.parquet")

    if start:
        close = close.loc[start:]
    if end:
        close = close.loc[:end]
    volume = volume.reindex_like(close)
    dv = dv.reindex_like(close)

    # --- screens, all applied with information available on the day ---------
    tradeable = close.notna()
    tradeable &= close >= min_price
    tradeable &= dv >= min_dollar_volume

    if top_n_by_liquidity:
        rank = dv.rank(axis=1, ascending=False, method="first")
        tradeable &= rank <= top_n_by_liquidity

    # --- staleness -----------------------------------------------------------
    # A close that has not moved for days is a halt, a data error, or a name
    # nobody is trading. It passes a dollar-volume screen on the days it does
    # trade and then sits still, which quietly feeds a momentum signal a
    # constant price and a zero return. Require recent price variation.
    unchanged = (close.diff().fillna(1) == 0).to_numpy()
    run = np.zeros(unchanged.shape, dtype=np.int32)
    for i in range(1, unchanged.shape[0]):
        run[i] = np.where(unchanged[i], run[i - 1] + 1, 0)
    tradeable &= pd.DataFrame(run <= max_stale_days,
                              index=close.index, columns=close.columns)

    keep = tradeable.sum() >= min_history_days
    close = close.loc[:, keep]
    volume = volume.loc[:, keep]
    dv = dv.loc[:, keep]
    tradeable = tradeable.loc[:, keep]

    # --- the trading calendar ------------------------------------------------
    # The set of dates in your files is NOT the exchange calendar. A single
    # ticker with one stray row on Christmas Day inserts a holiday into the
    # panel, and every strategy then trades a day that did not exist. Keep only
    # dates on which a real cross-section was actually open.
    per_day = tradeable.sum(axis=1)
    real_days = per_day[per_day >= min_names_per_day].index
    close = close.loc[real_days]
    volume = volume.loc[real_days]
    dv = dv.loc[real_days]
    tradeable = tradeable.loc[real_days]

    returns = close.pct_change(fill_method=None).where(tradeable)
    # A single day's move beyond +/-60% in a large, liquid equity is almost
    # always a data error rather than a trade you could have made.
    #
    # It is a *screen*, not a truth, and it is wrong for some instruments.
    # SVXY genuinely fell 83% on 6 February 2018; UVXY genuinely rose 66% the
    # day before. Both are real, tradeable, and the single most important
    # event in the history of volatility selling — and the default mask
    # deletes them. Chapter 10 passes `max_abs_return=None` for exactly this
    # reason, and Chapter 2.6's rule generalises: the screen is part of the
    # strategy, so it belongs to the strategy rather than to the data.
    if max_abs_return is not None:
        returns = returns.mask(returns.abs() > max_abs_return)

    manifest = {}
    mf = path / "manifest.json"
    if mf.exists():
        manifest = json.loads(mf.read_text())

    return {
        "prices": close.where(tradeable),
        "returns": returns,
        "volume": volume,
        "dollar_volume": dv,
        "alive": tradeable,
        "industry": None,        # supplied separately; see sector_map()
        "mcap": None,
        "truth": {"source": "real", **manifest},
    }


def sector_map(tickers, path: str | Path = DEFAULT_DIR) -> pd.Series:  # noqa: D401
    """Sector codes for neutralisation.

    If you have a sector file, drop it in as data/sectors.csv with columns
    ticker,sector. Without one, this returns a single bucket, which disables
    industry neutralisation rather than faking it — a fake sector map is worse
    than none, because it silently changes what the strategy is paid for.
    """
    f = Path(path) / "sectors.csv"
    if f.exists():
        s = pd.read_csv(f).set_index("ticker")["sector"]
        return s.reindex(tickers).fillna("UNKNOWN")
    return pd.Series("ALL", index=pd.Index(tickers), name="sector")


def universe_report(panel: dict) -> pd.Series:
    """A one-glance description of what you are actually trading."""
    alive = panel["alive"]
    n = alive.sum(axis=1)
    return pd.Series({
        "first_date": str(alive.index.min().date()),
        "last_date": str(alive.index.max().date()),
        "trading_days": len(alive),
        "tickers_ever": int(alive.any().sum()),
        "names_per_day_median": int(n.median()),
        "names_per_day_min": int(n.min()),
        "names_per_day_max": int(n.max()),
        "panel_density": round(float(alive.sum().sum() / alive.size), 3),
    })

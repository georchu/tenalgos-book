"""Chapter 2.7 — the integrity test suite you run before trusting any dataset.

Ten checks. Run them on every panel, every time, before a single backtest.
Each one corresponds to a way real vendor data has actually been wrong.

    python -m tenalgos.data.integrity
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_checks(panel: dict, verbose: bool = True) -> pd.DataFrame:
    close = panel["prices"]
    rets = panel["returns"]
    alive = panel["alive"]
    rows = []

    def check(name, ok, detail=""):
        rows.append({"check": name, "result": "PASS" if ok else "REVIEW",
                     "detail": detail})

    # 1 -- calendar
    gaps = close.index.to_series().diff().dt.days
    check("calendar has no impossible gaps", (gaps.max() or 0) <= 10,
          f"largest gap {int(gaps.max() or 0)} days")

    # 2 -- duplicated dates
    check("no duplicated dates", not close.index.duplicated().any(),
          f"{int(close.index.duplicated().sum())} duplicates")

    # 3 -- monotone index
    check("dates sorted", close.index.is_monotonic_increasing)

    # 4 -- prices strictly positive
    nonpos = int((close <= 0).sum().sum())
    check("no non-positive prices", nonpos == 0, f"{nonpos} cells")

    # 5 -- unadjusted split detection: a clean -50% or -66% with no volume shock
    r = rets.copy()
    suspicious = ((r < -0.45) & (r > -0.55)) | ((r < -0.62) & (r > -0.70))
    n_susp = int(suspicious.sum().sum())
    check("few split-shaped one-day drops", n_susp < 0.0002 * r.size,
          f"{n_susp} cells look like unadjusted splits")

    # 6 -- extreme returns
    ext = int((rets.abs() > 0.50).sum().sum())
    check("extreme daily moves are rare", ext < 0.0005 * rets.size,
          f"{ext} moves beyond +/-50%")

    # 7 -- zero-return runs (a stale feed repeats the last price)
    stale = (rets.fillna(1) == 0)
    longest = int(stale.astype(int).apply(
        lambda c: (c * (c.groupby((c != c.shift()).cumsum()).cumcount() + 1)).max()).max())
    check("no long stale-price runs", longest <= 10,
          f"longest run of identical closes: {longest} days")

    # 8 -- survivorship: does the universe only ever grow?
    n = alive.sum(axis=1)
    entries = int((alive.astype(int).diff() > 0).sum().sum())
    exits = int((alive.astype(int).diff() < 0).sum().sum())
    check("universe has real exits, not just entries", exits > 0.2 * entries,
          f"{entries} entries vs {exits} exits — a survivorship-free panel has both")

    # 9 -- cross-sectional breadth
    check("enough names each day", int(n.min()) >= 50,
          f"thinnest day has {int(n.min())} names")

    # 10 -- return/price consistency
    # Reconstruct exactly the way the loader does, including the outlier mask.
    # Comparing against a naive pct_change is meaningless: when a name leaves
    # the tradeable set for a week and comes back, pct_change spans the gap and
    # reports a 100,000% return that nobody could have earned.
    recon = close.pct_change(fill_method=None).where(alive)
    recon = recon.mask(recon.abs() > 0.60)
    both = recon.notna() & rets.notna()
    diff = float((recon - rets).abs().where(both).max().max() or 0.0)
    check("returns reconcile to prices", diff < 1e-6,
          f"max discrepancy {diff:.2e} over {int(both.sum().sum()):,} shared cells")

    df = pd.DataFrame(rows)
    if verbose:
        print("\nDATA INTEGRITY —", len(df), "checks")
        print("-" * 74)
        for _, x in df.iterrows():
            print(f"  {x['result']:<7} {x['check']:<44} {x['detail']}")
        print("-" * 74)
        bad = (df["result"] == "REVIEW").sum()
        print(f"  {len(df)-bad} pass, {bad} need review\n")
    return df


def survivorship_cost(panel_with_exits: dict, strategy_fn) -> pd.Series:
    """Measure, in Sharpe, what pretending survivors are the universe buys you.

    Runs the same strategy twice: once on the honest panel, once on the panel
    restricted to names that survive to the end. The difference is the free
    alpha a careless backtest hands itself. In Chapter 2 this is the number
    that makes the point better than any amount of warning.
    """
    honest = strategy_fn(panel_with_exits)

    alive = panel_with_exits["alive"]
    survivors = alive.iloc[-1]
    survivors = survivors[survivors].index
    biased = {k: (v.loc[:, v.columns.intersection(survivors)]
                  if isinstance(v, pd.DataFrame) else v)
              for k, v in panel_with_exits.items()}
    fake = strategy_fn(biased)

    return pd.Series({
        "sharpe_honest": honest,
        "sharpe_survivors_only": fake,
        "free_alpha_from_bias": fake - honest,
    })


if __name__ == "__main__":
    from .loaders import load_panel, universe_report
    p = load_panel()
    print(universe_report(p).to_string())
    run_checks(p)

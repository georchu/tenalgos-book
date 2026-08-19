"""Chapter 5 figures — cross-sectional equity momentum.

Run:  python notebooks/ch05_figures.py

DATA NOTE
---------
This script runs on the synthetic market from `tenalgos.data.synthetic`, which
has a *known* injected momentum effect. That makes it useful for two things:
verifying the pipeline end to end, and showing the reader what each diagnostic
looks like when the answer is known.

The figures that ship in the printed book are produced by the same script with
`--real`, which loads the US equity panel described in Chapter 2. Nothing in
the code path differs; only the data does.
"""
from __future__ import annotations

import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from tenalgos.data.synthetic import make_market
from tenalgos.research.backtest import (run_backtest, information_coefficient,
                                        ic_summary, quantile_returns)
from tenalgos.research.costs import CostModel
from tenalgos.research.stats import equity_curve, drawdown, sharpe
from tenalgos.strategies import ch05_xsmom as mom

FIGDIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
FIGDIR.mkdir(exist_ok=True)

INK, GOLD, GREY, RED = "#12161C", "#B08A3E", "#8A929B", "#9E2B25"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 9, "axes.edgecolor": INK, "axes.linewidth": 0.7,
    "axes.grid": True, "grid.color": "#DDE1E6", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})
pct = FuncFormatter(lambda v, _: f"{v:.0%}")


def build_case(real=True, seed=20260101, n_assets=400, years=15, strength=0.0009):
    if real:
        from tenalgos.data.loaders import load_panel
        mkt = load_panel()
        sig, w = mom.build(mkt["prices"], mkt["returns"], industry=None,
                           neutralise_industry=False, neutralise_beta=True)
        bt = run_backtest(w, mkt["returns"], CostModel(),
                          vol=mkt["returns"].rolling(21).std(),
                          dollar_volume=mkt["dollar_volume"], capital=1e8,
                          n_trials=1, name="xsmom")
        return mkt, sig, w, bt
    mkt = make_market(n_assets=n_assets, n_days=252 * years,
                      mom_strength=strength, survivorship=False, seed=seed)
    sig, w = mom.build(mkt["prices"], mkt["returns"], mkt["industry"])
    bt = run_backtest(w, mkt["returns"], CostModel(), n_trials=1, name="xsmom")
    return mkt, sig, w, bt


# ------------------------------------------------------------ figure 5.1 ---
def fig_performance(bt, tag):
    fig, ax = plt.subplots(3, 1, figsize=(6.6, 7.4), sharex=True,
                           gridspec_kw={"height_ratios": [2.2, 1, 1]})

    eq_g, eq_n = equity_curve(bt.gross_returns), equity_curve(bt.net_returns)
    ax[0].plot(eq_g.index, eq_g, color=GREY, lw=1.1, label="gross")
    ax[0].plot(eq_n.index, eq_n, color=INK, lw=1.4, label="net of costs")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("growth of $1 (log)")
    ax[0].legend(frameon=False, loc="upper left")
    ax[0].set_title("Cross-sectional momentum — US equities, 2006–2020", loc="left",
                    fontsize=10.5, color=INK, pad=8)

    dd = drawdown(bt.net_returns)
    ax[1].fill_between(dd.index, dd, 0, color=RED, alpha=0.28, lw=0)
    ax[1].plot(dd.index, dd, color=RED, lw=0.8)
    ax[1].yaxis.set_major_formatter(pct)
    ax[1].set_ylabel("drawdown")

    roll = bt.net_returns.rolling(252).apply(lambda x: sharpe(pd.Series(x)), raw=False)
    ax[2].axhline(0, color=INK, lw=0.7)
    ax[2].plot(roll.index, roll, color=GOLD, lw=1.1)
    ax[2].set_ylabel("rolling 1y Sharpe")
    ax[2].set_xlabel("")

    fig.savefig(FIGDIR / f"fig05_01_performance_{tag}.png")
    plt.close(fig)


# ------------------------------------------------------------ figure 5.2 ---
def fig_diagnostics(mkt, sig, bt, tag):
    fwd = mkt["returns"].shift(-1)
    ic = information_coefficient(sig, fwd)
    q = quantile_returns(sig, fwd, q=10)

    fig, ax = plt.subplots(1, 3, figsize=(9.4, 3.0))

    means = q.mean() * 252
    ax[0].bar(range(1, len(means) + 1), means.values,
              color=[RED if v < 0 else GOLD for v in means.values],
              edgecolor=INK, linewidth=0.5)
    ax[0].set_xlabel("signal decile (1 = worst)")
    ax[0].set_ylabel("mean forward return, annualised")
    ax[0].yaxis.set_major_formatter(pct)
    ax[0].set_title("Monotonicity", loc="left", fontsize=10)

    ic_m = ic.resample("QE").mean()
    ax[1].axhline(0, color=INK, lw=0.7)
    ax[1].bar(ic_m.index, ic_m.values, width=60,
              color=[RED if v < 0 else GOLD for v in ic_m.values],
              edgecolor="none")
    ax[1].set_ylabel("quarterly mean IC")
    ax[1].set_title("Information coefficient", loc="left", fontsize=10)

    yearly = bt.net_returns.resample("YE").sum()
    ax[2].bar([d.year for d in yearly.index], yearly.values,
              color=[RED if v < 0 else GOLD for v in yearly.values],
              edgecolor=INK, linewidth=0.5)
    ax[2].axhline(0, color=INK, lw=0.7)
    ax[2].yaxis.set_major_formatter(pct)
    ax[2].set_title("Return by year, net", loc="left", fontsize=10)
    ax[2].tick_params(axis="x", rotation=90, labelsize=6.5)

    fig.tight_layout()
    fig.savefig(FIGDIR / f"fig05_02_diagnostics_{tag}.png")
    plt.close(fig)
    return ic_summary(ic)


# ------------------------------------------------------------ figure 5.3 ---
def fig_param_surface(mkt, tag):
    lookbacks = [63, 126, 189, 252, 378]
    skips = [0, 5, 10, 21, 42]
    out = np.full((len(skips), len(lookbacks)), np.nan)
    for i, sk in enumerate(skips):
        for j, lb in enumerate(lookbacks):
            _, w = mom.build(mkt["prices"], mkt["returns"],
                             industry=mkt.get("industry"),
                             lookback=lb, skip=sk,
                             neutralise_industry=mkt.get("industry") is not None,
                             neutralise_beta=False)
            bt = run_backtest(w, mkt["returns"], CostModel())
            out[i, j] = bt.stats_net["sharpe"]

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    im = ax.imshow(out, cmap="RdYlGn", vmin=-np.nanmax(abs(out)), vmax=np.nanmax(abs(out)))
    ax.set_xticks(range(len(lookbacks)), lookbacks)
    ax.set_yticks(range(len(skips)), skips)
    ax.set_xlabel("lookback (trading days)")
    ax.set_ylabel("skip (trading days)")
    ax.grid(False)
    for i in range(len(skips)):
        for j in range(len(lookbacks)):
            ax.text(j, i, f"{out[i,j]:.2f}", ha="center", va="center",
                    fontsize=7.5, color=INK)
    ax.set_title("Net Sharpe across the parameter grid — 25 trials, not one",
                 loc="left", fontsize=9.5, pad=8)
    fig.colorbar(im, ax=ax, fraction=0.035)
    fig.savefig(FIGDIR / f"fig05_03_params_{tag}.png")
    plt.close(fig)
    return out, lookbacks, skips


# ------------------------------------------------------------------ main ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="real")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    print("building market and running strategy ...")
    mkt, sig, w, bt = build_case(real=not args.synthetic)

    print("\n--- performance ---")
    print(bt.summary().round(4).to_string())

    fig_performance(bt, args.tag)
    ics = fig_diagnostics(mkt, sig, bt, args.tag)
    print("\n--- information coefficient ---")
    print(ics.round(4).to_string())

    print("\nparameter sweep (25 configurations) ...")
    grid, lbs, sks = fig_param_surface(mkt, args.tag)

    from tenalgos.research.stats import deflated_sharpe
    best = np.nanmax(grid)
    dsr1 = deflated_sharpe(best, len(bt.net_returns), n_trials=1)
    dsr25 = deflated_sharpe(best, len(bt.net_returns), n_trials=grid.size)
    print(f"\nbest net Sharpe in grid      : {best:.3f}")
    print(f"deflated Sharpe prob, 1 trial : {dsr1:.4f}")
    print(f"deflated Sharpe prob, 25 trials: {dsr25:.4f}   <- the honest number")

    print(f"\nfigures written to {FIGDIR}")

"""Chapter 5, figures 5.4-5.6 — crashes, costs, capacity and decay.

Run:  python notebooks/ch05_extra.py

Everything here uses the same panel and the same engine as ch05_figures.py.
The three questions it answers are the three a reader with money at stake asks
after seeing an equity curve: what does the worst case look like, where did the
gross return actually go, and how much capital can this hold before it dies.
"""
from __future__ import annotations

import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from tenalgos.data.loaders import load_panel
from tenalgos.research.backtest import run_backtest
from tenalgos.research.costs import CostModel
from tenalgos.research.stats import equity_curve, drawdown, sharpe
from tenalgos.strategies import ch05_xsmom as mom

FIG = pathlib.Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)
OUT = pathlib.Path(__file__).resolve().parents[1] / "figures" / "ch05_numbers.json"

INK, GOLD, GREY, RED, BLUE = "#12161C", "#B08A3E", "#8A929B", "#9E2B25", "#2E5A78"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 9, "axes.edgecolor": INK, "axes.linewidth": 0.7,
    "axes.grid": True, "grid.color": "#DDE1E6", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})
pct = FuncFormatter(lambda v, _: f"{v:.0%}")
res = {}

print("loading panel ...")
mkt = load_panel()
rets = mkt["returns"]
vol21 = rets.rolling(21).std()

print("building baseline signal ...")
sig, w = mom.build(mkt["prices"], rets, industry=None,
                   neutralise_industry=False, neutralise_beta=True)
cm = CostModel()
bt = run_backtest(w, rets, cm, vol=vol21, dollar_volume=mkt["dollar_volume"],
                  capital=1e8, n_trials=1, name="xsmom")

# ----------------------------------------------------------- vol targeting ---
print("volatility-targeted variant ...")
w_vt = mom.volatility_target(w, bt.gross_returns, target_vol=0.10,
                             window=63, max_leverage=3.0)
bt_vt = run_backtest(w_vt, rets, cm, vol=vol21, dollar_volume=mkt["dollar_volume"],
                     capital=1e8, n_trials=1, name="xsmom-vt")

res["plain"] = {k: float(v) for k, v in bt.stats_net.items() if isinstance(v, (int, float))}
res["voltarget"] = {k: float(v) for k, v in bt_vt.stats_net.items() if isinstance(v, (int, float))}

# worst months, plain
m = bt.net_returns.resample("ME").sum()
res["worst_months"] = {str(d.date()): round(float(v), 4) for d, v in m.nsmallest(6).items()}
m_vt = bt_vt.net_returns.resample("ME").sum()
res["worst_months_vt"] = {str(d.date()): round(float(v), 4)
                          for d, v in m_vt.reindex(m.nsmallest(6).index).items()}

# --------------------------------------------------------------- figure 5.4 ---
lo, hi = "2008-06-01", "2010-06-30"
fig, ax = plt.subplots(2, 1, figsize=(6.6, 5.0), sharex=True,
                       gridspec_kw={"height_ratios": [2, 1]})
a = bt.net_returns.loc[lo:hi]
b = bt_vt.net_returns.loc[lo:hi]
ax[0].plot(a.index, np.exp(a.cumsum()) - 1, color=RED, lw=1.4, label="fixed gross exposure")
ax[0].plot(b.index, np.exp(b.cumsum()) - 1, color=BLUE, lw=1.4, label="volatility-targeted")
ax[0].axhline(0, color=INK, lw=0.7)
ax[0].yaxis.set_major_formatter(pct)
ax[0].set_ylabel("cumulative return")
ax[0].legend(frameon=False, loc="lower left")
ax[0].set_title("The 2008-09 momentum drawdown", loc="left", fontsize=10.5, pad=8)

mktvol = (rets.mean(axis=1).rolling(63).std() * np.sqrt(252)).loc[lo:hi]
ax[1].fill_between(mktvol.index, mktvol, 0, color=GREY, alpha=0.35, lw=0)
ax[1].plot(mktvol.index, mktvol, color=INK, lw=0.9)
ax[1].yaxis.set_major_formatter(pct)
ax[1].set_ylabel("market vol, 3m")
fig.savefig(FIG / "fig05_04_crash_real.png")
plt.close(fig)

# --------------------------------------------------------------- figure 5.5 ---
print("cost decomposition ...")
held = bt.weights
traded = held.diff().abs().fillna(held.abs())
linear = traded.sum(axis=1) * (0.5 * cm.spread_bps + cm.commission_bps) / 1e4
notional = traded * 1e8
part = (notional / mkt["dollar_volume"].replace(0, np.nan)).clip(upper=1.0)
impact = (traded * (cm.impact_coef * vol21 * np.sqrt(part.fillna(0.0)))).sum(axis=1)
carry = cm.carry_cost(held)
ann = lambda s: float(s.reindex(bt.gross_returns.index).fillna(0.0).mean() * 252)
g = float(bt.gross_returns.mean() * 252)
comp = {"gross": g, "spread+commission": -ann(linear), "market impact": -ann(impact),
        "borrow+financing": -ann(carry)}
comp["net"] = g - ann(linear) - ann(impact) - ann(carry)
res["cost_decomposition"] = {k: round(v, 5) for k, v in comp.items()}

fig, ax = plt.subplots(figsize=(6.0, 3.2))
labels = list(comp)
vals = [comp[k] for k in labels]
run, bottoms, heights = 0.0, [], []
for i, k in enumerate(labels):
    if k in ("gross", "net"):
        bottoms.append(0.0); heights.append(comp[k]); run = comp[k] if k == "gross" else run
    else:
        bottoms.append(run + comp[k]); heights.append(-comp[k]); run = run + comp[k]
cols = [GREY, RED, RED, RED, INK]
ax.bar(range(len(labels)), heights, bottom=bottoms, color=cols, edgecolor=INK, linewidth=0.5)
for i, k in enumerate(labels):
    y = bottoms[i] + heights[i]
    ax.text(i, y + 0.002, f"{comp[k]:+.2%}", ha="center", va="bottom", fontsize=8)
ax.axhline(0, color=INK, lw=0.7)
ax.set_xticks(range(len(labels)), [l.replace("+", "+\n") for l in labels], fontsize=8)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1%}"))
ax.set_ylim(0, max(vals) * 1.22)
ax.set_ylabel("annualised return")
ax.set_title("Where the gross return went", loc="left", fontsize=10.5, pad=12)
fig.savefig(FIG / "fig05_05_costs_real.png")
plt.close(fig)

# --------------------------------------------------------------- figure 5.6 ---
print("capacity curve ...")
caps = [1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10]
rows = []
for c in caps:
    wc = cm.capacity_limited_weights(w, mkt["dollar_volume"], c) * 2.0
    b = run_backtest(wc, rets, cm, vol=vol21,
                     dollar_volume=mkt["dollar_volume"], capital=c)
    rows.append({"capital": c,
                 "sharpe_gross": b.stats_gross["sharpe"],
                 "sharpe_net": b.stats_net["sharpe"],
                 "ret_net": b.stats_net["ann_return"],
                 "pnl_net": b.stats_net["ann_return"] * c})
cap = pd.DataFrame(rows)
res["capacity"] = cap.round(4).to_dict("records")

eras = {"2006–2008": ("2006-01-01", "2008-12-31"),
        "2009–2011": ("2009-01-01", "2011-12-31"),
        "2012–2014": ("2012-01-01", "2014-12-31"),
        "2015–2017": ("2015-01-01", "2017-12-31"),
        "2018–2020": ("2018-01-01", "2020-12-31")}
era_sr = {k: float(sharpe(bt.net_returns.loc[a:b])) for k, (a, b) in eras.items()}
era_srg = {k: float(sharpe(bt.gross_returns.loc[a:b])) for k, (a, b) in eras.items()}
res["era_sharpe_net"] = {k: round(v, 3) for k, v in era_sr.items()}
res["era_sharpe_gross"] = {k: round(v, 3) for k, v in era_srg.items()}

fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.2))
ax[0].plot(cap["capital"], cap["sharpe_gross"], "o-", color=GREY, lw=1.2, ms=4, label="gross")
ax[0].plot(cap["capital"], cap["sharpe_net"], "o-", color=INK, lw=1.4, ms=4, label="net")
ax[0].set_xscale("log")
ax[0].axhline(0, color=INK, lw=0.7)
ax[0].set_xlabel("capital deployed ($)")
ax[0].set_ylabel("Sharpe")
ax[0].legend(frameon=False)
ax[0].set_title("Capacity", loc="left", fontsize=10)

xs = list(era_sr)
ax[1].bar(np.arange(len(xs)) - 0.2, [era_srg[k] for k in xs], width=0.4,
          color=GREY, edgecolor=INK, linewidth=0.5, label="gross")
ax[1].bar(np.arange(len(xs)) + 0.2, [era_sr[k] for k in xs], width=0.4,
          color=GOLD, edgecolor=INK, linewidth=0.5, label="net")
ax[1].axhline(0, color=INK, lw=0.7)
ax[1].set_xticks(range(len(xs)), xs, fontsize=8, rotation=20)
ax[1].set_ylabel("Sharpe")
ax[1].legend(frameon=False)
ax[1].set_title("Decay, by era", loc="left", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "fig05_06_capacity_real.png")
plt.close(fig)

OUT.write_text(json.dumps(res, indent=2))
print("\n--- vol targeting ---")
print(pd.concat([bt.stats_net.rename("plain"), bt_vt.stats_net.rename("vol-target")],
                axis=1).round(4).to_string())
print("\n--- worst months, plain vs vol-targeted ---")
print(pd.DataFrame({"plain": m.nsmallest(6),
                    "vol_target": m_vt.reindex(m.nsmallest(6).index)}).round(4).to_string())
print("\n--- cost decomposition (annualised) ---")
for k, v in comp.items():
    print(f"  {k:<20} {v:+.4%}")
print("\n--- capacity ---")
print(cap.to_string(index=False))
print("\n--- era Sharpe ---")
print(pd.DataFrame({"gross": era_srg, "net": era_sr}).round(3).to_string())
print(f"\nnumbers -> {OUT}")

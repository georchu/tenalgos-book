"""Chapter 9 figures — statistical arbitrage, PCA residual reversion.

Run:  python notebooks/ch09_figures.py

The signal takes ~75s to build per factor count, so it is built once per
configuration and cached to figures/_ch09_cache/.
"""
from __future__ import annotations

import sys, pathlib, json, time, gc
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
from tenalgos.research.stats import equity_curve, drawdown, sharpe, deflated_sharpe
from tenalgos.strategies import ch09_statarb as sa

FIG = pathlib.Path(__file__).resolve().parents[1] / "figures"
CACHE = FIG / "_ch09_cache"; CACHE.mkdir(parents=True, exist_ok=True)
INK, GOLD, GREY, RED, BLUE, GREEN = ("#12161C", "#B08A3E", "#8A929B",
                                     "#9E2B25", "#2E5A78", "#3F6B4A")
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.edgecolor": INK, "axes.linewidth": 0.7, "axes.grid": True,
    "grid.color": "#DDE1E6", "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 200, "savefig.bbox": "tight",
})
pct = FuncFormatter(lambda v, _: f"{v:.0%}")
res = {}

print("loading ...")
mkt = load_panel(top_n_by_liquidity=600)
rets, dv, alive = mkt["returns"], mkt["dollar_volume"], mkt["alive"]
vol21 = rets.rolling(21).std()
res["panel"] = {"days": len(rets), "names": int(rets.shape[1])}


def signal_for(k: int):
    f = CACHE / f"w_k{k}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    t0 = time.time()
    _, w = sa.build(rets, alive=alive, n_factors=k, step=5, max_names=400)
    print(f"  k={k} built in {time.time()-t0:.0f}s")
    w.astype("float32").to_parquet(f)
    return w


def bt(w, cm=None, lag=1, cap=1e8, trials=1):
    return run_backtest(w, rets, cm or CostModel(), vol=vol21, dollar_volume=dv,
                        capital=cap, n_trials=trials, execution_lag=lag)


# ------------------------------------------------- how many factors matter ---
print("factor sweep ...")
fac = {}
for k in (1, 5, 15, 30):
    w = signal_for(k)
    b = bt(w)
    fac[k] = {"sharpe_gross": round(float(b.stats_gross["sharpe"]), 3),
              "sharpe_net": round(float(b.stats_net["sharpe"]), 3),
              "ann_gross": round(float(b.stats_gross["ann_return"]), 4),
              "ann_turnover": round(float(b.stats_net["ann_turnover"]), 1),
              "vol": round(float(b.stats_net["ann_vol"]), 4)}
    del w, b; gc.collect()
res["factor_sweep"] = fac

w15 = signal_for(15)
base = bt(w15)
res["headline"] = {k: round(float(v), 4) for k, v in base.summary()["net"].items()
                   if isinstance(v, (int, float))}
res["headline_gross"] = {k: round(float(v), 4) for k, v in base.summary()["gross"].items()
                         if isinstance(v, (int, float))}

# ------------------------------------------------------ cost sensitivity ---
print("cost sweep ...")
spreads = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
cost_rows = []
for s in spreads:
    b = bt(w15, CostModel(spread_bps=s))
    cost_rows.append({"spread_bps": s,
                      "sharpe_net": round(float(b.stats_net["sharpe"]), 3),
                      "ann_net": round(float(b.stats_net["ann_return"]), 4)})
    del b; gc.collect()
res["cost_sweep"] = cost_rows

print("borrow sweep ...")
borrows = [0, 40, 100, 300, 1000]
borrow_rows = []
for bp in borrows:
    b = bt(w15, CostModel(spread_bps=3.0, borrow_bps_annual=bp))
    borrow_rows.append({"borrow_bps": bp,
                        "sharpe_net": round(float(b.stats_net["sharpe"]), 3),
                        "ann_net": round(float(b.stats_net["ann_return"]), 4)})
    del b; gc.collect()
res["borrow_sweep"] = borrow_rows

# the breakeven: how cheap must execution be?
print("breakeven ...")
be = None
for s in np.arange(0.1, 3.1, 0.1):
    b = bt(w15, CostModel(spread_bps=float(s), commission_bps=0.05,
                          borrow_bps_annual=0.0, impact_coef=0.0))
    if b.stats_net["sharpe"] < 0 and be is None:
        be = round(float(s), 2)
    del b; gc.collect()
    if be:
        break
res["breakeven_spread_bps"] = be

# ------------------------------------------------------ execution latency ---
print("latency ...")
lat = []
for lag in (1, 2, 3, 5):
    b = bt(w15, CostModel(spread_bps=1.0, commission_bps=0.05,
                          borrow_bps_annual=0.0, impact_coef=0.0), lag=lag)
    lat.append({"lag_days": lag,
                "sharpe_gross": round(float(b.stats_gross["sharpe"]), 3),
                "sharpe_net": round(float(b.stats_net["sharpe"]), 3)})
    del b; gc.collect()
res["latency"] = lat

# ---------------------------------------------------------- the best case ---
# 1bp all-in, no borrow: what a firm with real execution and a good locate desk
# actually pays. This is the number the chapter argues about.
cm_pro = CostModel(spread_bps=1.0, commission_bps=0.05, borrow_bps_annual=40.0,
                   impact_coef=0.15)
bt_pro = bt(w15, cm_pro)
res["professional"] = {k: round(float(v), 4) for k, v in bt_pro.summary()["net"].items()
                       if isinstance(v, (int, float))}
srp = float(bt_pro.stats_net["sharpe"])
res["deflated_pro"] = {f"{t}_trials": round(float(deflated_sharpe(srp, len(rets), n_trials=t)), 4)
                       for t in (1, 12, 59)}
res["oos_pro"] = {"in_sample": round(float(sharpe(bt_pro.net_returns.loc[:"2013-12-31"])), 3),
                  "out_of_sample": round(float(sharpe(bt_pro.net_returns.loc["2014-01-01":])), 3)}

gross_avg = float(w15.abs().sum(axis=1).mean())
cap_rows = []
for c in (1e7, 5e7, 1e8, 5e8, 1e9):
    wc = cm_pro.capacity_limited_weights(w15, dv, c) * gross_avg
    b_ = run_backtest(wc, rets, cm_pro, vol=vol21, dollar_volume=dv, capital=c)
    cap_rows.append({"capital": c, "sharpe_net": round(float(b_.stats_net["sharpe"]), 3),
                     "ret_net": round(float(b_.stats_net["ann_return"]), 4),
                     "pnl_net": round(float(b_.stats_net["ann_return"] * c), 0)})
    del wc, b_; gc.collect()
res["capacity"] = cap_rows

# --------------------------------------------------------------- fig 9.1 ---
fig, ax = plt.subplots(2, 1, figsize=(6.6, 5.2), sharex=True,
                       gridspec_kw={"height_ratios": [2, 1]})
for lab, s, c, lw in [("gross of costs", base.gross_returns, GREY, 1.2),
                      ("net, retail costs (3bp + 40bp borrow)", base.net_returns, RED, 1.2),
                      ("net, professional costs (1bp)", bt_pro.net_returns, INK, 1.5)]:
    e = equity_curve(s)
    ax[0].plot(e.index, e, color=c, lw=lw, label=f"{lab}  (SR {sharpe(s):.2f})")
ax[0].set_yscale("log"); ax[0].set_ylabel("growth of $1 (log)")
ax[0].legend(frameon=False, loc="upper left", fontsize=8)
ax[0].set_title("Statistical arbitrage — the same signal, three cost assumptions",
                loc="left", fontsize=10.5, pad=8)
d = drawdown(bt_pro.net_returns)
ax[1].fill_between(d.index, d, 0, color=RED, alpha=0.28, lw=0)
ax[1].plot(d.index, d, color=RED, lw=0.8)
ax[1].yaxis.set_major_formatter(pct); ax[1].set_ylabel("drawdown, pro costs")
fig.savefig(FIG / "fig09_01_costs_real.png"); plt.close(fig)

# --------------------------------------------------------------- fig 9.2 ---
fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.1))
ks = list(fac)
ax[0].plot(ks, [fac[k]["sharpe_gross"] for k in ks], "o-", color=GREY, lw=1.3, ms=5, label="gross")
ax[0].plot(ks, [fac[k]["sharpe_net"] for k in ks], "o-", color=INK, lw=1.4, ms=5, label="net")
ax[0].axhline(0, color=INK, lw=0.7); ax[0].set_xlabel("PCA factors removed")
ax[0].set_ylabel("Sharpe"); ax[0].legend(frameon=False, fontsize=8)
ax[0].set_title("How many factors", loc="left", fontsize=10)

cs = pd.DataFrame(cost_rows)
ax[1].plot(cs.spread_bps, cs.sharpe_net, "o-", color=INK, lw=1.4, ms=5)
ax[1].axhline(0, color=RED, lw=0.9, ls="--")
ax[1].set_xlabel("quoted spread (bp)"); ax[1].set_ylabel("net Sharpe")
ax[1].set_title("Cost sensitivity", loc="left", fontsize=10)

bs = pd.DataFrame(borrow_rows)
ax[2].plot(bs.borrow_bps, bs.sharpe_net, "o-", color=INK, lw=1.4, ms=5)
ax[2].axhline(0, color=RED, lw=0.9, ls="--")
ax[2].set_xscale("symlog"); ax[2].set_xlabel("annual borrow cost (bp)")
ax[2].set_ylabel("net Sharpe")
ax[2].set_title("Borrow sensitivity", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "fig09_02_sensitivity_real.png"); plt.close(fig)

# --------------------------------------------------------------- fig 9.3 ---
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.2))
lt = pd.DataFrame(lat)
ax[0].bar(lt.lag_days - 0.19, lt.sharpe_gross, width=0.38, color=GREY,
          edgecolor=INK, linewidth=0.5, label="gross")
ax[0].bar(lt.lag_days + 0.19, lt.sharpe_net, width=0.38, color=GOLD,
          edgecolor=INK, linewidth=0.5, label="net, 1bp")
ax[0].axhline(0, color=INK, lw=0.7); ax[0].set_xticks(lt.lag_days)
ax[0].set_xlabel("execution lag (days)"); ax[0].set_ylabel("Sharpe")
ax[0].legend(frameon=False, fontsize=8)
ax[0].set_title("The edge decays in days", loc="left", fontsize=10)

cp = pd.DataFrame(cap_rows)
ax[1].plot(cp.capital, cp.sharpe_net, "o-", color=INK, lw=1.4, ms=5)
ax[1].axhline(0, color=RED, lw=0.9, ls="--"); ax[1].set_xscale("log")
ax[1].set_xlabel("capital deployed ($)"); ax[1].set_ylabel("net Sharpe, pro costs")
ax[1].set_title("Capacity", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "fig09_03_latency_real.png"); plt.close(fig)

(FIG / "ch09_numbers.json").write_text(json.dumps(res, indent=2, default=float))
print("\n--- headline (15 factors, retail costs) ---")
print(base.summary().round(4).to_string())
print("\n--- professional costs (1bp spread, 0.15 impact, 40bp borrow) ---")
print(bt_pro.summary().round(4).to_string())
print("\n--- factor sweep ---"); print(pd.DataFrame(fac).T.to_string())
print("\n--- cost sweep ---"); print(cs.to_string(index=False))
print("--- borrow sweep ---"); print(bs.to_string(index=False))
print(f"breakeven spread (no borrow, no impact): {res['breakeven_spread_bps']} bp")
print("\n--- latency ---"); print(lt.to_string(index=False))
print("\n--- capacity ---"); print(cp.to_string(index=False))
print("deflated:", res["deflated_pro"], " oos:", res["oos_pro"])

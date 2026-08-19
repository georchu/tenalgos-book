"""Chapter 8 figures — carry, measured from ETF distribution yields.

Run:  python notebooks/ch08_figures.py
"""
from __future__ import annotations

import sys, pathlib, json, gc
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from tenalgos.data.loaders import load_panel
from tenalgos.research.backtest import (run_backtest, information_coefficient,
                                        ic_summary, quantile_returns)
from tenalgos.research.costs import CostModel
from tenalgos.research.stats import equity_curve, drawdown, sharpe, deflated_sharpe
from tenalgos.strategies import ch05_xsmom as mom
from tenalgos.strategies import ch07_tsmom as tr
from tenalgos.strategies import ch08_carry as ca

FIG = pathlib.Path(__file__).resolve().parents[1] / "figures"
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
p = load_panel("etfs", min_dollar_volume=2_000_000, top_n_by_liquidity=250)
raw = pd.read_parquet("data/etfs/close_unadjusted.parquet")

cols = [c for c in ca.universe() if c in p["prices"].columns]
adj = p["prices"][cols]
rets = p["returns"][cols]
dv = p["dollar_volume"][cols]
unadj = raw.reindex(index=adj.index, columns=cols)
vol21 = rets.rolling(21).std()
groups = ca.group_series(cols)
print(f"  {len(cols)} of {len(ca.universe())} classified markets present")
res["n_markets"] = len(cols)
res["by_group"] = groups.value_counts().to_dict()

cm = CostModel(spread_bps=2.0, commission_bps=0.2, borrow_bps_annual=20.0)
def B(w, name="carry", trials=1, cap=1e8):
    return run_backtest(w, rets, cm, vol=vol21, dollar_volume=dv,
                        capital=cap, n_trials=trials, name=name)

# --------------------------------------------------------- the yield itself ---
y = ca.distribution_yield(adj, unadj)
res["yield_by_group"] = {g: round(float(y[[c for c in cols if groups.get(c) == g]]
                                        .median().median()), 4)
                         for g in groups.unique()}

# ------------------------------------------------------------- variants ---
print("running variants ...")
V = {}
_, w_raw = ca.build(adj, unadj, rets, groups=groups,
                    neutralise_groups=False, risk_adjust=False)
V["raw yield, no controls"] = B(w_raw, "carry-raw")

_, w_ra = ca.build(adj, unadj, rets, groups=groups,
                   neutralise_groups=False, risk_adjust=True)
V["risk-adjusted yield"] = B(w_ra, "carry-ra")

sig, w_gn = ca.build(adj, unadj, rets, groups=groups,
                     neutralise_groups=True, risk_adjust=True)
V["+ group-neutralised"] = B(w_gn, "carry-gn")

bt_gn = V["+ group-neutralised"]
w_vt = mom.volatility_target(w_gn, bt_gn.gross_returns, target_vol=0.10)
bt_carry = B(w_vt, "carry-final")
V["+ vol target"] = bt_carry

tbl = pd.DataFrame({k: v.stats_net for k, v in V.items()}).T
res["table_net"] = tbl[["ann_return", "ann_vol", "sharpe", "max_drawdown", "skew",
                        "excess_kurtosis", "ann_turnover"]].round(4).to_dict("index")
res["table_gross_sharpe"] = {k: round(float(v.stats_gross["sharpe"]), 4) for k, v in V.items()}

ic = information_coefficient(sig, rets.shift(-1))
res["ic"] = ic_summary(ic)[["mean_ic", "ic_ir", "t_stat", "pct_positive"]].round(4).to_dict()

q = quantile_returns(sig, rets.shift(-1), q=5)
res["quintiles"] = (q.mean() * 252).round(4).to_dict()

# ------------------------------------------------------ carry's left tail ---
n = bt_carry.net_returns
res["worst_months"] = {str(d.date()): round(float(v), 4)
                       for d, v in n.resample("ME").sum().nsmallest(5).items()}
res["skew_vs_trend"] = {}

# ------------------------------------------------ trend + carry, the pair ---
print("trend + carry ...")
pt = load_panel("etfs", min_dollar_volume=1_000_000, top_n_by_liquidity=400)
tc = [c for c in tr.universe() if c in pt["prices"].columns]
tr_rets = pt["returns"][tc]
tr_vol = tr_rets.rolling(21).std()
_, w_tr = tr.build(pt["prices"][tc], tr_rets)
cm_tr = CostModel(spread_bps=2.0, commission_bps=0.2, borrow_bps_annual=20.0)
bt_tr0 = run_backtest(w_tr, tr_rets, cm_tr, vol=tr_vol,
                      dollar_volume=pt["dollar_volume"][tc], capital=1e8)
w_trf = tr.portfolio_vol_target(w_tr, bt_tr0.gross_returns, target_vol=0.10)
bt_trend = run_backtest(w_trf, tr_rets, cm_tr, vol=tr_vol,
                        dollar_volume=pt["dollar_volume"][tc], capital=1e8)

pair = pd.DataFrame({"trend": bt_trend.net_returns, "carry": n}).dropna()
res["trend_carry_corr"] = round(float(pair.corr().iloc[0, 1]), 4)
combo = pair.mean(axis=1)
res["pair"] = {"trend_sharpe": round(float(sharpe(pair["trend"])), 3),
               "carry_sharpe": round(float(sharpe(pair["carry"])), 3),
               "combo_sharpe": round(float(sharpe(combo)), 3),
               "combo_maxdd": round(float(drawdown(combo).min()), 4),
               "trend_maxdd": round(float(drawdown(pair["trend"]).min()), 4),
               "carry_maxdd": round(float(drawdown(pair["carry"]).min()), 4)}
roll = pair["trend"].rolling(252).corr(pair["carry"])
res["pair_corr_range"] = [round(float(roll.min()), 3), round(float(roll.max()), 3)]

# crises, both sleeves
spy = pt["returns"]["SPY"].reindex(pair.index).fillna(0.0)
crises = {"GFC": ("2007-10-01", "2009-03-31"), "Euro 2011": ("2011-05-01", "2011-11-30"),
          "Q4 2018": ("2018-10-01", "2018-12-31"), "Covid": ("2020-02-19", "2020-04-01")}
res["crises"] = [{"episode": k, "SPY": round(float(spy.loc[a:b].sum()), 4),
                  "trend": round(float(pair["trend"].loc[a:b].sum()), 4),
                  "carry": round(float(pair["carry"].loc[a:b].sum()), 4),
                  "both": round(float(combo.loc[a:b].sum()), 4)}
                 for k, (a, b) in crises.items()]

srn = float(bt_carry.stats_net["sharpe"]); nd = len(n)
res["deflated"] = {f"{t}_trials": round(float(deflated_sharpe(srn, nd, n_trials=t)), 4)
                   for t in (1, 4, 47)}
res["oos"] = {"in_sample": round(float(sharpe(n.loc[:"2013-12-31"])), 3),
              "out_of_sample": round(float(sharpe(n.loc["2014-01-01":])), 3)}

gross_avg = float(w_vt.abs().sum(axis=1).mean())
cap_rows = []
for c in (1e8, 5e8, 1e9, 5e9):
    wc = cm.capacity_limited_weights(w_vt, dv, c) * gross_avg
    b_ = run_backtest(wc, rets, cm, vol=vol21, dollar_volume=dv, capital=c)
    cap_rows.append({"capital": c, "sharpe_net": round(float(b_.stats_net["sharpe"]), 3),
                     "ret_net": round(float(b_.stats_net["ann_return"]), 4),
                     "pnl_net": round(float(b_.stats_net["ann_return"] * c), 0)})
    del wc, b_; gc.collect()
res["capacity"] = cap_rows

# --------------------------------------------------------------- fig 8.1 ---
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.3))
ymed = y.median().dropna()
gord = ["rates", "credit", "equity_us", "equity_intl", "sector", "real"]
data = [ymed[[c for c in cols if groups.get(c) == g]].dropna().values for g in gord]
bp = ax[0].boxplot(data, patch_artist=True, widths=0.6,
                   medianprops=dict(color=INK, lw=1.2))
for b in bp["boxes"]:
    b.set(facecolor=GOLD, alpha=0.65, edgecolor=INK, linewidth=0.6)
ax[0].set_xticks(range(1, len(gord) + 1), [g.replace("_", "\n") for g in gord], fontsize=7.5)
ax[0].yaxis.set_major_formatter(pct)
ax[0].set_ylabel("median trailing 12m yield")
ax[0].set_title("Carry, by asset group", loc="left", fontsize=10)

qm = pd.Series(res["quintiles"])
ax[1].bar(range(1, len(qm) + 1), qm.values,
          color=[RED if v < 0 else GOLD for v in qm.values],
          edgecolor=INK, linewidth=0.5)
ax[1].set_xlabel("carry quintile (1 = lowest)")
ax[1].yaxis.set_major_formatter(pct)
ax[1].set_ylabel("mean forward return, annualised")
ax[1].set_title("Monotonicity", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "fig08_01_carry_real.png"); plt.close(fig)

# --------------------------------------------------------------- fig 8.2 ---
fig, ax = plt.subplots(2, 1, figsize=(6.6, 5.2), sharex=True,
                       gridspec_kw={"height_ratios": [2, 1]})
for k, c, lw in [("raw yield, no controls", GREY, 1.0),
                 ("risk-adjusted yield", GREEN, 1.1),
                 ("+ group-neutralised", BLUE, 1.2),
                 ("+ vol target", INK, 1.5)]:
    e = equity_curve(V[k].net_returns)
    ax[0].plot(e.index, e, color=c, lw=lw,
               label=f"{k} (SR {V[k].stats_net['sharpe']:.2f})")
ax[0].set_yscale("log"); ax[0].set_ylabel("growth of $1, net (log)")
ax[0].legend(frameon=False, loc="upper left", fontsize=8)
ax[0].set_title("Carry: what each control is worth", loc="left", fontsize=10.5, pad=8)
d = drawdown(n)
ax[1].fill_between(d.index, d, 0, color=RED, alpha=0.28, lw=0)
ax[1].plot(d.index, d, color=RED, lw=0.8)
ax[1].yaxis.set_major_formatter(pct); ax[1].set_ylabel("drawdown")
fig.savefig(FIG / "fig08_02_variants_real.png"); plt.close(fig)

# --------------------------------------------------------------- fig 8.3 ---
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.3))
for k, c, lw in [("trend", GREY, 1.1), ("carry", GOLD, 1.1)]:
    e = equity_curve(pair[k]); ax[0].plot(e.index, e, color=c, lw=lw, label=k)
e = equity_curve(combo)
ax[0].plot(e.index, e, color=INK, lw=1.6,
           label=f"both (SR {res['pair']['combo_sharpe']:.2f})")
ax[0].set_yscale("log"); ax[0].set_ylabel("growth of $1, net (log)")
ax[0].legend(frameon=False, fontsize=8, loc="upper left")
ax[0].set_title("The canonical two-sleeve book", loc="left", fontsize=10)

ax[1].axhline(0, color=INK, lw=0.7)
ax[1].plot(roll.index, roll, color=INK, lw=1.0)
ax[1].fill_between(roll.index, roll, 0, color=BLUE, alpha=0.18, lw=0)
ax[1].set_ylabel("1y correlation, trend vs carry")
ax[1].set_title(f"and how unstable it is  [{res['pair_corr_range'][0]:+.2f}, "
                f"{res['pair_corr_range'][1]:+.2f}]", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "fig08_03_pair_real.png"); plt.close(fig)

# ------------------------------------------------------- the diagnosis ---
# The headline result is negative. Before blaming carry, find out what the
# proxy is actually measuring. Run it inside each group, and then test the
# hypothesis directly.
print("diagnosing ...")
per_group = {}
for grp in sorted(groups.unique()):
    c2 = [c for c in cols if groups.get(c) == grp]
    if len(c2) < 25:
        continue
    r2, d2 = rets[c2], dv[c2]
    s2, w2 = ca.build(adj[c2], unadj[c2], r2, groups=None,
                      neutralise_groups=False, risk_adjust=True, n_bins=5)
    b2 = run_backtest(w2, r2, cm, vol=r2.rolling(21).std(),
                      dollar_volume=d2, capital=1e8)
    per_group[grp] = {
        "n": len(c2),
        "ic_t": round(float(ic_summary(information_coefficient(s2, r2.shift(-1)))["t_stat"]), 2),
        "sharpe_gross": round(float(b2.stats_gross["sharpe"]), 3),
        "sharpe_net": round(float(b2.stats_net["sharpe"]), 3)}
    del w2, b2; gc.collect()
res["per_group"] = per_group

# fixed income alone, the one place a distribution yield really is carry
c2 = [c for c in cols if groups.get(c) in ("rates", "credit")]
r2, d2 = rets[c2], dv[c2]
s2, w2 = ca.build(adj[c2], unadj[c2], r2, groups=groups[c2],
                  neutralise_groups=False, risk_adjust=True, n_bins=5)
b2 = run_backtest(w2, r2, cm, vol=r2.rolling(21).std(), dollar_volume=d2, capital=1e8)
res["fixed_income_only"] = {
    "n": len(c2),
    "ic_t": round(float(ic_summary(information_coefficient(s2, r2.shift(-1)))["t_stat"]), 2),
    "sharpe_gross": round(float(b2.stats_gross["sharpe"]), 3),
    "sharpe_net": round(float(b2.stats_net["sharpe"]), 3),
    "ann_return": round(float(b2.stats_net["ann_return"]), 4),
    "in_sample": round(float(sharpe(b2.net_returns.loc[:"2013-12-31"])), 3),
    "out_of_sample": round(float(sharpe(b2.net_returns.loc["2014-01-01":])), 3)}

# the direct test of the hypothesis: in equities, high yield IS value
hi = [c for c in ["SDY", "DVY", "VYM", "HDV", "IWD", "VTV", "IVE"] if c in cols]
lo = [c for c in ["VUG", "IWF", "IVW", "QQQ", "VGT", "XLK"] if c in cols]
spread = rets[hi].mean(axis=1) - rets[lo].mean(axis=1)
res["yield_is_value"] = {"long": hi, "short": lo,
                         "ann_return": round(float(spread.mean() * 252), 4),
                         "sharpe": round(float(sharpe(spread)), 3)}

(FIG / "ch08_numbers.json").write_text(json.dumps(res, indent=2, default=float))
print("\n--- net ---"); print(tbl[["ann_return","ann_vol","sharpe","max_drawdown","skew","ann_turnover"]].round(4).to_string())
print("\ngross:", res["table_gross_sharpe"])
print("\nyield by group:", res["yield_by_group"])
print("IC:", res["ic"]); print("quintiles:", res["quintiles"])
print("\ntrend/carry corr:", res["trend_carry_corr"], " range", res["pair_corr_range"])
print("pair:", res["pair"])
print("\ncrises:"); print(pd.DataFrame(res["crises"]).to_string(index=False))
print("\ndeflated:", res["deflated"], " oos:", res["oos"])
print("capacity:"); print(pd.DataFrame(res["capacity"]).to_string(index=False))
print("worst months:", res["worst_months"])
print("\n--- diagnosis: carry inside each group ---")
print(pd.DataFrame(res["per_group"]).T.to_string())
print("fixed income only:", res["fixed_income_only"])
print("high-yield equity vs growth:", res["yield_is_value"])

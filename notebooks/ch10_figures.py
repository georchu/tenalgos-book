"""Chapter 10 figures — the volatility risk premium.

Run:  python notebooks/ch10_figures.py

Loads the ETF panel with the outlier mask DISABLED, because the default +/-60%
screen deletes the real -83% SVXY move of 6 February 2018 — the single most
important day in the history of volatility selling.
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
from tenalgos.research.stats import equity_curve, drawdown, sharpe, deflated_sharpe
from tenalgos.strategies import ch10_volpremium as vp

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

print("loading (outlier mask OFF) ...")
p = load_panel("etfs", min_dollar_volume=500_000, top_n_by_liquidity=800,
               max_abs_return=None)
r_all = p["returns"]
START = "2011-10-10"                       # first day all four ETFs are live
tks = [t for t in vp.VOL_ETFS if t in r_all.columns] + ["SPY"]
R = r_all.loc[START:, tks].copy()
R["SPY"] = R["SPY"].fillna(0.0)
res["start"], res["end"] = str(R.index[0].date()), str(R.index[-1].date())
res["n_days"] = int(len(R))
print(f"  {len(R)} days, {res['start']} .. {res['end']}")

# what the mask would have deleted
masked = load_panel("etfs", min_dollar_volume=500_000, top_n_by_liquidity=800)
deleted = []
for t in ["SVXY", "UVXY", "VIXY"]:
    a, b = R[t], masked["returns"][t].reindex(R.index)
    d = R.index[a.notna() & b.isna()]
    for dt in d:
        deleted.append({"ticker": t, "date": str(dt.date()),
                        "true_return": round(float(a.loc[dt]), 4)})
res["deleted_by_default_mask"] = deleted
print(f"  default mask would delete {len(deleted)} genuine moves")

# ------------------------------------------------- 10.3 the premium itself ---
stats = {}
for t in [x for x in vp.VOL_ETFS if x in R.columns]:
    s = R[t].dropna()
    stats[t] = {"ann_return": round(float(s.mean() * 252), 4),
                "ann_vol": round(float(s.std() * np.sqrt(252)), 4),
                "sharpe": round(float(sharpe(s)), 3),
                "max_dd": round(float(drawdown(s).min()), 4),
                "worst_day": round(float(s.min()), 4),
                "best_day": round(float(s.max()), 4),
                "skew": round(float(s.skew()), 2)}
res["instruments"] = stats

decay = vp.premium_from_decay(R["VIXY"])
res["vixy_rolling_1y_decay"] = {
    "median": round(float(decay.median()), 4),
    "best": round(float(decay.max()), 4),
    "worst": round(float(decay.min()), 4),
    "pct_negative": round(float((decay < 0).mean()), 4)}

# ------------------------------------------------------ 10.4/10.5 strategies ---
COST_BPS = 8.0          # vol ETFs are wide; 8bp round trip is generous-but-fair
def net(gross_ret, weights_frame):
    traded = weights_frame.diff().abs().sum(axis=1).fillna(0.0)
    return gross_ret - traded * COST_BPS / 1e4

V = {}

w = pd.DataFrame({"VIXY": vp.short_vol_fixed_notional(R["VIXY"], gross=0.20)})
V["short VIXY, fixed 20%"] = net(vp.backtest_weights(
    {"VIXY": w["VIXY"]}, R), w)

w2 = pd.DataFrame({"SVXY": pd.Series(0.20, index=R.index)})
V["hold SVXY, fixed 20%"] = net(vp.backtest_weights(
    {"SVXY": w2["SVXY"]}, R), w2)

wv = vp.vol_targeted_short(R["VIXY"], target_vol=0.10)
w3 = pd.DataFrame({"VIXY": wv})
V["short VIXY, vol-targeted"] = net(vp.backtest_weights({"VIXY": wv}, R), w3)

ws, wh = vp.tail_hedged_short(R["VIXY"], R["VIXM"], target_vol=0.10,
                              hedge_frac=0.15)
w4 = pd.DataFrame({"VIXY": ws, "VIXM": wh})
V["vol-targeted + VIXM tail hedge"] = net(
    vp.backtest_weights({"VIXY": ws, "VIXM": wh}, R), w4)

tbl = {}
for k, s in V.items():
    s = s.dropna()
    tbl[k] = {"ann_return": round(float(s.mean() * 252), 4),
              "ann_vol": round(float(s.std() * np.sqrt(252)), 4),
              "sharpe": round(float(sharpe(s)), 3),
              "max_dd": round(float(drawdown(s).min()), 4),
              "skew": round(float(s.skew()), 2),
              "excess_kurt": round(float(s.kurtosis()), 1),
              "worst_day": round(float(s.min()), 4),
              "corr_spy": round(float(s.corr(R["SPY"])), 3)}
res["strategies"] = tbl

# ------------------------------------------------------- 10.6 the blow-up ---
EV = ("2018-02-01", "2018-02-12")
res["feb2018"] = {k: round(float(v.loc[EV[0]:EV[1]].sum()), 4) for k, v in V.items()}
res["feb2018"]["SPY"] = round(float(R["SPY"].loc[EV[0]:EV[1]].sum()), 4)
res["feb2018_svxy_3day"] = round(float(
    (1 + R["SVXY"].loc["2018-02-02":"2018-02-06"]).prod() - 1), 4)

COV = ("2020-02-19", "2020-04-01")
res["covid"] = {k: round(float(v.loc[COV[0]:COV[1]].sum()), 4) for k, v in V.items()}
res["covid"]["SPY"] = round(float(R["SPY"].loc[COV[0]:COV[1]].sum()), 4)

# worst days of the headline strategy
best = V["vol-targeted + VIXM tail hedge"].dropna()
res["worst_days_hedged"] = {str(d.date()): round(float(v), 4)
                            for d, v in best.nsmallest(5).items()}
naked = V["short VIXY, fixed 20%"].dropna()
res["worst_days_naked"] = {str(d.date()): round(float(v), 4)
                           for d, v in naked.nsmallest(5).items()}

# -------------------------------------------- 10.7 what the hedge costs -----
plain = V["short VIXY, vol-targeted"].dropna()
hedg = V["vol-targeted + VIXM tail hedge"].dropna()
common = plain.index.intersection(hedg.index)
res["hedge_cost"] = {
    "ann_return_give_up": round(float((plain[common].mean() - hedg[common].mean()) * 252), 4),
    "sharpe_plain": round(float(sharpe(plain[common])), 3),
    "sharpe_hedged": round(float(sharpe(hedg[common])), 3),
    "maxdd_plain": round(float(drawdown(plain[common]).min()), 4),
    "maxdd_hedged": round(float(drawdown(hedg[common]).min()), 4),
    "worst_day_plain": round(float(plain[common].min()), 4),
    "worst_day_hedged": round(float(hedg[common].min()), 4)}

srn = tbl["vol-targeted + VIXM tail hedge"]["sharpe"]
res["deflated"] = {f"{t}_trials": round(float(deflated_sharpe(srn, len(best), n_trials=t)), 4)
                   for t in (1, 4, 75)}
res["oos"] = {"in_sample": round(float(sharpe(best.loc[:"2016-12-31"])), 3),
              "out_of_sample": round(float(sharpe(best.loc["2017-01-01":])), 3)}

# --------------------------------------------------------------- fig 10.1 ---
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4))
for t, c in [("VIXY", RED), ("UVXY", "#C86A64"), ("VIXM", GOLD), ("SVXY", BLUE)]:
    if t in R.columns:
        e = equity_curve(R[t].dropna())
        ax[0].plot(e.index, e, color=c, lw=1.2,
                   label=f"{t} ({stats[t]['ann_return']:+.0%}/yr)")
ax[0].set_yscale("log"); ax[0].set_ylabel("growth of $1 (log)")
ax[0].legend(frameon=False, fontsize=8)
ax[0].set_title("The premium, as paid and as received", loc="left", fontsize=10)

ax[1].axhline(0, color=INK, lw=0.8)
ax[1].fill_between(decay.index, decay, 0, where=decay < 0, color=BLUE, alpha=0.35, lw=0)
ax[1].fill_between(decay.index, decay, 0, where=decay >= 0, color=RED, alpha=0.35, lw=0)
ax[1].plot(decay.index, decay, color=INK, lw=0.9)
ax[1].yaxis.set_major_formatter(pct)
ax[1].set_ylabel("VIXY rolling 1y return")
ax[1].set_title("What the insurance buyer paid", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "fig10_01_premium_real.png"); plt.close(fig)

# --------------------------------------------------------------- fig 10.2 ---
fig, ax = plt.subplots(2, 1, figsize=(6.8, 5.4), sharex=True,
                       gridspec_kw={"height_ratios": [2, 1]})
for k, c, lw in [("short VIXY, fixed 20%", RED, 1.1),
                 ("hold SVXY, fixed 20%", GREY, 1.1),
                 ("short VIXY, vol-targeted", GOLD, 1.2),
                 ("vol-targeted + VIXM tail hedge", INK, 1.6)]:
    e = equity_curve(V[k].dropna())
    ax[0].plot(e.index, e, color=c, lw=lw, label=f"{k} (SR {tbl[k]['sharpe']:.2f})")
ax[0].set_yscale("log"); ax[0].set_ylabel("growth of $1, net (log)")
ax[0].legend(frameon=False, fontsize=8, loc="upper left")
ax[0].set_title("Four ways to sell volatility", loc="left", fontsize=10.5, pad=8)
d = drawdown(V["vol-targeted + VIXM tail hedge"].dropna())
ax[1].fill_between(d.index, d, 0, color=RED, alpha=0.28, lw=0)
ax[1].plot(d.index, d, color=RED, lw=0.9)
ax[1].yaxis.set_major_formatter(pct); ax[1].set_ylabel("drawdown, hedged")
fig.savefig(FIG / "fig10_02_strategies_real.png"); plt.close(fig)

# --------------------------------------------------------------- fig 10.3 ---
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.3))
ev = R.loc["2018-01-25":"2018-02-20"]
for t, c in [("SVXY", BLUE), ("VIXY", RED), ("SPY", GREY)]:
    e = (1 + ev[t].fillna(0)).cumprod() - 1
    ax[0].plot(e.index, e, color=c, lw=1.4, label=t)
ax[0].axhline(0, color=INK, lw=0.7); ax[0].yaxis.set_major_formatter(pct)
ax[0].set_ylabel("cumulative return")
ax[0].tick_params(axis="x", rotation=30, labelsize=7)
ax[0].legend(frameon=False, fontsize=8)
ax[0].set_title("February 2018 — the claim arrives", loc="left", fontsize=10)

ks = list(V)
vals = [res["feb2018"][k] for k in ks]
ax[1].barh(range(len(ks)), vals, color=[RED if v < 0 else GOLD for v in vals],
           edgecolor=INK, linewidth=0.5)
ax[1].set_yticks(range(len(ks)), [k.replace(", ", ",\n") for k in ks], fontsize=7.5)
ax[1].invert_yaxis(); ax[1].axvline(0, color=INK, lw=0.7)
ax[1].xaxis.set_major_formatter(pct)
ax[1].set_xlabel("return, 1–12 Feb 2018")
ax[1].set_title("What each version lost", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "fig10_03_blowup_real.png"); plt.close(fig)

(FIG / "ch10_numbers.json").write_text(json.dumps(res, indent=2, default=float))
print("\n--- instruments ---"); print(pd.DataFrame(stats).T.to_string())
print("\n--- VIXY rolling 1y decay ---", res["vixy_rolling_1y_decay"])
print("\n--- strategies, net ---"); print(pd.DataFrame(tbl).T.to_string())
print("\n--- Feb 2018 ---", json.dumps(res["feb2018"], indent=1))
print("SVXY 2-6 Feb compounded:", res["feb2018_svxy_3day"])
print("\n--- Covid ---", json.dumps(res["covid"], indent=1))
print("\n--- hedge cost ---", json.dumps(res["hedge_cost"], indent=1))
print("\ndeleted by default mask:", json.dumps(res["deleted_by_default_mask"], indent=1))
print("deflated:", res["deflated"], " oos:", res["oos"])
print("worst days, naked:", res["worst_days_naked"])
print("worst days, hedged:", res["worst_days_hedged"])

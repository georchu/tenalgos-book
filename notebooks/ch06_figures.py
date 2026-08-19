"""Chapter 6 figures — multi-factor equity.

Run:  python notebooks/ch06_figures.py

Three price-only sleeves, run standalone and then combined two ways. Every
number in Chapter 6 comes from this script.
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
from tenalgos.research.backtest import (run_backtest, information_coefficient,
                                        ic_summary)
from tenalgos.research.costs import CostModel
from tenalgos.research.stats import equity_curve, drawdown, sharpe
from tenalgos.strategies import ch05_xsmom as mom
from tenalgos.strategies import ch06_multifactor as mf

FIG = pathlib.Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)
INK, GOLD, GREY, RED, BLUE, GREEN = ("#12161C", "#B08A3E", "#8A929B",
                                     "#9E2B25", "#2E5A78", "#3F6B4A")
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
px, rets = mkt["prices"], mkt["returns"]
vol21 = rets.rolling(21).std()
dv = mkt["dollar_volume"]
cm = CostModel()

def bt(w, name, trials=1, cap=1e8):
    return run_backtest(w, rets, cm, vol=vol21, dollar_volume=dv,
                        capital=cap, n_trials=trials, name=name)

# ------------------------------------------------------------- sleeves ---
print("sleeve 1: low risk ...")
market = rets.mean(axis=1)
beta = mom.trailing_beta(rets, market)
s_lowrisk = mf.low_risk_signal(rets)
w_lowrisk = mf.beta_balanced_weights(mf.hold_monthly(
    mom.zscore_cross_section(s_lowrisk)), beta)

print("sleeve 2: reversal ...")
s_rev = mf.reversal_signal(px)
s_rev_z = mom.zscore_cross_section(s_rev / mom.trailing_vol(rets, 126).replace(0, np.nan))
w_rev = mom.decile_weights(mf.hold_monthly(s_rev_z))

print("sleeve 3: momentum ...")
s_mom, w_mom = mom.build(px, rets, industry=None,
                         neutralise_industry=False, neutralise_beta=True)

sleeve_w = {"low risk": w_lowrisk, "reversal": w_rev, "momentum": w_mom}
sleeve_bt = {k: bt(v, k) for k, v in sleeve_w.items()}

# also: reversal at daily rebalance, to show why the sleeve is usually untradeable
w_rev_d = mom.decile_weights(s_rev_z)
bt_rev_d = bt(w_rev_d, "reversal-daily")
res["reversal_daily"] = {"sharpe_gross": float(bt_rev_d.stats_gross["sharpe"]),
                         "sharpe_net": float(bt_rev_d.stats_net["sharpe"]),
                         "ann_turnover": float(bt_rev_d.stats_net["ann_turnover"])}

# and low risk WITHOUT the beta balance, to show what it really is
w_lowrisk_naive = mom.decile_weights(mf.hold_monthly(
    mom.zscore_cross_section(s_lowrisk)))
bt_lr_naive = bt(w_lowrisk_naive, "low-risk-unbalanced")
nb = (bt_lr_naive.weights * beta.reindex_like(bt_lr_naive.weights)).sum(axis=1)
nb_bal = (sleeve_bt["low risk"].weights * beta.reindex_like(
    sleeve_bt["low risk"].weights)).sum(axis=1)
res["lowrisk_unbalanced"] = {
    "sharpe_net": float(bt_lr_naive.stats_net["sharpe"]),
    "mean_net_beta": round(float(nb.mean()), 4),
    "mean_net_beta_balanced": round(float(nb_bal.mean()), 4)}

sleeve_rets = pd.DataFrame({k: v.net_returns for k, v in sleeve_bt.items()})
res["sleeve_corr"] = sleeve_rets.corr().round(3).to_dict()

# ---------------------------------------------------------- combination ---
print("combining ...")
sleeves_sig = {"low risk": s_lowrisk, "reversal": s_rev_z, "momentum": s_mom}
c_eq = mf.combine_equal_weight(sleeves_sig)
c_rk = mf.combine_by_risk(sleeves_sig, sleeve_rets)
w_eq = mom.decile_weights(mf.hold_monthly(c_eq))
w_rk = mom.decile_weights(mf.hold_monthly(c_rk))
bt_eq, bt_rk = bt(w_eq, "combo-equal"), bt(w_rk, "combo-risk")

# naive alternative: just add the three portfolios, one third each
w_sum = sum(sleeve_w.values()) / 3.0
bt_sum = bt(w_sum, "combo-portfolio-sum")

# the two-sleeve book: drop reversal, which the pre-trade IC test already said
# carries no monthly-horizon information (t = 0.32)
keep = ["low risk", "momentum"]
w_two = sum(sleeve_w[k] for k in keep) / len(keep)
bt_two = bt(w_two, "combo-two-sleeve")

# and the same two, weighted inversely to their own trailing volatility at the
# PORTFOLIO level rather than the signal level
sv = sleeve_rets[keep].rolling(252, min_periods=120).std()
iv = (1.0 / sv.replace(0, np.nan)).shift(1)
iv = iv.div(iv.sum(axis=1), axis=0).fillna(1.0 / len(keep))
w_two_rp = sum(sleeve_w[k].mul(iv[k].reindex(sleeve_w[k].index).ffill(), axis=0)
               for k in keep)
bt_two_rp = bt(w_two_rp, "combo-two-riskparity")

# vol-targeted version of the best combination
w_vt = mom.volatility_target(w_two_rp, bt_two_rp.gross_returns, target_vol=0.10)
bt_vt = bt(w_vt, "combo-vt")

allbt = {**sleeve_bt,
         "equal-weight signal": bt_eq,
         "risk-weighted signal": bt_rk,
         "sum of 3 portfolios": bt_sum,
         "2 sleeves, equal": bt_two,
         "2 sleeves, risk-weighted": bt_two_rp,
         "2 sleeves, risk-wtd + vol target": bt_vt}
tbl = pd.DataFrame({k: v.stats_net for k, v in allbt.items()}).T
res["table_net"] = tbl[["ann_return", "ann_vol", "sharpe", "max_drawdown",
                        "skew", "ann_turnover"]].round(4).to_dict("index")
res["table_gross_sharpe"] = {k: round(float(v.stats_gross["sharpe"]), 4)
                             for k, v in allbt.items()}

# ic per sleeve
fwd = rets.shift(-1)
res["ic"] = {k: ic_summary(information_coefficient(s, fwd))[
                 ["mean_ic", "ic_ir", "t_stat", "pct_positive"]].round(4).to_dict()
             for k, s in sleeves_sig.items()}

# --------------------------------------------------------------- fig 6.1 ---
fig, ax = plt.subplots(figsize=(6.6, 3.8))
cols = {"low risk": BLUE, "reversal": GREEN, "momentum": GOLD}
for k, c in cols.items():
    e = equity_curve(sleeve_bt[k].net_returns)
    ax.plot(e.index, e, color=c, lw=1.3, label=f"{k} (SR {sleeve_bt[k].stats_net['sharpe']:.2f})")
ax.set_yscale("log")
ax.set_ylabel("growth of $1, net (log)")
ax.legend(frameon=False, loc="upper left")
ax.set_title("Three sleeves, standalone", loc="left", fontsize=10.5, pad=8)
fig.savefig(FIG / "fig06_01_sleeves_real.png"); plt.close(fig)

# --------------------------------------------------------------- fig 6.2 ---
fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.3))
c = sleeve_rets.corr()
im = ax[0].imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
ax[0].set_xticks(range(len(c)), c.columns, fontsize=8, rotation=20)
ax[0].set_yticks(range(len(c)), c.index, fontsize=8)
ax[0].grid(False)
for i in range(len(c)):
    for j in range(len(c)):
        ax[0].text(j, i, f"{c.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8.5)
ax[0].set_title("Sleeve correlation, net daily", loc="left", fontsize=10)
fig.colorbar(im, ax=ax[0], fraction=0.045)

roll = sleeve_rets.rolling(252).corr().unstack()[("momentum", "reversal")]
ax[1].axhline(0, color=INK, lw=0.7)
ax[1].plot(roll.index, roll, color=INK, lw=1.0)
ax[1].set_ylabel("1y correlation")
ax[1].set_title("Momentum vs reversal, rolling", loc="left", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "fig06_02_correlation_real.png"); plt.close(fig)

# --------------------------------------------------------------- fig 6.3 ---
fig, ax = plt.subplots(2, 1, figsize=(6.6, 5.2), sharex=True,
                       gridspec_kw={"height_ratios": [2, 1]})
for k, c, lw in [("equal-weight signal", RED, 1.1),
                 ("sum of 3 portfolios", GREY, 1.1),
                 ("2 sleeves, risk-weighted", BLUE, 1.2),
                 ("2 sleeves, risk-wtd + vol target", INK, 1.5)]:
    e = equity_curve(allbt[k].net_returns)
    ax[0].plot(e.index, e, color=c, lw=lw,
               label=f"{k} (SR {allbt[k].stats_net['sharpe']:.2f})")
ax[0].set_yscale("log"); ax[0].set_ylabel("growth of $1, net (log)")
ax[0].legend(frameon=False, loc="upper left", fontsize=8)
ax[0].set_title("Four ways to combine the same three signals", loc="left",
                fontsize=10.5, pad=8)
d = drawdown(allbt["2 sleeves, risk-wtd + vol target"].net_returns)
ax[1].fill_between(d.index, d, 0, color=RED, alpha=0.28, lw=0)
ax[1].plot(d.index, d, color=RED, lw=0.8)
ax[1].yaxis.set_major_formatter(pct); ax[1].set_ylabel("drawdown")
fig.savefig(FIG / "fig06_03_combination_real.png"); plt.close(fig)

# ------------------------------------------------- honesty: trials and OOS ---
from tenalgos.research.stats import deflated_sharpe
final = bt_vt
n, srn = len(final.net_returns), float(final.stats_net["sharpe"])
res["deflated"] = {f"{t}_trials": round(float(deflated_sharpe(srn, n, n_trials=t)), 4)
                   for t in (1, 11, 36)}

split = "2013-12-31"
res["oos"] = {
    "combo_in_sample": round(float(sharpe(final.net_returns.loc[:split])), 3),
    "combo_out_of_sample": round(float(sharpe(final.net_returns.loc["2014-01-01":])), 3),
    "lowrisk_in": round(float(sharpe(sleeve_bt["low risk"].net_returns.loc[:split])), 3),
    "lowrisk_out": round(float(sharpe(sleeve_bt["low risk"].net_returns.loc["2014-01-01":])), 3),
    "momentum_in": round(float(sharpe(sleeve_bt["momentum"].net_returns.loc[:split])), 3),
    "momentum_out": round(float(sharpe(sleeve_bt["momentum"].net_returns.loc["2014-01-01":])), 3),
}

# Free the intermediates before the capacity sweep: each run below allocates
# several full-panel frames, and holding every earlier backtest's weights at
# the same time is what turns this script into an out-of-memory error.
import gc
for _b in (bt_eq, bt_rk, bt_sum, bt_two, bt_two_rp, bt_rev_d, bt_lr_naive):
    _b.weights = None
for _name in ("c_eq", "c_rk", "w_eq", "w_rk", "w_sum", "w_two", "w_two_rp",
              "w_rev_d", "w_lowrisk_naive", "s_rev", "s_rev_z"):
    if _name in dir():
        del globals()[_name]
gc.collect()

gross_avg = float(w_vt.abs().sum(axis=1).mean())
cap_rows = []
for c in (1e7, 1e8, 3e8, 1e9, 3e9):
    wc = cm.capacity_limited_weights(w_vt, dv, c) * gross_avg
    b_ = run_backtest(wc, rets, cm, vol=vol21, dollar_volume=dv, capital=c)
    cap_rows.append({"capital": c,
                     "sharpe_net": round(float(b_.stats_net["sharpe"]), 3),
                     "ret_net": round(float(b_.stats_net["ann_return"]), 4),
                     "pnl_net": round(float(b_.stats_net["ann_return"] * c), 0)})
    del wc, b_
    gc.collect()
res["capacity"] = cap_rows

(FIG / "ch06_numbers.json").write_text(json.dumps(res, indent=2, default=float))

print("\n--- net stats ---")
print(tbl[["ann_return", "ann_vol", "sharpe", "max_drawdown", "skew",
           "ann_turnover"]].round(4).to_string())
print("\n--- gross Sharpe ---")
for k, v in res["table_gross_sharpe"].items():
    print(f"  {k:<28} {v:+.4f}")
print("\n--- sleeve correlation ---")
print(sleeve_rets.corr().round(3).to_string())
print("\n--- information coefficient ---")
print(pd.DataFrame(res["ic"]).round(4).to_string())
print("\n--- reversal at daily rebalance ---", res["reversal_daily"])
print("--- low risk without beta balancing ---", res["lowrisk_unbalanced"])
print("\n--- deflated Sharpe of the final book ---", res["deflated"])
print("--- in vs out of sample (split 2014) ---")
for k, v in res["oos"].items():
    print(f"  {k:<22} {v:+.3f}")
print("--- capacity ---")
print(pd.DataFrame(res["capacity"]).to_string(index=False))
print("\nfigures ->", FIG)

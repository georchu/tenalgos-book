"""Chapter 7 figures — time-series momentum on a 35-market ETF proxy book.

Run:  python notebooks/ch07_figures.py
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
from tenalgos.research.backtest import run_backtest
from tenalgos.research.costs import CostModel
from tenalgos.research.stats import equity_curve, drawdown, sharpe, deflated_sharpe
from tenalgos.strategies import ch07_tsmom as tr

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

print("loading ETF panel ...")
# ETFs are far more liquid per name and far fewer in number than the equity
# universe, so the Chapter 2 screen is loosened deliberately and the reason is
# recorded here rather than buried in a default argument.
mkt = load_panel("etfs", min_dollar_volume=1_000_000, top_n_by_liquidity=400,
                 min_price=5.0)
cols = [c for c in tr.universe() if c in mkt["prices"].columns]
px = mkt["prices"][cols]
rets = mkt["returns"][cols]
dv = mkt["dollar_volume"][cols]
vol21 = rets.rolling(21).std()
print(f"  {len(cols)} of {len(tr.universe())} markets present")
res["markets"] = cols
res["missing"] = [c for c in tr.universe() if c not in cols]

# ETFs quote tighter than single stocks and short interest is cheap on the
# large ones; 2bp spread and 20bp borrow are the conservative-but-fair numbers.
cm = CostModel(spread_bps=2.0, commission_bps=0.2, borrow_bps_annual=20.0,
               impact_coef=0.35, financing_bps_annual=100.0)

def B(w, name="trend", trials=1, cap=1e8):
    return run_backtest(w, rets, cm, vol=vol21, dollar_volume=dv,
                        capital=cap, n_trials=trials, name=name)

# ------------------------------------------------------------- variants ---
print("running variants ...")
variants = {}

# single speeds, to show what the ensemble is worth
for lab, pair in [("fast only (8/24)", ((8, 24),)),
                  ("medium only (32/96)", ((32, 96),)),
                  ("slow only (64/192)", ((64, 192),))]:
    _, w = tr.build(px, rets, pairs=pair)
    variants[lab] = B(w, lab)

_, w_ens = tr.build(px, rets)
variants["ensemble of 4 speeds"] = B(w_ens, "ensemble")

# no bucket normalisation, to price the correlation control
_, w_nb = tr.build(px, rets, normalise_buckets=False)
variants["ensemble, no bucket control"] = B(w_nb, "no-bucket")

# MOP sign-only, the original 2012 formulation
sig_sign = tr.tsmom_sign(px)
marks = sig_sign.resample("ME").last().index
sig_sign = sig_sign.where(pd.Series(sig_sign.index.isin(marks),
                                    index=sig_sign.index), np.nan).ffill()
w_sign = tr.bucket_normalise(tr.risk_scaled_positions(sig_sign, rets))
variants["sign of 12m return"] = B(w_sign, "mop-sign")

# the final book: ensemble + portfolio vol target
bt_ens = variants["ensemble of 4 speeds"]
w_final = tr.portfolio_vol_target(w_ens, bt_ens.gross_returns, target_vol=0.10)
bt_final = B(w_final, "trend-final")
variants["ensemble + vol target"] = bt_final

tbl = pd.DataFrame({k: v.stats_net for k, v in variants.items()}).T
res["table_net"] = tbl[["ann_return", "ann_vol", "sharpe", "max_drawdown",
                        "skew", "excess_kurtosis", "ann_turnover"]].round(4).to_dict("index")
res["table_gross_sharpe"] = {k: round(float(v.stats_gross["sharpe"]), 4)
                             for k, v in variants.items()}

# ------------------------------------------------------------ crisis alpha ---
spy = mkt["returns"]["SPY"].reindex(bt_final.net_returns.index).fillna(0.0)
n = bt_final.net_returns
res["corr_to_spy"] = round(float(n.corr(spy)), 4)
worst = spy.rolling(21).sum().nsmallest(400).index      # worst equity months
res["trend_in_worst_equity_months"] = {
    "spy_mean_21d": round(float(spy.rolling(21).sum().loc[worst].mean()), 4),
    "trend_mean_21d": round(float(n.rolling(21).sum().loc[worst].mean()), 4)}

crises = {"GFC 2007-09 to 2009-03": ("2007-10-01", "2009-03-31"),
          "Euro crisis 2011": ("2011-05-01", "2011-11-30"),
          "Q4 2018": ("2018-10-01", "2018-12-31"),
          "Covid 2020": ("2020-02-19", "2020-04-01")}
rows = []
for k, (a, b) in crises.items():
    rows.append({"episode": k,
                 "SPY": round(float(spy.loc[a:b].sum()), 4),
                 "trend_net": round(float(n.loc[a:b].sum()), 4)})
res["crises"] = rows

# beta to equities in down months vs up months
m_spy = spy.resample("ME").sum()
m_tr = n.resample("ME").sum()
up, dn = m_spy > 0, m_spy < 0
res["beta_up"] = round(float(np.polyfit(m_spy[up], m_tr[up], 1)[0]), 4)
res["beta_down"] = round(float(np.polyfit(m_spy[dn], m_tr[dn], 1)[0]), 4)

# ------------------------------------------------------- honesty and scale ---
srn = float(bt_final.stats_net["sharpe"])
nd = len(bt_final.net_returns)
res["deflated"] = {f"{t}_trials": round(float(deflated_sharpe(srn, nd, n_trials=t)), 4)
                   for t in (1, 7, 43)}
split = "2013-12-31"
res["oos"] = {"in_sample": round(float(sharpe(n.loc[:split])), 3),
              "out_of_sample": round(float(sharpe(n.loc["2014-01-01":])), 3)}
res["by_year"] = {str(d.year): round(float(v), 4)
                  for d, v in n.resample("YE").sum().items()}

gross_avg = float(w_final.abs().sum(axis=1).mean())
cap_rows = []
for c in (1e8, 1e9, 5e9, 1e10, 5e10):
    wc = cm.capacity_limited_weights(w_final, dv, c) * gross_avg
    b_ = run_backtest(wc, rets, cm, vol=vol21, dollar_volume=dv, capital=c)
    cap_rows.append({"capital": c,
                     "sharpe_net": round(float(b_.stats_net["sharpe"]), 3),
                     "ret_net": round(float(b_.stats_net["ann_return"]), 4),
                     "pnl_net": round(float(b_.stats_net["ann_return"] * c), 0)})
    del wc, b_; gc.collect()
res["capacity"] = cap_rows

# per-bucket contribution
buck = {}
for bname, ts in tr.MARKETS.items():
    c2 = [t for t in ts if t in cols]
    if not c2:
        continue
    wb = w_final[c2].reindex(columns=cols).fillna(0.0)
    buck[bname] = float(run_backtest(wb, rets, cm, vol=vol21,
                                     dollar_volume=dv).net_returns.sum())
res["bucket_pnl"] = {k: round(v, 4) for k, v in buck.items()}

# --------------------------------------------------------------- fig 7.1 ---
fig, ax = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True,
                       gridspec_kw={"height_ratios": [2.2, 1]})
e = equity_curve(bt_final.net_returns)
es = equity_curve(spy)
ax[0].plot(e.index, e, color=INK, lw=1.5, label=f"trend, net (SR {srn:.2f})")
ax[0].plot(es.index, es, color=GREY, lw=1.1, label=f"SPY (SR {sharpe(spy):.2f})")
ax[0].set_yscale("log"); ax[0].set_ylabel("growth of $1 (log)")
ax[0].legend(frameon=False, loc="upper left")
ax[0].set_title("Trend following, 35 markets, net of costs", loc="left",
                fontsize=10.5, pad=8)
d1, d2 = drawdown(bt_final.net_returns), drawdown(spy)
ax[1].plot(d2.index, d2, color=GREY, lw=0.9, label="SPY")
ax[1].fill_between(d1.index, d1, 0, color=RED, alpha=0.28, lw=0)
ax[1].plot(d1.index, d1, color=RED, lw=0.9, label="trend")
ax[1].yaxis.set_major_formatter(pct); ax[1].set_ylabel("drawdown")
ax[1].legend(frameon=False, loc="lower left", fontsize=8)
fig.savefig(FIG / "fig07_01_performance_real.png"); plt.close(fig)

# --------------------------------------------------------------- fig 7.2 ---
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.3))
ks = list(crises)
x = np.arange(len(ks))
ax[0].bar(x - 0.2, [r["SPY"] for r in rows], width=0.4, color=GREY,
          edgecolor=INK, linewidth=0.5, label="SPY")
ax[0].bar(x + 0.2, [r["trend_net"] for r in rows], width=0.4, color=BLUE,
          edgecolor=INK, linewidth=0.5, label="trend, net")
ax[0].axhline(0, color=INK, lw=0.7)
ax[0].set_xticks(x, [k.split(" ")[0] for k in ks], fontsize=8)
ax[0].yaxis.set_major_formatter(pct)
ax[0].legend(frameon=False, fontsize=8)
ax[0].set_title("Crisis alpha", loc="left", fontsize=10)

ax[1].scatter(m_spy, m_tr, s=14, color=INK, alpha=0.6, edgecolors="none")
xs = np.linspace(m_spy.min(), 0, 20)
ax[1].plot(xs, np.polyval(np.polyfit(m_spy[dn], m_tr[dn], 1), xs),
           color=RED, lw=1.4, label=f"down months, beta {res['beta_down']:+.2f}")
xs2 = np.linspace(0, m_spy.max(), 20)
ax[1].plot(xs2, np.polyval(np.polyfit(m_spy[up], m_tr[up], 1), xs2),
           color=GREEN, lw=1.4, label=f"up months, beta {res['beta_up']:+.2f}")
ax[1].axhline(0, color=INK, lw=0.6); ax[1].axvline(0, color=INK, lw=0.6)
ax[1].xaxis.set_major_formatter(pct); ax[1].yaxis.set_major_formatter(pct)
ax[1].set_xlabel("SPY, monthly"); ax[1].set_ylabel("trend, monthly")
ax[1].legend(frameon=False, fontsize=8)
ax[1].set_title("The smile", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "fig07_02_crisis_real.png"); plt.close(fig)

# --------------------------------------------------------------- fig 7.3 ---
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.3))
order = ["fast only (8/24)", "medium only (32/96)", "slow only (64/192)",
         "sign of 12m return", "ensemble, no bucket control",
         "ensemble of 4 speeds", "ensemble + vol target"]
sr_net = [variants[k].stats_net["sharpe"] for k in order]
ax[0].barh(range(len(order)), sr_net,
           color=[INK if k == "ensemble + vol target" else GOLD for k in order],
           edgecolor=INK, linewidth=0.5)
ax[0].set_yticks(range(len(order)), [k.replace(" (", "\n(") for k in order], fontsize=7.5)
ax[0].invert_yaxis(); ax[0].axvline(0, color=INK, lw=0.7)
ax[0].set_xlabel("net Sharpe")
ax[0].set_title("What each choice is worth", loc="left", fontsize=10)

bk = pd.Series(res["bucket_pnl"]).sort_values()
ax[1].barh(range(len(bk)), bk.values,
           color=[RED if v < 0 else GOLD for v in bk.values],
           edgecolor=INK, linewidth=0.5)
ax[1].set_yticks(range(len(bk)), [k.replace("_", " ") for k in bk.index], fontsize=8)
ax[1].axvline(0, color=INK, lw=0.7)
ax[1].xaxis.set_major_formatter(pct)
ax[1].set_xlabel("cumulative net return contribution")
ax[1].set_title("Where the money came from", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "fig07_03_variants_real.png"); plt.close(fig)

(FIG / "ch07_numbers.json").write_text(json.dumps(res, indent=2, default=float))

print("\n--- net stats ---")
print(tbl[["ann_return", "ann_vol", "sharpe", "max_drawdown", "skew",
           "excess_kurtosis", "ann_turnover"]].round(4).to_string())
print("\n--- gross Sharpe ---")
for k, v in res["table_gross_sharpe"].items():
    print(f"  {k:<30} {v:+.4f}")
print("\n--- crisis episodes (cumulative) ---")
print(pd.DataFrame(rows).to_string(index=False))
print(f"\ncorrelation to SPY, daily net : {res['corr_to_spy']:+.4f}")
print(f"beta in up months   : {res['beta_up']:+.3f}")
print(f"beta in down months : {res['beta_down']:+.3f}")
print("worst-400 equity 21d windows:", res["trend_in_worst_equity_months"])
print("\n--- deflated ---", res["deflated"], "\n--- oos ---", res["oos"])
print("\n--- capacity ---"); print(pd.DataFrame(res["capacity"]).to_string(index=False))
print("\n--- bucket contribution ---")
for k, v in sorted(res["bucket_pnl"].items(), key=lambda x: -x[1]):
    print(f"  {k:<14} {v:+.2%}")
print("\n--- by year ---", {k: f"{v:+.1%}" for k, v in res["by_year"].items()})
print("\nfigures ->", FIG)

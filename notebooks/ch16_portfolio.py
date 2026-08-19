"""Chapter 16 — the combined multi-sleeve portfolio.

Run:  python notebooks/ch16_portfolio.py

This is the chapter the book exists to reach. Every sleeve built in Part II is
run through the same engine, its NET daily return series is collected, and the
sleeves are combined several ways. The output answers the question on the cover
honestly.

Sleeve return series are cached to figures/_ch16_cache/ so this is fast to
re-run after the first pass.
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
from tenalgos.strategies import ch05_xsmom as mom
from tenalgos.strategies import ch06_multifactor as mf
from tenalgos.strategies import ch07_tsmom as tr
from tenalgos.strategies import ch08_carry as ca

FIG = pathlib.Path(__file__).resolve().parents[1] / "figures"
CACHE = FIG / "_ch16_cache"; CACHE.mkdir(parents=True, exist_ok=True)
INK, GOLD, GREY, RED, BLUE, GREEN, PURPLE = ("#12161C", "#B08A3E", "#8A929B",
                                             "#9E2B25", "#2E5A78", "#3F6B4A", "#6B4A6B")
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.edgecolor": INK, "axes.linewidth": 0.7, "axes.grid": True,
    "grid.color": "#DDE1E6", "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 200, "savefig.bbox": "tight",
})
pct = FuncFormatter(lambda v, _: f"{v:.0%}")
res = {}

SLEEVES = CACHE / "sleeves.parquet"
if SLEEVES.exists():
    S = pd.read_parquet(SLEEVES)
    print(f"loaded cached sleeves: {list(S.columns)}")
else:
    print("building sleeves ...")
    out = {}

    # --- equity sleeves (Chapters 5, 6, 9) --------------------------------
    eq = load_panel(top_n_by_liquidity=1500)
    r, dv = eq["returns"], eq["dollar_volume"]
    v21 = r.rolling(21).std()
    cm_eq = CostModel()

    def Beq(w, cap=1e8):
        return run_backtest(w, r, cm_eq, vol=v21, dollar_volume=dv, capital=cap)

    print("  ch5 momentum ...")
    _, w_mom = mom.build(eq["prices"], r, industry=None,
                         neutralise_industry=False, neutralise_beta=True)
    b_mom = Beq(w_mom)
    out["momentum"] = b_mom.net_returns

    print("  ch6 low risk ...")
    market = r.mean(axis=1)
    beta = mom.trailing_beta(r, market)
    w_lr = mf.beta_balanced_weights(
        mf.hold_monthly(mom.zscore_cross_section(mf.low_risk_signal(r))), beta)
    out["low_risk"] = Beq(w_lr).net_returns
    del beta; gc.collect()

    print("  ch9 stat arb (professional costs) ...")
    wsa = FIG / "_ch09_cache" / "w_k15.parquet"
    if wsa.exists():
        w_sa = pd.read_parquet(wsa)
        eq6 = load_panel(top_n_by_liquidity=600)
        r6 = eq6["returns"]
        cm_pro = CostModel(spread_bps=1.0, commission_bps=0.05,
                           borrow_bps_annual=40.0, impact_coef=0.15)
        b_sa = run_backtest(w_sa, r6, cm_pro, vol=r6.rolling(21).std(),
                            dollar_volume=eq6["dollar_volume"], capital=1e8)
        out["stat_arb"] = b_sa.net_returns
        del w_sa, eq6, r6, b_sa; gc.collect()
    del eq, r, dv, v21, w_mom, w_lr, b_mom; gc.collect()

    # --- macro sleeves (Chapters 7, 8) ------------------------------------
    print("  ch7 trend ...")
    pt = load_panel("etfs", min_dollar_volume=1_000_000, top_n_by_liquidity=400)
    tc = [c for c in tr.universe() if c in pt["prices"].columns]
    rt, dvt = pt["returns"][tc], pt["dollar_volume"][tc]
    vt = rt.rolling(21).std()
    cm_tr = CostModel(spread_bps=2.0, commission_bps=0.2, borrow_bps_annual=20.0)
    _, w_tr = tr.build(pt["prices"][tc], rt)
    b0 = run_backtest(w_tr, rt, cm_tr, vol=vt, dollar_volume=dvt, capital=1e8)
    w_trf = tr.portfolio_vol_target(w_tr, b0.gross_returns, target_vol=0.10)
    out["trend"] = run_backtest(w_trf, rt, cm_tr, vol=vt,
                                dollar_volume=dvt, capital=1e8).net_returns
    spy = pt["returns"]["SPY"]
    del pt, rt, dvt, vt, w_tr, w_trf, b0; gc.collect()

    print("  ch8 carry (reported, not recommended) ...")
    pc = load_panel("etfs", min_dollar_volume=2_000_000, top_n_by_liquidity=250)
    raw = pd.read_parquet("data/etfs/close_unadjusted.parquet")
    cc = [c for c in ca.universe() if c in pc["prices"].columns]
    rc, dvc = pc["returns"][cc], pc["dollar_volume"][cc]
    _, w_ca = ca.build(pc["prices"][cc], raw.reindex(index=rc.index, columns=cc),
                       rc, groups=ca.group_series(cc))
    out["carry"] = run_backtest(w_ca, rc, CostModel(spread_bps=2.0, commission_bps=0.2,
                                borrow_bps_annual=20.0), vol=rc.rolling(21).std(),
                                dollar_volume=dvc, capital=1e8).net_returns
    del pc, raw, rc, dvc, w_ca; gc.collect()

    S = pd.DataFrame(out).dropna(how="all")
    S["SPY"] = spy.reindex(S.index)
    S.to_parquet(SLEEVES)
    print("  cached.")

spy = S.pop("SPY").fillna(0.0)
S = S.loc["2007-01-01":].fillna(0.0)
spy = spy.reindex(S.index).fillna(0.0)
print(f"\nsleeves: {list(S.columns)}   {len(S)} days from {S.index[0].date()}")

res["sleeve_stats"] = {c: {"sharpe": round(float(sharpe(S[c])), 3),
                           "ann_return": round(float(S[c].mean() * 252), 4),
                           "ann_vol": round(float(S[c].std() * np.sqrt(252)), 4),
                           "max_dd": round(float(drawdown(S[c]).min()), 4)}
                       for c in S.columns}
res["correlation"] = S.corr().round(3).to_dict()

# ---------------------------------------------------- combination methods ---
def stats(x, name):
    return {"name": name, "sharpe": round(float(sharpe(x)), 3),
            "ann_return": round(float(x.mean() * 252), 4),
            "ann_vol": round(float(x.std() * np.sqrt(252)), 4),
            "max_dd": round(float(drawdown(x).min()), 4),
            "skew": round(float(x.skew()), 3),
            "corr_spy": round(float(x.corr(spy)), 3)}

POS = [c for c in S.columns if sharpe(S[c]) > 0]
res["positive_sleeves"] = POS

combos = {}
combos["all sleeves, equal"] = S.mean(axis=1)
combos["positive sleeves, equal"] = S[POS].mean(axis=1)

iv = (1.0 / S[POS].rolling(252, min_periods=120).std()).shift(1)
iv = iv.div(iv.sum(axis=1), axis=0).fillna(1.0 / len(POS))
combos["positive, inverse-vol"] = (S[POS] * iv).sum(axis=1)

# fractional Kelly on the trailing estimate, capped
mu = S[POS].rolling(504, min_periods=252).mean().shift(1) * 252
sd = S[POS].rolling(504, min_periods=252).std().shift(1) * np.sqrt(252)
kel = (mu / sd.pow(2)).clip(0, 3).fillna(0.0)
kel = kel.div(kel.sum(axis=1).replace(0, np.nan), axis=0).fillna(1.0 / len(POS))
combos["positive, half-Kelly"] = (S[POS] * kel).sum(axis=1)

res["combos"] = [stats(v, k) for k, v in combos.items()]

best = combos["positive, inverse-vol"]
target = 0.10
lev = (target / (best.rolling(63, min_periods=20).std() * np.sqrt(252))).shift(1)
lev = lev.clip(upper=3.0).fillna(1.0)
final = best * lev
res["final"] = stats(final, "inverse-vol + vol target")
res["mean_leverage"] = round(float(lev.mean()), 2)

nd = len(final); srf = float(sharpe(final))
res["deflated"] = {f"{t}_trials": round(float(deflated_sharpe(srf, nd, n_trials=t)), 4)
                   for t in (1, 8, 67)}
res["oos"] = {"in_sample": round(float(sharpe(final.loc[:"2013-12-31"])), 3),
              "out_of_sample": round(float(sharpe(final.loc["2014-01-01":])), 3)}

# crisis behaviour
crises = {"GFC": ("2007-10-01", "2009-03-31"), "Euro 2011": ("2011-05-01", "2011-11-30"),
          "Q4 2018": ("2018-10-01", "2018-12-31"), "Covid": ("2020-02-19", "2020-04-01")}
res["crises"] = [{"episode": k, "SPY": round(float(spy.loc[a:b].sum()), 4),
                  "portfolio": round(float(final.loc[a:b].sum()), 4)}
                 for k, (a, b) in crises.items()]

# rolling correlation instability
pairs = [("momentum", "low_risk"), ("trend", "momentum"), ("trend", "stat_arb")]
roll = {f"{a}|{b}": S[a].rolling(252).corr(S[b]) for a, b in pairs if a in S and b in S}
res["corr_ranges"] = {k: [round(float(v.min()), 2), round(float(v.max()), 2)]
                      for k, v in roll.items()}

# --------------------------------------- the answer to the book's question ---
# What capital does this portfolio need to earn $1bn a year, and is that
# capital within its capacity? Capacity ceilings come from each chapter.
CAP = {"momentum": 3e8, "low_risk": 1e9, "trend": 1e9, "stat_arb": 5e7, "carry": 0}
res["capacity_ceilings"] = CAP
ret = res["final"]["ann_return"]
res["capital_for_1bn"] = round(1e9 / ret) if ret > 0 else None
res["total_sleeve_capacity"] = sum(CAP[c] for c in POS if c in CAP)

# --------------------------------------------------------------- fig 16.1 ---
fig, ax = plt.subplots(2, 1, figsize=(6.8, 5.6), sharex=True,
                       gridspec_kw={"height_ratios": [2.2, 1]})
cmap = {"momentum": GOLD, "low_risk": BLUE, "trend": GREEN,
        "stat_arb": PURPLE, "carry": RED}
for c in S.columns:
    e = equity_curve(S[c])
    ax[0].plot(e.index, e, color=cmap.get(c, GREY), lw=0.9, alpha=0.75,
               label=f"{c} ({sharpe(S[c]):.2f})")
e = equity_curve(spy)
ax[0].plot(e.index, e, color=GREY, lw=1.0, ls="--", label=f"SPY ({sharpe(spy):.2f})")
e = equity_curve(final)
ax[0].plot(e.index, e, color=INK, lw=2.0, label=f"COMBINED ({srf:.2f})")
ax[0].set_yscale("log"); ax[0].set_ylabel("growth of $1, net (log)")
ax[0].legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
ax[0].set_title("Every sleeve in this book, and the portfolio they make",
                loc="left", fontsize=10.5, pad=8)
d1, d2 = drawdown(final), drawdown(spy)
ax[1].plot(d2.index, d2, color=GREY, lw=0.9, ls="--", label="SPY")
ax[1].fill_between(d1.index, d1, 0, color=RED, alpha=0.28, lw=0)
ax[1].plot(d1.index, d1, color=RED, lw=1.0, label="combined")
ax[1].yaxis.set_major_formatter(pct); ax[1].set_ylabel("drawdown")
ax[1].legend(frameon=False, fontsize=8, loc="lower left")
fig.savefig(FIG / "fig16_01_combined_real.png"); plt.close(fig)

# --------------------------------------------------------------- fig 16.2 ---
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4))
c = S.corr()
im = ax[0].imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
ax[0].set_xticks(range(len(c)), c.columns, fontsize=7.5, rotation=25)
ax[0].set_yticks(range(len(c)), c.index, fontsize=7.5); ax[0].grid(False)
for i in range(len(c)):
    for j in range(len(c)):
        ax[0].text(j, i, f"{c.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7.5)
ax[0].set_title("Sleeve correlation, net daily", loc="left", fontsize=10)
fig.colorbar(im, ax=ax[0], fraction=0.045)
ax[1].axhline(0, color=INK, lw=0.7)
for (k, v), col in zip(roll.items(), [GOLD, BLUE, PURPLE]):
    ax[1].plot(v.index, v, lw=1.0, color=col, label=k)
ax[1].legend(frameon=False, fontsize=7.5)
ax[1].set_ylabel("rolling 1y correlation")
ax[1].set_title("and how little it holds still", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "fig16_02_correlation_real.png"); plt.close(fig)

(FIG / "ch16_numbers.json").write_text(json.dumps(res, indent=2, default=float))
print("\n--- sleeves ---"); print(pd.DataFrame(res["sleeve_stats"]).T.to_string())
print("\n--- correlation ---"); print(S.corr().round(3).to_string())
print("\n--- combinations ---"); print(pd.DataFrame(res["combos"]).to_string(index=False))
print("\n--- FINAL ---"); print(json.dumps(res["final"], indent=2))
print("mean leverage:", res["mean_leverage"])
print("deflated:", res["deflated"], " oos:", res["oos"])
print("\ncrises:"); print(pd.DataFrame(res["crises"]).to_string(index=False))
print("\ncorr ranges:", res["corr_ranges"])
print(f"\ncapital needed for $1bn/yr: ${res['capital_for_1bn']:,}" if res["capital_for_1bn"] else "")
print(f"summed sleeve capacity:     ${res['total_sleeve_capacity']:,}")

"""Chapter 11 figures — the ML alpha ensemble, honest and leaky.

Run:  python notebooks/ch11_figures.py

Three models are trained on identical features:

    LEAKY-CV    plain K-fold. Adjacent overlapping labels sit on both sides of
                every fold boundary. This is what most published ML-in-finance
                work does, and it is the version that looks twice as good.
    LEAKY-FEAT  the same, plus one feature that quietly contains the future.
    HONEST      purged K-fold with an embargo, walk-forward only.

The gap between them is the chapter.
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
try:
    import lightgbm as lgb
except Exception as _err:                    # ImportError, or a load failure
    raise SystemExit(
        "\nChapter 11 needs LightGBM, which did not load:\n"
        f"    {type(_err).__name__}: {_err}\n\n"
        "    pip install -r requirements-ml.txt\n\n"
        "On macOS it also needs the OpenMP runtime, which Apple does not\n"
        "ship:\n"
        "    brew install libomp\n\n"
        "Nothing else in the book needs LightGBM. Chapters 5-10 and 16 run\n"
        "without it, and Chapter 11.7 concludes the ML sleeve is not worth\n"
        "its operational cost anyway.\n")
from tenalgos.data.loaders import load_panel
from tenalgos.research.backtest import (run_backtest, information_coefficient,
                                        ic_summary)
from tenalgos.research.costs import CostModel
from tenalgos.research.cv import PurgedKFold
from tenalgos.research.stats import equity_curve, drawdown, sharpe, deflated_sharpe
from tenalgos.strategies import ch05_xsmom as mom
from tenalgos.strategies import ch11_ml as ml

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
HORIZON, EMBARGO = 21, 10

print("loading ...")
mkt = load_panel(top_n_by_liquidity=800)
px, rets, dv = mkt["prices"], mkt["returns"], mkt["dollar_volume"]
vol21 = rets.rolling(21).std()
cm = CostModel()

print("building features ...")
F = ml.make_features(px, rets, dollar_volume=dv)
y = ml.make_label(rets, horizon=HORIZON)
res["features"] = list(F)

panel = ml.to_long(F, y)
dates = panel.index.get_level_values(0).unique().sort_values()
print(f"  {len(panel):,} rows, {len(F)} features, {len(dates)} dates")
res["n_rows"] = int(len(panel))

FEATS = list(F)
PARAMS = dict(objective="regression", n_estimators=300, learning_rate=0.03,
              num_leaves=31, min_child_samples=200, subsample=0.7,
              subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
              verbose=-1, n_jobs=4)


def fit_predict(splits, extra_col=None):
    """Out-of-fold predictions given a list of (train_dates, test_dates)."""
    cols = FEATS + ([extra_col] if extra_col else [])
    out = []
    for tr_d, te_d in splits:
        tr = panel[panel.index.get_level_values(0).isin(tr_d)]
        te = panel[panel.index.get_level_values(0).isin(te_d)]
        if len(tr) < 5000 or len(te) < 500:
            continue
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(tr[cols], tr["y"])
        out.append(pd.Series(m.predict(te[cols]), index=te.index))
    return pd.concat(out).sort_index()


# ------------------------------------------------------------ the splits ---
def plain_folds(k=6):
    b = np.linspace(0, len(dates), k + 1).astype(int)
    for i in range(k):
        te = dates[b[i]:b[i + 1]]
        tr = dates[~dates.isin(te)]
        yield tr, te


def purged_folds(k=6):
    pk = PurgedKFold(n_splits=k, label_horizon=HORIZON, embargo=EMBARGO)
    for tr_i, te_i in pk.split(dates):
        yield dates[tr_i], dates[te_i]


def walk_forward(n=6):
    b = np.linspace(0, len(dates), n + 2).astype(int)
    for i in range(1, n + 1):
        tr_end = b[i] - HORIZON - EMBARGO          # purge the boundary
        yield dates[:max(tr_end, 1)], dates[b[i]:b[i + 1]]


# ------------------------------------------------------------- the models ---
def evaluate(pred, name):
    sig = ml.predictions_to_signal(pred, rets.index, rets.columns)
    held = mom.hold_monthly(sig) if hasattr(mom, "hold_monthly") else sig
    marks = sig.resample("ME").last().index
    held = sig.where(pd.Series(sig.index.isin(marks), index=sig.index),
                     np.nan).ffill()
    w = mom.decile_weights(held)
    b = run_backtest(w, rets, cm, vol=vol21, dollar_volume=dv, capital=1e8)
    ic = ic_summary(information_coefficient(sig, rets.shift(-1)))
    return {"name": name,
            "sharpe_gross": round(float(b.stats_gross["sharpe"]), 3),
            "sharpe_net": round(float(b.stats_net["sharpe"]), 3),
            "ann_net": round(float(b.stats_net["ann_return"]), 4),
            "max_dd": round(float(b.stats_net["max_drawdown"]), 4),
            "turnover": round(float(b.stats_net["ann_turnover"]), 1),
            "ic_mean": round(float(ic["mean_ic"]), 4),
            "ic_t": round(float(ic["t_stat"]), 2)}, b, sig


print("\n[1/4] LEAKY-CV: plain K-fold ...")
p_leaky = fit_predict(list(plain_folds()))
s_leaky, bt_leaky, sig_leaky = evaluate(p_leaky, "plain K-fold (leaky)")
print("   ", s_leaky)

print("[2/4] LEAKY-FEAT: plain K-fold + a feature containing the future ...")
# The classic accident: a "volatility" feature computed on a centred window.
# It looks like an ordinary rolling statistic. It sees 10 days ahead.
panel_leak = panel.copy()
centred = rets.rolling(21, center=True, min_periods=15).std()
panel_leak["vol_centred"] = mom.zscore_cross_section(centred).stack(
    future_stack=True).reindex(panel.index)
panel_leak = panel_leak.dropna()
_panel_backup, panel = panel, panel_leak
dates_leak = panel.index.get_level_values(0).unique().sort_values()
_d_backup, dates = dates, dates_leak
p_lfeat = fit_predict(list(plain_folds()), extra_col="vol_centred")
panel, dates = _panel_backup, _d_backup
s_lfeat, bt_lfeat, _ = evaluate(p_lfeat, "plain K-fold + leaky feature")
print("   ", s_lfeat)

print("[3/4] HONEST: purged K-fold with embargo ...")
p_purged = fit_predict(list(purged_folds()))
s_purged, bt_purged, sig_purged = evaluate(p_purged, "purged K-fold + embargo")
print("   ", s_purged)

print("[4/4] HONEST: walk-forward ...")
p_wf = fit_predict(list(walk_forward()))
s_wf, bt_wf, sig_wf = evaluate(p_wf, "walk-forward")
print("   ", s_wf)

res["models"] = [s_leaky, s_lfeat, s_purged, s_wf]

# ------------------------------------------------------ feature importance ---
m_full = lgb.LGBMRegressor(**PARAMS)
m_full.fit(panel[FEATS], panel["y"])
imp = pd.Series(m_full.feature_importances_, index=FEATS).sort_values()
res["feature_importance"] = {k: int(v) for k, v in imp.items()}

# --------------------------------------------- does it ADD or RESTATE? ------
print("\ncomparing against the linear sleeves ...")
_, w_mom = mom.build(px, rets, industry=None, neutralise_industry=False,
                     neutralise_beta=True)
bt_mom = run_backtest(w_mom, rets, cm, vol=vol21, dollar_volume=dv, capital=1e8)

from tenalgos.strategies import ch06_multifactor as mf
market = rets.mean(axis=1)
beta = mom.trailing_beta(rets, market)
w_lr = mf.beta_balanced_weights(mf.hold_monthly(
    mom.zscore_cross_section(mf.low_risk_signal(rets))), beta)
bt_lr = run_backtest(w_lr, rets, cm, vol=vol21, dollar_volume=dv, capital=1e8)

comp = pd.DataFrame({"ml": bt_purged.net_returns,
                     "momentum": bt_mom.net_returns,
                     "low_risk": bt_lr.net_returns}).dropna()
res["correlation_to_sleeves"] = comp.corr().round(3).to_dict()

# regress ML on the two linear sleeves: is there residual alpha?
import numpy.linalg as la
Xr = np.column_stack([np.ones(len(comp)), comp["momentum"], comp["low_risk"]])
coef, *_ = la.lstsq(Xr, comp["ml"].to_numpy(), rcond=None)
resid = comp["ml"].to_numpy() - Xr @ coef
res["spanning_regression"] = {
    "alpha_ann": round(float(coef[0] * 252), 4),
    "beta_momentum": round(float(coef[1]), 3),
    "beta_low_risk": round(float(coef[2]), 3),
    "residual_sharpe": round(float(sharpe(pd.Series(resid, index=comp.index))), 3),
    "r_squared": round(float(1 - resid.var() / comp["ml"].var()), 3)}

# add ML to the two-sleeve book
two = (comp["momentum"] + comp["low_risk"]) / 2
three = (comp["momentum"] + comp["low_risk"] + comp["ml"]) / 3
res["ensemble"] = {"two_sleeve_sharpe": round(float(sharpe(two)), 3),
                   "three_sleeve_sharpe": round(float(sharpe(three)), 3)}

srn = s_purged["sharpe_net"]
nd = len(bt_purged.net_returns)
res["deflated"] = {f"{t}_trials": round(float(deflated_sharpe(srn, nd, n_trials=t)), 4)
                   for t in (1, 4, 71)}
res["oos"] = {"in_sample": round(float(sharpe(bt_purged.net_returns.loc[:"2013-12-31"])), 3),
              "out_of_sample": round(float(sharpe(bt_purged.net_returns.loc["2014-01-01":])), 3)}

# --------------------------------------------------------------- fig 11.1 ---
fig, ax = plt.subplots(2, 1, figsize=(6.8, 5.4), sharex=True,
                       gridspec_kw={"height_ratios": [2, 1]})
for b, lab, c, lw in [(bt_lfeat, "plain CV + leaky feature", RED, 1.3),
                      (bt_leaky, "plain K-fold", GOLD, 1.2),
                      (bt_purged, "purged K-fold + embargo", BLUE, 1.4),
                      (bt_wf, "walk-forward", INK, 1.6)]:
    e = equity_curve(b.net_returns)
    ax[0].plot(e.index, e, color=c, lw=lw,
               label=f"{lab}  (SR {b.stats_net['sharpe']:.2f})")
ax[0].set_yscale("log"); ax[0].set_ylabel("growth of $1, net (log)")
ax[0].legend(frameon=False, loc="upper left", fontsize=8)
ax[0].set_title("The same model and the same features, validated four ways",
                loc="left", fontsize=10.5, pad=8)
d = drawdown(bt_purged.net_returns)
ax[1].fill_between(d.index, d, 0, color=RED, alpha=0.28, lw=0)
ax[1].plot(d.index, d, color=RED, lw=0.8)
ax[1].yaxis.set_major_formatter(pct); ax[1].set_ylabel("drawdown, honest")
fig.savefig(FIG / "fig11_01_validation_real.png"); plt.close(fig)

# --------------------------------------------------------------- fig 11.2 ---
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4))
ax[0].barh(range(len(imp)), imp.values, color=GOLD, edgecolor=INK, linewidth=0.5)
ax[0].set_yticks(range(len(imp)), imp.index, fontsize=8)
ax[0].set_xlabel("LightGBM split importance")
ax[0].set_title("What the model actually used", loc="left", fontsize=10)

c = comp.corr()
im = ax[1].imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
ax[1].set_xticks(range(len(c)), c.columns, fontsize=8, rotation=20)
ax[1].set_yticks(range(len(c)), c.index, fontsize=8); ax[1].grid(False)
for i in range(len(c)):
    for j in range(len(c)):
        ax[1].text(j, i, f"{c.iloc[i,j]:.2f}", ha="center", va="center", fontsize=9)
ax[1].set_title("Does it add, or restate?", loc="left", fontsize=10)
fig.colorbar(im, ax=ax[1], fraction=0.045)
fig.tight_layout(); fig.savefig(FIG / "fig11_02_features_real.png"); plt.close(fig)

(FIG / "ch11_numbers.json").write_text(json.dumps(res, indent=2, default=float))
print("\n--- models ---"); print(pd.DataFrame(res["models"]).to_string(index=False))
print("\n--- feature importance ---")
print(imp.sort_values(ascending=False).to_string())
print("\n--- correlation to linear sleeves ---"); print(comp.corr().round(3).to_string())
print("\n--- spanning regression (ML on momentum + low risk) ---")
print(json.dumps(res["spanning_regression"], indent=2))
print("ensemble:", res["ensemble"])
print("deflated:", res["deflated"], " oos:", res["oos"])
print("\nfigures ->", FIG)

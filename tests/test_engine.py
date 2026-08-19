"""Prove the engine is correct before trusting a single result from it.

Four tests, each of which corresponds to a bug that has cost real firms real
money:

    1. NULL MARKET      no alpha injected -> the engine must report an IC and a
                        Sharpe indistinguishable from zero. A backtester with
                        look-ahead will happily report a Sharpe of 2 here.
    2. KNOWN ALPHA      alpha injected -> the engine must find it, and the
                        measured IC must rise monotonically with the injected
                        strength. This is the calibration test.
    3. NO LOOK-AHEAD    the oracle test. A signal equal to tomorrow's return
                        must be enormously profitable; a signal equal to
                        today's return must be worth nothing. Together they
                        pin the engine's timing to exactly one day.
    4. PURGING WORKS    purged K-fold must open a gap of at least the label
                        horizon between train and test; plain K-fold does not.

Run:  python tests/test_engine.py
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from tenalgos.data.synthetic import make_market
from tenalgos.research.backtest import (run_backtest, information_coefficient,
                                        ic_summary)
from tenalgos.research.costs import CostModel
from tenalgos.research.cv import PurgedKFold, plain_kfold
from tenalgos.strategies import ch05_xsmom as mom

RESULTS = {}
N_ASSETS, N_DAYS = 400, 252 * 15


def _build(mkt, **kw):
    return mom.build(mkt["prices"], mkt["returns"], mkt["industry"], **kw)


# ---------------------------------------------------------------- test 1 ---
def test_null_market():
    print("\n[1] NULL MARKET — no momentum injected, no delistings")
    mkt = make_market(n_assets=N_ASSETS, n_days=N_DAYS, mom_strength=0.0,
                      survivorship=False, seed=7)
    sig, w = _build(mkt)
    bt = run_backtest(w, mkt["returns"], CostModel(), name="xsmom-null")
    ic = ic_summary(information_coefficient(sig, mkt["returns"].shift(-1)))
    sr, se = bt.stats_gross["sharpe"], bt.stats_gross["sharpe_se"]
    print(f"    mean IC      = {ic['mean_ic']:+.5f}   t-stat = {ic['t_stat']:+.2f}")
    print(f"    gross Sharpe = {sr:+.3f}   (standard error {se:.3f}, so |t| = {abs(sr/se):.2f})")
    ok = abs(ic["t_stat"]) < 3.0 and abs(sr / se) < 3.0
    print(f"    -> {'PASS' if ok else 'FAIL'}  engine finds nothing where nothing exists")
    RESULTS["null"] = dict(ic=ic["mean_ic"], t=ic["t_stat"], sharpe=sr, se=se, ok=ok)
    return ok


# ---------------------------------------------------------------- test 2 ---
def test_known_alpha():
    print("\n[2] KNOWN ALPHA — measured IC must rise with injected strength")
    rows = []
    for strength in [0.0000, 0.0003, 0.0006, 0.0012]:
        mkt = make_market(n_assets=N_ASSETS, n_days=N_DAYS, mom_strength=strength,
                          survivorship=False, seed=11)
        sig, w = _build(mkt)
        bt = run_backtest(w, mkt["returns"], CostModel(), name="xsmom")
        ic = ic_summary(information_coefficient(sig, mkt["returns"].shift(-1)))
        rows.append(dict(strength=strength, ic=ic["mean_ic"], t=ic["t_stat"],
                         sr_gross=bt.stats_gross["sharpe"],
                         sr_net=bt.stats_net["sharpe"]))
        print(f"    injected {strength:.4f} -> IC {ic['mean_ic']:+.4f}  t {ic['t_stat']:+6.2f}"
              f"   Sharpe gross {bt.stats_gross['sharpe']:+.2f}  net {bt.stats_net['sharpe']:+.2f}")
    ics = [r["ic"] for r in rows]
    ok = all(ics[i] < ics[i + 1] for i in range(len(ics) - 1))
    print(f"    -> {'PASS' if ok else 'FAIL'}  IC is monotone in injected alpha")
    RESULTS["known_alpha"] = dict(rows=rows, ok=ok)
    return ok


# ---------------------------------------------------------------- test 3 ---
def test_no_lookahead():
    """The oracle test: the definitive check that signal[t] earns return[t+1].

    Two signals, both run through the engine identically:

        ORACLE   signal[t] = return[t+1]   -- tomorrow, known only in hindsight
        STALE    signal[t] = return[t]     -- today, already realised

    With a correct one-day execution lag, ORACLE must be enormously profitable
    (it is literally tomorrow's answer) and STALE must be worth roughly nothing
    (it is yesterday's news by the time you trade). If STALE also prints a large
    Sharpe, the engine is off by one and every number it has ever produced is
    fiction.
    """
    print("\n[3] NO LOOK-AHEAD — the oracle test")
    mkt = make_market(n_assets=N_ASSETS, n_days=252 * 8, mom_strength=0.0,
                      survivorship=False, seed=13)
    rets = mkt["returns"]

    def ls_weights(signal):
        return mom.decile_weights(signal, n_bins=10, gross=2.0)

    oracle = run_backtest(ls_weights(rets.shift(-1)), rets, CostModel(),
                          execution_lag=1, name="oracle")
    stale = run_backtest(ls_weights(rets), rets, CostModel(),
                         execution_lag=1, name="stale")
    o, s = oracle.stats_gross["sharpe"], stale.stats_gross["sharpe"]
    print(f"    ORACLE  signal[t] = return[t+1]  ->  Sharpe {o:+8.2f}")
    print(f"    STALE   signal[t] = return[t]    ->  Sharpe {s:+8.2f}")
    ok = o > 20.0 and abs(s) < 3.0
    print(f"    -> {'PASS' if ok else 'FAIL'}  signal[t] earns return[t+1], and never return[t]")
    RESULTS["lookahead"] = dict(oracle=o, stale=s, ok=ok)
    return ok


# ---------------------------------------------------------------- test 4 ---
def test_purging():
    print("\n[4] PURGING — purged K-fold must open a real gap; plain K-fold does not")
    H, EMB = 21, 10
    idx = pd.bdate_range("2010-01-04", periods=252 * 12)

    def gaps(splitter):
        before, after = [], []
        for tr, te in splitter:
            lo, hi = te[0], te[-1]
            left = tr[tr < lo]
            right = tr[tr > hi]
            if len(left):
                before.append(lo - left.max())
            if len(right):
                after.append(right.min() - hi)
        return (min(before) if before else np.inf,
                min(after) if after else np.inf)

    p_b, p_a = gaps(plain_kfold(idx, 6))
    q_b, q_a = gaps(PurgedKFold(6, label_horizon=H, embargo=EMB).split(idx))
    print(f"    label horizon {H}d, embargo {EMB}d")
    print(f"    plain  K-fold: gap before test = {p_b:>4}d   after = {p_a:>4}d")
    print(f"    purged K-fold: gap before test = {q_b:>4}d   after = {q_a:>4}d")
    ok = (q_b > H) and (q_a > EMB) and (p_b <= 1) and (p_a <= 1)
    print(f"    -> {'PASS' if ok else 'FAIL'}  purge covers the label window, embargo covers the tail")
    RESULTS["purging"] = dict(plain=(p_b, p_a), purged=(q_b, q_a), ok=ok)
    return ok


if __name__ == "__main__":
    print("=" * 70)
    print("ENGINE VALIDATION — tenalgos")
    print("=" * 70)
    res = [test_null_market(), test_known_alpha(), test_no_lookahead(), test_purging()]
    print("\n" + "=" * 70)
    print(f"{sum(res)}/{len(res)} tests passed")
    print("=" * 70)
    sys.exit(0 if all(res) else 1)

"""Add an unadjusted-close frame to a panel, so distribution yield is computable.

Chapter 8 needs carry, and for an ETF the cleanest price-only measure of carry
is its distribution yield. Both series are already in the raw cache:

    adjclose   total return  — price change plus reinvested distributions
    close      price only

so  (adjclose return) - (close return)  is the distribution, in return units.
Accumulate that over a year and you have a trailing yield per market, computed
from data everyone has, with no vendor and no assumptions.

    python build_yield_panel.py --panel etfs
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd


def main(panel: str) -> None:
    d = Path("data") / panel
    raw = d / "raw"
    if not raw.exists():
        raise SystemExit(f"no raw cache at {raw}")

    close = pd.read_parquet(d / "close.parquet")      # adjusted
    cols, idx = list(close.columns), close.index
    out = pd.DataFrame(np.nan, index=idx, columns=cols, dtype="float32")

    print(f"reading unadjusted close for {len(cols):,} tickers ...")
    for i, tk in enumerate(cols, 1):
        f = raw / f"{tk}.parquet"
        if not f.exists():
            continue
        try:
            s = pd.read_parquet(f, columns=["close"])["close"]
        except Exception:                                    # noqa: BLE001
            continue
        s = s[~s.index.duplicated(keep="last")]
        out[tk] = s.reindex(idx).astype("float32")
        if i % 500 == 0:
            print(f"  {i:,}/{len(cols):,}")

    out.to_parquet(d / "close_unadjusted.parquet")

    # sanity: the implied yield should be non-negative and small for most names
    tr = close.pct_change(fill_method=None)
    pr = out.pct_change(fill_method=None)
    dist = (tr - pr)
    ann = dist.rolling(252, min_periods=200).sum()
    med = float(ann.stack().median())
    p99 = float(ann.stack().quantile(0.99))
    neg = float((ann.stack() < -0.02).mean())
    print(f"\nimplied trailing 12m distribution yield")
    print(f"  median          {med:+.4f}")
    print(f"  99th percentile {p99:+.4f}")
    print(f"  share below -2% {neg:.4%}   (should be near zero)")
    (d / "yield_check.json").write_text(json.dumps(
        {"median": med, "p99": p99, "share_below_minus_2pct": neg}, indent=2))
    print(f"\nwrote {d/'close_unadjusted.parquet'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="etfs")
    main(ap.parse_args().panel)

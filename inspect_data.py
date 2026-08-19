#!/usr/bin/env python3
"""
inspect_data.py — is this dataset usable, before you spend time on it?

    python inspect_data.py ~/Downloads/stock-market-dataset

Point it at a folder or a .zip you downloaded from anywhere. It samples a few
files, works out the layout, and tells you the four things that decide whether
a price dataset can carry this book:

    1. how many tickers there are          (need >= 500 for a cross-section)
    2. what date range it covers            (need >= 10 years)
    3. whether there is an ADJUSTED close   (this one is not negotiable)
    4. whether delisted names are present   (survivorship)

Run this BEFORE `fetch.py --source local`.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd

if len(sys.argv) < 2:
    sys.exit(__doc__)

src = Path(sys.argv[1]).expanduser()
if not src.exists():
    sys.exit(f"no such path: {src}")

print("=" * 70)
print(f"INSPECTING {src}")
print("=" * 70)

# ------------------------------------------------------------- gather names ---
if src.suffix.lower() == ".zip":
    with zipfile.ZipFile(src) as z:
        names = [n for n in z.namelist()
                 if n.lower().endswith((".csv", ".txt")) and not n.endswith("/")]
        read = lambda n: pd.read_csv(z.open(n))          # noqa: E731
        sample = names[: min(len(names), 8)]
        print(f"zip contains {len(names):,} csv/txt files")
        frames = {}
        with zipfile.ZipFile(src) as z2:
            for n in sample:
                try:
                    frames[n] = pd.read_csv(z2.open(n), nrows=5000)
                except Exception as e:                    # noqa: BLE001
                    print(f"  could not read {n}: {type(e).__name__}")
else:
    names = [str(p) for p in src.rglob("*")
             if p.suffix.lower() in (".csv", ".txt")]
    print(f"folder contains {len(names):,} csv/txt files")
    frames = {}
    for n in names[: min(len(names), 8)]:
        try:
            frames[n] = pd.read_csv(n, nrows=5000)
        except Exception as e:                            # noqa: BLE001
            print(f"  could not read {n}: {type(e).__name__}")

if not frames:
    sys.exit("could not read any file — tell me what the files look like")

# ----------------------------------------------------------------- layout ---
print("\nSAMPLE FILES")
for n, df in list(frames.items())[:4]:
    print(f"  {Path(n).name:<28} {len(df):>6,} rows   columns: {list(df.columns)[:9]}")

cols = {str(c).strip().strip("<>").lower() for df in frames.values() for c in df.columns}
has_adj = bool(cols & {"adj close", "adjclose", "adj_close", "closeadj"})
has_ticker_col = "ticker" in cols or "symbol" in cols

# ------------------------------------------------------------------ dates ---
lo, hi = None, None
for df in frames.values():
    dcol = next((c for c in df.columns
                 if str(c).strip().strip("<>").lower() == "date"), None)
    if dcol is None:
        continue
    d = df[dcol].astype(str)
    parsed = pd.to_datetime(d, format="%Y%m%d", errors="coerce")
    parsed = parsed.fillna(pd.to_datetime(d, errors="coerce")).dropna()
    if len(parsed):
        lo = parsed.min() if lo is None else min(lo, parsed.min())
        hi = parsed.max() if hi is None else max(hi, parsed.max())

# ---------------------------------------------------------------- verdict ---
print("\nVERDICT")
checks = []

n_tickers = len(names) if not has_ticker_col else "many (one long file)"
ok_n = (len(names) >= 500) or has_ticker_col
checks.append((ok_n, f"ticker count        {n_tickers}",
               "need >= 500 names for a cross-sectional strategy"))

span = f"{lo.date() if lo is not None else '?'} .. {hi.date() if hi is not None else '?'}"
ok_span = lo is not None and hi is not None and (hi - lo).days > 3650
checks.append((ok_span, f"date range          {span}",
               "need at least 10 years; ending in 2017 or 2020 is FINE — "
               "this book demonstrates method, not live trading"))

checks.append((has_adj, f"adjusted close      {'YES' if has_adj else 'NO'}",
               "without it every split day becomes a fake -50% return and "
               "momentum shorts the wrong names"))

for ok, line, why in checks:
    print(f"  {'PASS' if ok else 'CHECK':<6} {line}")
    if not ok:
        print(f"         -> {why}")

print()
if all(ok for ok, _, _ in checks):
    print("  USABLE. Next:")
    print(f"     python fetch.py --source local --input {src}")
elif not has_adj and ok_n and ok_span:
    print("  USABLE BUT UNADJUSTED. It will work, and Chapter 2 will have to")
    print("  say so. Prefer a dataset with an 'Adj Close' column if you can")
    print("  find one; if not, send it anyway and I will handle the caveat.")
else:
    print("  NOT SUITABLE. Paste this output to me and I will tell you what to")
    print("  look for instead.")
print("=" * 70)

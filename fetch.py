#!/usr/bin/env python3
"""
fetch.py — build the US equity panel for THE QUANT BILLIONAIRE.

Run this once on your own machine. It needs no API key and no paid account.

    pip install pandas requests pyarrow
    python fetch.py

It will:

    1. download the official Nasdaq symbol directory (free, no key)
    2. keep ordinary common stock, dropping ETFs, funds, warrants, units,
       preferreds, rights and test issues
    3. download ~20 years of daily bars per ticker from Stooq (free, no key)
    4. write everything to ./data/ as parquet
    5. print a summary you can paste back to me

It is resumable. Interrupt it with Ctrl-C and run it again; it skips anything
already downloaded. Expect 40-90 minutes and roughly 300-500 MB.

    python fetch.py --limit 50        # quick smoke test first (2 minutes)
    python fetch.py                   # the real run
    python fetch.py --workers 4       # gentler on the source if you get 429s

------------------------------------------------------------------------------
ONE HONEST WARNING, WHICH IS ALSO CHAPTER 2 OF THE BOOK
------------------------------------------------------------------------------
Stooq serves *currently listed* tickers. Companies that went bankrupt, were
acquired or were delisted are largely missing. That is survivorship bias, and
it is the single most common reason an amateur backtest looks better than the
strategy really is.

This script does not pretend otherwise. It records the exact universe and dates
it captured so the book can *measure* the bias rather than ignore it, and
Chapter 2 shows how much it inflates a momentum backtest. If you later decide
you want a survivorship-free, point-in-time panel, Sharadar Core US Equities
(~$50/month via Nasdaq Data Link) is the cheapest good one, and this script has
a --source sharadar path ready for it.
"""
from __future__ import annotations

import argparse
import io
import re
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA = Path("data")
RAW = DATA / "raw"


def set_panel(name: str | None) -> None:
    """Point every write at data/<name>/ instead of data/.

    This exists because of a real accident. Ingesting a second dataset without
    it appends to the same raw cache and silently rebuilds ONE panel out of
    two different universes — US equities and ETFs, in our case — which then
    changes every backtest that had already been run and published. Different
    universes get different panels. Always.
    """
    global DATA, RAW
    DATA = Path("data") if not name else Path("data") / name
    RAW = DATA / "raw"
NASDAQ_DIR = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqtraded.txt"
STOOQ = "https://stooq.com/q/d/l/?s={sym}.us&i=d"

BULK_ZIP = "https://static.stooq.com/db/h/d_us_txt.zip"   # 401: paid/registered
NDL = "https://data.nasdaq.com/api/v3/datatables/SHARADAR"
SCREENER = ("https://api.nasdaq.com/api/screener/stocks"
            "?tableonly=true&limit=25000&download=true")
YAHOO = ("https://query2.finance.yahoo.com/v8/finance/chart/{sym}"
         "?period1=1072915200&period2=9999999999&interval=1d"
         "&events=div%7Csplit&includeAdjustedClose=true")

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "text/csv,text/plain,application/zip,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# --------------------------------------------------------------- universe ---
def get_universe(limit: int | None = None) -> pd.DataFrame:
    """The official Nasdaq-traded symbol directory, filtered to common stock."""
    cache = DATA / "universe.csv"
    if cache.exists():
        uni = pd.read_csv(cache)
        print(f"universe: {len(uni):,} tickers (cached)")
        return uni.head(limit) if limit else uni

    print("downloading the Nasdaq symbol directory ...")
    r = SESSION.get(NASDAQ_DIR, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep="|")
    df = df[df["Symbol"].notna()]
    df = df[df["Nasdaq Traded"] == "Y"]
    df = df[df["ETF"] != "Y"]
    df = df[df["Test Issue"] != "Y"]

    # Ordinary common stock only. Suffix letters and punctuation mark
    # warrants, units, rights, preferreds and when-issued lines.
    bad_words = ("ETF", "ETN", "Fund", "Trust", "Warrant", "Unit", "Right",
                 "Preferred", "Depositary", "Notes", "% ", "SPAC")
    df = df[~df["Security Name"].str.contains("|".join(bad_words), case=False, na=False)]
    df = df[df["Symbol"].str.fullmatch(r"[A-Z]{1,5}")]

    uni = (df[["Symbol", "Security Name", "Listing Exchange"]]
           .rename(columns={"Symbol": "ticker", "Security Name": "name",
                            "Listing Exchange": "exchange"})
           .drop_duplicates("ticker")
           .sort_values("ticker")
           .reset_index(drop=True))

    DATA.mkdir(exist_ok=True)
    uni.to_csv(cache, index=False)
    print(f"universe: {len(uni):,} common-stock tickers")
    return uni.head(limit) if limit else uni


def rank_by_size(uni: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Keep only the largest `top_n` names, using the free Nasdaq screener.

    This is the fix for the whole rate-limit problem. `load_panel` screens the
    universe down to the top 1,500 by liquidity anyway, so downloading 5,000
    tickers spends four times the requests to build a panel that is then thrown
    away. Screen first, download second.
    """
    cache = DATA / "marketcap.csv"
    if cache.exists():
        caps = pd.read_csv(cache)
    else:
        print("downloading market caps from the Nasdaq screener ...")
        try:
            r = SESSION.get(SCREENER, timeout=90,
                            headers={"Accept": "application/json"})
            rows = r.json()["data"]["rows"]
            caps = pd.DataFrame(rows)[["symbol", "marketCap"]]
            caps["marketCap"] = pd.to_numeric(
                caps["marketCap"].astype(str).str.replace(r"[$,]", "", regex=True),
                errors="coerce")
            caps = caps.rename(columns={"symbol": "ticker"}).dropna()
            caps.to_csv(cache, index=False)
        except Exception as e:                          # noqa: BLE001
            print(f"  screener unavailable ({type(e).__name__}); "
                  f"keeping the first {top_n:,} alphabetically instead")
            return uni.head(top_n)

    merged = uni.merge(caps, on="ticker", how="inner")
    merged = merged.sort_values("marketCap", ascending=False).head(top_n)
    print(f"universe screened to the largest {len(merged):,} by market cap "
          f"(smallest kept: ${merged['marketCap'].min()/1e9:.2f}bn)")
    return merged.reset_index(drop=True)


# ----------------------------------------------------------------- prices ---
def fetch_one(ticker: str, retries: int = 3) -> tuple[str, str]:
    """Download one ticker to parquet. Returns (ticker, status)."""
    out = RAW / f"{ticker}.parquet"
    if out.exists() and out.stat().st_size > 500:
        return ticker, "cached"

    url = STOOQ.format(sym=ticker.lower())
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=45)
            if r.status_code == 429:
                time.sleep(8 + attempt * 12 + random.random() * 4)
                continue
            if r.status_code != 200 or len(r.text) < 120:
                return ticker, "empty"
            if r.text.lstrip().startswith("<"):
                return ticker, "blocked"

            df = pd.read_csv(io.StringIO(r.text))
            need = {"Date", "Open", "High", "Low", "Close", "Volume"}
            if not need.issubset(df.columns) or len(df) < 250:
                return ticker, "too short"

            df["Date"] = pd.to_datetime(df["Date"])
            df = df.rename(columns=str.lower).set_index("date").sort_index()
            df = df[df.index >= "2004-01-01"]
            if len(df) < 250:
                return ticker, "too short"

            df["ticker"] = ticker
            df.to_parquet(out)
            return ticker, "ok"
        except Exception as e:                       # noqa: BLE001
            if attempt == retries - 1:
                return ticker, f"error: {type(e).__name__}"
            time.sleep(3 + attempt * 5)
    return ticker, "failed"


def fetch_either(ticker: str) -> tuple[str, str]:
    """Try Yahoo, then Stooq. Two free sources fail in different ways, so
    trying both roughly doubles the chance any given ticker lands."""
    tk, s = fetch_one_yahoo(ticker, retries=2)
    if s in ("ok", "cached"):
        return tk, s
    tk, s2 = fetch_one(ticker, retries=2)
    return tk, s2 if s2 in ("ok", "cached") else f"{s} / {s2}"


def fetch_all(tickers: list[str], workers: int = 6, pause: float = 0.25,
              source: str = "yahoo"):
    one = {"yahoo": fetch_one_yahoo, "stooq": fetch_one,
           "both": fetch_either}.get(source, fetch_one_yahoo)
    RAW.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for tk in tickers:
            futures[ex.submit(one, tk)] = tk
            time.sleep(pause / max(workers, 1))
        for fut in as_completed(futures):
            tk, status = fut.result()
            key = status.split(":")[0]
            stats[key] = stats.get(key, 0) + 1
            done += 1

            # Fail fast. A run that is producing nothing should say so in
            # thirty seconds, not in thirteen hours.
            if done >= 20 and stats.get("ok", 0) + stats.get("cached", 0) == 0:
                ex.shutdown(wait=False, cancel_futures=True)
                raise SystemExit(
                    f"\naborting: {done} tickers attempted, none succeeded.\n"
                    f"statuses seen: {dict(sorted(stats.items()))}\n\n"
                    f"Run this and send me the output:\n"
                    f"    python fetch.py --diagnose --proxy <your proxy>")

            if done % 25 == 0 or done == len(tickers):
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(tickers) - done) / rate / 60 if rate else 0
                print(f"  {done:>5}/{len(tickers)}  "
                      f"ok={stats.get('ok',0)} cached={stats.get('cached',0)} "
                      f"skip={done-stats.get('ok',0)-stats.get('cached',0)}  "
                      f"{rate:.1f}/s  ETA {eta:.0f} min", flush=True)
    return stats


# ------------------------------------------------------------------ yahoo ---
_CRUMB = {"value": None}


def yahoo_auth() -> str | None:
    """Get the cookie + crumb pair Yahoo now requires.

    Since 2024 the chart endpoint returns 401 to anonymous callers. The
    handshake is: pick up cookies from fc.yahoo.com, then exchange them for a
    short crumb which must be appended to every request.
    """
    if _CRUMB["value"]:
        return _CRUMB["value"]
    try:
        SESSION.get("https://fc.yahoo.com", timeout=20)      # 404, but sets cookies
    except Exception:                                        # noqa: BLE001
        pass
    try:
        r = SESSION.get("https://query2.finance.yahoo.com/v1/test/getcrumb",
                        timeout=20)
        crumb = r.text.strip()
        if r.status_code == 200 and crumb and len(crumb) < 40 and "<" not in crumb:
            _CRUMB["value"] = crumb
            return crumb
        print(f"    crumb handshake failed: HTTP {r.status_code} {crumb[:60]!r}")
    except Exception as e:                                   # noqa: BLE001
        print(f"    crumb handshake error: {type(e).__name__}")
    return None


def diagnose(ticker: str = "AAPL") -> None:
    """One request, everything printed. Stop guessing, look at the response."""
    print(f"\ndiagnosing {ticker}\n" + "-" * 60)
    crumb = yahoo_auth()
    print(f"crumb            : {crumb!r}")
    print(f"cookies held     : {len(SESSION.cookies)} "
          f"{[c.name for c in SESSION.cookies][:6]}")
    url = YAHOO.format(sym=ticker) + (f"&crumb={crumb}" if crumb else "")
    try:
        r = SESSION.get(url, timeout=30)
        print(f"HTTP             : {r.status_code}")
        print(f"content-type     : {r.headers.get('content-type')}")
        body = r.text[:400].replace("\n", " ")
        print(f"first 400 chars  : {body}")
        if r.status_code == 200:
            j = r.json()
            res = (j.get("chart") or {}).get("result")
            err = (j.get("chart") or {}).get("error")
            print(f"chart.error      : {err}")
            if res:
                n = len(res[0].get("timestamp") or [])
                cols = list(res[0].get("indicators", {}).keys())
                print(f"rows returned    : {n:,}   indicator blocks: {cols}")
                print("\n-> YAHOO IS USABLE")
                return
    except Exception as e:                                   # noqa: BLE001
        print(f"exception        : {type(e).__name__}: {e}")
    print("\n-> YAHOO IS NOT USABLE from this network right now")



def fetch_one_yahoo(ticker: str, retries: int = 3) -> tuple[str, str]:
    """Download one ticker from the Yahoo chart endpoint.

    Free, no key, and it returns a split- and dividend-adjusted close, which is
    what a backtest must use. Unadjusted closes put a fake -50% return on every
    split day, and Chapter 2 shows what that does to a momentum signal.
    """
    out = RAW / f"{ticker}.parquet"
    if out.exists() and out.stat().st_size > 500:
        return ticker, "cached"

    crumb = yahoo_auth()
    url = YAHOO.format(sym=ticker) + (f"&crumb={crumb}" if crumb else "")
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=45)
            if r.status_code in (429, 999):
                time.sleep(2 + attempt * 4 + random.random() * 2)
                continue
            if r.status_code == 401:
                _CRUMB["value"] = None          # crumb expired; get a new one
                if attempt == 0:
                    yahoo_auth()
                    url = YAHOO.format(sym=ticker) + (
                        f"&crumb={_CRUMB['value']}" if _CRUMB["value"] else "")
                    continue
                return ticker, "http 401 (auth)"
            if r.status_code == 404:
                return ticker, "not found"
            if r.status_code != 200:
                return ticker, f"http {r.status_code}"

            j = r.json()
            res = (j.get("chart") or {}).get("result")
            if not res:
                return ticker, "empty"
            res = res[0]
            ts = res.get("timestamp")
            if not ts or len(ts) < 250:
                return ticker, "too short"

            q = res["indicators"]["quote"][0]
            adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose")

            df = pd.DataFrame({
                "open": q.get("open"), "high": q.get("high"),
                "low": q.get("low"), "close": q.get("close"),
                "volume": q.get("volume"),
                "adjclose": adj if adj else q.get("close"),
            }, index=pd.to_datetime(ts, unit="s").normalize())
            df.index.name = "date"
            df = df.dropna(subset=["adjclose"])
            if len(df) < 250:
                return ticker, "too short"

            df["ticker"] = ticker
            df.to_parquet(out)
            return ticker, "ok"
        except Exception as e:                          # noqa: BLE001
            if attempt == retries - 1:
                return ticker, f"error: {type(e).__name__}"
            time.sleep(1 + attempt * 2)
    return ticker, "failed"


def probe_sources() -> None:
    """Try every source on three well-known tickers and report what works.

    Data sources change their terms without notice. Rather than guess, ask.
    """
    tests = ["AAPL", "MSFT", "KO"]
    print("\nprobing data sources ...\n")

    print("  YAHOO  (chart API, free, adjusted close)")
    ok = 0
    for tk in tests:
        RAW.mkdir(parents=True, exist_ok=True)
        f = RAW / f"{tk}.parquet"
        if f.exists():
            f.unlink()
        _, s = fetch_one_yahoo(tk)
        n = len(pd.read_parquet(f)) if f.exists() else 0
        print(f"    {tk:6} {s:<14} {n:>6,} rows")
        ok += s == "ok"
    print(f"    -> {'USABLE' if ok == len(tests) else 'NOT USABLE'}\n")

    print("  STOOQ  (per-symbol CSV, free, daily IP cap)")
    for tk in tests:
        f = RAW / f"_probe_{tk}.parquet"
        try:
            r = SESSION.get(STOOQ.format(sym=tk.lower()), timeout=30)
            body = r.text.lstrip()[:40].replace("\n", " ")
            state = ("blocked/HTML" if body.startswith("<")
                     else "ok" if "Date" in body else f"http {r.status_code}")
        except Exception as e:                          # noqa: BLE001
            state = f"error: {type(e).__name__}"
        print(f"    {tk:6} {state}")
    print()

    print("  STOOQ BULK ARCHIVE")
    try:
        r = SESSION.head(BULK_ZIP, timeout=30, allow_redirects=True)
        print(f"    HTTP {r.status_code} "
              f"({'usable' if r.status_code == 200 else 'requires a paid account'})")
    except Exception as e:                              # noqa: BLE001
        print(f"    error: {type(e).__name__}")
    print("\nRun the full download with whichever source says USABLE, e.g.\n"
          "    python fetch.py --source yahoo --proxy http://127.0.0.1:15236\n")


# --------------------------------------------------------------- sharadar ---
def fetch_sharadar(api_key: str, start: str = "2004-01-01") -> int:
    """Sharadar Core US Equities via Nasdaq Data Link.

    This is the dataset the book actually wants: survivorship-free, so
    delisted and acquired companies are present, with a properly adjusted
    close and a sector map. One bulk export, then a local parse.

    It is a paid table (about $50/month). Everything about Part II is more
    honest with it, and Chapter 2 stops being an apology.
    """
    DATA.mkdir(exist_ok=True)

    # ---- 1. sector map, which the free sources cannot give us at all -------
    print("downloading the ticker/sector table ...")
    r = SESSION.get(f"{NDL}/TICKERS.csv",
                    params={"table": "SEP", "api_key": api_key,
                            "qopts.columns": "ticker,name,sector,industry,"
                                             "exchange,category,isdelisted"},
                    timeout=180)
    if r.status_code == 403:
        sys.exit("Nasdaq Data Link rejected the key (403). Check that your "
                 "subscription to SHARADAR/SEP is active.")
    r.raise_for_status()
    tickers = pd.read_csv(io.StringIO(r.text))
    tickers.to_csv(DATA / "tickers.csv", index=False)
    sec = tickers[["ticker", "sector"]].dropna().drop_duplicates("ticker")
    sec.to_csv(DATA / "sectors.csv", index=False)
    print(f"  {len(tickers):,} tickers, {sec['sector'].nunique()} sectors, "
          f"{int((tickers.get('isdelisted') == 'Y').sum()):,} already delisted "
          f"-> this panel is survivorship-free")

    # ---- 2. bulk price export ---------------------------------------------
    import zipfile as _zf
    zpath = DATA / "sharadar_sep.zip"
    # A truncated download is not a valid zip, which is a better resume check
    # than a size threshold.
    if not (zpath.exists() and _zf.is_zipfile(zpath)):
        print("requesting the bulk price export (Nasdaq builds it server-side) ...")
        link = None
        for attempt in range(40):
            rr = SESSION.get(f"{NDL}/SEP.json",
                             params={"qopts.export": "true", "api_key": api_key},
                             timeout=120)
            rr.raise_for_status()
            f = rr.json()["datatable_bulk_download"]["file"]
            status = f.get("status")
            print(f"  attempt {attempt+1:>2}: {status}")
            if status == "fresh":
                link = f["link"]
                break
            time.sleep(15)
        if not link:
            sys.exit("the export never became ready. Try again in a few minutes.")

        print("downloading the export ...")
        with SESSION.get(link, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            got = 0
            with open(zpath, "wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    fh.write(chunk)
                    got += len(chunk)
                    if total:
                        print(f"\r  {got/1e6:6.0f} / {total/1e6:.0f} MB", end="", flush=True)
        print()
    else:
        print(f"export already downloaded ({zpath.stat().st_size/1e6:.0f} MB)")

    # ---- 3. parse ----------------------------------------------------------
    import zipfile
    RAW.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        name = z.namelist()[0]
        print(f"parsing {name} ...")
        with z.open(name) as fh:
            df = pd.read_csv(fh, parse_dates=["date"],
                             usecols=lambda c: c in {"ticker", "date", "open", "high",
                                                     "low", "close", "closeadj", "volume"})
    df = df[df["date"] >= start]
    print(f"  {len(df):,} rows, {df['ticker'].nunique():,} tickers")

    n = 0
    for tk, g in df.groupby("ticker", sort=False):
        if len(g) < 250 or not str(tk).isalpha() or len(str(tk)) > 5:
            continue
        g = g.set_index("date").sort_index()
        g = g.rename(columns={"closeadj": "adjclose"})
        g["ticker"] = tk
        g.to_parquet(RAW / f"{tk}.parquet")
        n += 1
        if n % 1000 == 0:
            print(f"  wrote {n:,}")
    print(f"kept {n:,} tickers")
    return n


# ------------------------------------------------------------- bulk archive ---
def fetch_bulk() -> int:
    """Download every US daily history in one archive.

    Stooq caps per-symbol downloads per IP per day, so pulling 5,000 tickers
    one at a time gets you a block page long before you finish. The bulk
    archive is a single request for the same data. It is the right way to do
    this and it is what the book will tell the reader to do.
    """
    import zipfile as _zf
    zpath = DATA / "d_us_txt.zip"

    def _usable(f):
        """A truncated download is not a valid zip, so testing the archive
        itself is a better resume check than testing the file size."""
        try:
            return f.exists() and _zf.is_zipfile(f)
        except Exception:
            return False

    if not _usable(zpath):
        print(f"downloading the bulk archive (one request, ~150-300 MB) ...")
        with SESSION.get(BULK_ZIP, stream=True, timeout=180) as r:
            ctype = r.headers.get("content-type", "")
            if r.status_code != 200 or "zip" not in ctype and "octet" not in ctype:
                sys.exit(f"bulk archive refused (HTTP {r.status_code}, {ctype}). "
                         f"Try again through your proxy:\n"
                         f"    python fetch.py --proxy http://127.0.0.1:15236")
            total = int(r.headers.get("content-length", 0))
            got = 0
            DATA.mkdir(exist_ok=True)
            with open(zpath, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
                    got += len(chunk)
                    if total:
                        print(f"\r  {got/1e6:6.0f} / {total/1e6:.0f} MB "
                              f"({100*got/total:4.1f}%)", end="", flush=True)
            print()
    else:
        print(f"bulk archive already downloaded ({zpath.stat().st_size/1e6:.0f} MB)")

    import zipfile
    RAW.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zpath) as z:
        members = [m for m in z.namelist()
                   if m.endswith(".txt") and "/us/" in m.replace("\\", "/")
                   and "stocks" in m.lower()]          # stocks only, no ETFs
        print(f"archive holds {len(members):,} US stock files; parsing ...")
        for i, m in enumerate(members):
            ticker = Path(m).stem.upper().replace(".US", "")
            if not ticker.isalpha() or len(ticker) > 5:
                continue
            out = RAW / f"{ticker}.parquet"
            if out.exists():
                n += 1
                continue
            try:
                raw = z.read(m).decode("utf-8", "ignore")
                if len(raw) < 200:
                    continue
                df = pd.read_csv(io.StringIO(raw))
                df.columns = [c.strip("<>").lower() for c in df.columns]
                if "date" not in df.columns or "close" not in df.columns:
                    continue
                df["date"] = pd.to_datetime(df["date"], format="%Y%m%d",
                                            errors="coerce")
                df = df.dropna(subset=["date"]).set_index("date").sort_index()
                df = df[df.index >= "2004-01-01"]
                if len(df) < 250:
                    continue
                df = df[[c for c in ("open", "high", "low", "close", "vol", "volume")
                         if c in df.columns]]
                df = df.rename(columns={"vol": "volume"})
                df["ticker"] = ticker
                df.to_parquet(out)
                n += 1
            except Exception:                          # noqa: BLE001
                continue
            if (i + 1) % 500 == 0:
                print(f"  {i+1:,}/{len(members):,} parsed, {n:,} kept", flush=True)
    print(f"kept {n:,} tickers with at least one year of history")
    return n


# ------------------------------------------------------------------ local ---
def ingest_local(path: str, min_tickers: int = 100) -> int:
    """Ingest price data you obtained by ANY means.

    This exists because getting the bytes and building the panel are two
    different problems, and only the first one depends on somebody else's
    rate limiter. Point this at a folder or a zip of per-ticker files and it
    will normalise whatever it finds.

    Accepts .csv, .txt, .parquet, or a .zip of those, in any of the common
    layouts: a Yahoo CSV export, a Stooq CSV, the Stooq <TICKER>,<PER>,<DATE>
    archive format, or one long file with a `ticker` column.

    The ticker is taken from the file name unless the file has a ticker column.
    """
    import json
    import zipfile

    src = Path(path)
    if not src.exists():
        sys.exit(f"no such path: {src}")
    RAW.mkdir(parents=True, exist_ok=True)

    def norm(df, ticker):
        df = df.rename(columns=lambda c: str(c).strip().strip("<>").lower())
        df = df.rename(columns={"adj close": "adjclose", "adj_close": "adjclose",
                                "closeadj": "adjclose", "vol": "volume"})
        if "date" not in df.columns or "close" not in df.columns:
            return None
        d = df["date"].astype(str)
        parsed = pd.to_datetime(d, format="%Y%m%d", errors="coerce")
        parsed = parsed.fillna(pd.to_datetime(d, errors="coerce"))
        df = df.assign(date=parsed).dropna(subset=["date"])
        if "adjclose" not in df.columns:
            df["adjclose"] = df["close"]
        if "volume" not in df.columns:
            df["volume"] = np.nan
        df = df.set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df[df.index >= "2004-01-01"]
        if len(df) < 250:
            return None
        keep = [c for c in ("open", "high", "low", "close", "adjclose", "volume")
                if c in df.columns]
        out = df[keep].copy()
        out["ticker"] = ticker
        return out

    def handle(name, reader):
        stem = Path(name).stem.upper()
        for suffix in (".US", ".USA", "_DAILY", "-DAILY"):
            stem = stem.replace(suffix, "")
        try:
            df = reader()
        except Exception:
            return 0
        cols = [str(c).lower() for c in df.columns]
        if "ticker" in cols and df[df.columns[cols.index("ticker")]].nunique() > 1:
            tcol = df.columns[cols.index("ticker")]
            n = 0
            for tk, g in df.groupby(tcol):
                tk = str(tk).upper().replace(".US", "")
                if not tk.isalpha() or len(tk) > 5:
                    continue
                nd = norm(g.drop(columns=[tcol]), tk)
                if nd is not None:
                    nd.to_parquet(RAW / f"{tk}.parquet")
                    n += 1
            return n
        if not stem.isalpha() or len(stem) > 5:
            return 0
        out = RAW / f"{stem}.parquet"
        if out.exists():
            return 1
        nd = norm(df, stem)
        if nd is None:
            return 0
        nd.to_parquet(out)
        return 1

    def is_etf(path_str: str) -> bool:
        """Several bulk datasets ship a stocks/ folder and an etfs/ folder.
        An ETF in an equity cross-section is not a mistake you notice quickly:
        it just quietly makes the momentum signal a sector-rotation signal."""
        low = path_str.replace("\\", "/").lower()
        return "/etf" in low or low.startswith("etf")

    kept = 0
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            members = [m for m in z.namelist()
                       if m.lower().endswith((".csv", ".txt", ".parquet"))
                       and not is_etf(m)]
            print(f"reading {len(members):,} files from {src.name} ...")
            for i, m in enumerate(members):
                def rd(m=m, z=z):
                    with z.open(m) as fh:
                        return (pd.read_parquet(fh) if m.endswith(".parquet")
                                else pd.read_csv(fh))
                kept += handle(m, rd)
                if (i + 1) % 500 == 0:
                    print(f"  {i+1:,}/{len(members):,}, {kept:,} kept")
    else:
        files = [f for f in src.rglob("*")
                 if f.suffix.lower() in (".csv", ".txt", ".parquet")
                 and not is_etf(str(f.relative_to(src)))]
        print(f"reading {len(files):,} files from {src} ...")
        for i, f in enumerate(files):
            def rd(f=f):
                return (pd.read_parquet(f) if f.suffix.lower() == ".parquet"
                        else pd.read_csv(f))
            kept += handle(f.name, rd)
            if (i + 1) % 500 == 0:
                print(f"  {i+1:,}/{len(files):,}, {kept:,} kept")

    # Record exactly which tickers this ingest produced. build_panel uses this
    # list rather than "whatever parquet files happen to be in the folder", so
    # a stale file from an earlier experiment can never silently join the panel.
    produced = sorted(f.stem for f in RAW.glob("*.parquet"))
    (DATA / "provenance.json").write_text(json.dumps({
        "source": f"local ingest of {src}",
        "survivorship_free": False,
        "ingested_at": pd.Timestamp.utcnow().isoformat(),
        "n_tickers": len(produced),
        "tickers": produced,
    }, indent=2))

    print(f"ingested {kept:,} tickers with at least one year of history")
    if kept < min_tickers:
        sys.exit(f"only {kept} tickers — fewer than the {min_tickers} needed for "
                 f"a cross-sectional strategy. Check the file layout and tell me "
                 f"what the files look like.")
    return kept


# ------------------------------------------------------------------ panel ---
def build_panel() -> dict:
    """Assemble the wide price/volume panels from the per-ticker parquet cache.

    Built column by column rather than by concatenating everything into one
    long frame and pivoting. With ~6,000 tickers the long-frame route peaks at
    several gigabytes and dies on a laptop; this route holds one ticker at a
    time and a growing dict of Series.
    """
    files = sorted(RAW.glob("*.parquet"))
    if not files:
        sys.exit("no raw files found — run the download or ingest first")

    # --- provenance: only use tickers a recorded ingest actually produced ---
    prov_file = DATA / "provenance.json"
    provenance = None
    if prov_file.exists():
        import json as _json
        provenance = _json.loads(prov_file.read_text())
        allowed = set(provenance["tickers"])
        before = len(files)
        files = [f for f in files if f.stem in allowed]
        if before != len(files):
            print(f"  ignoring {before - len(files)} parquet files not part of "
                  f"the recorded ingest")

    # --- fallback guard, only when there is no provenance record ------------
    # The provenance list above is authoritative. This pattern is a safety net
    # for a folder of unknown origin, and it runs ONLY then: TST is a real
    # ticker (TheStreet Inc), so applying this to a legitimate ingest would
    # silently drop a real company.
    SYNTH = re.compile(r"^TST\d*$|^T\d{4}$|^SY\d{4}$")
    fixtures = ([f.stem for f in files if SYNTH.match(f.stem)]
                if provenance is None else [])
    if fixtures:
        sys.exit(f"refusing to build: {len(fixtures)} synthetic test tickers "
                 f"are present ({', '.join(fixtures[:6])}...). "
                 f"Start clean:\n    rm -rf data\n"
                 f"then re-run the ingest.")

    print(f"\nassembling panel from {len(files):,} tickers ...")
    price_cols, vol_cols = {}, {}
    price_name = None
    skipped = 0

    for i, f in enumerate(files):
        try:
            df = pd.read_parquet(f)
        except Exception:                                   # noqa: BLE001
            skipped += 1
            continue
        if price_name is None:
            price_name = "adjclose" if "adjclose" in df.columns else "close"
        col = price_name if price_name in df.columns else "close"
        if col not in df.columns:
            skipped += 1
            continue
        tk = f.stem
        price_cols[tk] = pd.to_numeric(df[col], errors="coerce")
        if "volume" in df.columns:
            vol_cols[tk] = pd.to_numeric(df["volume"], errors="coerce")
        if (i + 1) % 1000 == 0:
            print(f"  read {i+1:,}/{len(files):,}", flush=True)

    if not price_cols:
        sys.exit("no usable price columns found")
    print(f"  using '{price_name}' as the price series"
          + (f"  ({skipped} files skipped)" if skipped else ""))

    close = pd.concat(price_cols, axis=1).sort_index()
    close.index.name = "date"
    volume = (pd.concat(vol_cols, axis=1).reindex_like(close)
              if vol_cols else pd.DataFrame(np.nan, index=close.index,
                                            columns=close.columns))

    # Trim days with no listed names. A single stray date in one file
    # otherwise leaves an empty row at the end of the panel, which is what
    # produced "names_last_day 0" and would quietly poison the last backtest bar.
    per_day = close.notna().sum(axis=1)
    live = per_day[per_day >= 20].index
    if len(live):
        close = close.loc[live[0]:live[-1]]
        volume = volume.reindex(close.index)

    close = close.astype("float32")
    volume = volume.astype("float32")

    returns = close.pct_change(fill_method=None)
    dollar_volume = (close * volume).rolling(21, min_periods=5).median().astype("float32")

    DATA.mkdir(exist_ok=True)
    close.to_parquet(DATA / "close.parquet")
    returns.astype("float32").to_parquet(DATA / "returns.parquet")
    volume.to_parquet(DATA / "volume.parquet")
    dollar_volume.to_parquet(DATA / "dollar_volume.parquet")

    listed = close.notna()
    manifest = {
        "tickers": int(close.shape[1]),
        "days": int(close.shape[0]),
        "start": str(close.index.min().date()),
        "end": str(close.index.max().date()),
        "median_names_per_day": int(listed.sum(axis=1).median()),
        "names_first_day": int(listed.iloc[0].sum()),
        "names_last_day": int(listed.iloc[-1].sum()),
        "cells_present": int(listed.sum().sum()),
        "density": round(float(listed.sum().sum() / listed.size), 4),
        "price_column": price_name,
        "source": (provenance["source"] if provenance
                   else "sharadar" if (DATA / "tickers.csv").exists()
                   else "unknown — no provenance record"),
        "survivorship_free": (provenance["survivorship_free"] if provenance
                              else bool((DATA / "tickers.csv").exists())),
        "built": pd.Timestamp.utcnow().isoformat(),
    }
    pd.Series(manifest).to_json(DATA / "manifest.json", indent=2)
    return manifest


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None,
                    help="only fetch the first N tickers (use 50 for a smoke test)")
    ap.add_argument("--top-n", type=int, default=1500,
                    help="download only the N largest companies (default 1500). "
                         "The loader screens to the top 1500 by liquidity anyway, "
                         "so fetching 5,000 wastes requests and invites throttling.")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--source", default="both",
                    choices=["both", "yahoo", "stooq", "stooq-bulk", "sharadar", "local"],

                    help="where to get prices (default: both = yahoo, then stooq)")
    ap.add_argument("--probe", action="store_true",
                    help="test every source on three tickers and stop")
    ap.add_argument("--diagnose", nargs="?", const="AAPL", default=None,
                    metavar="TICKER",
                    help="make ONE request and print the whole response, then stop")
    ap.add_argument("--panel-only", action="store_true",
                    help="skip downloading, just rebuild the panel from ./data/raw")
    ap.add_argument("--panel", default=None, metavar="NAME",
                    help="write to data/NAME/ instead of data/. Use a separate "
                         "panel for every distinct universe — e.g. --panel "
                         "equities, --panel etfs. Mixing universes in one panel "
                         "silently invalidates every backtest built on it.")
    ap.add_argument("--input", default=None, metavar="PATH",
                    help="for --source local: a folder or .zip of per-ticker "
                         "CSV/TXT/parquet files, however you obtained them")
    ap.add_argument("--api-key", default=os.environ.get("NASDAQ_DATA_LINK_KEY"),
                    help="Nasdaq Data Link key, for --source sharadar "
                         "(or set NASDAQ_DATA_LINK_KEY)")
    ap.add_argument("--proxy", default=None,
                    help="route all requests through a local proxy, e.g. "
                         "http://127.0.0.1:15236 or socks5h://127.0.0.1:15236 "
                         "(socks needs: pip install 'requests[socks]')")
    args = ap.parse_args()
    set_panel(args.panel)
    if args.panel:
        print(f"panel: {DATA}")

    if args.proxy:
        SESSION.proxies.update({"http": args.proxy, "https": args.proxy})
        SESSION.trust_env = False        # the flag wins over any env vars
        print(f"routing through proxy: {args.proxy}")
        try:
            ip = SESSION.get("https://api.ipify.org", timeout=20).text.strip()
            print(f"outbound IP as seen by the internet: {ip}")
        except Exception as e:                       # noqa: BLE001
            sys.exit(f"proxy is not reachable: {type(e).__name__}: {e}\n"
                     f"check the port, and whether it is HTTP or SOCKS5.")

    DATA.mkdir(exist_ok=True)

    if args.diagnose:
        diagnose(args.diagnose)
        return

    if args.probe:
        probe_sources()
        return

    if not args.panel_only:
        if args.source == "local":
            if not args.input:
                sys.exit("--source local needs --input PATH")
            ingest_local(args.input)
        elif args.source == "sharadar":
            if not args.api_key:
                sys.exit("--source sharadar needs --api-key (or the "
                         "NASDAQ_DATA_LINK_KEY environment variable)")
            fetch_sharadar(args.api_key)
        elif args.source == "stooq-bulk":
            fetch_bulk()
        else:
            uni = get_universe(args.limit)
            if args.top_n and not args.limit:
                uni = rank_by_size(uni, args.top_n)
            print(f"\ndownloading daily bars for {len(uni):,} tickers "
                  f"from {args.source} ({args.workers} workers) ...")
            stats = fetch_all(uni["ticker"].tolist(), workers=args.workers,
                              source=args.source)
            print("\ndownload summary:", dict(sorted(stats.items())))
            if stats.get("ok", 0) + stats.get("cached", 0) < 100:
                sys.exit("\ntoo few tickers downloaded to build a panel. "
                         "Run `python fetch.py --probe` to see which source works.")

    m = build_panel()

    print("\n" + "=" * 66)
    print("PANEL BUILT — paste this block back to me")
    print("=" * 66)
    for k, v in m.items():
        print(f"  {k:24} {v}")
    print("=" * 66)
    size = sum(f.stat().st_size for f in DATA.rglob("*") if f.is_file())
    print(f"  {'total size on disk':24} {size/1e6:.0f} MB")
    print(f"\nSend me the four files in ./data/ "
          f"(close, returns, volume, dollar_volume + manifest.json).")
    print("You do NOT need to send ./data/raw — it is only the download cache.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
diagnose_net.py — find out what is actually broken, in one run.

    python diagnose_net.py
    python diagnose_net.py --proxy http://127.0.0.1:15236

Tests, in order of how likely they are to be the real problem:

  A. your local TLS stack        (LibreSSL 2.8.3 cannot talk to some servers)
  B. plain connectivity          (can you reach anything at all)
  C. six different price URLs    (query1 vs query2, crumb vs no crumb, stooq
                                  http vs https, and a third free source)

Prints a verdict at the end. Paste the whole output back.
"""
from __future__ import annotations

import argparse
import ssl
import sys

import requests

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

ap = argparse.ArgumentParser()
ap.add_argument("--proxy", default=None)
args = ap.parse_args()

S = requests.Session()
S.headers.update({"User-Agent": UA, "Accept": "*/*",
                  "Accept-Language": "en-US,en;q=0.9"})
if args.proxy:
    S.proxies.update({"http": args.proxy, "https": args.proxy})
    S.trust_env = False

print("=" * 72)
print("NETWORK AND TLS DIAGNOSIS")
print("=" * 72)

# ---------------------------------------------------------------- A. TLS ---
print("\n[A] LOCAL TLS STACK")
print(f"    python          : {sys.version.split()[0]}")
print(f"    ssl library     : {ssl.OPENSSL_VERSION}")
libressl = "LibreSSL" in ssl.OPENSSL_VERSION
if libressl:
    print("    *** THIS IS THE macOS SYSTEM PYTHON, LINKED AGAINST LibreSSL 2.8.3.")
    print("    *** It cannot complete TLS 1.3 handshakes with a number of hosts,")
    print("    *** which shows up as SSLError. This may be the whole problem.")
else:
    print("    OpenSSL 1.1.1+ — fine")

# ------------------------------------------------------- B. connectivity ---
print("\n[B] BASIC CONNECTIVITY")
for name, url in [("ipify (who am I)", "https://api.ipify.org"),
                  ("example.com", "https://example.com")]:
    try:
        r = S.get(url, timeout=20)
        body = r.text.strip()[:60]
        print(f"    {name:20} HTTP {r.status_code}  {body}")
    except Exception as e:                                # noqa: BLE001
        print(f"    {name:20} {type(e).__name__}: {str(e)[:70]}")

# ---------------------------------------------------------- C. the sources ---
print("\n[C] PRICE SOURCES — six variants")

def show(label, url, want="json"):
    try:
        r = S.get(url, timeout=30)
        ct = r.headers.get("content-type", "")[:28]
        head = r.text.lstrip()[:70].replace("\n", " ")
        ok = False
        if r.status_code == 200:
            if want == "json" and "json" in ct:
                try:
                    res = r.json().get("chart", {}).get("result")
                    n = len(res[0]["timestamp"]) if res else 0
                    ok = n > 200
                    head = f"{n:,} rows"
                except Exception:                          # noqa: BLE001
                    pass
            elif want == "csv" and "Date" in r.text[:200]:
                ok = r.text.count("\n") > 200
                head = f"{r.text.count(chr(10)):,} rows"
        print(f"    {'OK ' if ok else '   '} {label:34} HTTP {r.status_code:<4} {ct:<28} {head}")
        return ok
    except Exception as e:                                 # noqa: BLE001
        print(f"        {label:34} {type(e).__name__}: {str(e)[:50]}")
        return False

P = "period1=1072915200&period2=9999999999&interval=1d"
results = {}

results["yahoo q1 no crumb"] = show(
    "yahoo query1, no crumb",
    f"https://query1.finance.yahoo.com/v8/finance/chart/AAPL?{P}")

results["yahoo q2 no crumb"] = show(
    "yahoo query2, no crumb",
    f"https://query2.finance.yahoo.com/v8/finance/chart/AAPL?{P}")

# warm cookies then try again
try:
    S.get("https://finance.yahoo.com/quote/AAPL", timeout=20)
    print(f"        (warmed cookies from finance.yahoo.com: "
          f"{[c.name for c in S.cookies][:5]})")
except Exception as e:                                     # noqa: BLE001
    print(f"        (cookie warm-up failed: {type(e).__name__})")

results["yahoo q1 warmed"] = show(
    "yahoo query1, after cookie warm-up",
    f"https://query1.finance.yahoo.com/v8/finance/chart/AAPL?{P}")

results["stooq https"] = show("stooq https",
                              "https://stooq.com/q/d/l/?s=aapl.us&i=d", want="csv")
results["stooq http"] = show("stooq http (no TLS at all)",
                             "http://stooq.com/q/d/l/?s=aapl.us&i=d", want="csv")
results["stooq pl"] = show("stooq.pl mirror",
                           "https://stooq.pl/q/d/l/?s=aapl.us&i=d", want="csv")

# --------------------------------------------- what is stooq actually saying ---
print("\n[D] WHAT STOOQ IS ACTUALLY SAYING")
try:
    r = S.get("https://stooq.com/q/d/l/?s=aapl.us&i=d", timeout=30)
    body = r.text
    import re as _re
    text = _re.sub(r"<[^>]+>", " ", body)
    text = _re.sub(r"\s+", " ", text).strip()
    print(f"    page length     : {len(body):,} chars")
    print(f"    visible text    : {text[:400]}")
    for probe, meaning in [
        ("limit", "you have hit the DAILY REQUEST CAP -> it resets, try tomorrow"),
        ("Przekroczony", "Polish for 'limit exceeded' -> daily cap"),
        ("captcha", "a captcha wall -> automation is blocked"),
        ("login", "it wants an account"),
        ("robots", "it is serving the robots/consent page, not data"),
    ]:
        if probe.lower() in body.lower():
            print(f"    -> contains {probe!r}: {meaning}")
except Exception as e:                                     # noqa: BLE001
    print(f"    {type(e).__name__}: {e}")

# ---------------------------------------------------------------- verdict ---
print("\n" + "=" * 72)
working = [k for k, v in results.items() if v]
if working:
    print(f"VERDICT: these work -> {', '.join(working)}")
    print("Tell me which one and I will point fetch.py at it.")
elif libressl:
    print("VERDICT: nothing works, AND you are on the LibreSSL system Python.")
    print("Fix that first — it is the most likely single cause:")
    print("    brew install python@3.12")
    print("    cd repo && rm -rf .venv")
    print("    /opt/homebrew/bin/python3.12 -m venv .venv")
    print("    source .venv/bin/activate && pip install pandas requests pyarrow")
    print("    python diagnose_net.py --proxy http://127.0.0.1:15236")
else:
    print("VERDICT: nothing works and TLS is fine, so it is the network path.")
    print("Try a different VPN node, or a node in a different country.")
print("=" * 72)

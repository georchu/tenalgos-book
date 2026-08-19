"""Check the environment before anything else. Run this first.

    python3 preflight.py

Reports the Python version, whether every dependency loads, and whether the
data panels are present — in that order, because that is the order in which
things go wrong.

It never installs anything, never touches the network, and never raises. A
diagnostic tool that ends in a traceback has failed at its only job: the first
version of this file caught ImportError but not OSError, so a LightGBM install
missing its OpenMP runtime crashed the very script meant to explain it.
"""
from __future__ import annotations

import importlib
import pathlib
import platform
import sys

MIN = (3, 9)
RECOMMENDED = (3, 11)

# (import name, pinned version, required?)
DEPS = [
    ("numpy", "1.26.4", True), ("pandas", "2.2.2", True),
    ("pyarrow", "16.1.0", True), ("scipy", "1.13.1", True),
    ("matplotlib", "3.9.0", True), ("sklearn", "1.5.0", True),
    ("requests", "2.32.3", True),
    ("lightgbm", "4.6.0", False),        # Chapter 11 only
]

ROOT = pathlib.Path(__file__).resolve().parent
ok = True

print("=" * 68)
print("TENALGOS PREFLIGHT")
print("=" * 68)

# ---- 1. Python ------------------------------------------------------------
v = sys.version_info
print(f"  python        {platform.python_version()}  ({sys.executable})")
if v[:2] < MIN:
    ok = False
    print(f"  ERROR         needs Python >= {MIN[0]}.{MIN[1]}")
    print("                brew install python@3.12, then rebuild the venv")
elif v[:2] < RECOMMENDED:
    print(f"  note          {RECOMMENDED[0]}.{RECOMMENDED[1]}+ preferred; "
          f"{v[0]}.{v[1]} is supported and everything runs")

# ---- 2. dependencies ------------------------------------------------------
missing, broken, drift = [], [], []
for mod, want, required in DEPS:
    try:
        m = importlib.import_module(mod)
        got = getattr(m, "__version__", "?")
        if got != want:
            drift.append((mod, want, got))
    except ImportError:
        (missing if required else broken).append((mod, required, "not installed"))
    except Exception as e:                      # OSError, and anything else
        # A package that installed but cannot load — almost always a missing
        # system library rather than a Python problem.
        broken.append((mod, required, f"{type(e).__name__}: {e}"))

req_missing = [m for m, r, _ in missing if r]
if req_missing:
    ok = False
    print(f"  ERROR         required, not installed: {', '.join(req_missing)}")
    print("                pip install -r requirements.txt")
else:
    n_req = sum(1 for _, _, r in DEPS if r)
    print(f"  dependencies  all {n_req} required packages load")

for mod, required, why in broken:
    tag = "ERROR  " if required else "optional"
    print(f"  {tag}      {mod} — {why.splitlines()[0][:80]}")
    if mod == "lightgbm" and "libomp" in why:
        print("                LightGBM needs the OpenMP runtime, which macOS")
        print("                does not ship. Either:")
        print("                    brew install libomp")
        print("                or skip it — only Chapter 11 uses LightGBM, and")
        print("                Chapters 5-10 and 16 run without it.")
    elif mod == "lightgbm":
        print("                Chapter 11 only. See requirements-ml.txt.")
    if required:
        ok = False

for mod, want, got in drift:
    print(f"  note          {mod} is {got}, book used {want} "
          f"(last-decimal differences possible)")

# ---- 3. data --------------------------------------------------------------
dd = ROOT / "data"
panels = ([p for p in dd.iterdir() if p.is_dir() and (p / "close.parquet").exists()]
          if dd.exists() else [])
if panels:
    print(f"  data          {len(panels)} panel(s): "
          f"{', '.join(sorted(p.name for p in panels))}")
else:
    print("  data          none yet — expected on a fresh clone")
    print("                see 'Getting the data' in README.md")

# ---------------------------------------------------------------------------
print("-" * 68)
if not ok:
    print("  FIX THE ERRORS ABOVE FIRST")
    sys.exit(1)
print("  READY.  Next:  python3 tests/test_engine.py    -> expect 4/4")
print("                 (needs no data; run it now)")
if not panels:
    print("          Then get the data and run:")
    print("                 python3 -m tenalgos.data.integrity  -> expect 10/10")
print("=" * 68)

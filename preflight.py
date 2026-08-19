"""Check the environment before anything else. Run this first.

    python3 preflight.py

Reports the Python version, whether every dependency imports, and whether the
data panels are present — in that order, because that is the order in which
things go wrong. It never installs anything and never touches the network.
"""
from __future__ import annotations

import importlib
import pathlib
import platform
import sys

MIN = (3, 9)
RECOMMENDED = (3, 11)

DEPS = [
    ("numpy", "1.26.4"), ("pandas", "2.2.2"), ("pyarrow", "16.1.0"),
    ("scipy", "1.13.1"), ("matplotlib", "3.9.0"), ("sklearn", "1.5.0"),
    ("lightgbm", "4.6.0"), ("requests", "2.32.3"),
]

ROOT = pathlib.Path(__file__).resolve().parent
ok = True

print("=" * 66)
print("TENALGOS PREFLIGHT")
print("=" * 66)

# ---- 1. Python ------------------------------------------------------------
v = sys.version_info
print(f"  python        {platform.python_version()}  ({sys.executable})")
if v[:2] < MIN:
    ok = False
    print(f"  ERROR         needs Python >= {MIN[0]}.{MIN[1]}")
    print()
    print("  macOS ships 3.9 as `python3`, which is fine. If you are older:")
    print("    brew install python@3.12")
    print("    /opt/homebrew/bin/python3.12 -m venv .venv && source .venv/bin/activate")
elif v[:2] < RECOMMENDED:
    print(f"  note          {RECOMMENDED[0]}.{RECOMMENDED[1]}+ is recommended, "
          f"but {v[0]}.{v[1]} is supported and everything will run")

# ---- 2. dependencies ------------------------------------------------------
missing, wrong = [], []
for mod, want in DEPS:
    try:
        m = importlib.import_module(mod)
        got = getattr(m, "__version__", "?")
        if got != want:
            wrong.append((mod, want, got))
    except ImportError:
        missing.append(mod)

if missing:
    ok = False
    print(f"  ERROR         not installed: {', '.join(missing)}")
    print("                pip install -r requirements.txt")
else:
    print(f"  dependencies  all {len(DEPS)} import")
for mod, want, got in wrong:
    print(f"  note          {mod} is {got}, book used {want} "
          f"(results may differ in the last decimal)")

# ---- 3. data --------------------------------------------------------------
panels = [p for p in (ROOT / "data").iterdir()
          if p.is_dir() and (p / "close.parquet").exists()] if (ROOT / "data").exists() else []
if panels:
    print(f"  data          {len(panels)} panel(s): "
          f"{', '.join(sorted(p.name for p in panels))}")
else:
    print("  data          none yet — this is expected on a fresh clone")
    print("                see 'Getting the data' in README.md; Chapters 5-11")
    print("                need it, but tests/test_engine.py does not")

print("-" * 66)
if not ok:
    print("  FIX THE ERRORS ABOVE FIRST")
    sys.exit(1)
print("  READY.  Next:  python tests/test_engine.py     -> expect 4/4")
if not panels:
    print("          Then get the data, and:")
    print("                 python -m tenalgos.data.integrity  -> expect 10/10")
print("=" * 66)

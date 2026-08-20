# tenalgos — the code for *The Quant Billionaire*

**github.com/georchu/tenalgos-book**

Every number, table and figure in **The Quant Billionaire: Ten Algorithms and
the Firm That Earns a Billion a Year** is produced by the code in this
repository. Nothing in the book is asserted that is not reproducible here.

The book is by George Chu — paperback and Kindle,
[ISBN 979-8-19363-612-0](https://www.amazon.com/dp/B0HFRXJNR2). This repository
is the companion to it and is useless as a trading system — see the warning at
the bottom.

You do not need the book to use this code, and you do not need the code to read
the book. They are more useful together: the book explains why each choice was
made, and the repository lets you disagree with one and rerun it.

---

## Quick start

**Python 3.9 or newer.** The stock `python3` on macOS is 3.9 and works;
3.11+ is recommended. Note the `3` — on macOS and most Linux boxes there is no
bare `python` command.

```bash
git clone https://github.com/georchu/tenalgos-book.git
cd tenalgos-book
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip            # the shipped pip is often too old
pip install -r requirements.txt
python3 preflight.py                 # checks Python, imports and data
```

**One macOS note.** LightGBM installs fine but loads a system OpenMP library
Apple does not ship, so `import lightgbm` fails until you run:

```bash
brew install libomp
```

That affects exactly one script — `notebooks/ch11_figures.py`. If you skip it,
`preflight.py` reports LightGBM as optional and everything else runs normally.
You lose one sleeve out of eleven, and Chapter 11.7 concludes that sleeve is
not worth its operational cost anyway.

`preflight.py` tells you what is wrong and what to do about it, which is more
useful than a hundred lines of pip output. Then, **in this order** — the first
two commands are the whole method of the book, and if they do not pass,
nothing after them means anything:

```bash
python3 tests/test_engine.py           # 4 checks on the engine  -> 4/4
python3 -m tenalgos.data.integrity     # 10 checks on your data  -> 10/10
python3 notebooks/ch05_figures.py      # the first strategy, end to end
```

`test_engine.py` needs no data and runs immediately. The other two need the
panels — `data/` starts empty; see **Getting the data** below.

---

## What is here

| path | what it is | book chapter |
|---|---|---|
| `fetch.py` | data acquisition; `--panel NAME`, one panel per universe | 2.6 |
| `inspect_data.py` | check a downloaded dataset before ingesting it | 2.5 |
| `build_yield_panel.py` | adds unadjusted close, for the carry measure | 8.3 |
| `tenalgos/data/loaders.py` | the liquidity screen that makes a backtest honest | 2.6 |
| `tenalgos/data/integrity.py` | the ten data-integrity checks | 2.7 |
| `tenalgos/data/synthetic.py` | a market with known injected alpha, for testing | 4.2 |
| `tenalgos/research/backtest.py` | the engine; **it owns the execution lag** | 3.5 |
| `tenalgos/research/costs.py` | spread, market impact, borrow, financing | 3.6 |
| `tenalgos/research/stats.py` | performance report, deflated Sharpe ratio | 3.3, 3.8 |
| `tenalgos/research/cv.py` | purged K-fold with embargo | 3.4 |
| `tenalgos/strategies/ch05_xsmom.py` | cross-sectional equity momentum | 5 |
| `tenalgos/strategies/ch06_multifactor.py` | low risk, reversal, combination | 6 |
| `tenalgos/strategies/ch07_tsmom.py` | trend following, 35 markets | 7 |
| `tenalgos/strategies/ch08_carry.py` | carry — **a negative result** | 8 |
| `tenalgos/strategies/ch09_statarb.py` | PCA residual reversion | 9 |
| `tenalgos/strategies/ch10_volpremium.py` | the volatility risk premium | 10 |
| `tenalgos/strategies/ch11_ml.py` | gradient-boosted alpha ensemble | 11 |
| `notebooks/*.py` | every figure in the book; plain scripts, not notebooks | — |
| `tests/test_engine.py` | four checks the engine must pass | 3.4, 3.5 |
| `figures/` | the figures as printed, plus `ch0N_numbers.json` | — |

**`notebooks/` are scripts, not Jupyter notebooks.** Each runs top to bottom and
writes both its PNGs and a `ch0N_numbers.json` of every number it computed. If a
number in the printed book disagrees with the JSON, the JSON is right and the
author made a transcription error.

---

## Getting the data

`data/` is empty on purpose. The panels are about 1.8 GB and are built from a
free bulk historical dataset that you download yourself in a browser — no API
key, no vendor, no subscription. Total cost: nothing.

```bash
# one panel per universe. ALWAYS pass --panel.
python fetch.py --source local --panel equities --input ~/Downloads/archive/stocks
python fetch.py --source local --panel etfs     --input ~/Downloads/archive/etfs
python build_yield_panel.py --panel etfs        # only needed for Chapter 8
```

**Always pass `--panel`.** Without it a second ingest appends to the same raw
cache and rebuilds one panel out of two different universes, silently changing
every backtest already run against it. This happened during the writing of the
book; see Chapter 2.6.

The panel used in the book runs 3 January 2006 to 1 April 2020. A dataset
ending in 2020 is not a limitation — nothing in Part II needs last week's
prices, and a fixed historical window makes every result here reproducible.

---

## Reproducing the whole book

```bash
python -m tenalgos.data.integrity          # must be 10/10
python tests/test_engine.py                # must be 4/4
for n in 05 06 07 08 09 10 11; do python notebooks/ch${n}_figures.py; done
python notebooks/ch05_extra.py
python notebooks/ch16_portfolio.py
```

Runtime is a few minutes per chapter on an ordinary laptop, except Chapter 9's
rolling PCA (~75 seconds per configuration) and Chapter 11's gradient boosting
(~6 minutes over 2 million rows). No GPU, no cluster, no cloud.

---

## The results, so you know what you should get

All net of costs, on the panel described above:

| sleeve | net Sharpe | net return | max drawdown | capacity |
|---|---|---|---|---|
| trend, 35 markets (ch 7) | +0.514 | 5.86% | −16.6% | $1.0bn |
| low risk (ch 6) | +0.375 | 4.12% | −41.9% | $1.0bn |
| momentum (ch 5) | +0.067 | 1.01% | −54.8% | $0.3bn |
| statistical arbitrage (ch 9) | +0.023 | 0.12% | −20.4% | $0.05bn |
| ML ensemble (ch 11), walk-forward | −0.011 | — | — | — |
| carry (ch 8) | −0.486 | −3.20% | −41.6% | **do not run** |
| **combined (ch 16)** | **+0.666** | **7.00%** | **−25.5%** | **$2.35bn** |

Deflated for all 75 strategy configurations tried across the book, the combined
result has a **49.9% probability of being real**. That is the honest number and
it is the point of the book.

---

## Licence

MIT — see `LICENSE`. The code is free to use, copy and modify.

The book itself is not: its text, its figures as arranged and captioned, and
its structure are copyright © 2026 George Chu, all rights reserved. `NOTICE.md`
sets out the boundary, and explains why that note is not inside the licence
file.

---

## Warning

**This is not a trading system and it is not investment advice.**

Several strategies here lose money, and one of them loses money in every
configuration tested — deliberately, because the book is about how to find that
out before you fund something rather than after. Backtested results are
hypothetical, benefit from hindsight, and do not reflect actual trading. The
data window ends in April 2020 and several of the effects measured had already
decayed by then.

Read the Risk Disclosure at the front of the book before using any of this.

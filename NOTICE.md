# Notice — what the licence covers, and what it does not

## Scope

`LICENSE` is the MIT licence and it applies to **the source code in this
repository**: the data layer, the backtest engine, the cost model, the ten
strategy modules, the figure scripts and the tests. Use it, copy it, modify it,
sell what you build with it.

It does **not** apply to the book. The text, the figures as arranged and
captioned in the book, and the structure of *The Quant Billionaire* are
copyright © 2026 George Chu (Xingxiong Zhu), all rights reserved.

The licence file is kept as the unmodified MIT text on purpose: GitHub detects
a licence by matching the file against known templates, and appending even a
short paragraph to it is enough to break that match and leave the repository
showing no licence at all. Scope notes therefore live here rather than there.

## This is not a trading system

It is education, and it is not investment advice.

Several strategies in this repository lose money. One of them — the carry sleeve
in Chapter 8 — loses money in **every** configuration tested. That is
deliberate. The book is about finding that out before you fund something rather
than after, and a repository containing only the strategies that worked would
teach the opposite lesson.

Backtested results are hypothetical. They benefit from hindsight in ways no
amount of methodology fully removes, they do not reflect actual trading, and
the data window ends in April 2020 — by which point several of the effects
measured here had already decayed.

Deflated for all 75 strategy configurations tried across the book, the combined
portfolio's Sharpe ratio has a **49.9% probability of being real**. Read the
Risk Disclosure at the front of the book before using any of this.

## Third-party data

No market data is distributed with this repository. `data/` is empty and you
build the panels yourself from a public bulk historical dataset, under whatever
terms that dataset carries. See the README and the book's Chapter 2.

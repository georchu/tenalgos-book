"""Purged K-fold cross-validation with an embargo.

Chapter 3.4. Standard K-fold is invalid on financial panels for two reasons:

1. **Overlap.** If your label is the forward 20-day return, an observation at
   time t shares 19 days of future with the observation at t+1. Putting one in
   train and the other in test leaks the answer.
2. **Serial correlation.** Even non-overlapping observations are correlated
   across the fold boundary, so the model sees a near-copy of the test set.

The fix (López de Prado, 2018) is to *purge* training observations whose label
window overlaps the test window, and then *embargo* a further band immediately
after the test set.

`leakage_demo` in tests/ shows the size of the effect: on a market with zero
true alpha, plain K-fold reports a positive information coefficient and purged
K-fold does not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class PurgedKFold:
    """K-fold over a time index, with purging and an embargo.

    Parameters
    ----------
    n_splits
        Number of folds.
    label_horizon
        Number of periods the label looks forward. Training observations whose
        [t, t + label_horizon] window intersects the test window are purged.
    embargo
        Number of periods after the test window also removed from training.
        A common choice is about 1% of the sample.
    """

    def __init__(self, n_splits: int = 6, label_horizon: int = 21, embargo: int = 10):
        self.n_splits = n_splits
        self.label_horizon = label_horizon
        self.embargo = embargo

    def split(self, index: pd.Index):
        n = len(index)
        fold_bounds = np.linspace(0, n, self.n_splits + 1).astype(int)

        for k in range(self.n_splits):
            test_start, test_end = fold_bounds[k], fold_bounds[k + 1]
            test_idx = np.arange(test_start, test_end)

            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False

            # purge: any training obs whose label window reaches into the test set
            purge_lo = max(0, test_start - self.label_horizon)
            train_mask[purge_lo:test_start] = False

            # embargo: a band after the test set
            emb_hi = min(n, test_end + self.embargo)
            train_mask[test_end:emb_hi] = False

            yield np.where(train_mask)[0], test_idx

    def describe(self, index: pd.Index) -> pd.DataFrame:
        rows = []
        for k, (tr, te) in enumerate(self.split(index)):
            rows.append({
                "fold": k,
                "n_train": len(tr),
                "n_test": len(te),
                "test_start": str(index[te[0]])[:10],
                "test_end": str(index[te[-1]])[:10],
                "purged_and_embargoed": len(index) - len(tr) - len(te),
            })
        return pd.DataFrame(rows)


def plain_kfold(index: pd.Index, n_splits: int = 6):
    """Deliberately naive K-fold, kept so the book can measure what it costs."""
    n = len(index)
    bounds = np.linspace(0, n, n_splits + 1).astype(int)
    for k in range(n_splits):
        test_idx = np.arange(bounds[k], bounds[k + 1])
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        yield train_idx, test_idx


def walk_forward(index: pd.Index, n_splits: int = 8, min_train: float = 0.3):
    """Expanding-window walk-forward: the only scheme that matches live trading."""
    n = len(index)
    start = int(n * min_train)
    bounds = np.linspace(start, n, n_splits + 1).astype(int)
    for k in range(n_splits):
        yield np.arange(0, bounds[k]), np.arange(bounds[k], bounds[k + 1])

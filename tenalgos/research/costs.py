"""A transaction-cost model that survives contact with reality.

Chapter 3.6. Three components, because a model with fewer than three will
flatter every high-turnover strategy in the book:

    spread      you cross half the quoted spread on entry and on exit
    impact      you move the price against yourself, roughly with the square
                root of participation (Almgren et al., 2005; Kyle, 1985)
    financing   shorts cost borrow, and leverage costs the broker's rate

The default parameters are deliberately conservative for US large caps. If your
strategy only works with optimistic costs, it does not work.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


class CostModel:
    def __init__(
        self,
        spread_bps: float = 3.0,
        impact_coef: float = 0.35,
        commission_bps: float = 0.3,
        borrow_bps_annual: float = 40.0,
        financing_bps_annual: float = 100.0,
        participation_cap: float = 0.05,
    ):
        """
        Parameters
        ----------
        spread_bps
            Full quoted spread in basis points. You pay half of it per side.
            3bp is reasonable for S&P 500 names; small caps are 15-60bp.
        impact_coef
            Coefficient in  impact = coef * sigma * sqrt(participation).
            0.35 is a common empirical calibration for equities.
        borrow_bps_annual
            Cost of borrowing to short. 40bp is a general-collateral rate;
            hard-to-borrow names run 300-3000bp and Chapter 9 shows what that
            does to statistical arbitrage.
        """
        self.spread_bps = spread_bps
        self.impact_coef = impact_coef
        self.commission_bps = commission_bps
        self.borrow_bps_annual = borrow_bps_annual
        self.financing_bps_annual = financing_bps_annual
        self.participation_cap = participation_cap

    # ------------------------------------------------------------------ #
    def trade_cost(
        self,
        traded: pd.DataFrame,
        vol: pd.DataFrame | None = None,
        dollar_volume: pd.DataFrame | None = None,
        capital: float = 1e8,
    ) -> pd.Series:
        """Cost in return units per day, given |Δweight| per name.

        `traded` is absolute weight change per asset per day.
        """
        traded = traded.abs()

        # spread + commission, paid on every dollar traded
        linear = traded.sum(axis=1) * (0.5 * self.spread_bps + self.commission_bps) / 1e4

        # square-root impact
        if vol is not None and dollar_volume is not None:
            notional = traded * capital
            participation = (notional / dollar_volume.replace(0, np.nan)).clip(upper=1.0)
            impact = self.impact_coef * vol * np.sqrt(participation.fillna(0.0))
            impact_cost = (traded * impact).sum(axis=1)
        else:
            impact_cost = pd.Series(0.0, index=traded.index)

        return linear.add(impact_cost, fill_value=0.0)

    def carry_cost(self, weights: pd.DataFrame) -> pd.Series:
        """Daily borrow on the short book plus financing on gross above 1x."""
        short_notional = weights.clip(upper=0).abs().sum(axis=1)
        gross = weights.abs().sum(axis=1)
        excess_leverage = (gross - 1.0).clip(lower=0.0)
        daily = (short_notional * self.borrow_bps_annual
                 + excess_leverage * self.financing_bps_annual) / 1e4 / TRADING_DAYS
        return daily

    def capacity_limited_weights(
        self,
        weights: pd.DataFrame,
        dollar_volume: pd.DataFrame,
        capital: float,
    ) -> pd.DataFrame:
        """Cap each position so a day's trade stays under participation_cap.

        This is how a capacity curve is actually produced: raise `capital`,
        re-run, and watch the strategy's own constraints eat the return.
        """
        max_notional = dollar_volume * self.participation_cap
        max_weight = (max_notional / capital).clip(upper=1.0)
        capped = weights.clip(lower=-max_weight, upper=max_weight)
        gross = capped.abs().sum(axis=1)
        return capped.div(gross.replace(0, np.nan), axis=0).fillna(0.0)

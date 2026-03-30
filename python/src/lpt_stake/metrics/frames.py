"""Presentation helpers for the dashboard's framing tabs."""

from __future__ import annotations

import pandas as pd


def expense_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return columns relevant to the expense framing."""

    return df[
        [
            "date",
            "issuance_lpt",
            "lpt_price_usd",
            "tbc_cost_usd",
            "market_cap_usd",
            "tbc_pct_market_cap",
            "tbc_pct_fees",
        ]
    ].copy()


def dilution_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return columns relevant to the dilution framing."""

    return df[
        [
            "date",
            "issuance_lpt",
            "total_supply_lpt",
            "bonded_lpt",
            "participation_rate",
            "dilution_rate",
            "forward_inflation_rate",
        ]
    ].copy()

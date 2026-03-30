"""KPI formulas adapted from the deep research report."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lpt_stake.config import ROUNDS_PER_YEAR


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_numeric = pd.to_numeric(numerator, errors="coerce")
    denominator_numeric = pd.to_numeric(denominator, errors="coerce")
    safe_denominator = denominator_numeric.where(denominator_numeric != 0, np.nan)
    result = numerator_numeric / safe_denominator
    return result.where(np.isfinite(result), np.nan)


def compute_issuance_lpt(df: pd.DataFrame) -> pd.Series:
    """Estimate daily issuance from changes in total supply."""

    issuance = df["total_supply_lpt"].diff().fillna(0)
    return issuance.clip(lower=0)


def compute_market_cap_usd(df: pd.DataFrame) -> pd.Series:
    """Compute market cap from supply and token price."""

    return df["total_supply_lpt"] * df["lpt_price_usd"]


def compute_tbc_cost_usd(df: pd.DataFrame) -> pd.Series:
    """Compute the token-based compensation cost in USD."""

    return df["issuance_lpt"] * df["lpt_price_usd"]


def compute_dilution_rate(df: pd.DataFrame) -> pd.Series:
    """Compute realized dilution as issuance divided by post-issuance supply."""

    previous_supply = df["total_supply_lpt"].shift(1)
    denominator = previous_supply + df["issuance_lpt"]
    dilution = _safe_divide(df["issuance_lpt"], denominator)
    return dilution.fillna(0)


def compute_tbc_pct_market_cap(df: pd.DataFrame) -> pd.Series:
    """Compute TBC as a share of market cap."""

    return _safe_divide(df["tbc_cost_usd"], df["market_cap_usd"])


def compute_tbc_pct_fees(df: pd.DataFrame) -> pd.Series:
    """Compute TBC as a share of protocol fees."""

    return _safe_divide(df["tbc_cost_usd"], df["fees_usd"])


def compute_cost_per_work_unit(df: pd.DataFrame) -> pd.Series:
    """Compute cost per unit of work when throughput data exist."""

    return _safe_divide(df["tbc_cost_usd"], df["work_units"])


def compute_forward_inflation_rate(df: pd.DataFrame) -> pd.Series:
    """Annualize the protocol's per-round inflation setting."""

    per_round = df["inflation_ppb"] / 1_000_000_000
    return (1 + per_round) ** ROUNDS_PER_YEAR - 1


def build_kpi_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Append dashboard KPIs to the canonical daily dataset."""

    result = df.copy()

    if "lpt_price_usd" not in result.columns:
        result["lpt_price_usd"] = np.nan
    if "fees_usd" not in result.columns:
        result["fees_usd"] = np.nan
    if "work_units" not in result.columns:
        result["work_units"] = np.nan

    result["issuance_lpt"] = compute_issuance_lpt(result)
    result["market_cap_usd"] = compute_market_cap_usd(result)
    result["tbc_cost_usd"] = compute_tbc_cost_usd(result)
    result["dilution_rate"] = compute_dilution_rate(result)
    result["tbc_pct_market_cap"] = compute_tbc_pct_market_cap(result)
    result["tbc_pct_fees"] = compute_tbc_pct_fees(result)
    result["cost_per_work_unit"] = compute_cost_per_work_unit(result)
    result["forward_inflation_rate"] = compute_forward_inflation_rate(result)

    ordered_columns = [
        "date",
        "issuance_lpt",
        "lpt_price_usd",
        "tbc_cost_usd",
        "total_supply_lpt",
        "bonded_lpt",
        "participation_rate",
        "market_cap_usd",
        "dilution_rate",
        "fees_eth",
        "fees_usd",
        "winning_ticket_transfers",
        "tbc_pct_market_cap",
        "tbc_pct_fees",
        "work_units",
        "delivery_usage_mins",
        "storage_usage_mins",
        "cost_per_work_unit",
        "forward_inflation_rate",
        "eth_price_usd",
        "inflation_ppb",
    ]

    return result.reindex(columns=ordered_columns)

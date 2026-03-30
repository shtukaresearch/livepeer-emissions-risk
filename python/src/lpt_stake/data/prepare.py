"""Build canonical daily datasets for the internal dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from lpt_stake.config import (
    CHAIN_STATE_PATH_ENV,
    DEFAULT_CHAIN_STATE_CANDIDATES,
    DEFAULT_FEES_DATA_CANDIDATES,
    DEFAULT_MARKET_DATA_CANDIDATES,
    DEFAULT_WORK_DATA_CANDIDATES,
    FEES_DATA_PATH_ENV,
    MARKET_DATA_PATH_ENV,
    WORK_DATA_PATH_ENV,
    env_path,
)
from lpt_stake.data.io import read_csv, read_json


def _first_existing_path(
    env_override: Path | None, candidates: Iterable[Path]
) -> Path | None:
    if env_override is not None:
        return env_override

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _require_path(path: Path | None, dataset_name: str) -> Path:
    if path is None:
        raise FileNotFoundError(
            f"Could not find {dataset_name}. Set the corresponding environment "
            "variable or place the file in one of the documented default paths."
        )
    return path


def _normalize_token_units(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    median = numeric.dropna().abs().median()
    if pd.notna(median) and median > 1e12:
        return numeric / 1e18
    return numeric


def _prepare_date_column(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.normalize()
    prepared = prepared.dropna(subset=["date"]).sort_values("date")
    return prepared


def _empty_daily_frame(column_name: str) -> pd.DataFrame:
    return pd.DataFrame(columns=["date", column_name])


def load_chain_state() -> pd.DataFrame:
    """Load and normalize historic chain state."""

    path = _require_path(
        _first_existing_path(env_path(CHAIN_STATE_PATH_ENV), DEFAULT_CHAIN_STATE_CANDIDATES),
        "chain state dataset",
    )

    raw = read_json(path)
    df = _prepare_date_column(raw)

    rename_map = {
        "total-supply": "total_supply_raw",
        "bonded": "bonded_raw",
        "inflation": "inflation_ppb",
    }
    df = df.rename(columns=rename_map)

    missing = {"total_supply_raw", "bonded_raw", "inflation_ppb"} - set(df.columns)
    if missing:
        raise ValueError(
            "Chain state dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df["total_supply_lpt"] = _normalize_token_units(df["total_supply_raw"])
    df["bonded_lpt"] = _normalize_token_units(df["bonded_raw"])
    df["inflation_ppb"] = pd.to_numeric(df["inflation_ppb"], errors="coerce")
    df["participation_rate"] = df["bonded_lpt"] / df["total_supply_lpt"]

    return df[
        [
            "date",
            "inflation_ppb",
            "total_supply_lpt",
            "bonded_lpt",
            "participation_rate",
        ]
    ].drop_duplicates(subset=["date"])


def load_market_data() -> pd.DataFrame:
    """Load daily market data when available."""

    path = _first_existing_path(
        env_path(MARKET_DATA_PATH_ENV), DEFAULT_MARKET_DATA_CANDIDATES
    )
    if path is None:
        return _empty_daily_frame("lpt_price_usd")

    df = _prepare_date_column(read_csv(path))
    if "lpt_price_usd" not in df.columns:
        raise ValueError("Market dataset must include an 'lpt_price_usd' column.")

    market_columns = [
        column
        for column in [
            "date",
            "lpt_price_usd",
            "btc_price_usd",
            "eth_price_usd",
            "fear_greed_index",
        ]
        if column in df.columns
    ]
    return df[market_columns].drop_duplicates(subset=["date"])


def load_fee_data() -> pd.DataFrame:
    """Load optional protocol fee data."""

    path = _first_existing_path(env_path(FEES_DATA_PATH_ENV), DEFAULT_FEES_DATA_CANDIDATES)
    if path is None:
        return _empty_daily_frame("fees_eth")

    df = _prepare_date_column(read_csv(path))
    rename_map = {
        "fees": "fees_usd",
        "feesUSD": "fees_usd",
        "protocol_fees_usd": "fees_usd",
        "amount_eth": "fees_eth",
    }
    df = df.rename(columns=rename_map)
    if "fees_usd" not in df.columns and "fees_eth" not in df.columns:
        raise ValueError("Fees dataset must include a 'fees_usd' or 'fees_eth' column.")

    fee_columns = [
        column
        for column in ["date", "fees_usd", "fees_eth", "winning_ticket_transfers"]
        if column in df.columns
    ]
    return df[fee_columns].drop_duplicates(subset=["date"])


def load_work_data() -> pd.DataFrame:
    """Load optional work or throughput data."""

    path = _first_existing_path(env_path(WORK_DATA_PATH_ENV), DEFAULT_WORK_DATA_CANDIDATES)
    if path is None:
        return _empty_daily_frame("work_units")

    df = _prepare_date_column(read_csv(path))
    rename_map = {
        "minutes_processed": "work_units",
        "transcoding_minutes": "work_units",
        "work": "work_units",
        "TotalUsageMins": "work_units",
        "DeliveryUsageMins": "delivery_usage_mins",
        "StorageUsageMins": "storage_usage_mins",
    }
    df = df.rename(columns=rename_map)
    if "work_units" not in df.columns:
        raise ValueError("Work dataset must include a 'work_units' column.")

    work_columns = [
        column
        for column in ["date", "work_units", "delivery_usage_mins", "storage_usage_mins"]
        if column in df.columns
    ]
    return df[work_columns].drop_duplicates(subset=["date"])


def build_base_daily_dataset() -> pd.DataFrame:
    """Join raw datasets into a canonical daily metrics base table."""

    base = load_chain_state()
    merged = base.merge(load_market_data(), on="date", how="left")
    merged = merged.merge(load_fee_data(), on="date", how="left")
    merged = merged.merge(load_work_data(), on="date", how="left")

    if "fees_eth" in merged.columns and "fees_usd" not in merged.columns:
        merged["fees_usd"] = pd.to_numeric(merged["fees_eth"], errors="coerce") * pd.to_numeric(
            merged.get("eth_price_usd"), errors="coerce"
        )

    return merged.sort_values("date").reset_index(drop=True)

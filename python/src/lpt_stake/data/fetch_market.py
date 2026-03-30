"""Fetch external market context for the dashboard."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf


def _normalize_start(value: date | datetime) -> date:
    return value.date() if isinstance(value, datetime) else value


def _normalize_end(value: date | datetime) -> date:
    return value.date() if isinstance(value, datetime) else value


def fetch_market_prices(start_date: date | datetime, end_date: date | datetime) -> pd.DataFrame:
    """Fetch daily price and volume context from Yahoo Finance."""

    start = _normalize_start(start_date).strftime("%Y-%m-%d")
    end_exclusive = (_normalize_end(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    tickers = ["LPT-USD", "BTC-USD", "ETH-USD"]
    market_raw = yf.download(
        tickers,
        start=start,
        end=end_exclusive,
        interval="1d",
        progress=False,
        auto_adjust=False,
    )

    if market_raw.empty:
        return pd.DataFrame(columns=["date", "lpt_price_usd"])

    market_close = market_raw["Close"].reset_index()
    market_volume = market_raw["Volume"].reset_index()

    market_close.rename(
        columns={
            "Date": "date",
            "LPT-USD": "lpt_price_usd",
            "BTC-USD": "btc_price_usd",
            "ETH-USD": "eth_price_usd",
        },
        inplace=True,
    )
    market_volume.rename(
        columns={
            "Date": "date",
            "LPT-USD": "lpt_volume",
            "BTC-USD": "btc_volume",
            "ETH-USD": "eth_volume",
        },
        inplace=True,
    )

    market_df = pd.merge(market_close, market_volume, on="date", how="inner")
    market_df["date"] = pd.to_datetime(market_df["date"]).dt.normalize()
    return market_df.sort_values("date").reset_index(drop=True)


def fetch_fear_greed_index(
    start_date: date | datetime, end_date: date | datetime
) -> pd.DataFrame:
    """Fetch daily fear and greed values from Alternative.me."""

    response = requests.get("https://api.alternative.me/fng/?limit=0", timeout=20)
    response.raise_for_status()

    rows = response.json().get("data", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "fear_greed_index"])

    numeric_values = pd.to_numeric(df["timestamp"], errors="coerce")
    parsed_from_unix = pd.to_datetime(numeric_values, unit="s", errors="coerce", utc=True)
    parsed_from_text = pd.to_datetime(
        df["timestamp"].where(numeric_values.isna()),
        errors="coerce",
        utc=True,
    )
    df["date"] = parsed_from_unix.fillna(parsed_from_text).dt.tz_convert(None).dt.normalize()
    df = df[["date", "value"]].rename(columns={"value": "fear_greed_index"})
    df["fear_greed_index"] = pd.to_numeric(df["fear_greed_index"], errors="coerce")

    start = pd.Timestamp(_normalize_start(start_date))
    end = pd.Timestamp(_normalize_end(end_date))
    return (
        df[(df["date"] >= start) & (df["date"] <= end)]
        .sort_values("date")
        .reset_index(drop=True)
    )


def build_market_context(start_date: date | datetime, end_date: date | datetime) -> pd.DataFrame:
    """Build the merged market context table used by the dashboard."""

    market_df = fetch_market_prices(start_date, end_date)
    fear_greed_df = fetch_fear_greed_index(start_date, end_date)

    if market_df.empty:
        raise ValueError("Market download returned no rows for the requested date range.")

    if fear_greed_df.empty:
        return market_df

    final_df = market_df.merge(fear_greed_df, on="date", how="left")
    return final_df.sort_values("date").reset_index(drop=True)


def write_market_context(df: pd.DataFrame, output_path: Path) -> None:
    """Persist fetched market context to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

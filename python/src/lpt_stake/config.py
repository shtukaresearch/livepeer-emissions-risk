"""Shared configuration for local research workflows."""

from __future__ import annotations

import os
from pathlib import Path

ROUNDS_PER_YEAR = 417

PYTHON_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PYTHON_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DERIVED_DIR = DATA_DIR / "derived"
RAW_CHAIN_DIR = RAW_DIR / "chain"
RAW_ENRICHED_DIR = RAW_DIR / "enriched"
RAW_FEES_DIR = RAW_DIR / "fees"
RAW_WORK_DIR = RAW_DIR / "work"

DAILY_METRICS_PATH = DERIVED_DIR / "daily_metrics.parquet"
CHAIN_TICKS_PATH = RAW_CHAIN_DIR / "arbitrum-daily-blocks.json"
CHAIN_STATE_PATH = RAW_CHAIN_DIR / "lpt-daily-data.json"
MARKET_DATA_PATH = RAW_ENRICHED_DIR / "market_context.csv"
FEES_DATA_PATH = RAW_FEES_DIR / "fees.csv"
WORK_DATA_PATH = RAW_WORK_DIR / "work.csv"

CHAIN_STATE_PATH_ENV = "LPT_CHAIN_STATE_PATH"
MARKET_DATA_PATH_ENV = "LPT_MARKET_DATA_PATH"
FEES_DATA_PATH_ENV = "LPT_FEES_DATA_PATH"
WORK_DATA_PATH_ENV = "LPT_WORK_DATA_PATH"
LIVEPEER_STUDIO_API_KEY_ENV = "LIVEPEER_STUDIO_API_KEY"
LIVEPEER_CREATOR_ID_ENV = "LIVEPEER_CREATOR_ID"
LIVEPEER_API_BASE_URL_ENV = "LIVEPEER_API_BASE_URL"

DEFAULT_CHAIN_STATE_CANDIDATES = [
    CHAIN_STATE_PATH,
    DATA_DIR / "lpt-daily-data.json",
    DATA_DIR / "lpt-daily-data-22-24.json",
    DATA_DIR / "lpt-daily-data-22-25.json",
]

DEFAULT_MARKET_DATA_CANDIDATES = [
    MARKET_DATA_PATH,
    RAW_ENRICHED_DIR / "Data2022-2025.csv",
    DATA_DIR / "Data2022-2025.csv",
]

DEFAULT_FEES_DATA_CANDIDATES = [
    FEES_DATA_PATH,
]

DEFAULT_WORK_DATA_CANDIDATES = [
    WORK_DATA_PATH,
]


def env_path(name: str) -> Path | None:
    """Return an environment-configured path when set."""

    raw_value = os.getenv(name)
    return Path(raw_value).expanduser() if raw_value else None

"""Small IO helpers for local dashboard workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: Path) -> pd.DataFrame:
    """Read a JSON file into a dataframe."""

    with path.open() as handle:
        data: Any = json.load(handle)

    if isinstance(data, dict):
        return pd.DataFrame(data)
    if isinstance(data, list):
        return pd.DataFrame(data)
    raise ValueError(f"Unsupported JSON structure in {path}")


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file into a dataframe."""

    return pd.read_csv(path)


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file into a dataframe."""

    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a dataframe to Parquet, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a dataframe to CSV, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def merge_daily_frames(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Append new daily rows and keep the latest row for each date."""

    frames = [frame.copy() for frame in (existing, new_rows) if not frame.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "date" not in combined.columns:
        raise ValueError("Expected a 'date' column when merging daily frames.")

    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.normalize()
    combined = combined.dropna(subset=["date"])
    combined = combined.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return combined.reset_index(drop=True)

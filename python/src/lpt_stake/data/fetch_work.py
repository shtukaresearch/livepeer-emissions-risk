"""Fetch work/usage metrics from the Livepeer Studio usage API."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pytz import UTC

DEFAULT_BASE_URL = "https://livepeer.studio/api"


def _normalize_day(value: date | datetime) -> date:
    return value.date() if isinstance(value, datetime) else value


def _to_millis(day: date, exclusive: bool = False) -> int:
    ts = datetime.combine(day, time.min, tzinfo=UTC)
    if exclusive:
        ts += timedelta(days=1)
    return int(ts.timestamp() * 1000)


def _extract_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return payload["data"]
        if isinstance(payload.get("metrics"), list):
            return payload["metrics"]
        return [payload]
    raise ValueError("Unsupported usage API response shape.")


def _parse_row_date(row: dict[str, Any], fallback_start: date) -> pd.Timestamp:
    for key in ("date", "timestamp", "time", "TimeIntervalStart", "Date"):
        value = row.get(key)
        if value is None:
            continue
        parsed = pd.to_datetime(value, unit="ms", errors="coerce", utc=True)
        if pd.notna(parsed):
            return parsed.tz_convert(None).normalize()
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.notna(parsed):
            return pd.Timestamp(parsed).normalize()
    return pd.Timestamp(fallback_start)


def fetch_usage_series(
    start_date: date | datetime,
    end_date: date | datetime,
    api_key: str,
    creator_id: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> pd.DataFrame:
    """Fetch daily usage metrics from Livepeer Studio."""

    start = _normalize_day(start_date)
    end = _normalize_day(end_date)
    params: dict[str, Any] = {
        "from": _to_millis(start),
        "to": _to_millis(end, exclusive=True),
        "timeStep": "day",
    }
    if creator_id:
        params["creatorId"] = creator_id

    response = requests.get(
        f"{base_url.rstrip('/')}/data/usage/query",
        headers={"Authorization": f"Bearer {api_key}"},
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    rows = _extract_rows(response.json())
    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "work_units",
                "delivery_usage_mins",
                "storage_usage_mins",
            ]
        )

    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {
                "date": _parse_row_date(row, start).date().isoformat(),
                "work_units": row.get("TotalUsageMins", row.get("totalUsageMins")),
                "delivery_usage_mins": row.get(
                    "DeliveryUsageMins", row.get("deliveryUsageMins")
                ),
                "storage_usage_mins": row.get(
                    "StorageUsageMins", row.get("storageUsageMins")
                ),
            }
        )

    return pd.DataFrame(normalized_rows).sort_values("date").reset_index(drop=True)


def write_usage_series(df: pd.DataFrame, output_path: Path) -> None:
    """Write usage metrics to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

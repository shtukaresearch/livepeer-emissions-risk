"""Fetch daily protocol fee data from onchain ticket redemption events."""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
import requests
from pytz import UTC

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DEPLOYMENTS_DIR = REPO_ROOT / "protocol" / "deployments" / "arbitrumMainnet"
DEFAULT_TICKS_PATH = REPO_ROOT / "python" / "data" / "raw" / "chain" / "arbitrum-daily-blocks.json"
API_URL = "https://api.etherscan.io/v2/api?chainid=42161"
WINNING_TICKET_TRANSFER_TOPIC = (
    "0x8b87351a208c06e3ceee59d80725fd77a23b4129e1b51ca231fc89b40712649c"
)
FEE_CHUNK_DAYS = 14


def _normalize_day(value: date | datetime) -> date:
    return value.date() if isinstance(value, datetime) else value


def _midnight_utc(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=UTC)


def _load_ticket_broker_address(deployments_dir: Path) -> str:
    deployment_path = deployments_dir / "TicketBroker.json"
    if not deployment_path.exists():
        raise FileNotFoundError(
            "Missing TicketBroker deployment JSON. Initialize the protocol "
            "submodule or set LPT_DEPLOYMENTS_DIR to a valid deployment directory."
        )

    deployment = json.load(deployment_path.open())
    return deployment["address"]


def _load_tick_map(ticks_path: Path = DEFAULT_TICKS_PATH) -> dict[str, int]:
    if not ticks_path.exists():
        return {}

    payload = json.load(ticks_path.open())
    dates = payload.get("date", [])
    blocks = payload.get("block", [])
    return {date_value: int(block_value) for date_value, block_value in zip(dates, blocks, strict=False)}


def get_block_number_by_time(apikey: str, timestamp: datetime, closest: str) -> int:
    response = requests.get(
        API_URL,
        params={
            "module": "block",
            "action": "getblocknobytime",
            "timestamp": int(timestamp.timestamp()),
            "closest": closest,
            "apikey": apikey,
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "1":
        raise ValueError(f"Etherscan block lookup failed: {payload.get('message')}")
    return int(payload["result"])


def _fetch_logs_page(
    apikey: str,
    address: str,
    from_block: int,
    to_block: int,
    page: int,
    offset: int = 1000,
) -> list[dict]:
    response = requests.get(
        API_URL,
        params={
            "module": "logs",
            "action": "getLogs",
            "fromBlock": from_block,
            "toBlock": to_block,
            "address": address,
            "topic0": WINNING_TICKET_TRANSFER_TOPIC,
            "page": page,
            "offset": offset,
            "apikey": apikey,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") == "0" and "No records found" in str(payload.get("message")):
        return []
    if not isinstance(payload.get("result"), list):
        raise ValueError(f"Etherscan log lookup failed: {payload}")
    return payload["result"]


def _fetch_logs(apikey: str, address: str, from_block: int, to_block: int) -> list[dict]:
    logs: list[dict] = []
    page = 1
    while True:
        page_rows = _fetch_logs_page(
            apikey=apikey,
            address=address,
            from_block=from_block,
            to_block=to_block,
            page=page,
        )
        if not page_rows:
            break
        logs.extend(page_rows)
        if len(page_rows) < 1000:
            break
        page += 1
    return logs


def _block_for_day(
    tick_map: dict[str, int],
    apikey: str,
    day: date,
    closest: str,
) -> int:
    cached = tick_map.get(day.isoformat())
    if cached is not None:
        return cached
    return get_block_number_by_time(apikey, _midnight_utc(day), closest)


def build_fee_series(
    start_date: date | datetime,
    end_date: date | datetime,
    apikey: str,
    deployments_dir: Path | None = None,
) -> pd.DataFrame:
    """Build a daily series of protocol fees from ticket redemption events."""

    start = _normalize_day(start_date)
    end = _normalize_day(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    deployments = deployments_dir or DEFAULT_DEPLOYMENTS_DIR
    address = _load_ticket_broker_address(deployments)
    tick_map = _load_tick_map()

    all_days = pd.date_range(start=start, end=end, freq="D")
    logs: list[dict] = []

    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=FEE_CHUNK_DAYS - 1), end)
        print(f"Fetching fee chunk {chunk_start.isoformat()} -> {chunk_end.isoformat()}")
        from_block = _block_for_day(tick_map, apikey, chunk_start, "after")
        to_block = _block_for_day(tick_map, apikey, chunk_end + timedelta(days=1), "before")

        if to_block >= from_block:
            logs.extend(
                _fetch_logs(
                    apikey=apikey,
                    address=address,
                    from_block=from_block,
                    to_block=to_block,
                )
            )

        chunk_start = chunk_end + timedelta(days=1)

    if not logs:
        return pd.DataFrame(
            {
                "date": [day.date().isoformat() for day in all_days],
                "fees_eth": 0.0,
                "winning_ticket_transfers": 0,
            }
        )

    events = pd.DataFrame(
        {
            "date": [
                datetime.fromtimestamp(int(log["timeStamp"], 16), tz=UTC).date().isoformat()
                for log in logs
            ],
            "fees_eth": [int(log["data"], 16) / 1e18 for log in logs],
        }
    )

    grouped = (
        events.groupby("date", as_index=False)
        .agg(
            fees_eth=("fees_eth", "sum"),
            winning_ticket_transfers=("fees_eth", "size"),
        )
        .sort_values("date")
    )

    full_range = pd.DataFrame({"date": [day.date().isoformat() for day in all_days]})
    result = full_range.merge(grouped, on="date", how="left").fillna(
        {"fees_eth": 0.0, "winning_ticket_transfers": 0}
    )
    result["winning_ticket_transfers"] = result["winning_ticket_transfers"].astype(int)
    return result


def write_fee_series(df: pd.DataFrame, output_path: Path) -> None:
    """Write the fee series to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

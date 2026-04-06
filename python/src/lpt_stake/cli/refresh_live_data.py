"""Incrementally refresh raw datasets and rebuild the dashboard metrics."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from lpt_stake.cli.build_metrics import main as build_metrics_main
from lpt_stake.config import (
    CHAIN_STATE_PATH,
    CHAIN_STATE_PATH_ENV,
    FEES_DATA_PATH,
    FEES_DATA_PATH_ENV,
    LIVEPEER_API_BASE_URL_ENV,
    LIVEPEER_CREATOR_ID_ENV,
    LIVEPEER_STUDIO_API_KEY_ENV,
    MARKET_DATA_PATH,
    MARKET_DATA_PATH_ENV,
    WORK_DATA_PATH,
    WORK_DATA_PATH_ENV,
    env_path,
)
from lpt_stake.data.fetch_fees import build_fee_series
from lpt_stake.data.fetch_market import build_market_context
from lpt_stake.data.fetch_work import fetch_usage_series
from lpt_stake.data.io import merge_daily_frames, read_csv, read_json, write_csv


def _date_arg(value: str) -> date:
    return date.fromisoformat(value)


def _default_end_date() -> date:
    return datetime.now(UTC).date() - timedelta(days=1)


def _load_daily_dates(path: Path) -> pd.Series:
    if path.suffix == ".json":
        df = read_json(path)
    else:
        df = read_csv(path)

    if "date" not in df.columns:
        raise ValueError(f"{path} does not contain a 'date' column.")
    return pd.to_datetime(df["date"], errors="coerce").dropna().dt.normalize().sort_values()


def _existing_bounds(path: Path) -> tuple[date, date] | None:
    if not path.exists():
        return None

    dates = _load_daily_dates(path)
    if dates.empty:
        return None
    return dates.iloc[0].date(), dates.iloc[-1].date()


def _next_missing_day(path: Path) -> date | None:
    bounds = _existing_bounds(path)
    if bounds is None:
        return None
    _, end_date = bounds
    return end_date + timedelta(days=1)


def _merge_and_write(path: Path, new_rows: pd.DataFrame) -> int:
    existing = read_csv(path) if path.exists() else pd.DataFrame()
    merged = merge_daily_frames(existing, new_rows)
    write_csv(merged, path)
    return len(merged)


def _run_chain_refresh(state_path: Path, target_end_date: date, bootstrap_start: date | None) -> None:
    script_path = Path(__file__).resolve().parents[3] / "script" / "fetch-data.py"
    bounds = _existing_bounds(state_path)

    if bounds is None:
        if bootstrap_start is None:
            raise ValueError(
                "Chain state file is missing. Provide --start-date to bootstrap the incremental refresher."
            )
        start_date = bootstrap_start
        cmd = [
            sys.executable,
            str(script_path),
            "--ticks",
            "--state",
            "--start-date",
            start_date.isoformat(),
            "--end-date",
            (target_end_date + timedelta(days=1)).isoformat(),
        ]
        print(f"Bootstrapping chain state {start_date} -> {target_end_date}")
    else:
        existing_start, existing_end = bounds
        missing_start = existing_end + timedelta(days=1)
        if missing_start > target_end_date:
            print(f"Chain state already covers through {existing_end}")
            return

        cmd = [
            sys.executable,
            str(script_path),
            "--extend",
            str(state_path),
            "--ticks",
            "--state",
            "--start-date",
            existing_start.isoformat(),
            "--end-date",
            (target_end_date + timedelta(days=1)).isoformat(),
        ]
        print(f"Appending chain state {missing_start} -> {target_end_date}")

    subprocess.run(cmd, check=True)


def _refresh_market(target_end_date: date, bootstrap_start: date | None) -> None:
    output_path = env_path(MARKET_DATA_PATH_ENV) or MARKET_DATA_PATH
    missing_start = _next_missing_day(output_path)

    if missing_start is None:
        if bootstrap_start is None:
            raise ValueError(
                "Market context file is missing. Provide --start-date to bootstrap the incremental refresher."
            )
        missing_start = bootstrap_start

    if missing_start > target_end_date:
        print(f"Market data already covers through {target_end_date}")
        return

    print(f"Appending market data {missing_start} -> {target_end_date}")
    row_count = _merge_and_write(output_path, build_market_context(missing_start, target_end_date))
    print(f"Market dataset now has {row_count} rows")


def _refresh_fees(target_end_date: date, bootstrap_start: date | None) -> None:
    output_path = env_path(FEES_DATA_PATH_ENV) or FEES_DATA_PATH
    missing_start = _next_missing_day(output_path)

    if missing_start is None:
        if bootstrap_start is None:
            raise ValueError(
                "Fee dataset is missing. Provide --start-date to bootstrap the incremental refresher."
            )
        missing_start = bootstrap_start

    if missing_start > target_end_date:
        print(f"Fee data already covers through {target_end_date}")
        return

    api_key = os.getenv("ETHERSCAN_API_KEY")
    if not api_key:
        raise ValueError("ETHERSCAN_API_KEY must be set to refresh fee data.")

    print(f"Appending fee data {missing_start} -> {target_end_date}")
    row_count = _merge_and_write(
        output_path,
        build_fee_series(
            start_date=missing_start,
            end_date=target_end_date,
            apikey=api_key,
        ),
    )
    print(f"Fee dataset now has {row_count} rows")


def _refresh_work(target_end_date: date, bootstrap_start: date | None) -> None:
    output_path = env_path(WORK_DATA_PATH_ENV) or WORK_DATA_PATH
    api_key = os.getenv(LIVEPEER_STUDIO_API_KEY_ENV)
    if not api_key:
        print("Skipping work data refresh because LIVEPEER_STUDIO_API_KEY is not set")
        return

    missing_start = _next_missing_day(output_path)
    if missing_start is None:
        if bootstrap_start is None:
            raise ValueError(
                "Work dataset is missing. Provide --start-date to bootstrap the incremental refresher."
            )
        missing_start = bootstrap_start

    if missing_start > target_end_date:
        print(f"Work data already covers through {target_end_date}")
        return

    print(f"Appending work data {missing_start} -> {target_end_date}")
    row_count = _merge_and_write(
        output_path,
        fetch_usage_series(
            start_date=missing_start,
            end_date=target_end_date,
            api_key=api_key,
            creator_id=os.getenv(LIVEPEER_CREATOR_ID_ENV),
            base_url=os.getenv(LIVEPEER_API_BASE_URL_ENV, "https://livepeer.studio/api"),
        ),
    )
    print(f"Work dataset now has {row_count} rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh only the missing daily rows for the internal dashboard datasets."
    )
    parser.add_argument(
        "--start-date",
        type=_date_arg,
        help="Bootstrap start date when a raw dataset does not exist yet.",
    )
    parser.add_argument(
        "--end-date",
        type=_date_arg,
        default=_default_end_date(),
        help="Inclusive UTC end date to refresh through. Defaults to the last closed UTC day.",
    )
    parser.add_argument("--skip-chain", action="store_true", help="Skip chain-state refresh.")
    parser.add_argument("--skip-market", action="store_true", help="Skip market refresh.")
    parser.add_argument("--skip-fees", action="store_true", help="Skip fee refresh.")
    parser.add_argument("--skip-work", action="store_true", help="Skip work refresh.")
    parser.add_argument("--skip-build", action="store_true", help="Skip rebuilding derived metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_path = env_path(CHAIN_STATE_PATH_ENV) or CHAIN_STATE_PATH

    try:
        if not args.skip_chain:
            _run_chain_refresh(state_path, args.end_date, args.start_date)
        if not args.skip_market:
            _refresh_market(args.end_date, args.start_date)
        if not args.skip_fees:
            _refresh_fees(args.end_date, args.start_date)
        if not args.skip_work:
            _refresh_work(args.end_date, args.start_date)
        if not args.skip_build:
            build_metrics_main()
    except (
        FileNotFoundError,
        ValueError,
        requests.RequestException,
        subprocess.CalledProcessError,
    ) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()

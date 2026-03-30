"""CLI entrypoint for fetching Livepeer usage/work metrics."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from lpt_stake.config import (
    LIVEPEER_API_BASE_URL_ENV,
    LIVEPEER_CREATOR_ID_ENV,
    LIVEPEER_STUDIO_API_KEY_ENV,
    RAW_DIR,
)
from lpt_stake.data.fetch_work import fetch_usage_series, write_usage_series

DEFAULT_OUTPUT_PATH = RAW_DIR / "work" / "work.csv"


def _date_arg(value: str) -> date:
    return date.fromisoformat(value)


def _infer_dates_from_state(path: Path) -> tuple[date, date]:
    with path.open() as handle:
        raw = json.load(handle)

    if isinstance(raw, dict):
        raw_dates = raw.get("date", [])
    else:
        raw_dates = [row.get("date") for row in raw]

    dates = pd.to_datetime(pd.Series(raw_dates), errors="coerce").dropna().sort_values()
    if dates.empty:
        raise ValueError("Could not infer dates from state dataset.")
    return dates.iloc[0].date(), dates.iloc[-1].date()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch daily work metrics from the Livepeer Studio usage API."
    )
    parser.add_argument("--start-date", type=_date_arg, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=_date_arg, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--state-path", type=Path, help="Infer date range from a state JSON file.")
    parser.add_argument("--creator-id", type=str, help="Optional creatorId filter.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination CSV path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    args = parser.parse_args()

    using_dates = args.start_date is not None and args.end_date is not None
    if not using_dates and args.state_path is None:
        parser.error("Provide either --start-date/--end-date or --state-path.")
    if (args.start_date is None) ^ (args.end_date is None):
        parser.error("Provide both --start-date and --end-date together.")
    return args


def main() -> None:
    args = parse_args()
    api_key = os.getenv(LIVEPEER_STUDIO_API_KEY_ENV)
    if not api_key:
        raise SystemExit(
            "LIVEPEER_STUDIO_API_KEY must be set to fetch work metrics from Livepeer Studio."
        )

    try:
        if args.state_path is not None:
            start_date, end_date = _infer_dates_from_state(args.state_path)
        else:
            start_date, end_date = args.start_date, args.end_date

        df = fetch_usage_series(
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
            creator_id=args.creator_id or os.getenv(LIVEPEER_CREATOR_ID_ENV),
            base_url=os.getenv(LIVEPEER_API_BASE_URL_ENV, "https://livepeer.studio/api"),
        )
        write_usage_series(df, args.output_path)
    except (requests.RequestException, ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Wrote {len(df)} rows to {args.output_path}")

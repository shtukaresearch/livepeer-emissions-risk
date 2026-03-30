"""CLI entrypoint for building derived dashboard metrics."""

from __future__ import annotations

import sys

from lpt_stake.config import DAILY_METRICS_PATH
from lpt_stake.data.io import write_parquet
from lpt_stake.data.prepare import build_base_daily_dataset
from lpt_stake.metrics.kpis import build_kpi_dataset


def main() -> None:
    try:
        base = build_base_daily_dataset()
        metrics = build_kpi_dataset(base)
        write_parquet(metrics, DAILY_METRICS_PATH)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Wrote {len(metrics)} rows to {DAILY_METRICS_PATH}")

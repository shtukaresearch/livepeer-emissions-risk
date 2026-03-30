# lpt-stake

Internal dashboard and supporting Python code for Livepeer emissions risk
analysis.

## Quick Start

The branch includes bundled raw and derived data, so you can open the dashboard
without fetching anything.

```bash
uv sync
uv run marimo run app/internal_dashboard.py
```

If you want to rebuild the derived table from the bundled raw files first:

```bash
uv sync
python script/build-metrics.py
uv run marimo run app/internal_dashboard.py
```

## Included Data

The bundled dashboard dataset includes:

- `data/raw/chain/arbitrum-daily-blocks.json`
- `data/raw/chain/lpt-daily-data.json`
- `data/raw/enriched/market_context.csv`
- `data/raw/fees/fees.csv`
- `data/derived/daily_metrics.parquet`

## Dashboard Files

- `app/internal_dashboard.py`: Marimo app entrypoint
- `app/pages/`: page layout and dashboard copy
- `src/lpt_stake/`: reusable data loading, KPI, and chart helpers
- `script/build-metrics.py`: rebuild the canonical derived table

## Refreshing Data

Only refresh if you want newer data than what shipped in the branch.

From the `python/` directory:

```bash
python script/refresh-live-data.py
```

That command appends only missing days to the bundled datasets and rebuilds
`data/derived/daily_metrics.parquet`. It requires the appropriate API
credentials in your environment.

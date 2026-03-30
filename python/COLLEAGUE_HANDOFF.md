# Colleague Handoff

This dashboard bundle already includes the fetched raw and derived data, so a
first run does **not** require any fetching or API keys.

## Included data

- `data/raw/chain/arbitrum-daily-blocks.json`
- `data/raw/chain/lpt-daily-data.json`
- `data/raw/enriched/market_context.csv`
- `data/raw/fees/fees.csv`
- `data/derived/daily_metrics.parquet`

## First run

From the `python/` directory:

```bash
uv sync
uv run marimo run app/internal_dashboard.py
```

## Optional rebuild from bundled raw data

If you want to regenerate the derived table locally before opening the
dashboard:

```bash
uv sync
python script/build-metrics.py
uv run marimo run app/internal_dashboard.py
```

## Only if newer data are needed

Fetching is not required for the bundled handoff. Refresh only if you want data
newer than what shipped in this branch:

```bash
python script/refresh-live-data.py
```

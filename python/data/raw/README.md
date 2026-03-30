# Raw Data

Use this directory for locally managed inputs that should not be hard-coded into
the dashboard app.

Recommended layout:

- `chain/lpt-daily-data.json`
- `chain/arbitrum-daily-blocks.json`
- `enriched/market_context.csv`
- `fees/fees.csv`
- `work/work.csv`

Sources:

- `chain/*` is produced by `python/script/fetch-data.py`
- `enriched/market_context.csv` is produced by `uv run lpt-refresh-market-data`
- `fees/fees.csv` is produced by `uv run lpt-refresh-fee-data`
- `work/work.csv` is produced by `uv run lpt-refresh-work-data` and requires
  `LIVEPEER_STUDIO_API_KEY`

The metrics build step can run with just the chain state dataset. Fees and work
inputs are optional in the current MVP scaffold.

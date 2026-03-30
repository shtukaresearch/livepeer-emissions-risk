# lpt-stake

Python libraries, notebooks, and an internal dashboard scaffold for Livepeer
emissions risk analysis.

## Layout

- `src/lpt_stake/` contains reusable data loading, KPI, and chart helpers.
- `script/build-metrics.py` builds the canonical dashboard dataset.
- `app/internal_dashboard.py` is the internal Marimo dashboard entrypoint.
- `data/raw/` is for locally managed raw inputs.
- `data/derived/` is for generated dashboard-ready tables.

## Expected inputs

The dashboard expects a real chain state dataset and can optionally join market,
fee, and work datasets.

- Chain state: `data/raw/chain/lpt-daily-data.json`
- Market data: `data/raw/enriched/market_context.csv`
- Fees: `data/raw/fees/fees.csv`
- Work: `data/raw/work/work.csv`

Environment variables can override those defaults:

- `LPT_CHAIN_STATE_PATH`
- `LPT_MARKET_DATA_PATH`
- `LPT_FEES_DATA_PATH`
- `LPT_WORK_DATA_PATH`

## Commands

From the `python/` directory:

```bash
uv sync
python script/fetch-data.py --ticks --state --start-date 2025-01-01 --end-date 2025-02-01
uv run lpt-refresh-market-data --state-path data/raw/chain/lpt-daily-data.json
uv run lpt-refresh-fee-data --state-path data/raw/chain/lpt-daily-data.json
# optional, requires LIVEPEER_STUDIO_API_KEY
uv run lpt-refresh-work-data --state-path data/raw/chain/lpt-daily-data.json
uv run lpt-build-metrics
uv run marimo run app/internal_dashboard.py
```

Once the historical backfill exists, the normal daily update path is:

```bash
uv run lpt-refresh-live-data
```

That command refreshes only the missing UTC days in the raw chain, market, fee,
and optional work datasets, then rebuilds `data/derived/daily_metrics.parquet`.
By default it refreshes through the last fully closed UTC day, so it avoids
partial-current-day gaps in fee-based metrics.

## Quick Start From Bundled Data

If the repo already includes the fetched raw data and
`data/derived/daily_metrics.parquet`, you do not need to fetch anything to open
the dashboard.

From the `python/` directory:

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

Only run the refresh/fetch commands if you want newer data than what shipped in
the branch.

## Daily scheduler

For a local macOS machine, the repo now includes a tiny `launchd` wrapper.

1. Create `python/.env.refresh` with the credentials the refresher needs:

```bash
ETHERSCAN_API_KEY=...
ARB_RPC_URL=...
# optional
LIVEPEER_STUDIO_API_KEY=...
LIVEPEER_CREATOR_ID=...
```

2. Generate the plist:

```bash
cd python
uv run lpt-generate-launchd-plist --hour 6 --minute 15
```

This writes a repo-local plist at `python/ops/org.livepeer.emissions-risk.refresh.plist`
and a runner script at `python/script/run-daily-refresh.sh` will source
`python/.env.refresh`, run `script/refresh-live-data.py` through the project
virtualenv when available, and append logs to
`python/data/derived/logs/daily-refresh.log`.

3. Install it into your user LaunchAgents:

```bash
cp ops/org.livepeer.emissions-risk.refresh.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/org.livepeer.emissions-risk.refresh.plist
```

To stop it later:

```bash
launchctl unload ~/Library/LaunchAgents/org.livepeer.emissions-risk.refresh.plist
```

Notes:

- `script/fetch-data.py` requires `ETHERSCAN_API_KEY`, `ARB_RPC_URL`, and the
  `protocol/` submodule or an equivalent `LPT_DEPLOYMENTS_DIR`.
- `lpt-refresh-market-data` pulls market prices from Yahoo Finance and sentiment
  data from Alternative.me.
- `lpt-refresh-fee-data` derives protocol fees from `WinningTicketTransfer`
  events emitted by the deployed TicketBroker proxy and converts them into USD
  using the ETH price series.
- `lpt-refresh-work-data` uses the official Livepeer Studio usage endpoint and
  requires `LIVEPEER_STUDIO_API_KEY`. `LIVEPEER_CREATOR_ID` is optional.
- `lpt-refresh-live-data` is the recommended command for a live internal
  dashboard after the initial backfill. It appends only new days instead of
  fetching the full historical range again.
- `lpt-generate-launchd-plist` creates a macOS scheduler config that points at
  the repo-local refresh runner.
- If work inputs are not available yet, the build step will still succeed and
  leave those columns empty in the derived dataset.

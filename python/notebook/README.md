# Data Collection

This project enriches Livepeer state data with two off-chain data sources.

Market data are retrieved from Yahoo Finance using `yfinance`. The script downloads
daily close prices and daily trading volumes for `LPT-USD`, `BTC-USD`, and `ETH-USD`.
These are then labeled as price and volume columns for each asset.

Fear & Greed data are retrieved from the Alternative.me API. The response is parsed
into daily records and stored as a single sentiment field, `fear_greed_index`.

Both datasets are aligned by date with the existing Livepeer dataset so they can be
used together in downstream analysis.

# Data Preparation
After the external data are collected, data-fetching.py does two main kinds of manipulation:

First, it standardizes and merges the daily datasets. The Yahoo Finance data are split into close-price and volume tables, their columns are renamed into clearer names like lpt_price_usd and btc_volume, and dates are normalized into plain daily values. The Fear & Greed response is also cleaned into a simple two-column daily table with date and fear_greed_index. After that, both off-chain datasets are merged with the existing Livepeer state data on date, producing one combined daily dataset that is saved as Data2022-2025.csv.

Second, it derives a round-based dataset from that daily table. It copies the merged daily data, interpolates missing numeric values, adds a sequential daily index, and maps those daily observations onto synthetic Livepeer round positions using a 24 / 21 rounds-per-day assumption. Then it selects the nearest daily values for each round position, builds a round-indexed table, and saves that as Data2022-2025[perRound].csv
# Data Collection

This project enriches Livepeer state data with two off-chain data sources.

Market data are retrieved from Yahoo Finance using `yfinance`. The script downloads
daily close prices and daily trading volumes for `LPT-USD`, `BTC-USD`, and `ETH-USD`.
These are then labeled as price and volume columns for each asset.

Fear & Greed data are retrieved from the Alternative.me API. The response is parsed
into daily records and stored as a single sentiment field, `fear_greed_index`.

Both datasets are aligned by date with the existing Livepeer dataset so they can be
used together in downstream analysis.

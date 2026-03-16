"""Data loading and daily-to-round reindexing.

Handles two concerns:

1. **Ingestion** — loading raw daily data from the JSON output of
   ``script/fetch-data.py`` (on-chain) and from exogenous CSV sources
   (prices, volumes, fear & greed — stubs for now).

2. **Reindexing** — converting daily-resolution data to a contiguous
   round-indexed DataFrame using the estimated time conversion from
   ``time.py``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from lpt_stake.time import estimate_round_number

_REQUIRED_ONCHAIN_COLUMNS = {"date", "block", "inflation", "total-supply", "bonded"}


# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------


def load_onchain_daily(path: str | Path) -> pl.DataFrame:
    """Load daily on-chain data from the JSON output of ``fetch-data.py``.

    The JSON is expected to have parallel-list structure with at least the
    keys: ``date``, ``block``, ``inflation``, ``total-supply``, ``bonded``.

    Parameters
    ----------
    path
        Path to the JSON file.

    Returns
    -------
    pl.DataFrame
        DataFrame with ``date`` as ``pl.Date`` and numeric columns cast to
        appropriate types.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r") as f:
        raw = json.load(f)

    missing = _REQUIRED_ONCHAIN_COLUMNS - set(raw.keys())
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = pl.DataFrame(raw)

    return df.with_columns(
        pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d"),
        pl.col("total-supply").cast(pl.Float64),
        pl.col("bonded").cast(pl.Float64),
    )


def load_exogenous_daily(path: str | Path) -> pl.DataFrame:
    """Load daily exogenous data (prices, volumes, fear & greed).

    Not yet implemented — awaiting documentation of the exogenous data
    pipeline.

    Raises
    ------
    NotImplementedError
        Always.
    """
    raise NotImplementedError("Exogenous data loading not yet implemented.")


def merge_daily(
    onchain: pl.DataFrame, exogenous: pl.DataFrame
) -> pl.DataFrame:
    """Merge on-chain and exogenous daily DataFrames on date.

    Not yet implemented — awaiting documentation of the exogenous data
    pipeline.

    Raises
    ------
    NotImplementedError
        Always.
    """
    raise NotImplementedError("Daily merge not yet implemented.")


# ---------------------------------------------------------------------------
# Daily-to-round reindexing
# ---------------------------------------------------------------------------


def reindex_daily_to_rounds(daily_df: pl.DataFrame) -> pl.DataFrame:
    """Convert a date-indexed daily DataFrame to a contiguous round index.

    Each date is assigned to the round containing midnight UTC on that date.
    The output has one row per round from the minimum to maximum assigned
    round, with gaps forward-filled from the most recent observation.

    The ``date`` column is preserved: rows corresponding to actual
    observations carry their original date; forward-filled rows have
    ``null`` in the date column.

    Parameters
    ----------
    daily_df
        DataFrame with a ``date`` column of type ``pl.Date``.

    Returns
    -------
    pl.DataFrame
        Round-indexed DataFrame with a ``round`` column, sorted by round.
    """
    # Assign each date to a round
    dates = daily_df["date"].to_list()
    rounds = [
        estimate_round_number(
            datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        )
        for d in dates
    ]

    with_round = daily_df.with_columns(pl.Series("round", rounds))

    # If multiple dates map to the same round, keep the last (most recent)
    with_round = with_round.unique(subset=["round"], keep="last")

    # Build a complete round index from min to max
    min_round = with_round["round"].min()
    max_round = with_round["round"].max()
    all_rounds = pl.DataFrame(
        {"round": list(range(min_round, max_round + 1))}
    )

    # Join and forward-fill gaps
    # The date column should be null for forward-filled rows, so we exclude
    # it from the forward fill.
    data_cols = [c for c in with_round.columns if c not in ("round", "date")]

    result = (
        all_rounds.join(with_round, on="round", how="left")
        .sort("round")
        .with_columns(pl.col(c).forward_fill() for c in data_cols)
    )

    return result

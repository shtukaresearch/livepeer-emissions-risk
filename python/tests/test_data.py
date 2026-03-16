"""Tests for lpt_stake.data.

Testing strategy for reindex_daily_to_rounds
---------------------------------------------
Daily data has one row per calendar date.  Rounds are ~21.3 hours, so
consecutive calendar dates (midnight-to-midnight = 24 h) always advance by
at least one round.  A gap of N > 1 calendar days always spans more than
one round, which guarantees forward-fill is needed for intermediate rounds.

Tests derive expected round numbers from the input dates using
``_date_to_round`` (a thin wrapper around ``estimate_round_number``), so
they remain correct if the reference point changes.  No test hard-codes a
round number.
"""

import json
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from lpt_stake.data import (
    load_exogenous_daily,
    load_onchain_daily,
    merge_daily,
    reindex_daily_to_rounds,
)
from lpt_stake.time import estimate_round_number


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _date_to_round(d: date) -> int:
    """What round does midnight UTC on this date fall in?"""
    return estimate_round_number(datetime(d.year, d.month, d.day, tzinfo=timezone.utc))


def _make_daily_df(dates: list[date]) -> pl.DataFrame:
    """Build a daily DataFrame from explicit calendar dates."""
    return pl.DataFrame(
        {
            "date": dates,
            "inflation": [200_000 - i * 500 for i in range(len(dates))],
            "total-supply": [3e25] * len(dates),
            "bonded": [1.5e25] * len(dates),
        }
    )


# ---------------------------------------------------------------------------
# load_onchain_daily
# ---------------------------------------------------------------------------


@pytest.fixture()
def onchain_json(tmp_path: Path) -> Path:
    """Write a minimal valid on-chain JSON fixture and return its path."""
    data = {
        "date": ["2025-12-28", "2025-12-29", "2025-12-30"],
        "block": [414_800_000, 415_100_000, 415_400_000],
        "inflation": [200_000, 200_000, 199_500],
        "total-supply": [
            "30000000000000000000000000",
            "30000100000000000000000000",
            "30000200000000000000000000",
        ],
        "bonded": [
            "15000000000000000000000000",
            "15000050000000000000000000",
            "15000100000000000000000000",
        ],
    }
    path = tmp_path / "lpt-daily-data.json"
    path.write_text(json.dumps(data))
    return path


class TestLoadOnchainDaily:
    """Test loading on-chain daily JSON from fetch-data.py output."""

    def test_returns_dataframe(self, onchain_json):
        result = load_onchain_daily(onchain_json)
        assert isinstance(result, pl.DataFrame)

    def test_expected_columns(self, onchain_json):
        result = load_onchain_daily(onchain_json)
        for col in ("date", "block", "inflation", "total-supply", "bonded"):
            assert col in result.columns

    def test_date_column_is_date_type(self, onchain_json):
        result = load_onchain_daily(onchain_json)
        assert result["date"].dtype == pl.Date

    def test_numeric_columns(self, onchain_json):
        result = load_onchain_daily(onchain_json)
        assert result["total-supply"].dtype in (pl.Float64, pl.Int64, pl.UInt64)
        assert result["bonded"].dtype in (pl.Float64, pl.Int64, pl.UInt64)
        assert result["inflation"].dtype in (pl.Float64, pl.Int64, pl.UInt64)

    def test_row_count(self, onchain_json):
        result = load_onchain_daily(onchain_json)
        assert len(result) == 3

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_onchain_daily(tmp_path / "nonexistent.json")

    def test_missing_column_raises(self, tmp_path):
        data = {"date": ["2025-12-28"], "block": [100]}
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="[Mm]issing"):
            load_onchain_daily(path)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class TestStubs:
    """Stub functions raise NotImplementedError."""

    def test_load_exogenous_raises(self, tmp_path):
        with pytest.raises(NotImplementedError):
            load_exogenous_daily(tmp_path / "exog.csv")

    def test_merge_daily_raises(self):
        dummy = pl.DataFrame({"x": [1]})
        with pytest.raises(NotImplementedError):
            merge_daily(dummy, dummy)


# ---------------------------------------------------------------------------
# reindex_daily_to_rounds
# ---------------------------------------------------------------------------


class TestReindexDailyToRounds:
    """Test daily-to-round reindexing."""

    def test_output_has_round_column(self):
        df = _make_daily_df([date(2026, 1, 1), date(2026, 1, 2)])
        result = reindex_daily_to_rounds(df)
        assert "round" in result.columns
        assert result["round"].dtype in (pl.Int32, pl.Int64)

    def test_date_column_preserved(self):
        df = _make_daily_df([date(2026, 1, 1), date(2026, 1, 2)])
        result = reindex_daily_to_rounds(df)
        assert "date" in result.columns

    def test_data_columns_carried_through(self):
        df = _make_daily_df([date(2026, 1, 1), date(2026, 1, 2)])
        result = reindex_daily_to_rounds(df)
        for col in ("inflation", "total-supply", "bonded"):
            assert col in result.columns

    def test_round_index_is_contiguous(self):
        """Output should have a row for every round from min to max, with
        gaps filled by forward-fill."""
        dates = [date(2026, 1, 1), date(2026, 1, 10)]
        df = _make_daily_df(dates)
        result = reindex_daily_to_rounds(df)

        rounds = result["round"].to_list()
        expected = list(range(min(rounds), max(rounds) + 1))
        assert rounds == expected

    def test_row_count_equals_round_span(self):
        dates = [date(2026, 1, 1), date(2026, 1, 10)]
        df = _make_daily_df(dates)
        result = reindex_daily_to_rounds(df)

        rounds = result["round"]
        expected_count = rounds.max() - rounds.min() + 1
        assert len(result) == expected_count

    def test_forward_fill_values(self):
        """Rounds between two observations should carry the earlier
        observation's values."""
        dates = [date(2026, 1, 1), date(2026, 1, 10)]
        df = _make_daily_df(dates)
        result = reindex_daily_to_rounds(df)

        first_round = _date_to_round(dates[0])
        last_round = _date_to_round(dates[1])
        first_inflation = df["inflation"][0]

        for r in range(first_round + 1, last_round):
            row = result.filter(pl.col("round") == r)
            assert len(row) == 1
            assert row["inflation"][0] == first_inflation

    def test_forward_filled_rows_have_null_date(self):
        """Rows created by forward-fill should have null in the date column."""
        dates = [date(2026, 1, 1), date(2026, 1, 10)]
        df = _make_daily_df(dates)
        result = reindex_daily_to_rounds(df)

        first_round = _date_to_round(dates[0])
        last_round = _date_to_round(dates[1])

        for r in range(first_round + 1, last_round):
            row = result.filter(pl.col("round") == r)
            assert row["date"][0] is None

    def test_observed_rows_retain_original_dates(self):
        """Rows that correspond to actual observations should carry their
        original date values."""
        dates = [date(2026, 1, 1), date(2026, 1, 5), date(2026, 1, 10)]
        df = _make_daily_df(dates)
        original_dates = set(dates)

        result = reindex_daily_to_rounds(df)
        output_dates = set(
            result.filter(pl.col("date").is_not_null())["date"].to_list()
        )
        assert original_dates == output_dates

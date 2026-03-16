"""Tests for lpt_stake.time."""

from datetime import datetime, timedelta, timezone

import pytest

from lpt_stake.constants import (
    REFERENCE_DATETIME,
    REFERENCE_ROUND,
    SECONDS_PER_ROUND,
)
from lpt_stake.time import (
    estimate_datetime,
    estimate_round_count,
    estimate_round_number,
    estimate_timedelta,
)


class TestEstimateRoundNumber:
    """Test estimate_round_number (datetime → round)."""

    def test_reference_point_returns_reference_round(self):
        """The reference datetime must map exactly to the reference round."""
        assert estimate_round_number(REFERENCE_DATETIME) == REFERENCE_ROUND

    def test_returns_int(self):
        assert isinstance(estimate_round_number(REFERENCE_DATETIME), int)

    def test_one_round_later(self):
        dt = REFERENCE_DATETIME + timedelta(seconds=SECONDS_PER_ROUND)
        assert estimate_round_number(dt) == REFERENCE_ROUND + 1

    def test_one_round_earlier(self):
        dt = REFERENCE_DATETIME - timedelta(seconds=SECONDS_PER_ROUND)
        assert estimate_round_number(dt) == REFERENCE_ROUND - 1

    def test_floors_like_contract(self):
        """A datetime halfway through a round should still return that round
        (integer division floors)."""
        dt = REFERENCE_DATETIME + timedelta(seconds=SECONDS_PER_ROUND / 2)
        assert estimate_round_number(dt) == REFERENCE_ROUND

    def test_just_before_next_round(self):
        dt = REFERENCE_DATETIME + timedelta(seconds=SECONDS_PER_ROUND - 1)
        assert estimate_round_number(dt) == REFERENCE_ROUND

    def test_exactly_at_next_round(self):
        dt = REFERENCE_DATETIME + timedelta(seconds=SECONDS_PER_ROUND)
        assert estimate_round_number(dt) == REFERENCE_ROUND + 1

    def test_naive_datetime_raises(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            estimate_round_number(datetime(2026, 1, 1))

    def test_non_utc_timezone_works(self):
        """A non-UTC timezone should still give a correct result."""
        eastern = timezone(timedelta(hours=-5))
        dt_eastern = REFERENCE_DATETIME.astimezone(eastern)
        assert estimate_round_number(dt_eastern) == REFERENCE_ROUND


class TestEstimateDatetime:
    """Test estimate_datetime (round → datetime)."""

    def test_reference_round_returns_reference_datetime(self):
        """The reference round must map exactly to the reference datetime."""
        assert estimate_datetime(REFERENCE_ROUND) == REFERENCE_DATETIME

    def test_result_is_utc(self):
        dt = estimate_datetime(REFERENCE_ROUND + 100)
        assert dt.tzinfo is timezone.utc

    def test_one_round_later(self):
        dt = estimate_datetime(REFERENCE_ROUND + 1)
        expected = REFERENCE_DATETIME + timedelta(seconds=SECONDS_PER_ROUND)
        assert dt == expected

    def test_fractional_round(self):
        dt = estimate_datetime(REFERENCE_ROUND + 0.5)
        expected = REFERENCE_DATETIME + timedelta(seconds=SECONDS_PER_ROUND / 2)
        assert dt == expected


class TestRoundTrip:
    """Verify that estimate_datetime → estimate_round_number recovers the
    round (for integer inputs), and that estimate_round_number →
    estimate_datetime recovers the start of that round."""

    @pytest.mark.parametrize("round_num", [0, 1000, 4048, 5000, 10000])
    def test_round_to_datetime_to_round(self, round_num):
        """estimate_datetime of an int round, fed back into
        estimate_round_number, should return the same int."""
        dt = estimate_datetime(round_num)
        recovered = estimate_round_number(dt)
        assert recovered == round_num

    @pytest.mark.parametrize(
        "dt",
        [
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 6, 15, 12, 30, 0, tzinfo=timezone.utc),
            datetime(2027, 6, 15, 12, 30, 0, tzinfo=timezone.utc),
        ],
    )
    def test_datetime_to_round_to_datetime_gives_round_start(self, dt):
        """Going datetime → round → datetime should give the *start* of that
        round, which is ≤ the original datetime and within one round of it."""
        r = estimate_round_number(dt)
        round_start = estimate_datetime(r)
        assert round_start <= dt
        assert (dt - round_start).total_seconds() < SECONDS_PER_ROUND


class TestEstimateRoundCount:
    """Test estimate_round_count (timedelta → rounds)."""

    def test_zero_delta(self):
        assert estimate_round_count(timedelta(0)) == 0.0

    def test_one_round_duration(self):
        delta = timedelta(seconds=SECONDS_PER_ROUND)
        assert estimate_round_count(delta) == pytest.approx(1.0)

    def test_one_year(self):
        delta = timedelta(days=365.25)
        from lpt_stake.constants import ROUNDS_PER_YEAR

        assert estimate_round_count(delta) == pytest.approx(ROUNDS_PER_YEAR)

    def test_negative_delta(self):
        delta = timedelta(seconds=-SECONDS_PER_ROUND)
        assert estimate_round_count(delta) == pytest.approx(-1.0)


class TestEstimateTimedelta:
    """Test estimate_timedelta (rounds → timedelta)."""

    def test_zero_rounds(self):
        assert estimate_timedelta(0) == timedelta(0)

    def test_one_round(self):
        td = estimate_timedelta(1)
        assert td.total_seconds() == pytest.approx(SECONDS_PER_ROUND)

    def test_negative_rounds(self):
        td = estimate_timedelta(-1)
        assert td.total_seconds() == pytest.approx(-SECONDS_PER_ROUND)


class TestTimedeltaRoundTrip:
    """Verify that estimate_round_count and estimate_timedelta are inverses."""

    @pytest.mark.parametrize("rounds", [0, 1, 10.5, 412.4, -100])
    def test_rounds_to_timedelta_to_rounds(self, rounds):
        td = estimate_timedelta(rounds)
        recovered = estimate_round_count(td)
        assert recovered == pytest.approx(rounds, abs=1e-9)

    @pytest.mark.parametrize(
        "delta",
        [
            timedelta(0),
            timedelta(days=1),
            timedelta(days=365),
            timedelta(hours=12, minutes=30),
        ],
    )
    def test_timedelta_to_rounds_to_timedelta(self, delta):
        r = estimate_round_count(delta)
        recovered = estimate_timedelta(r)
        assert abs((recovered - delta).total_seconds()) < 0.001

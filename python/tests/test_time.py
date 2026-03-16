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
    """Test estimate_round_number (datetime → round number)."""

    def test_reference_point_returns_reference_round(self):
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
        """Halfway through a round still returns that round number."""
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
        eastern = timezone(timedelta(hours=-5))
        dt_eastern = REFERENCE_DATETIME.astimezone(eastern)
        assert estimate_round_number(dt_eastern) == REFERENCE_ROUND


class TestEstimateDatetime:
    """Test estimate_datetime (round number → datetime)."""

    def test_reference_round_returns_reference_datetime(self):
        assert estimate_datetime(REFERENCE_ROUND) == REFERENCE_DATETIME

    def test_result_is_utc(self):
        dt = estimate_datetime(REFERENCE_ROUND + 100)
        assert dt.tzinfo is timezone.utc

    def test_one_round_later(self):
        dt = estimate_datetime(REFERENCE_ROUND + 1)
        expected = REFERENCE_DATETIME + timedelta(seconds=SECONDS_PER_ROUND)
        assert dt == expected


class TestEstimateRoundCount:
    """Test estimate_round_count (timedelta → int round count)."""

    def test_returns_int(self):
        assert isinstance(estimate_round_count(timedelta(days=1)), int)

    def test_zero_delta(self):
        assert estimate_round_count(timedelta(0)) == 0

    def test_exactly_one_round(self):
        assert estimate_round_count(timedelta(seconds=SECONDS_PER_ROUND)) == 1

    def test_floors_incomplete_round(self):
        """A duration of 1.5 rounds should count as 1 complete round."""
        delta = timedelta(seconds=int(SECONDS_PER_ROUND * 1.5))
        assert estimate_round_count(delta) == 1

    def test_just_under_one_round(self):
        delta = timedelta(seconds=SECONDS_PER_ROUND - 1)
        assert estimate_round_count(delta) == 0

    def test_negative_delta(self):
        delta = timedelta(seconds=-SECONDS_PER_ROUND)
        assert estimate_round_count(delta) == -1

    def test_one_year(self):
        delta = timedelta(days=365)
        count = estimate_round_count(delta)
        assert isinstance(count, int)
        assert 410 < count < 415


class TestEstimateTimedelta:
    """Test estimate_timedelta (int round count → timedelta)."""

    def test_zero_rounds(self):
        assert estimate_timedelta(0) == timedelta(0)

    def test_one_round(self):
        td = estimate_timedelta(1)
        assert td.total_seconds() == SECONDS_PER_ROUND

    def test_negative_rounds(self):
        td = estimate_timedelta(-1)
        assert td.total_seconds() == -SECONDS_PER_ROUND


class TestRoundNumberRoundTrip:
    """Test lossiness of round_number ↔ datetime composition."""

    @pytest.mark.parametrize("round_num", [0, 1000, 4048, 5000, 10000])
    def test_round_to_datetime_to_round_is_exact(self, round_num):
        """Starting from an integer round, going to datetime and back should
        be lossless because estimate_datetime lands exactly on a round
        boundary."""
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
    def test_datetime_to_round_to_datetime_loses_sub_round(self, dt):
        """Starting from an arbitrary datetime, going to round number and back
        loses the sub-round offset.  The recovered datetime should be the
        start of the round containing *dt* — i.e. ≤ dt and within one round
        of it."""
        r = estimate_round_number(dt)
        round_start = estimate_datetime(r)
        assert round_start <= dt
        delta = (dt - round_start).total_seconds()
        assert 0 <= delta < SECONDS_PER_ROUND


class TestRoundCountRoundTrip:
    """Test lossiness of round_count ↔ timedelta composition."""

    @pytest.mark.parametrize("rounds", [0, 1, 100, 412])
    def test_rounds_to_timedelta_to_rounds_is_exact(self, rounds):
        """Starting from an integer round count, going to timedelta and back
        should be lossless because estimate_timedelta produces an exact
        multiple of SECONDS_PER_ROUND."""
        td = estimate_timedelta(rounds)
        recovered = estimate_round_count(td)
        assert recovered == rounds

    @pytest.mark.parametrize(
        "delta",
        [
            timedelta(days=1),
            timedelta(days=365),
            timedelta(hours=12, minutes=30),
        ],
    )
    def test_timedelta_to_rounds_to_timedelta_loses_sub_round(self, delta):
        """Starting from an arbitrary timedelta, going to round count and
        back loses the sub-round remainder.  The recovered timedelta should
        be ≤ the original and differ by less than one round."""
        r = estimate_round_count(delta)
        recovered = estimate_timedelta(r)
        assert recovered <= delta
        lost = (delta - recovered).total_seconds()
        assert 0 <= lost < SECONDS_PER_ROUND

    def test_zero_is_lossless(self):
        assert estimate_timedelta(estimate_round_count(timedelta(0))) == timedelta(0)

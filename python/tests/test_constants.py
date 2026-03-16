"""Tests for lpt_stake.constants."""

from datetime import timezone

from lpt_stake.constants import (
    ETH_SLOTS_PER_ROUND,
    REFERENCE_DATETIME,
    REFERENCE_ROUND,
    ROUNDS_PER_YEAR,
    SECONDS_PER_ETH_SLOT,
    SECONDS_PER_ROUND,
)


class TestRoundGeometry:
    """Verify that the round geometry constants are self-consistent and match
    known protocol values."""

    def test_seconds_per_round_is_product(self):
        assert SECONDS_PER_ROUND == SECONDS_PER_ETH_SLOT * ETH_SLOTS_PER_ROUND

    def test_eth_slot_is_12s(self):
        assert SECONDS_PER_ETH_SLOT == 12

    def test_slots_per_round_is_lip83_value(self):
        """LIP-83 set roundLength to 6377 on 2022-10-06."""
        assert ETH_SLOTS_PER_ROUND == 6377

    def test_seconds_per_round_value(self):
        assert SECONDS_PER_ROUND == 76_524

    def test_rounds_per_year_approximate(self):
        """Rounds per year should be roughly 412–413."""
        assert 412 < ROUNDS_PER_YEAR < 413

    def test_rounds_per_year_consistent_with_seconds(self):
        julian_year_seconds = 365.25 * 86_400
        expected = julian_year_seconds / SECONDS_PER_ROUND
        assert ROUNDS_PER_YEAR == expected

    def test_round_duration_is_roughly_21_hours(self):
        hours = SECONDS_PER_ROUND / 3600
        assert 21.0 < hours < 21.5


class TestReferencePoint:
    """Verify the reference point is well-formed."""

    def test_reference_round_is_positive(self):
        assert REFERENCE_ROUND > 0

    def test_reference_datetime_is_utc(self):
        assert REFERENCE_DATETIME.tzinfo is timezone.utc

    def test_reference_datetime_is_2026(self):
        assert REFERENCE_DATETIME.year == 2026
        assert REFERENCE_DATETIME.month == 1
        assert REFERENCE_DATETIME.day == 1

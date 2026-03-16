"""Tests for lpt_stake.emissions_schedule."""

import numpy as np
import pytest

from lpt_stake.emissions_schedule import ClampedSignedStepSchedule, SignedStepSchedule


# ---------------------------------------------------------------------------
# SignedStepSchedule (no floor/ceiling — current protocol status quo)
# ---------------------------------------------------------------------------


class TestSignedStepSchedule:
    """Test the unclamped signed-step emissions schedule."""

    @pytest.fixture()
    def schedule(self):
        return SignedStepSchedule(
            target_participation_rate=0.5,
            emissions_change=500.0,
        )

    def test_below_target_increases_emissions(self, schedule):
        """Participation below target → positive step → emissions increase."""
        emissions = np.array([100_000.0])
        participation = np.array([0.3])
        result = schedule.update(emissions, participation)
        assert result[0] == pytest.approx(100_500.0)

    def test_above_target_decreases_emissions(self, schedule):
        """Participation above target → negative step → emissions decrease."""
        emissions = np.array([100_000.0])
        participation = np.array([0.7])
        result = schedule.update(emissions, participation)
        assert result[0] == pytest.approx(99_500.0)

    def test_at_target_no_change(self, schedule):
        """Participation exactly at target → np.sign(0) == 0 → no change."""
        emissions = np.array([100_000.0])
        participation = np.array([0.5])
        result = schedule.update(emissions, participation)
        assert result[0] == pytest.approx(100_000.0)

    def test_vectorised_mixed_paths(self, schedule):
        """Array of paths with mixed above/below/at-target states."""
        emissions = np.array([100_000.0, 200_000.0, 150_000.0])
        participation = np.array([0.3, 0.7, 0.5])
        result = schedule.update(emissions, participation)
        expected = np.array([100_500.0, 199_500.0, 150_000.0])
        np.testing.assert_allclose(result, expected)

    def test_does_not_mutate_input(self, schedule):
        """update() returns a new array, doesn't modify the input."""
        emissions = np.array([100_000.0])
        participation = np.array([0.3])
        original = emissions.copy()
        schedule.update(emissions, participation)
        np.testing.assert_array_equal(emissions, original)

    def test_emissions_can_go_negative(self, schedule):
        """Without clamping, emissions can cross zero."""
        emissions = np.array([200.0])
        participation = np.array([0.7])
        result = schedule.update(emissions, participation)
        assert result[0] == pytest.approx(-300.0)

    def test_multi_step_bounded_drift(self, schedule):
        """After n steps, emissions has moved at most n * emissions_change
        from its starting value, regardless of the participation path."""
        rng = np.random.default_rng(42)
        n_steps = 100
        n_paths = 50
        emissions = np.full(n_paths, 100_000.0)
        initial = emissions.copy()

        for _ in range(n_steps):
            participation = rng.uniform(0.0, 1.0, size=n_paths)
            emissions = schedule.update(emissions, participation)

        max_drift = n_steps * schedule.emissions_change
        np.testing.assert_array_less(
            np.abs(emissions - initial), max_drift + 1e-10
        )


# ---------------------------------------------------------------------------
# ClampedSignedStepSchedule (with floor/ceiling)
# ---------------------------------------------------------------------------


class TestClampedSignedStepSchedule:
    """Test the clamped signed-step emissions schedule."""

    @pytest.fixture()
    def schedule(self):
        return ClampedSignedStepSchedule(
            target_participation_rate=0.5,
            emissions_change=500.0,
            emissions_floor=50_000.0,
            emissions_ceiling=500_000.0,
        )

    def test_below_target_increases_emissions(self, schedule):
        emissions = np.array([100_000.0])
        participation = np.array([0.3])
        result = schedule.update(emissions, participation)
        assert result[0] == pytest.approx(100_500.0)

    def test_above_target_decreases_emissions(self, schedule):
        emissions = np.array([200_000.0])
        participation = np.array([0.7])
        result = schedule.update(emissions, participation)
        assert result[0] == pytest.approx(199_500.0)

    def test_at_target_no_change(self, schedule):
        emissions = np.array([100_000.0])
        participation = np.array([0.5])
        result = schedule.update(emissions, participation)
        assert result[0] == pytest.approx(100_000.0)

    def test_clamps_at_floor(self, schedule):
        """Emissions at floor with negative step → stays at floor."""
        emissions = np.array([50_000.0])
        participation = np.array([0.7])
        result = schedule.update(emissions, participation)
        assert result[0] == pytest.approx(50_000.0)

    def test_clamps_at_ceiling(self, schedule):
        """Emissions at ceiling with positive step → stays at ceiling."""
        emissions = np.array([500_000.0])
        participation = np.array([0.3])
        result = schedule.update(emissions, participation)
        assert result[0] == pytest.approx(500_000.0)

    def test_step_past_floor_clamps(self, schedule):
        """Step would push below floor → clamped to floor."""
        emissions = np.array([50_200.0])
        participation = np.array([0.7])
        result = schedule.update(emissions, participation)
        # 50_200 - 500 = 49_700, clamped to 50_000
        assert result[0] == pytest.approx(50_000.0)

    def test_vectorised_mixed_paths(self, schedule):
        emissions = np.array([100_000.0, 50_000.0, 500_000.0, 250_000.0])
        participation = np.array([0.3, 0.7, 0.3, 0.5])
        result = schedule.update(emissions, participation)
        expected = np.array([100_500.0, 50_000.0, 500_000.0, 250_000.0])
        np.testing.assert_allclose(result, expected)

    def test_multi_step_stays_within_bounds(self, schedule):
        """Over many steps with random participation, emissions never
        leaves [floor, ceiling]."""
        rng = np.random.default_rng(42)
        n_steps = 200
        n_paths = 50
        emissions = np.full(n_paths, 200_000.0)

        for _ in range(n_steps):
            participation = rng.uniform(0.0, 1.0, size=n_paths)
            emissions = schedule.update(emissions, participation)
            assert np.all(emissions >= schedule.emissions_floor - 1e-10)
            assert np.all(emissions <= schedule.emissions_ceiling + 1e-10)

    def test_floor_must_be_below_ceiling(self):
        with pytest.raises(ValueError, match="floor.*ceiling"):
            ClampedSignedStepSchedule(
                target_participation_rate=0.5,
                emissions_change=500.0,
                emissions_floor=500_000.0,
                emissions_ceiling=50_000.0,
            )

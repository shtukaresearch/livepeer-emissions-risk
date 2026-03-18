"""Tests for lpt_stake.util."""

from __future__ import annotations

import numpy as np
import pytest

from lpt_stake.util import compute_total_supply_paths


class TestComputeTotalSupplyPaths:
    """Total supply grows by (1 + emissions_rate / 1e9) each round."""

    def test_initial_supply_preserved(self):
        emissions = np.full((3, 11), 200_000.0)
        result = compute_total_supply_paths(1000.0, emissions)
        np.testing.assert_array_equal(result[:, 0], 1000.0)

    def test_shape(self):
        emissions = np.full((5, 21), 100_000.0)
        result = compute_total_supply_paths(1.0, emissions)
        assert result.shape == (5, 21)

    def test_constant_emissions_hand_computed(self):
        """With constant emissions rate, supply follows geometric growth."""
        rate_ppb = 500_000.0  # 0.05%
        horizon = 10
        emissions = np.full((1, horizon + 1), rate_ppb)
        initial = 1_000_000.0
        result = compute_total_supply_paths(initial, emissions)

        for t in range(horizon + 1):
            expected = initial * (1.0 + rate_ppb / 1e9) ** t
            assert result[0, t] == pytest.approx(expected, rel=1e-10)

    def test_zero_emissions(self):
        """Zero emissions rate means supply is constant."""
        emissions = np.zeros((2, 6))
        result = compute_total_supply_paths(42.0, emissions)
        np.testing.assert_allclose(result, 42.0)

    def test_varying_emissions(self):
        """Each step uses its own emissions rate."""
        # rates: [100, 200, 300] ppb
        emissions = np.array([[100.0, 200.0, 300.0]])
        result = compute_total_supply_paths(1.0, emissions)

        assert result[0, 0] == pytest.approx(1.0)
        assert result[0, 1] == pytest.approx(1.0 * (1 + 100.0 / 1e9))
        assert result[0, 2] == pytest.approx(
            1.0 * (1 + 100.0 / 1e9) * (1 + 200.0 / 1e9)
        )

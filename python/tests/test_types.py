"""Tests for lpt_stake.types."""

import numpy as np
import pytest

from lpt_stake.types import SimulationResult, SimulationState


class TestSimulationState:
    """Validate SimulationState construction and constraints."""

    def test_valid_construction(self):
        s = SimulationState(participation_rate=0.5, emissions_rate_per_round=200_000)
        assert s.participation_rate == 0.5
        assert s.emissions_rate_per_round == 200_000

    def test_boundary_participation_zero(self):
        s = SimulationState(participation_rate=0.0, emissions_rate_per_round=100)
        assert s.participation_rate == 0.0

    def test_boundary_participation_one(self):
        s = SimulationState(participation_rate=1.0, emissions_rate_per_round=100)
        assert s.participation_rate == 1.0

    def test_participation_below_zero_raises(self):
        with pytest.raises(ValueError, match="participation_rate"):
            SimulationState(participation_rate=-0.01, emissions_rate_per_round=100)

    def test_participation_above_one_raises(self):
        with pytest.raises(ValueError, match="participation_rate"):
            SimulationState(participation_rate=1.01, emissions_rate_per_round=100)

    def test_negative_emissions_raises(self):
        with pytest.raises(ValueError, match="emissions_rate_per_round"):
            SimulationState(participation_rate=0.5, emissions_rate_per_round=-1)

    def test_zero_emissions_allowed(self):
        s = SimulationState(participation_rate=0.5, emissions_rate_per_round=0)
        assert s.emissions_rate_per_round == 0

    def test_frozen(self):
        s = SimulationState(participation_rate=0.5, emissions_rate_per_round=100)
        with pytest.raises(AttributeError):
            s.participation_rate = 0.6


class TestSimulationResult:
    """Validate SimulationResult construction and constraints."""

    def test_valid_construction(self):
        p = np.ones((10, 51))
        e = np.ones((10, 51))
        r = SimulationResult(
            participation_rate_paths=p,
            emissions_rate_per_round_paths=e,
        )
        assert r.participation_rate_paths.shape == (10, 51)

    def test_shape_mismatch_raises(self):
        p = np.ones((10, 51))
        e = np.ones((10, 50))
        with pytest.raises(ValueError, match="same shape"):
            SimulationResult(
                participation_rate_paths=p,
                emissions_rate_per_round_paths=e,
            )

    def test_1d_array_raises(self):
        p = np.ones(10)
        e = np.ones(10)
        with pytest.raises(ValueError, match="2-dimensional"):
            SimulationResult(
                participation_rate_paths=p,
                emissions_rate_per_round_paths=e,
            )

    def test_3d_array_raises(self):
        p = np.ones((10, 51, 2))
        e = np.ones((10, 51, 2))
        with pytest.raises(ValueError, match="2-dimensional"):
            SimulationResult(
                participation_rate_paths=p,
                emissions_rate_per_round_paths=e,
            )

    def test_frozen(self):
        p = np.ones((10, 51))
        e = np.ones((10, 51))
        r = SimulationResult(
            participation_rate_paths=p,
            emissions_rate_per_round_paths=e,
        )
        with pytest.raises(AttributeError):
            r.participation_rate_paths = np.ones((5, 51))

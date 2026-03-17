"""Tests for noise model implementations in lpt_stake.simulation."""

import numpy as np
import pytest

from lpt_stake.simulation import BootstrapNoise, GaussianNoise


# ---------------------------------------------------------------------------
# GaussianNoise
# ---------------------------------------------------------------------------


class TestGaussianNoiseShape:
    """Output shape must be (n,)."""

    def test_shape(self):
        noise = GaussianNoise(sigma=1.0)
        result = noise(100, rng=np.random.default_rng(0))
        assert result.shape == (100,)

    def test_single_sample(self):
        noise = GaussianNoise(sigma=1.0)
        result = noise(1, rng=np.random.default_rng(0))
        assert result.shape == (1,)


class TestGaussianNoiseReproducibility:
    """Fixed seed must produce identical output."""

    def test_reproducible(self):
        noise = GaussianNoise(sigma=0.5)
        a = noise(50, rng=np.random.default_rng(42))
        b = noise(50, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)


class TestGaussianNoiseStatistics:
    """Large sample should have mean ~0 and std ~sigma."""

    def test_mean_and_std(self):
        sigma = 2.0
        noise = GaussianNoise(sigma=sigma)
        samples = noise(100_000, rng=np.random.default_rng(42))
        assert np.mean(samples) == pytest.approx(0.0, abs=0.05)
        assert np.std(samples) == pytest.approx(sigma, abs=0.05)

    def test_zero_sigma(self):
        noise = GaussianNoise(sigma=0.0)
        samples = noise(100, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(samples, 0.0)


# ---------------------------------------------------------------------------
# BootstrapNoise
# ---------------------------------------------------------------------------


class TestBootstrapNoiseShape:
    """Output shape must be (n,)."""

    def test_shape(self):
        residuals = np.random.default_rng(0).normal(0, 1, size=200)
        noise = BootstrapNoise(residuals=residuals, block_size=10)
        result = noise(50, rng=np.random.default_rng(1))
        assert result.shape == (50,)

    def test_larger_than_residuals(self):
        """Can request more samples than the length of residuals."""
        residuals = np.random.default_rng(0).normal(0, 1, size=20)
        noise = BootstrapNoise(residuals=residuals, block_size=5)
        result = noise(100, rng=np.random.default_rng(1))
        assert result.shape == (100,)

    def test_n_not_multiple_of_block_size(self):
        residuals = np.random.default_rng(0).normal(0, 1, size=50)
        noise = BootstrapNoise(residuals=residuals, block_size=7)
        result = noise(20, rng=np.random.default_rng(1))
        assert result.shape == (20,)


class TestBootstrapNoiseValues:
    """All sampled values must come from the residuals array."""

    def test_values_in_residuals(self):
        residuals = np.array([0.1, -0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8, -0.9, 1.0])
        noise = BootstrapNoise(residuals=residuals, block_size=3)
        samples = noise(100, rng=np.random.default_rng(42))
        residual_set = set(residuals.tolist())
        for v in samples:
            assert v in residual_set


class TestBootstrapNoiseReproducibility:
    """Fixed seed must produce identical output."""

    def test_reproducible(self):
        residuals = np.random.default_rng(0).normal(0, 1, size=200)
        noise = BootstrapNoise(residuals=residuals, block_size=10)
        a = noise(50, rng=np.random.default_rng(42))
        b = noise(50, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)

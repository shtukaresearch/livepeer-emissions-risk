"""Tests for lpt_stake.exogenous."""

import numpy as np
import pytest

from lpt_stake.exogenous import AR1Sampler, BootstrapSampler


# ---------------------------------------------------------------------------
# BootstrapSampler
# ---------------------------------------------------------------------------


class TestBootstrapSamplerShape:
    """Output shape must be (n_paths, horizon, n_features)."""

    def test_single_feature(self):
        hist = np.random.default_rng(0).random((100, 1))
        sampler = BootstrapSampler(block_size=10)
        result = sampler.sample(hist, n_paths=5, horizon=50, rng=np.random.default_rng(1))
        assert result.shape == (5, 50, 1)

    def test_multi_feature(self):
        hist = np.random.default_rng(0).random((100, 3))
        sampler = BootstrapSampler(block_size=10)
        result = sampler.sample(hist, n_paths=8, horizon=30, rng=np.random.default_rng(1))
        assert result.shape == (8, 30, 3)

    def test_horizon_not_multiple_of_block_size(self):
        hist = np.random.default_rng(0).random((100, 2))
        sampler = BootstrapSampler(block_size=7)
        result = sampler.sample(hist, n_paths=3, horizon=20, rng=np.random.default_rng(1))
        assert result.shape == (3, 20, 2)


class TestBootstrapSamplerValues:
    """All sampled values must come from the historical data."""

    def test_values_in_historical_range(self):
        rng = np.random.default_rng(42)
        hist = rng.random((100, 2))
        sampler = BootstrapSampler(block_size=10)
        result = sampler.sample(hist, n_paths=20, horizon=50, rng=np.random.default_rng(1))
        for j in range(hist.shape[1]):
            hist_set = set(hist[:, j].tolist())
            sampled = result[:, :, j].ravel()
            for v in sampled:
                assert v in hist_set


class TestBootstrapSamplerReproducibility:
    """Fixed seed must produce identical output."""

    def test_reproducible(self):
        hist = np.random.default_rng(0).random((100, 2))
        sampler = BootstrapSampler(block_size=10)
        a = sampler.sample(hist, n_paths=5, horizon=30, rng=np.random.default_rng(42))
        b = sampler.sample(hist, n_paths=5, horizon=30, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)

    def test_different_seed_differs(self):
        hist = np.random.default_rng(0).random((100, 2))
        sampler = BootstrapSampler(block_size=10)
        a = sampler.sample(hist, n_paths=5, horizon=30, rng=np.random.default_rng(1))
        b = sampler.sample(hist, n_paths=5, horizon=30, rng=np.random.default_rng(2))
        assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# AR1Sampler
# ---------------------------------------------------------------------------


class TestAR1SamplerShape:
    """Output shape must be (n_paths, horizon, n_features)."""

    def test_single_feature(self):
        hist = np.random.default_rng(0).random((100, 1))
        sampler = AR1Sampler()
        result = sampler.sample(hist, n_paths=5, horizon=50, rng=np.random.default_rng(1))
        assert result.shape == (5, 50, 1)

    def test_multi_feature(self):
        hist = np.random.default_rng(0).random((100, 3))
        sampler = AR1Sampler()
        result = sampler.sample(hist, n_paths=8, horizon=30, rng=np.random.default_rng(1))
        assert result.shape == (8, 30, 3)


class TestAR1SamplerStartValue:
    """First sampled value should continue from last historical value."""

    def test_first_step_is_last_historical(self):
        rng = np.random.default_rng(0)
        # Constant series => AR(1) with phi~1, sigma~0
        # First value should be very close to the last historical value
        hist = np.full((100, 1), 5.0)
        sampler = AR1Sampler()
        result = sampler.sample(hist, n_paths=10, horizon=20, rng=np.random.default_rng(1))
        # With constant input, phi=1 and sigma=0, all values should be ~5.0
        np.testing.assert_allclose(result, 5.0, atol=1e-10)


class TestAR1SamplerReproducibility:
    """Fixed seed must produce identical output."""

    def test_reproducible(self):
        hist = np.random.default_rng(0).random((100, 2))
        sampler = AR1Sampler()
        a = sampler.sample(hist, n_paths=5, horizon=30, rng=np.random.default_rng(42))
        b = sampler.sample(hist, n_paths=5, horizon=30, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)


class TestAR1SamplerCoefficient:
    """On a known AR(1) process, the fitted phi should be close to the true value."""

    def test_recovers_coefficient(self):
        # Generate a long AR(1) series with known phi
        true_phi = 0.8
        true_sigma = 0.5
        rng = np.random.default_rng(42)
        n = 5000
        series = np.zeros(n)
        series[0] = 0.0
        for t in range(1, n):
            series[t] = true_phi * series[t - 1] + rng.normal(0, true_sigma)

        hist = series.reshape(-1, 1)
        sampler = AR1Sampler()
        result = sampler.sample(hist, n_paths=1000, horizon=100, rng=np.random.default_rng(0))

        # The sample mean at step 1 should be close to phi * last_value
        # More robust: check that the sample autocorrelation of the
        # generated paths is close to phi
        paths = result[:, :, 0]  # (1000, 100)
        # Empirical lag-1 autocorrelation across the time axis, averaged over paths
        lag1_corrs = []
        for i in range(paths.shape[0]):
            p = paths[i]
            if np.std(p) > 1e-10:
                lag1_corrs.append(np.corrcoef(p[:-1], p[1:])[0, 1])
        mean_lag1 = np.mean(lag1_corrs)
        assert mean_lag1 == pytest.approx(true_phi, abs=0.1)

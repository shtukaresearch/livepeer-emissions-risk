"""Tests for ChainStateSimulation, ParticipationSimulation, and Simulator."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from lpt_stake.emissions_schedule import SignedStepSchedule
from lpt_stake.exogenous import BootstrapSampler
from lpt_stake.simulation import (
    ChainStateSimulation,
    GaussianNoise,
    ParticipationSimulation,
    RawParticipationModel,
    Simulator,
)
from lpt_stake.types import SimulationOverrunError, SimulationState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCHEDULE = SignedStepSchedule(
    target_participation_rate=0.5,
    emissions_change=1.0,
)

ZERO_NOISE = GaussianNoise(sigma=0.0)


def _make_df() -> pl.DataFrame:
    """Simple DataFrame with participation_rate, emissions_rate, and one
    exogenous column."""
    n = 30
    rng = np.random.default_rng(42)
    p = np.clip(0.4 + 0.01 * np.arange(n) + rng.normal(0, 0.01, n), 0.01, 0.99)
    return pl.DataFrame({
        "participation_rate": p,
        "emissions_rate": np.full(n, 200_000.0),
        "fear_greed": rng.uniform(20, 80, size=n),
    })


def _fitted_model(df: pl.DataFrame) -> RawParticipationModel:
    """Return a fitted RawParticipationModel on df."""
    model = RawParticipationModel(
        features=["participation_rate", "fear_greed"],
        target="participation_rate",
        alpha=0.1,
        noise=ZERO_NOISE,
    )
    model.fit(df)
    return model


# ---------------------------------------------------------------------------
# ChainStateSimulation
# ---------------------------------------------------------------------------


class TestChainStateFromInitial:
    """from_initial sets correct shape and initial values."""

    def test_shapes(self):
        state = SimulationState(participation_rate=0.45, emissions_rate_per_round=200_000)
        chain = ChainStateSimulation.from_initial(state, SCHEDULE, n_paths=10, horizon=50)
        assert chain.participation_rate.shape == (10, 51)
        assert chain.emissions_rate.shape == (10, 51)

    def test_initial_values(self):
        state = SimulationState(participation_rate=0.45, emissions_rate_per_round=200_000)
        chain = ChainStateSimulation.from_initial(state, SCHEDULE, n_paths=5, horizon=20)
        np.testing.assert_array_equal(chain.participation_rate[:, 0], 0.45)
        np.testing.assert_array_equal(chain.emissions_rate[:, 0], 200_000)

    def test_counter_starts_at_zero(self):
        state = SimulationState(participation_rate=0.5, emissions_rate_per_round=100)
        chain = ChainStateSimulation.from_initial(state, SCHEDULE, n_paths=2, horizon=5)
        assert chain._t == 0


class TestChainStateStep:
    """step() writes correct values and advances counter."""

    def test_step_writes_response_and_emissions(self):
        state = SimulationState(participation_rate=0.5, emissions_rate_per_round=100.0)
        chain = ChainStateSimulation.from_initial(state, SCHEDULE, n_paths=3, horizon=5)

        chain.step(np.full(3, 0.6))

        # P(1) = response
        np.testing.assert_array_equal(chain.participation_rate[:, 1], 0.6)

        # Schedule sees P(1)=0.6 > target=0.5, so step is -1.0
        # I(1) = I(0) + (-1.0) = 100.0 - 1.0 = 99.0
        np.testing.assert_allclose(chain.emissions_rate[:, 1], 99.0)

    def test_counter_advances(self):
        state = SimulationState(participation_rate=0.5, emissions_rate_per_round=100.0)
        chain = ChainStateSimulation.from_initial(state, SCHEDULE, n_paths=2, horizon=5)
        chain.step(np.full(2, 0.6))
        assert chain._t == 1
        chain.step(np.full(2, 0.4))
        assert chain._t == 2

    def test_two_steps_hand_computed(self):
        """Two steps with known responses and schedule."""
        state = SimulationState(participation_rate=0.5, emissions_rate_per_round=100.0)
        chain = ChainStateSimulation.from_initial(state, SCHEDULE, n_paths=1, horizon=5)

        # Step 1: response=0.6, P(1)=0.6, above target -> I decreases
        chain.step(np.array([0.6]))
        assert chain.participation_rate[0, 1] == pytest.approx(0.6)
        assert chain.emissions_rate[0, 1] == pytest.approx(99.0)

        # Step 2: response=0.4, P(2)=0.4, below target -> I increases
        chain.step(np.array([0.4]))
        assert chain.participation_rate[0, 2] == pytest.approx(0.4)
        assert chain.emissions_rate[0, 2] == pytest.approx(100.0)

    def test_step_past_horizon_raises(self):
        state = SimulationState(participation_rate=0.5, emissions_rate_per_round=100.0)
        chain = ChainStateSimulation.from_initial(state, SCHEDULE, n_paths=1, horizon=2)
        chain.step(np.array([0.5]))
        chain.step(np.array([0.5]))
        with pytest.raises(SimulationOverrunError):
            chain.step(np.array([0.5]))


# ---------------------------------------------------------------------------
# ParticipationSimulation
# ---------------------------------------------------------------------------


class TestParticipationSimulationFromModel:
    """from_model creates correct FeatureSimulations."""

    def test_creates_feature_sims(self):
        df = _make_df()
        model = _fitted_model(df)
        part = ParticipationSimulation.from_model(
            model, df, n_paths=5, horizon=10,
        )
        assert len(part.feature_sims) == len(model.features)

    def test_counter_starts_at_zero(self):
        df = _make_df()
        model = _fitted_model(df)
        part = ParticipationSimulation.from_model(
            model, df, n_paths=5, horizon=10,
        )
        assert part._t == 0


class TestParticipationSimulationStep:
    """step() routes observations and returns valid response."""

    def test_step_returns_valid_response(self):
        df = _make_df()
        model = _fitted_model(df)
        state = SimulationState(participation_rate=0.45, emissions_rate_per_round=200_000)
        chain = ChainStateSimulation.from_initial(state, SCHEDULE, n_paths=5, horizon=10)

        rng = np.random.default_rng(42)
        exog_paths = rng.uniform(20, 80, size=(5, 10, 1))

        part = ParticipationSimulation.from_model(
            model, df, n_paths=5, horizon=10,
        )
        response = part.step(chain, exog_paths, rng)

        assert response.shape == (5,)
        assert np.all(response >= 0.0)
        assert np.all(response <= 1.0)

    def test_counter_advances(self):
        df = _make_df()
        model = _fitted_model(df)
        state = SimulationState(participation_rate=0.45, emissions_rate_per_round=200_000)
        chain = ChainStateSimulation.from_initial(state, SCHEDULE, n_paths=3, horizon=10)
        exog_paths = np.random.default_rng(0).uniform(20, 80, size=(3, 10, 1))

        part = ParticipationSimulation.from_model(
            model, df, n_paths=3, horizon=10,
        )
        rng = np.random.default_rng(0)
        part.step(chain, exog_paths, rng)
        assert part._t == 1

    def test_step_past_horizon_raises(self):
        df = _make_df()
        model = _fitted_model(df)
        state = SimulationState(participation_rate=0.45, emissions_rate_per_round=200_000)
        chain = ChainStateSimulation.from_initial(state, SCHEDULE, n_paths=2, horizon=2)
        exog_paths = np.random.default_rng(0).uniform(20, 80, size=(2, 2, 1))

        part = ParticipationSimulation.from_model(
            model, df, n_paths=2, horizon=2,
        )
        rng = np.random.default_rng(0)
        part.step(chain, exog_paths, rng)
        chain.step(part.step(chain, exog_paths, rng))
        with pytest.raises(SimulationOverrunError):
            part.step(chain, exog_paths, rng)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class TestSimulatorOutputShape:
    """Simulator.run() returns SimulationResult with correct shapes."""

    def test_shapes(self):
        df = _make_df()
        model = _fitted_model(df)
        sim = Simulator(
            schedule=SCHEDULE,
            model=model,
            exog_sampler=BootstrapSampler(block_size=5),
        )
        result = sim.run(
            initial_state=SimulationState(
                participation_rate=0.45,
                emissions_rate_per_round=200_000,
            ),
            n_paths=10,
            horizon=20,
            df=df,
            rng=np.random.default_rng(42),
        )
        assert result.participation_rate_paths.shape == (10, 21)
        assert result.emissions_rate_per_round_paths.shape == (10, 21)


class TestSimulatorReproducibility:
    """Same seed produces identical results."""

    def test_reproducible(self):
        df = _make_df()
        model = _fitted_model(df)
        sim = Simulator(
            schedule=SCHEDULE,
            model=model,
            exog_sampler=BootstrapSampler(block_size=5),
        )
        state = SimulationState(
            participation_rate=0.45,
            emissions_rate_per_round=200_000,
        )
        r1 = sim.run(state, n_paths=10, horizon=20, df=df, rng=np.random.default_rng(42))
        r2 = sim.run(state, n_paths=10, horizon=20, df=df, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(
            r1.participation_rate_paths, r2.participation_rate_paths,
        )
        np.testing.assert_array_equal(
            r1.emissions_rate_per_round_paths, r2.emissions_rate_per_round_paths,
        )


class TestSimulatorValueRanges:
    """Participation rates should be in [0, 1]."""

    def test_participation_in_range(self):
        df = _make_df()
        model = _fitted_model(df)
        sim = Simulator(
            schedule=SCHEDULE,
            model=model,
            exog_sampler=BootstrapSampler(block_size=5),
        )
        result = sim.run(
            initial_state=SimulationState(
                participation_rate=0.45,
                emissions_rate_per_round=200_000,
            ),
            n_paths=50,
            horizon=100,
            df=df,
            rng=np.random.default_rng(42),
        )
        assert np.all(result.participation_rate_paths >= 0.0)
        assert np.all(result.participation_rate_paths <= 1.0)


class TestSimulatorInitialValues:
    """First column of output should match initial state."""

    def test_initial_state_preserved(self):
        df = _make_df()
        model = _fitted_model(df)
        sim = Simulator(
            schedule=SCHEDULE,
            model=model,
            exog_sampler=BootstrapSampler(block_size=5),
        )
        state = SimulationState(
            participation_rate=0.45,
            emissions_rate_per_round=200_000,
        )
        result = sim.run(state, n_paths=5, horizon=10, df=df, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(result.participation_rate_paths[:, 0], 0.45)
        np.testing.assert_array_equal(result.emissions_rate_per_round_paths[:, 0], 200_000)


# ---------------------------------------------------------------------------
# Numeric integration tests
# ---------------------------------------------------------------------------


class TestConstantParticipation:
    """With a trivial model that always returns the initial P, the only
    dynamics come from the emissions schedule.  P stays constant and
    emissions follow a deterministic arithmetic sequence.
    """

    def test_emissions_monotone_increase(self):
        """P=0.3 < target=0.5 => schedule increases emissions every round."""
        # Build a df where P is constant at 0.3
        n = 20
        df = pl.DataFrame({
            "participation_rate": np.full(n, 0.3),
            "emissions_rate": np.full(n, 100.0),
        })
        # Fit a model with no features — prediction depends only on intercept.
        # With P constant at 0.3, target is P[1:] = 0.3, intercept fits to 0.3.
        model = RawParticipationModel(
            features=[],
            target="participation_rate",
            alpha=0.0,
            noise=ZERO_NOISE,
        )
        model.fit(df)

        schedule = SignedStepSchedule(
            target_participation_rate=0.5,
            emissions_change=10.0,
        )
        sim = Simulator(
            schedule=schedule,
            model=model,
            exog_sampler=BootstrapSampler(block_size=5),
        )
        horizon = 50
        result = sim.run(
            initial_state=SimulationState(
                participation_rate=0.3,
                emissions_rate_per_round=100.0,
            ),
            n_paths=1,
            horizon=horizon,
            df=df,
            rng=np.random.default_rng(0),
        )

        p_path = result.participation_rate_paths[0]
        i_path = result.emissions_rate_per_round_paths[0]

        # P should be approximately constant at 0.3
        np.testing.assert_allclose(p_path, 0.3, atol=1e-6)

        # Emissions should increase by 10.0 each round (P < target)
        expected_i = 100.0 + 10.0 * np.arange(horizon + 1)
        np.testing.assert_allclose(i_path, expected_i)

    def test_emissions_monotone_decrease(self):
        """P=0.7 > target=0.5 => schedule decreases emissions every round."""
        n = 20
        df = pl.DataFrame({
            "participation_rate": np.full(n, 0.7),
            "emissions_rate": np.full(n, 1000.0),
        })
        model = RawParticipationModel(
            features=[],
            target="participation_rate",
            alpha=0.0,
            noise=ZERO_NOISE,
        )
        model.fit(df)

        schedule = SignedStepSchedule(
            target_participation_rate=0.5,
            emissions_change=5.0,
        )
        sim = Simulator(
            schedule=schedule,
            model=model,
            exog_sampler=BootstrapSampler(block_size=5),
        )
        horizon = 30
        result = sim.run(
            initial_state=SimulationState(
                participation_rate=0.7,
                emissions_rate_per_round=1000.0,
            ),
            n_paths=1,
            horizon=horizon,
            df=df,
            rng=np.random.default_rng(0),
        )

        p_path = result.participation_rate_paths[0]
        i_path = result.emissions_rate_per_round_paths[0]

        np.testing.assert_allclose(p_path, 0.7, atol=1e-6)

        # Emissions decrease by 5.0 each round (P > target)
        expected_i = 1000.0 - 5.0 * np.arange(horizon + 1)
        np.testing.assert_allclose(i_path, expected_i)

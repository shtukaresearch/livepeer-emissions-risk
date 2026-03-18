"""Core data types and pluggable-interface protocols for the lpt_stake library.

Dataclasses define the concrete data containers that flow between pipeline
stages.  Protocols define the interfaces that the simulation stage depends on,
allowing alternative implementations to be swapped in without changing the
simulation loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from numpy.random import Generator

    import polars as pl

# ---------------------------------------------------------------------------
# Dataclasses — concrete data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationState:
    """Starting point for a simulation run.

    Fields
    ------
    participation_rate : float
        Last observed participation rate, in [0, 1].
    emissions_rate_per_round : float
        Last observed emissions rate, in parts per billion (the raw
        ``inflation`` field from the Minter contract).
    """

    participation_rate: float
    emissions_rate_per_round: float

    def __post_init__(self) -> None:
        """Validate field constraints."""
        if not 0.0 <= self.participation_rate <= 1.0:
            raise ValueError(
                f"participation_rate must be in [0, 1], got {self.participation_rate}"
            )
        if self.emissions_rate_per_round < 0:
            raise ValueError(
                f"emissions_rate_per_round must be non-negative, "
                f"got {self.emissions_rate_per_round}"
            )


@dataclass(frozen=True)
class SimulationResult:
    """Output of a simulation run.

    Fields
    ------
    participation_rate_paths : NDArray
        Shape ``(n_paths, horizon + 1)``.  Each row is a simulated
        participation-rate trajectory in [0, 1].
    emissions_rate_per_round_paths : NDArray
        Shape ``(n_paths, horizon + 1)``.  Each row is a simulated
        emissions-rate trajectory in parts per billion.
    """

    participation_rate_paths: NDArray[np.floating]
    emissions_rate_per_round_paths: NDArray[np.floating]

    def __post_init__(self) -> None:
        """Validate array shape constraints."""
        if self.participation_rate_paths.shape != self.emissions_rate_per_round_paths.shape:
            raise ValueError(
                "participation_rate_paths and emissions_rate_per_round_paths "
                "must have the same shape, got "
                f"{self.participation_rate_paths.shape} and "
                f"{self.emissions_rate_per_round_paths.shape}"
            )
        if self.participation_rate_paths.ndim != 2:
            raise ValueError(
                "Path arrays must be 2-dimensional (n_paths, horizon + 1), "
                f"got {self.participation_rate_paths.ndim} dimensions"
            )


# ---------------------------------------------------------------------------
# Protocols — pluggable interfaces
# ---------------------------------------------------------------------------


class EmissionsSchedule(Protocol):
    """State machine for the per-round emissions-rate update rule.

    Implementations receive the current emissions rate and participation rate
    (both as arrays of shape ``(n_paths,)``) and return the next-round
    emissions rate.

    The current Livepeer protocol rule is ``SignedStepSchedule`` in
    ``emissions_schedule.py``.
    """

    def update(
        self,
        emissions_rate_per_round: NDArray[np.floating],
        participation_rate: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Return next-round emissions rate given current state.

        Parameters
        ----------
        emissions_rate_per_round
            Current emissions rate in parts per billion, shape ``(n_paths,)``.
        participation_rate
            Current participation rate in [0, 1], shape ``(n_paths,)``.

        Returns
        -------
        NDArray
            Next-round emissions rate, shape ``(n_paths,)``.
        """
        ...


class Feature(Protocol):
    """A stateful feature with column dependencies.

    Stateful features (e.g. trailing yield) implement this protocol.
    They carry their column dependencies as column name strings and
    provide two entry points:

    - ``evaluate(df)`` for fitting: stateless evaluation over a full
      DataFrame.
    - ``make_simulation(historical, n_paths)`` for simulation: construct
      a ``FeatureSimulation`` initialized from historical data.

    For simple column features (strings in the model constructor), the
    model handles both paths directly — no ``Feature`` object needed.
    """

    columns: list[str]

    def evaluate(self, df: pl.DataFrame) -> NDArray[np.floating]:
        """Evaluate feature over a full DataFrame.

        Used by ``ParticipationModel.fit()`` to build the design matrix.

        Parameters
        ----------
        df
            Round-indexed DataFrame containing the columns listed in
            ``self.columns``.

        Returns
        -------
        NDArray
            Feature values, shape ``(T,)``.
        """
        ...

    def make_simulation(
        self,
        historical: pl.DataFrame,
        n_paths: int,
    ) -> FeatureSimulation:
        """Construct a ``FeatureSimulation`` initialized from history.

        Parameters
        ----------
        historical
            Historical DataFrame for cache initialization.
        n_paths
            Number of Monte Carlo paths.

        Returns
        -------
        FeatureSimulation
        """
        ...


class FeatureSimulation(Protocol):
    """Stateful feature accumulator for the simulation loop.

    Initialized from historical data to set up cache shape and fill
    lookback.  At each step, receives fresh observations as arguments
    and returns the feature value.  Does not hold a reference to any
    domain array.

    For simple column features, this is a pass-through that returns
    the observation unchanged.  For stateful features (e.g. trailing
    yield), this maintains a rolling accumulator.
    """

    def step(self, observation: NDArray[np.floating]) -> NDArray[np.floating]:
        """Receive fresh observation, update cache, return feature value.

        Parameters
        ----------
        observation
            Current-step values for this feature's input columns.
            Shape ``(n_paths,)`` for single-column,
            ``(n_paths, k)`` for multi-column.

        Returns
        -------
        NDArray
            Feature value, shape ``(n_paths,)``.
        """
        ...


class NoiseModel(Protocol):
    """Draws residual noise samples for the participation model."""

    def __call__(self, n: int, rng: Generator) -> NDArray[np.floating]:
        """Draw *n* noise samples.

        Parameters
        ----------
        n
            Number of samples to draw.
        rng
            Numpy random generator.

        Returns
        -------
        NDArray
            Noise samples, shape ``(n,)``.
        """
        ...


class ExogenousSampler(Protocol):
    """Samples forward paths for exogenous variables from historical data."""

    def sample(
        self,
        historical: NDArray[np.floating],
        n_paths: int,
        horizon: int,
        rng: Generator,
    ) -> NDArray[np.floating]:
        """Sample exogenous variable paths.

        Parameters
        ----------
        historical
            Historical exogenous data as a numpy array, shape
            ``(T, n_features)``.
        n_paths
            Number of Monte Carlo paths.
        horizon
            Number of forward steps.
        rng
            Numpy random generator.

        Returns
        -------
        NDArray
            Sampled paths, shape ``(n_paths, horizon, n_features)``.
        """
        ...

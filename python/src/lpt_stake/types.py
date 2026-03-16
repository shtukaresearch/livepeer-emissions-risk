"""Core data types and pluggable-interface protocols for the lpt_stake library.

Dataclasses define the concrete data containers that flow between pipeline
stages.  Protocols define the interfaces that the simulation stage depends on,
allowing alternative implementations to be swapped in without changing the
simulation loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

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


@runtime_checkable
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


@runtime_checkable
class ParticipationModel(Protocol):
    """Predicts the next-round participation rate from current state.

    Implementations encapsulate all internal transforms (logit, differencing,
    etc.) and map raw state to the next participation rate in [0, 1].
    """

    def predict_next(
        self,
        participation_rate: NDArray[np.floating],
        emissions_rate: NDArray[np.floating],
        step: int,
        rng: Generator,
    ) -> NDArray[np.floating]:
        """Predict next-round participation rate.

        Parameters
        ----------
        participation_rate
            Current participation rate in [0, 1], shape ``(n_paths,)``.
        emissions_rate
            Current emissions rate in parts per billion, shape ``(n_paths,)``.
        step
            Time index into pre-sampled exogenous paths (0 .. horizon-1).
        rng
            Numpy random generator for noise sampling.

        Returns
        -------
        NDArray
            Predicted participation rate in [0, 1], shape ``(n_paths,)``.
        """
        ...

    def prepare(
        self,
        n_paths: int,
        horizon: int,
        rng: Generator,
    ) -> None:
        """Pre-sample stochastic components before the simulation loop.

        Called once by the ``Simulator`` before entering the step loop.  Use
        this to pre-sample exogenous variable paths, noise vectors, or any
        other random state that should be fixed for the entire run.

        Parameters
        ----------
        n_paths
            Number of Monte Carlo paths.
        horizon
            Number of forward steps.
        rng
            Numpy random generator.
        """
        ...


@runtime_checkable
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


@runtime_checkable
class ExogenousSampler(Protocol):
    """Samples forward paths for exogenous variables from historical data."""

    def sample(
        self,
        historical: pl.DataFrame,
        n_paths: int,
        horizon: int,
        rng: Generator,
    ) -> NDArray[np.floating]:
        """Sample exogenous variable paths.

        Parameters
        ----------
        historical
            Historical exogenous data as a Polars DataFrame.
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

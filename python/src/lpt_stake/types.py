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
    """An unbound feature with column dependencies.

    Stateful features (e.g. trailing yield) implement this protocol and
    carry their column dependencies as column name strings.  For simple
    column features (strings in the model constructor), the model wraps
    them internally in a ``BoundColumnFeature`` without going through
    ``Feature``.
    """

    def bind(self, data: NDArray[np.floating]) -> BoundFeature:
        """Bind to column data.

        Parameters
        ----------
        data
            Column data sliced by the caller: shape ``(T,)`` for fitting,
            ``(n_paths, T)`` for simulation.  For multi-column features,
            shape ``(T, k)`` or ``(n_paths, T, k)``.

        Returns
        -------
        BoundFeature
        """
        ...


class BoundFeature(Protocol):
    """A feature bound to a specific data array.

    Provides vectorised access (``evaluate``) and incremental access
    (``step``).  For stateless features, ``step(t)`` is equivalent to
    ``evaluate()[..., t]``.  For stateful features, ``evaluate`` with
    ``cache=True`` prepares internal accumulators for subsequent
    ``step`` calls.
    """

    def evaluate(
        self,
        end: int | None = ...,
        cache: bool = ...,
    ) -> NDArray[np.floating]:
        """Compute feature values over the data, up to index *end*.

        Parameters
        ----------
        end
            Slice the time axis to ``[:end]``.  ``None`` means the full
            extent.
        cache
            If ``True``, save intermediate state for subsequent ``step``
            calls.  Ignored by stateless features.

        Returns
        -------
        NDArray
            Shape ``(T,)`` or ``(end,)`` for 1-D data;
            ``(n_paths, T)`` or ``(n_paths, end)`` for 2-D data.
        """
        ...

    def step(self, t: int) -> NDArray[np.floating]:
        """Feature value at time *t* (incremental).

        Returns
        -------
        NDArray
            Scalar for 1-D data, shape ``(n_paths,)`` for 2-D data.
        """
        ...

    def rebind(self, data: NDArray[np.floating]) -> BoundFeature:
        """Create a new ``BoundFeature`` on *data*, same transform.

        Returns
        -------
        BoundFeature
        """
        ...


class ParticipationModel(Protocol):
    """Central simulation object.  Owns features, fitting, and prediction.

    The three concrete implementations (``RawParticipationModel``,
    ``LogitParticipationModel``, ``DiffLogitParticipationModel``) differ
    only in how they transform the target and inverse-transform predictions.
    """

    def fit(self) -> None:
        """Fit ridge model from ``sample_path_historic``.

        Calls ``evaluate()`` on each ``BoundFeature`` to build the design
        matrix, computes the target (class-specific transform), calls
        ``fit_ridge``.
        """
        ...

    def prepare(
        self,
        sample_path_simulated: NDArray[np.floating],
        t0: int,
    ) -> None:
        """Rebind features to simulation array and initialise.

        Calls ``rebind`` on each ``BoundFeature`` to produce new
        ``BoundFeature`` instances on *sample_path_simulated*, then calls
        ``evaluate(end=t0, cache=True)`` on each.

        Parameters
        ----------
        sample_path_simulated
            Simulation array, shape ``(n_paths, lookback + horizon + 1,
            n_cols)``.
        t0
            Index at which simulation starts (i.e. the lookback length).
        """
        ...

    def predict_next(
        self,
        t: int,
        rng: Generator,
    ) -> NDArray[np.floating]:
        """Predict next participation rate from state at time *t*.

        Calls ``step(t)`` on each ``BoundFeature``, assembles feature
        vector with intercept, applies ridge predict, adds noise,
        inverse-transforms.

        Parameters
        ----------
        t
            Current time index into ``sample_path_simulated``.
        rng
            Numpy random generator for noise sampling.

        Returns
        -------
        NDArray
            Predicted participation rate in [0, 1], shape ``(n_paths,)``.
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

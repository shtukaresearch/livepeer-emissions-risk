"""Simulation components for the lpt_stake library.

Provides:

- Noise models: :class:`GaussianNoise`, :class:`BootstrapNoise`.
- Participation models: :class:`RawParticipationModel`,
  :class:`LogitParticipationModel`, :class:`DiffLogitParticipationModel`.
- Simulation subsystems: :class:`ChainStateSimulation`,
  :class:`ParticipationSimulation`, :class:`Simulator`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from lpt_stake.exogenous import block_bootstrap
from lpt_stake.features import np_expit, np_logit
from lpt_stake.model import fit_ridge
from lpt_stake.types import SimulationOverrunError, SimulationResult, SimulationState

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.random import Generator
    from numpy.typing import NDArray

    from lpt_stake.model import RidgeResult
    from lpt_stake.types import (
        EmissionsSchedule,
        ExogenousSampler,
        Feature,
        FeatureSimulation,
        NoiseModel,
    )


# ---------------------------------------------------------------------------
# Noise models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GaussianNoise:
    """Gaussian residual noise model.

    Draws i.i.d. samples from ``N(0, sigma)``.

    Fields
    ------
    sigma : float
        Standard deviation of the noise.
    """

    sigma: float

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
        if self.sigma == 0.0:
            return np.zeros(n)
        return rng.normal(0.0, self.sigma, size=n)


@dataclass(frozen=True)
class BootstrapNoise:
    """Block-bootstrap residual noise model.

    Resamples contiguous blocks from fitted residuals using
    :func:`~lpt_stake.exogenous.block_bootstrap`.  This preserves
    temporal correlation structure in the noise.

    Fields
    ------
    residuals : NDArray
        1-D array of fitted residuals to resample from.
    block_size : int
        Length of each contiguous block.
    """

    residuals: NDArray[np.floating]
    block_size: int

    def __call__(self, n: int, rng: Generator) -> NDArray[np.floating]:
        """Draw *n* noise samples by block-bootstrapping residuals.

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
        # block_bootstrap returns (n_paths, horizon) for 1-D input.
        # We want a single flat draw of length n, so n_paths=1.
        return block_bootstrap(self.residuals, n_paths=1, horizon=n,
                               block_size=self.block_size, rng=rng).ravel()


# ---------------------------------------------------------------------------
# Participation models
# ---------------------------------------------------------------------------


class _BaseParticipationModel:
    """Common logic for participation model variants.

    Subclasses implement :meth:`_compute_target` and
    :meth:`_inverse_transform` to define the target transform and its
    inverse.

    Parameters
    ----------
    features
        List of column names (strings) and/or :class:`Feature` objects.
    target
        Name of the participation-rate column in the DataFrame.
    alpha
        Ridge regularisation strength.
    noise
        Noise model for adding residual noise to predictions.
    """

    def __init__(
        self,
        features: list[str | Feature],
        target: str,
        alpha: float,
        noise: NoiseModel,
    ) -> None:
        self.features = features
        self.target = target
        self.alpha = alpha
        self.noise = noise
        self.ridge_result: RidgeResult | None = None
        self._feature_names: list[str] = []

    def fit(self, df: pl.DataFrame) -> None:
        """Fit the ridge model from a round-indexed DataFrame.

        For each feature, extracts the column (string) or calls
        ``feature.evaluate(df)`` (Feature object).  The target is the
        participation-rate column with the class-specific transform
        applied.  An internal shift aligns features at round *n* with
        the target derived from round *n* + 1 — the caller should
        **not** pre-shift the target column.

        Parameters
        ----------
        df
            Round-indexed DataFrame with columns for all features and
            the target.
        """
        # Evaluate features
        feature_arrays: list[NDArray[np.floating]] = []
        self._feature_names = []
        for i, f in enumerate(self.features):
            if isinstance(f, str):
                feature_arrays.append(df[f].to_numpy().astype(np.float64))
                self._feature_names.append(f)
            else:
                feature_arrays.append(f.evaluate(df))
                self._feature_names.append(f"feature_{i}")

        # Compute target with class-specific transform (length T-1)
        p = df[self.target].to_numpy().astype(np.float64)
        target_values = self._compute_target(p)

        # Align: features at [0, T-2], target has length T-1
        n_target = len(target_values)
        trimmed = [a[:n_target] for a in feature_arrays]

        # Build design matrix as Polars DataFrame
        dm_data: dict[str, object] = {
            "intercept": np.ones(n_target),
        }
        for name, arr in zip(self._feature_names, trimmed):
            dm_data[name] = arr
        dm_data["target"] = target_values
        dm = pl.DataFrame(dm_data)

        self.ridge_result = fit_ridge(dm, "target", self.alpha)

    def predict(
        self,
        feature_vector: NDArray[np.floating],
        current_p: NDArray[np.floating],
        rng: Generator,
    ) -> NDArray[np.floating]:
        """Predict next-round participation rate.

        Applies ridge coefficients to the feature vector, adds noise,
        and applies the class-specific inverse transform.

        Parameters
        ----------
        feature_vector
            Feature values for the current round, shape ``(n_paths,
            n_features)``.  Same order as features passed to the
            constructor, **without** intercept.
        current_p
            Current participation rate, shape ``(n_paths,)``.  Used by
            ``DiffLogitParticipationModel``; ignored by other variants.
        rng
            Numpy random generator for noise sampling.

        Returns
        -------
        NDArray
            Predicted participation rate in [0, 1], shape ``(n_paths,)``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self.ridge_result is None:
            raise RuntimeError("fit() must be called before predict()")

        n = feature_vector.shape[0]
        X = np.column_stack([np.ones(n), feature_vector])
        y_hat = self.ridge_result.predict(X)
        noise_samples = self.noise(n, rng)
        return self._inverse_transform(y_hat + noise_samples, current_p)

    def _compute_target(self, p: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute the regression target from the participation-rate column.

        The returned array has length ``len(p) - 1`` because features
        at round *n* predict the target derived from round *n* + 1.
        """
        raise NotImplementedError

    def _inverse_transform(
        self,
        y: NDArray[np.floating],
        current_p: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Inverse-transform predictions back to [0, 1]."""
        raise NotImplementedError


class RawParticipationModel(_BaseParticipationModel):
    """Participation model with identity target transform.

    Target is ``P(n+1)`` directly.  Predictions are clipped to [0, 1].
    """

    def _compute_target(self, p: NDArray[np.floating]) -> NDArray[np.floating]:
        """Target is the next-round participation rate."""
        return p[1:]

    def _inverse_transform(
        self,
        y: NDArray[np.floating],
        current_p: NDArray[np.floating],  # noqa: ARG002
    ) -> NDArray[np.floating]:
        """Clip to [0, 1]."""
        return np.clip(y, 0.0, 1.0)


class LogitParticipationModel(_BaseParticipationModel):
    """Participation model with logit target transform.

    Target is ``logit(P(n+1))``.  Predictions are inverse-transformed
    via expit, which guarantees output in (0, 1).
    """

    def _compute_target(self, p: NDArray[np.floating]) -> NDArray[np.floating]:
        """Target is logit of the next-round participation rate."""
        return np_logit(p[1:])

    def _inverse_transform(
        self,
        y: NDArray[np.floating],
        current_p: NDArray[np.floating],  # noqa: ARG002
    ) -> NDArray[np.floating]:
        """Apply expit."""
        return np_expit(y)


class DiffLogitParticipationModel(_BaseParticipationModel):
    """Participation model with diff-logit target transform.

    Target is ``logit(P(n+1)) - logit(P(n))``.  Predictions are
    inverse-transformed via ``expit(logit(current_p) + ŷ)``.
    """

    def _compute_target(self, p: NDArray[np.floating]) -> NDArray[np.floating]:
        """Target is the first difference of logit(P)."""
        logit_p = np_logit(p)
        return logit_p[1:] - logit_p[:-1]

    def _inverse_transform(
        self,
        y: NDArray[np.floating],
        current_p: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Apply expit(logit(current_p) + ŷ)."""
        return np_expit(np_logit(current_p) + y)


# ---------------------------------------------------------------------------
# Pass-through FeatureSimulation for simple column features
# ---------------------------------------------------------------------------


class _PassthroughFeatureSim:
    """FeatureSimulation that returns the observation unchanged."""

    def step(self, observation: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return the observation as-is."""
        return observation


# ---------------------------------------------------------------------------
# Observation routing
# ---------------------------------------------------------------------------

# Chain state column names that are read from ChainStateSimulation.
_CHAIN_STATE_COLS = frozenset({"participation_rate", "emissions_rate"})


def _build_observation_routes(
    features: list[str | Feature],
    exog_col_names: list[str],
) -> list[Callable[[ChainStateSimulation, NDArray[np.floating]], NDArray[np.floating]]]:
    """Build a list of callables that extract the observation for each feature.

    Each callable has signature ``(chain_state, exog_paths) -> NDArray``
    and reads from the appropriate source at the current time index.

    Parameters
    ----------
    features
        The model's feature list (strings and/or Feature objects).
    exog_col_names
        Ordered list of exogenous column names, matching the last axis
        of the ``exog_paths`` array.
    """
    routes: list[Callable] = []
    exog_index = {name: i for i, name in enumerate(exog_col_names)}

    for f in features:
        col = f if isinstance(f, str) else f.columns[0]

        if col == "participation_rate":
            routes.append(
                lambda cs, ep: cs.participation_rate[:, cs._t]  # noqa: B023
            )
        elif col == "emissions_rate":
            routes.append(
                lambda cs, ep: cs.emissions_rate[:, cs._t]  # noqa: B023
            )
        elif col in exog_index:
            j = exog_index[col]
            routes.append(
                lambda cs, ep, _j=j: ep[:, cs._t, _j]
            )
        else:
            raise ValueError(
                f"Feature column '{col}' not found in chain state or "
                f"exogenous columns {exog_col_names}"
            )

    return routes


# ---------------------------------------------------------------------------
# ChainStateSimulation
# ---------------------------------------------------------------------------


class ChainStateSimulation:
    """Participation rate and emissions rate trajectories.

    Owns its step counter.  The schedule is held as a reference so that
    ``step()`` needs only the participation response.

    Fields
    ------
    participation_rate : NDArray
        Shape ``(n_paths, horizon + 1)``.
    emissions_rate : NDArray
        Shape ``(n_paths, horizon + 1)``.
    schedule : EmissionsSchedule
        The emissions update rule.
    """

    def __init__(
        self,
        participation_rate: NDArray[np.floating],
        emissions_rate: NDArray[np.floating],
        schedule: EmissionsSchedule,
    ) -> None:
        self.participation_rate = participation_rate
        self.emissions_rate = emissions_rate
        self.schedule = schedule
        self._t: int = 0
        self._horizon: int = participation_rate.shape[1] - 1

    @classmethod
    def from_initial(
        cls,
        initial_state: SimulationState,
        schedule: EmissionsSchedule,
        n_paths: int,
        horizon: int,
    ) -> ChainStateSimulation:
        """Allocate arrays and set initial values.

        Parameters
        ----------
        initial_state
            Starting participation rate and emissions rate.
        schedule
            Emissions schedule (held by reference for ``step()``).
        n_paths
            Number of Monte Carlo paths.
        horizon
            Number of simulation steps.
        """
        p = np.empty((n_paths, horizon + 1))
        e = np.empty((n_paths, horizon + 1))
        p[:, 0] = initial_state.participation_rate
        e[:, 0] = initial_state.emissions_rate_per_round
        return cls(p, e, schedule)

    def step(self, response: NDArray[np.floating]) -> None:
        """Write chainstate at ``_t + 1`` and advance the counter.

        Applies the participation response, then computes the new
        emissions rate using the schedule (which sees the post-response
        participation rate).

        Parameters
        ----------
        response
            Predicted participation rate, shape ``(n_paths,)``.

        Raises
        ------
        SimulationOverrunError
            If the simulation has already reached the end of the
            allocated horizon.
        """
        if self._t >= self._horizon:
            raise SimulationOverrunError(
                f"Cannot step past horizon ({self._horizon} steps)"
            )
        self.participation_rate[:, self._t + 1] = response
        self.emissions_rate[:, self._t + 1] = self.schedule.update(
            self.emissions_rate[:, self._t],
            self.participation_rate[:, self._t + 1],
        )
        self._t += 1


# ---------------------------------------------------------------------------
# ParticipationSimulation
# ---------------------------------------------------------------------------


class ParticipationSimulation:
    """Owns FeatureSimulation instances and produces the participation
    response at each simulation step.

    Constructed from a fitted :class:`_BaseParticipationModel` via
    :meth:`from_model`.

    Fields
    ------
    model : _BaseParticipationModel
        The fitted participation model (for predict).
    feature_sims : list[FeatureSimulation]
        One per feature.
    feature_log : NDArray or None
        If ``save_history`` was set, shape ``(n_paths, horizon,
        n_features)``.
    """

    def __init__(
        self,
        model: _BaseParticipationModel,
        feature_sims: list[FeatureSimulation],
        observation_routes: list[Callable],
        horizon: int,
        feature_log: NDArray[np.floating] | None = None,
    ) -> None:
        self.model = model
        self.feature_sims = feature_sims
        self._observation_routes = observation_routes
        self._horizon = horizon
        self.feature_log = feature_log
        self._t: int = 0

    @classmethod
    def from_model(
        cls,
        model: _BaseParticipationModel,
        df: pl.DataFrame,
        n_paths: int,
        horizon: int,
        save_history: bool = False,
    ) -> ParticipationSimulation:
        """Build FeatureSimulations from a fitted model's features.

        Parameters
        ----------
        model
            A fitted participation model.
        df
            Historical DataFrame (for Feature.make_simulation and for
            determining exogenous column names).
        n_paths
            Number of Monte Carlo paths.
        horizon
            Number of simulation steps.
        save_history
            If True, allocate a feature log array.
        """
        feature_sims: list[FeatureSimulation] = []
        for f in model.features:
            if isinstance(f, str):
                feature_sims.append(_PassthroughFeatureSim())
            else:
                feature_sims.append(f.make_simulation(df, n_paths))

        # Determine exogenous column names (everything not chain state)
        exog_col_names = [
            c for c in df.columns if c not in _CHAIN_STATE_COLS
        ]

        routes = _build_observation_routes(model.features, exog_col_names)

        feature_log = None
        if save_history:
            feature_log = np.empty((n_paths, horizon, len(model.features)))

        return cls(model, feature_sims, routes, horizon, feature_log)

    def step(
        self,
        chain_state: ChainStateSimulation,
        exog_paths: NDArray[np.floating],
        rng: Generator,
    ) -> NDArray[np.floating]:
        """Observe features, predict response, advance.

        Parameters
        ----------
        chain_state
            Current chain state (read at ``chain_state._t``).
        exog_paths
            Exogenous paths, shape ``(n_paths, horizon, n_exog)``.
        rng
            Numpy random generator.

        Returns
        -------
        NDArray
            Predicted participation rate, shape ``(n_paths,)``.

        Raises
        ------
        SimulationOverrunError
            If the simulation has already reached the end of the
            allocated horizon.
        """
        if self._t >= self._horizon:
            raise SimulationOverrunError(
                f"Cannot step past horizon ({self._horizon} steps)"
            )

        # 1. Observe: route observations and compute features
        feature_values = []
        for route, fsim in zip(self._observation_routes, self.feature_sims):
            obs = route(chain_state, exog_paths)
            feature_values.append(fsim.step(obs))

        # 2. Log if requested
        if self.feature_log is not None:
            for i, fv in enumerate(feature_values):
                self.feature_log[:, self._t, i] = fv

        # 3. Respond: assemble feature vector and predict
        if feature_values:
            feature_matrix = np.column_stack(feature_values)
        else:
            # No features — predict from intercept only
            n_paths = chain_state.participation_rate.shape[0]
            feature_matrix = np.empty((n_paths, 0))

        current_p = chain_state.participation_rate[:, chain_state._t]
        response = self.model.predict(feature_matrix, current_p, rng)

        self._t += 1
        return response


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


@dataclass
class Simulator:
    """Orchestrates the simulation subsystems.

    Fields
    ------
    schedule : EmissionsSchedule
        The emissions update rule.
    model : _BaseParticipationModel
        A fitted participation model.
    exog_sampler : ExogenousSampler
        Sampler for exogenous variable forward paths.
    """

    schedule: EmissionsSchedule
    model: _BaseParticipationModel
    exog_sampler: ExogenousSampler

    def run(
        self,
        initial_state: SimulationState,
        n_paths: int,
        horizon: int,
        df: pl.DataFrame,
        rng: Generator,
        save_feature_history: bool = False,
    ) -> SimulationResult:
        """Run Monte Carlo simulation.

        Parameters
        ----------
        initial_state
            Starting chain state.
        n_paths
            Number of Monte Carlo paths.
        horizon
            Number of simulation steps.
        df
            Historical DataFrame for exogenous sampling and feature
            initialization.
        rng
            Numpy random generator.
        save_feature_history
            If True, the ParticipationSimulation will log feature values.

        Returns
        -------
        SimulationResult
        """
        # 1. Sample exogenous paths
        exog_col_names = [
            c for c in df.columns if c not in _CHAIN_STATE_COLS
        ]
        if exog_col_names:
            historical_exog = df.select(exog_col_names).to_numpy()
            exog_paths = self.exog_sampler.sample(
                historical_exog, n_paths, horizon, rng,
            )
        else:
            exog_paths = np.empty((n_paths, horizon, 0))

        # 2. Initialise subsystems
        chain = ChainStateSimulation.from_initial(
            initial_state, self.schedule, n_paths, horizon,
        )
        part = ParticipationSimulation.from_model(
            self.model, df, n_paths, horizon,
            save_history=save_feature_history,
        )

        # 3. Simulation loop
        for _ in range(horizon):
            response = part.step(chain, exog_paths, rng)
            chain.step(response)

        # 4. Return result
        return SimulationResult(
            participation_rate_paths=chain.participation_rate,
            emissions_rate_per_round_paths=chain.emissions_rate,
        )

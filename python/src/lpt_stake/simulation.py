"""Simulation components for the lpt_stake library.

Provides:

- Noise models: :class:`GaussianNoise`, :class:`BootstrapNoise`.
- Participation models: :class:`RawParticipationModel`,
  :class:`LogitParticipationModel`, :class:`DiffLogitParticipationModel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from lpt_stake.exogenous import block_bootstrap
from lpt_stake.features import np_expit, np_logit
from lpt_stake.model import fit_ridge

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.typing import NDArray

    from lpt_stake.model import RidgeResult
    from lpt_stake.types import Feature, NoiseModel


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

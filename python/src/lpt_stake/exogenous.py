"""Exogenous variable sampling for the lpt_stake simulation.

Provides two implementations of the
:class:`~lpt_stake.types.ExogenousSampler` protocol:

- :class:`BootstrapSampler` — block bootstrap from historical data.
- :class:`AR1Sampler` — fit per-feature AR(1) to historical data and
  sample forward.

Both return arrays of shape ``(n_paths, horizon, n_features)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Block bootstrap (shared logic)
# ---------------------------------------------------------------------------


def block_bootstrap(
    data: NDArray[np.floating],
    n_paths: int,
    horizon: int,
    block_size: int,
    rng: Generator,
) -> NDArray[np.floating]:
    """Block-bootstrap sample from a 1-D or 2-D historical array.

    Draws random contiguous blocks of length *block_size* from *data*,
    concatenates them, and trims to *horizon*.  Vectorised over paths.

    Parameters
    ----------
    data
        Historical data.  Shape ``(T,)`` for a single series or
        ``(T, k)`` for *k* series bootstrapped jointly.
    n_paths
        Number of independent sample paths.
    horizon
        Length of each sample path.
    block_size
        Length of each contiguous block.
    rng
        Numpy random generator.

    Returns
    -------
    NDArray
        Shape ``(n_paths, horizon)`` if *data* is 1-D, or
        ``(n_paths, horizon, k)`` if *data* is 2-D.
    """
    T = data.shape[0]
    n_blocks = int(np.ceil(horizon / block_size))
    max_start = T - block_size + 1

    # (n_paths, n_blocks) random start indices
    starts = rng.integers(0, max_start, size=(n_paths, n_blocks))

    # Build (n_paths, n_blocks * block_size, ...) then trim
    # Offsets within each block: 0, 1, ..., block_size-1
    offsets = np.arange(block_size)  # (block_size,)

    # (n_paths, n_blocks, block_size) absolute indices
    indices = starts[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]

    # Flatten blocks: (n_paths, n_blocks * block_size)
    indices = indices.reshape(n_paths, -1)[:, :horizon]

    # Gather from data
    return data[indices]


# ---------------------------------------------------------------------------
# BootstrapSampler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapSampler:
    """Block-bootstrap sampler for exogenous variables.

    Draws random contiguous blocks of length *block_size* from historical
    data, concatenates them, and trims to the requested horizon.

    Fields
    ------
    block_size : int
        Length of each contiguous block.
    """

    block_size: int

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
            Historical exogenous data, shape ``(T, n_features)``.
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
        return block_bootstrap(historical, n_paths, horizon, self.block_size, rng)


# ---------------------------------------------------------------------------
# AR1Sampler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AR1Sampler:
    """AR(1) sampler for exogenous variables.

    Fits a per-feature AR(1) model to the historical data and samples
    forward.  Each feature is modelled independently::

        x[t+1] = phi * x[t] + eps,  eps ~ N(0, sigma)

    where *phi* and *sigma* are estimated from the historical series by
    OLS.
    """

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
            Historical exogenous data, shape ``(T, n_features)``.
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
        T, n_features = historical.shape
        result = np.empty((n_paths, horizon, n_features))

        for j in range(n_features):
            series = historical[:, j]
            y_lag = series[:-1]
            y_curr = series[1:]

            # OLS estimate of phi
            denom = np.dot(y_lag, y_lag)
            if denom == 0.0:
                # Constant series
                phi = 1.0
                sigma = 0.0
            else:
                phi = np.dot(y_lag, y_curr) / denom
                sigma = float(np.std(y_curr - phi * y_lag, ddof=0))

            # Sample forward from last historical value
            noise = rng.normal(0.0, sigma, size=(n_paths, horizon)) if sigma > 0 else np.zeros((n_paths, horizon))
            paths = np.empty((n_paths, horizon))
            paths[:, 0] = phi * series[-1] + noise[:, 0]
            for t in range(1, horizon):
                paths[:, t] = phi * paths[:, t - 1] + noise[:, t]

            result[:, :, j] = paths

        return result

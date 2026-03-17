"""Simulation components for the lpt_stake library.

Provides noise model implementations:

- :class:`GaussianNoise` — i.i.d. Gaussian residual noise.
- :class:`BootstrapNoise` — block-bootstrap resample from fitted residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lpt_stake.exogenous import block_bootstrap

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.typing import NDArray


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

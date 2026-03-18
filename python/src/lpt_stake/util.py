"""Utility functions for the lpt_stake library.

Provides post-simulation computations that operate on
:class:`~lpt_stake.types.SimulationResult` arrays.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_total_supply_paths(
    initial_supply: float,
    emissions_rate_paths: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute total supply trajectories from emissions rate paths.

    Each round, total supply grows by the emissions rate:

    .. code-block:: text

        supply[t+1] = supply[t] * (1 + emissions_rate[t] / 1e9)

    The emissions rate is expected in parts per billion (the raw
    ``inflation`` field from the Minter contract), matching
    ``SimulationResult.emissions_rate_per_round_paths``.

    Parameters
    ----------
    initial_supply
        Total supply at the start of the simulation (e.g. in wei or
        any consistent unit).
    emissions_rate_paths
        Emissions rate per round in parts per billion, shape
        ``(n_paths, horizon + 1)``.

    Returns
    -------
    NDArray
        Total supply trajectories, shape ``(n_paths, horizon + 1)``.
        ``result[:, 0]`` is *initial_supply* for all paths.
    """
    growth_factors = 1.0 + emissions_rate_paths / 1e9
    cumulative = np.cumprod(growth_factors, axis=1)
    # cumprod gives cumulative product starting from column 0.
    # We want supply[0] = initial_supply, supply[t] = initial_supply * prod(growth[0:t]).
    # Shift right by one: prepend 1.0 column, drop last.
    result = np.empty_like(emissions_rate_paths)
    result[:, 0] = initial_supply
    result[:, 1:] = initial_supply * cumulative[:, :-1]
    return result

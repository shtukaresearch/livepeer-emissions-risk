"""Emissions schedule implementations for the lpt_stake simulation.

Each class implements the :class:`~lpt_stake.types.EmissionsSchedule` protocol:
given current emissions rate and participation rate (both as arrays of shape
``(n_paths,)``), return the next-round emissions rate.

Two implementations are provided:

- :class:`SignedStepSchedule` — the current Livepeer protocol rule, which
  steps the emissions rate up or down by a fixed amount each round depending
  on whether participation is below or above the target.  No floor or ceiling.
- :class:`ClampedSignedStepSchedule` — same signed-step rule with the
  addition of a floor and ceiling that clamp the emissions rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SignedStepSchedule:
    """Signed-step emissions schedule without bounds.

    This is the current Livepeer protocol rule.  Each round, the emissions
    rate moves by a fixed step whose sign depends on whether participation
    is below or above the target:

    .. code-block:: text

        I' = I + emissions_change * sign(target - P)

    When participation equals the target exactly, ``sign(0) == 0`` and
    the emissions rate is unchanged.

    Vectorised: operates on arrays of shape ``(n_paths,)`` for all Monte
    Carlo paths simultaneously.

    Fields
    ------
    target_participation_rate : float
        Target participation rate, in [0, 1].
    emissions_change : float
        Per-round step size in parts per billion.
    """

    target_participation_rate: float
    emissions_change: float

    def update(
        self,
        emissions_rate_per_round: NDArray[np.floating],
        participation_rate: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Return next-round emissions rate.

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
        step = self.emissions_change * np.sign(
            self.target_participation_rate - participation_rate
        )
        return emissions_rate_per_round + step


@dataclass(frozen=True)
class ClampedSignedStepSchedule:
    """Signed-step emissions schedule with floor and ceiling.

    Same rule as :class:`SignedStepSchedule`, but the result is clamped to
    ``[emissions_floor, emissions_ceiling]`` after each step:

    .. code-block:: text

        I' = clip(I + emissions_change * sign(target - P), floor, ceiling)

    Fields
    ------
    target_participation_rate : float
        Target participation rate, in [0, 1].
    emissions_change : float
        Per-round step size in parts per billion.
    emissions_floor : float
        Minimum emissions rate per round in parts per billion.
    emissions_ceiling : float
        Maximum emissions rate per round in parts per billion.
    """

    target_participation_rate: float
    emissions_change: float
    emissions_floor: float
    emissions_ceiling: float

    def __post_init__(self) -> None:
        """Validate that floor < ceiling."""
        if self.emissions_floor >= self.emissions_ceiling:
            raise ValueError(
                f"emissions_floor ({self.emissions_floor}) must be less than "
                f"emissions_ceiling ({self.emissions_ceiling})"
            )

    def update(
        self,
        emissions_rate_per_round: NDArray[np.floating],
        participation_rate: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Return next-round emissions rate, clamped to [floor, ceiling].

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
        step = self.emissions_change * np.sign(
            self.target_participation_rate - participation_rate
        )
        return np.clip(
            emissions_rate_per_round + step,
            self.emissions_floor,
            self.emissions_ceiling,
        )

"""lpt-stake: libraries for Livepeer emissions risk analysis."""

from lpt_stake.emissions_schedule import ClampedSignedStepSchedule, SignedStepSchedule
from lpt_stake.exogenous import AR1Sampler, BootstrapSampler
from lpt_stake.simulation import (
    BootstrapNoise,
    ChainStateSimulation,
    DiffLogitParticipationModel,
    GaussianNoise,
    LogitParticipationModel,
    ParticipationSimulation,
    RawParticipationModel,
    Simulator,
)
from lpt_stake.types import SimulationOverrunError, SimulationResult, SimulationState
from lpt_stake.util import compute_total_supply_paths

__all__ = [
    # Types
    "SimulationState",
    "SimulationResult",
    "SimulationOverrunError",
    # Emissions schedules
    "SignedStepSchedule",
    "ClampedSignedStepSchedule",
    # Exogenous samplers
    "BootstrapSampler",
    "AR1Sampler",
    # Noise models
    "GaussianNoise",
    "BootstrapNoise",
    # Participation models
    "RawParticipationModel",
    "LogitParticipationModel",
    "DiffLogitParticipationModel",
    # Simulation
    "ChainStateSimulation",
    "ParticipationSimulation",
    "Simulator",
    # Utilities
    "compute_total_supply_paths",
]

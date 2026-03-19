# `lpt_stake` Library Architecture

## Overview

`lpt_stake` is a Python library for Livepeer emissions risk analysis. It models how participants respond to protocol conditions and simulates forward trajectories of participation rate and emissions rate under Monte Carlo sampling.

The library separates concerns into:

- **Data preparation** — loading, reindexing, feature engineering (Polars)
- **Model fitting** — ridge regression with diagnostics (numpy)
- **Simulation** — three independent subsystems with their own state arrays

Notebooks are a thin presentation layer over the library.

## Module Map

```
src/lpt_stake/
    constants.py          Protocol-level numeric constants
    types.py              Dataclasses, protocols, SimulationOverrunError
    time.py               Round↔datetime conversion (linear estimate + Etherscan lookup)
    data.py               Raw data loading and daily-to-round reindexing
    features.py           Polars expression plugins, numpy logit/expit, design matrix assembly
    model.py              Ridge regression solver (closed-form, intercept unpenalised)
    emissions_schedule.py EmissionsSchedule implementations
    exogenous.py          ExogenousSampler implementations (block bootstrap, AR(1))
    simulation.py         ParticipationModel classes, simulation subsystems, Simulator
    util.py               Post-simulation utilities (total supply computation)
```

## Simulation Architecture

### Time Model

The simulation operates on Livepeer rounds. At each round *n*:

1. **Observe** — participants observe features based on current chain state and exogenous world state: `features(n) = f(chainstate(n), exog(n))`
2. **Respond** — the participation model predicts the population response: `response = predict(features(n)) + noise`
3. **Boundary** — at the round boundary, the response is applied and emissions are updated: `chainstate(n+1) = apply_emissions(apply_response(chainstate(n), response))`

The emissions schedule sees the **post-response** participation rate — the protocol adjusts emissions based on where participation actually ended up.

### Three Subsystems

Each subsystem owns its state arrays independently. The historical DataFrame is used to initialise all three and then set aside.

#### ChainStateSimulation

Owns the participation rate and emissions rate trajectories.

```python
ChainStateSimulation:
    participation_rate: NDArray   # (n_paths, horizon + 1)
    emissions_rate: NDArray       # (n_paths, horizon + 1)
    schedule: EmissionsSchedule   # held by reference
    _t: int                       # step counter
```

`step(response)` writes `participation_rate[:, _t+1] = response`, then computes `emissions_rate[:, _t+1]` via the schedule, and advances the counter.

#### ParticipationSimulation

Owns `FeatureSimulation` instances and produces the participation response at each step.

```python
ParticipationSimulation:
    model: ParticipationModel     # fitted model (for predict)
    feature_sims: list            # one FeatureSimulation per feature
    observation_routes: list      # how to read each feature's inputs
    feature_log: NDArray | None   # optional (n_paths, horizon, n_features)
    _t: int                       # step counter
```

`step(chain_state, exog_paths, rng)` routes observations from chain state or exogenous paths to each FeatureSimulation, assembles the feature vector, calls model.predict, and returns the response.

**Observation routing**: at construction time, each feature's column name is classified as chain state (`participation_rate`, `emissions_rate`) or exogenous (anything else). A routing table maps each feature to the correct source array and index.

#### Exogenous Paths

A plain numpy array `(n_paths, horizon, n_exog)` produced by the `ExogenousSampler` before the simulation loop. Read-only during simulation.

### Simulator

Orchestrates the subsystems:

```python
for _ in range(horizon):
    response = participation_sim.step(chain_state, exog_paths, rng)
    chain_state.step(response)
```

Both subsystems own step counters — the caller does not manage time indices.

## Participation Model

Models the population's participation response to world state over the course of a round. **Stateless with respect to data** — holds unbound features, target column name, fitted parameters, and noise model.

### Variants

Three concrete classes, differing only in target transform and inverse:

| Class | Target | Inverse | Output guarantee |
|---|---|---|---|
| `RawParticipationModel` | P(n+1) | clip to [0, 1] | [0, 1] |
| `LogitParticipationModel` | logit(P(n+1)) | expit | (0, 1) |
| `DiffLogitParticipationModel` | logit(P(n+1)) − logit(P(n)) | expit(logit(P) + ŷ) | (0, 1) |

### Fitting

`fit(df)` takes a round-indexed Polars DataFrame:

1. For each feature: if string, extract column; if `Feature` object, call `evaluate(df)`.
2. Compute target by applying the class-specific transform to the participation rate column.
3. Internally shift to align features at round *n* with target at round *n*+1. The caller does **not** pre-shift.
4. Assemble design matrix with intercept, call `fit_ridge`, store `RidgeResult`.

### Prediction

`predict(feature_vector, current_p, rng)`:

1. Prepend intercept to feature vector.
2. Apply ridge coefficients → ŷ.
3. Add noise from the noise model.
4. Apply class-specific inverse transform → P(n+1).

`current_p` is only used by `DiffLogitParticipationModel`.

## Feature System

Features can be:

- **String column names** — extracted directly from the DataFrame for fitting, passed through as-is during simulation.
- **Feature objects** — implement the `Feature` protocol with `evaluate(df)` for fitting and `make_simulation(historical_df, n_paths)` for simulation.

During simulation, each feature becomes a `FeatureSimulation` — a stateful accumulator that receives fresh observations each step and returns the feature value. Simple column features use a pass-through (no cache). Stateful features (e.g. trailing yield) maintain rolling accumulators.

## Emissions Schedule

Pluggable interface for the per-round emissions rate update rule.

- **`SignedStepSchedule`** — current Livepeer protocol rule: `I' = I + change * sign(target - P)`. No bounds.
- **`ClampedSignedStepSchedule`** — same rule with floor and ceiling: `I' = clip(I + change * sign(target - P), floor, ceiling)`.

Both are vectorised over `(n_paths,)`.

## Ridge Regression

`fit_ridge(dm, target_col, alpha)` — closed-form ridge with the intercept excluded from regularisation (Hastie et al., ESL Section 3.4.1).

Returns `RidgeResult` with:
- `coefficients`, `coefficient_std_errors` — dict keyed by column name
- `residuals`, `residual_std` — training residuals
- `effective_df` — trace of the hat matrix
- `aic`, `bic` — information criteria
- `predict(X)` — apply coefficients to a feature matrix

## Exogenous Sampling

Two implementations of the `ExogenousSampler` protocol:

- **`BootstrapSampler(block_size)`** — block bootstrap: draws random contiguous blocks from historical data, concatenates, trims to horizon.
- **`AR1Sampler()`** — fits per-feature AR(1) by OLS, samples forward.

Both return `(n_paths, horizon, n_features)`.

## Data Flow

```
                    ┌─────────────┐
                    │  DataFrame   │
                    │  (Polars)    │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
      ┌──────────┐  ┌───────────┐  ┌──────────┐
      │  Model   │  │  Exog     │  │  Initial │
      │  .fit()  │  │  Sampler  │  │  State   │
      └────┬─────┘  └─────┬─────┘  └────┬─────┘
           │              │              │
           ▼              ▼              ▼
      ┌────────────────────────────────────────┐
      │              Simulator.run()            │
      │                                        │
      │   exog_paths ← sampler.sample()        │
      │   chain ← ChainState.from_initial()    │
      │   part ← ParticipationSim.from_model() │
      │                                        │
      │   for each round:                      │
      │     response ← part.step(chain, exog)  │
      │     chain.step(response)               │
      │                                        │
      └────────────────┬───────────────────────┘
                       │
                       ▼
               ┌──────────────┐
               │ Simulation   │
               │ Result       │
               │  .participation_rate_paths   │
               │  .emissions_rate_per_round_paths │
               └──────────────┘
```

## Usage

### Minimal Example

```python
import polars as pl
from numpy.random import default_rng
from lpt_stake import (
    LogitParticipationModel, GaussianNoise, SignedStepSchedule,
    BootstrapSampler, SimulationState, Simulator,
)
from lpt_stake.features import logit, annualise_ppb
from lpt_stake.constants import ROUNDS_PER_YEAR

# 1. Load and prepare data
raw = pl.read_csv("data.csv", glob=False)
df = raw.select(
    pl.col("round").cast(pl.Int64),
    (pl.col("bonded") / pl.col("total-supply")).alias("participation_rate"),
    pl.col("inflation").alias("emissions_rate"),
    pl.col("fear_greed_index"),
).with_columns(
    pl.col("participation_rate").pipe(logit).alias("logit_participation"),
    pl.col("emissions_rate").pipe(annualise_ppb, ROUNDS_PER_YEAR).alias("inflation_annual"),
).drop_nulls()

# 2. Fit participation model
model = LogitParticipationModel(
    features=["logit_participation", "inflation_annual", "fear_greed_index"],
    target="participation_rate",
    alpha=0.1,
    noise=GaussianNoise(sigma=0.005),
)
model.fit(df)

# 3. Run simulation
sim = Simulator(
    schedule=SignedStepSchedule(target_participation_rate=0.5, emissions_change=500),
    model=model,
    exog_sampler=BootstrapSampler(block_size=10),
)
result = sim.run(
    initial_state=SimulationState(participation_rate=0.45, emissions_rate_per_round=687_000),
    n_paths=1000,
    horizon=500,
    df=df,
    rng=default_rng(42),
)

# 4. Analyse results
print(result.participation_rate_paths.shape)      # (1000, 501)
print(result.emissions_rate_per_round_paths.shape) # (1000, 501)
```

### Diagnostics

Use `ParticipationModel` directly for model comparison:

```python
from lpt_stake import RawParticipationModel, DiffLogitParticipationModel

# Compare model variants
for cls in [RawParticipationModel, LogitParticipationModel, DiffLogitParticipationModel]:
    m = cls(features=[...], target="participation_rate", alpha=0.1, noise=noise)
    m.fit(df_train)
    print(f"{cls.__name__}: AIC={m.ridge_result.aic:.2f}, residual_std={m.ridge_result.residual_std:.6f}")
```

### Notebooks

Two marimo notebooks are provided:

- **`notebook/inference.py`** — fit model, run simulation, display fan charts. Interactive sliders for alpha, noise, horizon, paths, seed.
- **`notebook/diagnostics.py`** — compare model variants with train/test split, coefficient tables, residual plots, alpha sweep.

Run with:
```bash
LPT_DATA_SOURCE=/path/to/data.csv uv run marimo edit notebook/inference.py
```

## Protocol Constants

| Constant | Value | Source |
|---|---|---|
| `SECONDS_PER_ETH_SLOT` | 12 | Ethereum consensus spec |
| `ETH_SLOTS_PER_ROUND` | 6377 | `RoundsManager.roundLength()` (post-LIP-83) |
| `SECONDS_PER_ROUND` | 76,524 | Derived |
| `ROUNDS_PER_YEAR` | ~412.5 | Derived (Julian year) |
| `REFERENCE_ROUND` | 4048 | Verified on-chain 2026-03-15 |
| `REFERENCE_DATETIME` | 2025-12-31T13:01:11Z | L1 block timestamp of round start |

## Extending

### Custom Emissions Schedule

Implement the `EmissionsSchedule` protocol:

```python
class MySchedule:
    def update(self, emissions_rate_per_round, participation_rate):
        # Both arrays shape (n_paths,), return shape (n_paths,)
        return ...
```

### Custom Features

Implement the `Feature` protocol for stateful features that need accumulators during simulation:

```python
class TrailingYield:
    columns = ["emissions_rate", "participation_rate"]

    def evaluate(self, df):
        # Compute over full DataFrame for fitting
        ...

    def make_simulation(self, historical, n_paths):
        # Return a FeatureSimulation with initialized cache
        ...
```

### Custom Noise Model

Implement the `NoiseModel` protocol:

```python
class MyNoise:
    def __call__(self, n, rng):
        # Return shape (n,)
        return ...
```

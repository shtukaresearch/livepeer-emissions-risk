# Design Plan: `lpt_stake` Library Restructuring

## Context

The Livepeer emissions risk analysis code currently lives in marimo notebooks (`ForecastTool_RoundBasis.py`, `emissions-history.py`) that interleave computational logic with UI controls. This makes the code hard to test, reuse, and iterate on. Several open GitHub issues (#2, #5, #6, #7, #13) call for cleaner separation of concerns.

The goal is to extract the computational pipeline into a proper Python library (`lpt_stake`) that notebooks can call as a thin presentation layer. The user's handwritten data flow diagram defines the pipeline stages: fetch → preprocess → feature engineering → fit → simulate → evaluate admissibility.

## Module Structure

```
src/lpt_stake/
    __init__.py              # Public API exports
    constants.py             # Protocol-level numeric constants (seconds per round, etc.)
    types.py                 # Dataclasses (SimulationState, SimulationResult) and Protocols
    time.py                  # Estimated round↔datetime conversion + real timestamp→block lookup
    data.py                  # Raw daily data loading and daily-to-round reindexing
    features.py              # Polars expression plugins, Feature/FeatureSimulation protocols, design matrix assembly
    model.py                 # Ridge regression fitting (closed-form, intercept unpenalised)
    emissions_schedule.py    # EmissionsSchedule protocol + SignedStepSchedule, ClampedSignedStepSchedule
    exogenous.py             # ExogenousSampler protocol + BootstrapSampler, AR1Sampler
    simulation.py            # ParticipationModel classes, ChainStateSimulation, ParticipationSimulation, Simulator, NoiseModel
    util.py                  # Post-simulation utilities (total supply computation)
```

## Key Types

### Dataclasses

- **`SimulationState`** — simulation starting point: `participation_rate` (float, [0,1]), `emissions_rate_per_round` (float, parts per billion).
- **`SimulationResult`** — simulation output: `participation_rate_paths` (ndarray), `emissions_rate_per_round_paths` (ndarray). Both shape `(n_paths, horizon+1)`.
- **`RidgeResult`** — fitted ridge model with coefficients, residuals, standard errors, effective df, AIC, BIC, and `.predict()`.
- **`SignedStepSchedule`** / **`ClampedSignedStepSchedule`** — the current Livepeer emissions rule (unbounded / with floor and ceiling).
- **`GaussianNoise`** / **`BootstrapNoise`** — concrete noise model implementations.
- **`ChainStateSimulation`** — owns participation rate and emissions rate trajectory arrays, step counter.
- **`ParticipationSimulation`** — owns FeatureSimulations and optional feature log, step counter.
- **`Simulator`** — orchestrates ChainStateSimulation, ParticipationSimulation, and exogenous sampling.

### Protocols (pluggable interfaces)

- **`Feature`** — stateful transform with column dependencies. `evaluate(df) -> ndarray` for fitting, `make_simulation(historical_df, n_paths) -> FeatureSimulation` for simulation. String column names are handled directly by the model.
- **`FeatureSimulation`** — `step(observation) -> ndarray`. Stateful accumulator for the simulation loop. Initialized from history, receives fresh observations each step.
- **`EmissionsSchedule`** — `update(emissions_rate_per_round, participation_rate) -> ndarray`.
- **`ParticipationModel`** — models population response to world state. Stateless w.r.t. data. Three concrete implementations: `RawParticipationModel`, `LogitParticipationModel`, `DiffLogitParticipationModel`. See Stage 4.
- **`NoiseModel`** — `__call__(n: int, rng: Generator) -> ndarray`.
- **`ExogenousSampler`** — `sample(historical, n_paths, horizon, rng) -> ndarray`.

## Data Flow and Contracts

### Stage 1: Load and reindex data (`data.py`)

This stage has two concerns: (a) ingesting raw daily data from various sources, and (b) reindexing from daily to round-indexed form.

#### 1a. Raw data ingestion

Raw data arrives as daily-resolution time series from multiple sources:

- **On-chain data** (`inflation`, `total-supply`, `bonded`): fetched from Arbitrum via Etherscan API + archive RPC node. The existing `script/fetch-data.py` does this and outputs daily JSON. The library should wrap this as `load_onchain_daily(path) -> DataFrame`.
- **Prices and volumes** (BTC, ETH, LPT): source and fetch method TBD — awaiting documentation of current manual process.
- **Fear & Greed index**: source is CoinMarketCap. Fetch method TBD — awaiting documentation of current manual process.

Functions:
- `load_onchain_daily(path) -> DataFrame` — load daily JSON/CSV from `fetch-data.py` output
- `load_exogenous_daily(path) -> DataFrame` — load pre-assembled exogenous CSV (prices, volumes, F&G)
- `merge_daily(onchain, exogenous) -> DataFrame` — join on date

> **TODO:** The exogenous data pipeline (prices, volumes, F&G) is currently an undocumented manual process. Once documented, replace `load_exogenous_daily` with proper fetch functions (e.g. `fetch_prices`, `fetch_fear_greed`) or keep the manual CSV path as a supported fallback.

#### 1b. Daily-to-round reindexing

Raw data is daily; the model operates on rounds. The library is responsible for this conversion using the constants and reference point from `constants.py`.

- `reindex_daily_to_rounds(daily_df, reference_round, reference_datetime) -> DataFrame` — assign each daily row to its nearest round, interpolate or forward-fill as needed for any gaps.

**Contract out:** DataFrame indexed by round. `inflation` column is raw ppb from chain (NOT annualized, fixing issue #2). `total_supply` and `bonded` in wei. Exogenous columns (prices, volumes, F&G) carried through from the daily merge.

### Stage 2: Feature engineering (`features.py`)

Features that don't depend on simulation output are pre-computed in Polars during data preparation and added as columns to the input DataFrame. The model constructor receives column names (strings) for these. For stateful features that depend on simulation output (e.g. trailing yield), the user passes a `Feature` object.

#### Polars expression plugins

These take and return `pl.Expr`, composing with `.pipe()`. Used during data preparation in notebooks:

- `logit(expr)` — `log(x / (1-x))`, for participation rate
- `expit(expr)` — inverse logit, `1 / (1 + exp(-x))`
- `annualise_ppb(expr, rounds_per_year)` — `(1 + x/1e9)^rounds_per_year - 1`

These are for data preparation only. The model operates on numpy arrays.

#### Feature protocol

See Stage 4 for the full `Feature` and `FeatureSimulation` protocol definitions. In summary:

- A `Feature` object carries column dependencies and parameters for a stateful transform (e.g. trailing yield). It provides `evaluate(df)` for fitting (stateless, over a full DataFrame) and `make_simulation(historical_df, n_paths)` to construct a `FeatureSimulation` for the simulation loop.
- Plain column names (strings) are handled directly by `ParticipationModel` — no `Feature` object needed.

#### How the model processes features

The `ParticipationModel` receives a list where each element is either a column name (string) or a `Feature` object:

```python
model = LogitParticipationModel(
    features=[
        "participation_rate",     # pre-computed column
        "inflation_annual",       # pre-computed: annualise_ppb applied in data prep
        "fear_greed",             # raw column
        trailing_yield("inflation", "participation_rate", window=412),  # Feature
    ],
    target="participation_rate",
    alpha=0.1,
    noise=GaussianNoise(0.01),
)
model.fit(round_df)
```

In `fit(df)`, for each element:
1. **String**: extract column from df as numpy.
2. **`Feature` object**: call `feature.evaluate(df)` to get a numpy array.

Assemble into design matrix with intercept, compute target, call `fit_ridge`.

> **Note — future convenience methods:** It will be useful to allow certain lightweight transformations to be passed in the features list without pre-computing columns. For example, `lag("fear_greed", n)` would let the user iterate over different lag values without creating N differently-named columns in the DataFrame. These can be added as convenience factories that return Feature objects. Not needed now, but the `Feature` protocol supports this cleanly.

Derived columns that don't depend on simulation output (ratios, rescaling, etc.) are computed in Polars during data preparation and added as new columns to the input DataFrame before model construction.

#### Design matrix assembly

`build_design_matrix` is retained as a utility for interactive use in the diagnostics notebook. The `ParticipationModel.fit()` method assembles the design matrix internally from feature evaluations.

No `train_test_split` in the library — that's a diagnostic notebook concern (see Notebook Structure below).

### Stage 3: Fit model (`model.py`)

`fit_ridge(dm, target_col, alpha) -> RidgeResult` — closed-form ridge regression with the intercept excluded from regularisation (Hastie et al., ESL Section 3.4.1). Returns `RidgeResult` with `.predict()`, `.coefficients`, `.residuals`, `.residual_std`, `.coefficient_std_errors`, `.effective_df`, `.aic`, `.bic`.

Alpha selection is not in the library — diagnostic notebook concern.

`fit_ridge` is a low-level function. In the simulation pipeline, `ParticipationModel.fit(df)` evaluates features and target from the DataFrame, then delegates to `fit_ridge`. The diagnostics notebook can also call `fit_ridge` directly for exploring configurations.

**Contract out:** `RidgeResult`. The simulation stage needs `.predict()` and `.residuals` only. The diagnostics notebook uses the full interface.

### Stage 4: Simulate (`simulation.py`, `emissions_schedule.py`, `exogenous.py`)

#### Time model

At the beginning of round n, participants observe features based on the current chain state and exogenous world state. They react over the course of round n, yielding a response. At the boundary of rounds n and n+1, the response is applied to chain state and emissions are computed:

1. **Observe**: features(n) = f(chainstate(n), exog(n))
2. **Respond**: response = predict(features(n)) + noise
3. **Boundary**: chainstate(n+1) = apply\_emissions(apply\_response(chainstate(n), response))

Note: the on-chain smart contracts use the term `inflation` for the emissions rate. We use "emissions" everywhere in our own code and documentation, and only use `inflation` when referring directly to the contract field.

#### ParticipationModel

Models the population response to world state over the course of a round. Stateless with respect to data — holds unbound features, target column name, model parameters, noise model.

**Three concrete classes**, differing only in target transform and inverse:

- **`RawParticipationModel`** — target is P(n+1). Inverse is identity.
- **`LogitParticipationModel`** — target is logit(P(n+1)). Inverse is expit.
- **`DiffLogitParticipationModel`** — target is logit(P(n+1)) − logit(P(n)). Inverse is expit(logit(P(n)) + ŷ).

The target transform is internal to the model class — not a user-supplied callable, and not part of the Feature system. The model's output is a *response* — the participation response to current conditions. It is not itself state; it is an impulse applied to chain state.

Constructor:

```python
model = LogitParticipationModel(
    features=["logit_participation", "inflation_annual", "fear_greed"],
    target="participation_rate",
    alpha=0.1,
    noise=GaussianNoise(sigma=0.01),
)
model.fit(round_df)
```

The model holds:
- `features`: list of column names (strings) and/or `Feature` objects
- `target`: name of the participation rate column
- `alpha`: ridge penalty
- `noise`: NoiseModel
- `ridge_result`: populated by `fit()`, `None` before

`fit(df)`:
1. For each feature: if string, extract column from df as numpy; if Feature, call `feature.evaluate(df)`. Assemble into design matrix with intercept.
2. Compute target: extract target column, apply class-specific transform (identity / logit / diff-logit), shift as needed.
3. Call `fit_ridge`. Store `RidgeResult`.

`predict(feature_vector, current_p, rng)`:
Apply ridge coefficients to feature vector, add noise, apply class-specific inverse transform. `current_p` is needed only by `DiffLogitParticipationModel` (to compute expit(logit(P) + ŷ)). Returns predicted P(n+1) of shape `(n_paths,)`.

The diagnostics notebook uses ParticipationModel directly: inspect coefficients, residuals, compare model variants, evaluate on held-out data.

Construction in the inference notebook:

```python
model = LogitParticipationModel(
    features=[
        "logit_participation",   # pre-computed in data prep
        "inflation_annual",      # pre-computed in data prep
        "fear_greed",
    ],
    target="participation_rate",
    alpha=0.1,
    noise=GaussianNoise(sigma=0.01),
)
model.fit(round_df)

sim = Simulator(schedule=SignedStepSchedule(...), model=model)
result = sim.run(
    initial_state=SimulationState(participation_rate=0.45, emissions_rate_per_round=38),
    n_paths=1000, horizon=500,
    df=round_df,
    exog_sampler=BootstrapSampler(block_size=10),
    rng=default_rng(42),
)
```

#### Feature protocol

A `Feature` object represents a stateful transform that depends on multiple columns and/or requires internal accumulators (e.g. trailing yield). It carries the parameters needed to evaluate over a DataFrame (fitting) and to construct a `FeatureSimulation` (simulation).

```python
class Feature(Protocol):
    columns: list[str]

    def evaluate(self, df: pl.DataFrame) -> NDArray:
        """Evaluate feature over a full DataFrame. Used by fit()."""
        ...

    def make_simulation(
        self, historical: pl.DataFrame, n_paths: int,
    ) -> FeatureSimulation:
        """Construct a FeatureSimulation initialized from history."""
        ...
```

For plain column names (strings), `ParticipationModel` handles both paths directly: extract column for fitting, wrap in a pass-through `FeatureSimulation` for simulation.

#### FeatureSimulation

Stateful feature accumulator for the simulation loop. Initialized from historical data to set up cache shape and fill lookback. At each step, receives fresh observations and returns the feature value. Does not hold a reference to any domain array — it receives observations as arguments and maintains its own internal accumulators.

```python
class FeatureSimulation(Protocol):
    def step(self, observation: NDArray) -> NDArray:
        """Receive fresh observation, update cache, return feature value.

        Parameters
        ----------
        observation
            Current-step values for this feature's input columns.
            Shape (n_paths,) for single-column, (n_paths, k) for multi-column.

        Returns
        -------
        NDArray
            Feature value, shape (n_paths,).
        """
        ...
```

For simple column features: pass-through, returns the observation unchanged, no cache.

For stateful features (e.g. trailing yield): maintains a rolling accumulator initialized from history. `step()` appends the observation, drops the oldest value, returns the updated statistic.

#### Three simulation subsystems

Each subsystem owns its state. The DataFrame is used to initialise all three and then set aside.

##### ChainStateSimulation

Participation rate and emissions rate trajectories. Owns its step counter.

```python
@dataclass
class ChainStateSimulation:
    participation_rate: NDArray   # (n_paths, horizon + 1)
    emissions_rate: NDArray       # (n_paths, horizon + 1)
    _t: int                       # current step index
```

`from_initial(initial_state, n_paths, horizon)`: allocates arrays, sets `[:, 0]` from `SimulationState`, `_t = 0`.

`step(response, schedule)`:
```python
self.participation_rate[:, self._t + 1] = response
self.emissions_rate[:, self._t + 1] = schedule.update(
    self.emissions_rate[:, self._t],
    self.participation_rate[:, self._t + 1],
)
self._t += 1
```

The schedule sees the post-response participation rate — the protocol adjusts emissions based on where participation ended up.

##### ParticipationSimulation

Owns `FeatureSimulation` instances and optionally a feature log. Constructed from a fitted `ParticipationModel`. Owns its step counter.

```python
@dataclass
class ParticipationSimulation:
    model: ParticipationModel
    feature_sims: list[FeatureSimulation]
    observation_routes: list[...]  # how to read each feature's inputs
    feature_log: NDArray | None    # (n_paths, horizon, n_features), optional
    _t: int                        # current step index
```

`from_model(model, historical_df, n_paths, horizon, save_history=False)`:
1. For each feature in the model: if string, create pass-through `FeatureSimulation`; if `Feature`, call `feature.make_simulation(historical_df, n_paths)`.
2. Classify each feature's columns as chain state (`participation_rate`, `emissions_rate`) or exogenous. Build observation routing table.
3. If `save_history`, allocate `feature_log`.

`step(chain_state, exog_paths, rng)`:
1. **Observe**: for each FeatureSimulation, route the appropriate observation from `chain_state` or `exog_paths` at current indices, call `step(observation)`, collect feature values.
2. If logging, write into `feature_log[:, _t, :]`.
3. **Respond**: assemble feature vector with intercept, call `model.predict(feature_vector, current_p, rng)`.
4. `_t += 1`.
5. Return response, shape `(n_paths,)`.

##### Observation routing

At `from_model` time, each feature's column name(s) are classified:
- `"participation_rate"` → read from `chain_state.participation_rate[:, chain_state._t]`
- `"emissions_rate"` → read from `chain_state.emissions_rate[:, chain_state._t]`
- Anything else → exogenous, read from `exog_paths[:, _t, j]` where j is resolved at construction

This mapping is stored as `observation_routes` and used by `step()`.

#### Simulator

Orchestrates the subsystems.

```python
@dataclass
class Simulator:
    schedule: EmissionsSchedule
    model: ParticipationModel
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
        # 1. Sample exogenous paths
        exog_cols = [col for col in self._exogenous_columns(df)]
        historical_exog = df.select(exog_cols).to_numpy()
        exog_paths = self.exog_sampler.sample(
            historical_exog, n_paths, horizon, rng,
        )

        # 2. Initialise subsystems
        chain = ChainStateSimulation.from_initial(
            initial_state, n_paths, horizon,
        )
        part = ParticipationSimulation.from_model(
            self.model, df, n_paths, horizon,
            save_history=save_feature_history,
        )

        # 3. Simulation loop
        for _ in range(horizon):
            response = part.step(chain, exog_paths, rng)
            chain.step(response, self.schedule)

        # 4. Return result
        return SimulationResult(
            participation_rate_paths=chain.participation_rate,
            emissions_rate_per_round_paths=chain.emissions_rate,
        )
```

The Simulator receives the full DataFrame, extracts exogenous columns (everything that isn't `participation_rate` or `emissions_rate`) for the sampler, and passes the DataFrame to `ParticipationSimulation.from_model` for feature initialization.

**Contract out:** `SimulationResult` with `participation_rate_paths` in natural [0,1] units and `emissions_rate_per_round_paths` in parts per billion. Both arrays of shape `(n_paths, horizon+1)`.

#### Noise models — IMPLEMENTED

```python
class NoiseModel(Protocol):
    def __call__(self, n: int, rng: Generator) -> NDArray:
        """Draw n residual noise samples."""
        ...
```

- **`GaussianNoise(sigma)`** — `rng.normal(0, sigma, n)`
- **`BootstrapNoise(residuals, block_size)`** — block-bootstrap resample from fitted residuals

#### Emissions schedule (`emissions_schedule.py`) — IMPLEMENTED

`SignedStepSchedule` and `ClampedSignedStepSchedule`. No changes needed.

#### Exogenous variable sampling (`exogenous.py`) — IMPLEMENTED

- `BootstrapSampler(block_size)` — block bootstrap from historical exogenous data
- `AR1Sampler()` — fit AR(1) to historical data, sample forward

Both implement the `ExogenousSampler` protocol. No changes needed.

### Stage 5: Utilities (`util.py`)

Post-simulation utility functions.  In principle total supply tracking belongs in `ChainStateSimulation`; for now it is a standalone function.

- `compute_total_supply_paths(initial_supply: float, emissions_rate_paths: ndarray) -> ndarray` — cumulative product: `supply[t+1] = supply[t] * (1 + emissions_rate[t] / 1e9)`. Shape `(n_paths, horizon+1)`. Emissions rate is in parts per billion, matching `SimulationResult`.

Other derived quantities (yield, dilution, etc.) are left to notebooks — they are one-liners in numpy and different analyses may define them differently.

Admissibility is not formalised in the library. It is a predicate over the simulation results: the notebook computes statistics (quantiles, exceedance probabilities), evaluates inequalities, and decides pass/fail.

## Constants (`constants.py`)

Protocol-level numeric constants only:

```
SECONDS_PER_ETH_SLOT = 12
ETH_SLOTS_PER_ROUND = 6377
SECONDS_PER_ROUND = SECONDS_PER_ETH_SLOT * ETH_SLOTS_PER_ROUND  # 76524
ROUNDS_PER_YEAR ≈ 365.25 * 86400 / SECONDS_PER_ROUND ≈ 412.5
```

Reference point for estimated time conversion (verified on-chain 2026-03-15 via
`RoundsManager.currentRound()` and `currentRoundStartBlock()`; start block
L1 24132771 resolved to timestamp via Etherscan):

```
REFERENCE_ROUND = 4048
REFERENCE_DATETIME = datetime(2025, 12, 31, 13, 1, 11, tzinfo=timezone.utc)
```

### Historical notes on round length

The `roundLength` parameter has changed once in protocol history:

- **Original value: 5760** L1 blocks. At the pre-Merge average block time of ~13.29s, this gave rounds of ~76,525s (~21.3h).
- **LIP-83 (2022-10-06): changed to 6377** L1 blocks. The Ethereum Merge (2022-09-15) fixed block time at 12s, which would have shortened rounds by ~10%. LIP-83 adjusted `roundLength` to preserve the historical round duration: 76,525s / 12s ≈ 6377.

The on-chain round number is computed by the `RoundsManager` contract as:

```
currentRound = lastRoundLengthUpdateRound + (blockNum - lastRoundLengthUpdateStartBlock) / roundLength
```

where `blockNum` is `block.number`. On Arbitrum, `block.number` currently returns a value that tracks the L1 block number (synced approximately every minute). If Arbitrum changes this semantic in a future ArbOS upgrade, the protocol would likely need another `setRoundLength` call, and we would need to update our constants and reference point.

## Round/Time Conversion (`time.py`)

Round↔datetime conversion is handled within the library using a linear approximation anchored to the reference point in `constants.py`. This replaces the earlier plan (issue #13) to develop a separate time-conversion library.

### Estimated conversions (linear approximation)

These use `SECONDS_PER_ROUND` and the reference point. Function names include "estimate" to make clear they are approximations — real round boundaries depend on actual L1 block production, which is not perfectly periodic.

- `estimate_round_number(dt) -> float` — estimate the round number at a given datetime
- `estimate_datetime(round_num) -> datetime` — estimate the datetime at which a given round starts
- `estimate_round_count(delta) -> float` — convert a timedelta to an estimated number of rounds (no reference point needed, pure arithmetic)
- `estimate_timedelta(rounds) -> timedelta` — convert a number of rounds to an estimated timedelta (no reference point needed)

### Real timestamp → block number conversion

The Etherscan API provides an authoritative timestamp→block lookup. This is used by the data-fetching pipeline (`fetch-data.py`) and is not an estimate. `time.py` takes ownership of this function so that `fetch-data.py` imports it rather than defining it inline.

- `fetch_block_at_timestamp(timestamp, api_key) -> int` — query Etherscan for the Arbitrum block number closest to a given timestamp

### Where each conversion is used

1. **Data ingestion** (Stage 1) — `reindex_daily_to_rounds` uses `estimate_round_number` to assign each daily row to its nearest round
2. **Simulation horizon** — notebooks convert a user-supplied end date to a round count using `estimate_round_count`
3. **Notebook visualization** — `estimate_datetime` generates calendar-date tick labels for round-indexed plots
4. **Data fetching** — `fetch_block_at_timestamp` is the real (non-estimated) conversion, used only in the data collection pipeline

### Accuracy of the linear estimate

The estimate assumes rounds are exactly `SECONDS_PER_ROUND` apart, which is only approximately true — real rounds depend on L1 block production, which has some jitter. Over short horizons the error is small. Over long historical ranges spanning the LIP-83 boundary (before 2022-10-06), the estimate will be systematically wrong because the round duration was different. For current purposes this is acceptable: historical data is daily-resolution (~1.1 rounds/day), so sub-round timing errors do not affect round assignment. If higher accuracy is needed in future (e.g. per-round historical data, or fitting `SECONDS_PER_ROUND` from observed data), this module is the place to extend.

## Notebook Structure

The current `ForecastTool_RoundBasis.py` tries to be an interactive playground and a forecasting tool simultaneously, which leads to UI contortions (e.g. setting the "validation" window to zero to get a genuine forecast). Split into separate notebooks with distinct purposes:

### 1. Diagnostics / feature selection notebook

Exploratory. Iterate over candidate configurations (feature sets, transforms, differencing, alpha values) and compare them side-by-side. For each combination: fit on a training window, evaluate on held-out data, report AIC/RMSE/coefficient stability. Output is a comparison table. Train/test splitting lives here, not in the library. Can use `fit_ridge` directly or construct `ParticipationModel` instances to compare model types.

### 2. Inference / forecasting notebook

Takes a fixed configuration as input: feature specs, model class (Raw/Logit/DiffLogit), alpha, noise model, protocol params, risk objectives. Fits on all available data, runs simulation, evaluates admissibility. This is the "production" notebook — no exploratory UI, just a clear pipeline from config to results.

### 3. Historical exploration notebook

`emissions-history.py` already serves this role — visualising raw data, trends, protocol behaviour over time.

## What's NOT Included

- **Loss function** (issue #8 breach-count loss) — deferred
- **Feature convenience factories** (e.g. `lag("col", n)`, combinators) — future iteration once patterns stabilise
- **Plotting/visualization** — stays in notebooks
- **Pandas support** — Polars for data loading, numpy for simulation
- **Daily-resolution pipeline** — November notebook is deprecated

## Dependency Changes to `pyproject.toml`

Library core deps: `polars`, `numpy`

Optional: `requests` (for Etherscan API in `time.py`), `web3` (for on-chain fetching in `data.py`)

Move to dev/notebook deps: `marimo`, `altair`, `matplotlib`, `arrow`, `pytz`, `pyzmq`

Remove: `pandas` (replaced by `polars` throughout), `statsmodels` (replaced by closed-form solver in `model.py`)

## Implementation Order

1. `constants.py` + `types.py` — foundational, no deps — DONE
2. `time.py` — estimated round↔datetime conversion, real timestamp→block lookup — DONE
3. `features.py` — Polars expression plugins (logit, expit, annualise_ppb), build_design_matrix — DONE
4. `data.py` — daily data loading and round reindexing — DONE
5. `model.py` — ridge fitting, closed-form solver — DONE
6. `emissions_schedule.py` — `EmissionsSchedule` protocol + `SignedStepSchedule`, `ClampedSignedStepSchedule` — DONE
7. `exogenous.py` — `ExogenousSampler` protocol + `BootstrapSampler`, `AR1Sampler` — DONE
8. `simulation.py` — `NoiseModel` implementations (`GaussianNoise`, `BootstrapNoise`) — DONE
9. `types.py` + `features.py` revision — replace `BoundFeature`/`BoundColumnFeature`/`ParticipationModel` protocol with new `Feature`/`FeatureSimulation` protocols. Delete `BoundColumnFeature`. — DONE
10. `simulation.py` — `ParticipationModel` classes (Raw, Logit, DiffLogit) + tests — DONE
11. `simulation.py` — `ChainStateSimulation`, `ParticipationSimulation`, `Simulator` + tests — DONE
12. `util.py` — `compute_total_supply_paths` + tests — DONE
13. `__init__.py` + `pyproject.toml` — public API exports, dependency cleanup — DONE
14. Build the inference and diagnostics notebooks using the library

## Verification

- Unit tests for each module using pytest (in `tests/` directory)
- Key things to test: forward/inverse transform roundtrip, ParticipationModel fit/predict, ChainStateSimulation step mechanics, observation routing, simulation reproducibility with fixed seed, derived quantity computations against hand-calculated examples
- Integration test: run full pipeline (load sample data → features → fit → simulate → derive) and check output types and shapes
- Regression test: compare library output against current notebook output for a known parameter set to ensure extraction didn't change behaviour

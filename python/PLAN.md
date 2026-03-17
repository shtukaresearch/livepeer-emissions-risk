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
    features.py              # Feature protocol, ColumnFeature, numpy transforms, design matrix assembly
    model.py                 # Ridge regression fitting (closed-form, intercept unpenalised)
    emissions_schedule.py    # EmissionsSchedule protocol + SignedStepSchedule, ClampedSignedStepSchedule
    exogenous.py             # ExogenousSampler protocol + BootstrapSampler, AR1Sampler
    simulation.py            # ParticipationModel classes, NoiseModel, Simulator, Ω management
    derived.py               # Post-simulation derived quantities (total supply, yield, dilution)
```

## Key Types

### Dataclasses

- **`SimulationState`** — simulation starting point: `participation_rate` (float, [0,1]), `emissions_rate_per_round` (float, parts per billion).
- **`SimulationResult`** — simulation output: `participation_rate_paths` (ndarray), `emissions_rate_per_round_paths` (ndarray). Both shape `(n_paths, horizon+1)`.
- **`RidgeResult`** — fitted ridge model with coefficients, residuals, standard errors, effective df, AIC, BIC, and `.predict()`.
- **`SignedStepSchedule`** / **`ClampedSignedStepSchedule`** — the current Livepeer emissions rule (unbounded / with floor and ceiling).
- **`GaussianNoise`** / **`BootstrapNoise`** — concrete noise model implementations.
- **`Simulator`** — allocates sample path arrays, composes an `EmissionsSchedule` and a `ParticipationModel`, runs simulation loop.

### Protocols (pluggable interfaces)

- **`Feature`** — unbound transform with column dependencies. `bind(data) -> BoundFeature`. For stateful features (e.g. trailing yield). String column names are wrapped internally by the model constructor.
- **`BoundFeature`** — `evaluate(end, cache) -> ndarray`, `step(t) -> ndarray`, `rebind(data) -> BoundFeature`. Holds reference to data array.
- **`EmissionsSchedule`** — `update(emissions_rate_per_round, participation_rate) -> ndarray`.
- **`ParticipationModel`** — central simulation object. Owns features, fitting, and prediction. Three concrete implementations: `RawParticipationModel`, `LogitParticipationModel`, `DiffLogitParticipationModel`. See Stage 4.
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

#### Feature and BoundFeature protocols

A `Feature` is an unbound transform with column dependencies that can be bound to data. A `BoundFeature` is the bound version — it holds a reference to data and can compute values either vectorised or incrementally.

```python
class Feature(Protocol):
    """An unbound feature with column dependencies.

    Stateful features (e.g. trailing_yield) implement this protocol
    and carry their column dependencies as column name strings.
    """

    def bind(self, data: NDArray) -> BoundFeature:
        """Bind to column data.

        Parameters
        ----------
        data
            Column data sliced by the caller: shape (T,) for fitting,
            (n_paths, T) for simulation.  For multi-column features,
            shape (T, k) or (n_paths, T, k).
        """
        ...

class BoundFeature(Protocol):
    def evaluate(self, end: int | None = None, cache: bool = False) -> NDArray:
        """Compute feature values over the data, up to index end.

        With end=None: full time axis (fitting).
        With end=t0, cache=True: compute up to t0 and prepare
        intermediate state for subsequent step() calls.

        1D data: returns shape (T,) or (end,).
        2D data: returns shape (n_paths, T) or (n_paths, end).
        """
        ...

    def step(self, t: int) -> NDArray:
        """Feature value at time t (incremental).
        Scalar for 1D data, shape (n_paths,) for 2D data."""
        ...

    def rebind(self, data: NDArray) -> BoundFeature:
        """Create a new BoundFeature on different data, same transform."""
        ...
```

#### How the model constructor processes features

The `ParticipationModel` constructor receives a list where each element is either a column name (string) or a `Feature` object:

```python
model = LogitParticipationModel(
    features=[
        "participation_rate",     # pre-computed column (could be logit-transformed in data prep)
        "inflation_annual",       # pre-computed: annualise_ppb applied in data prep
        "fear_greed",             # raw column
        trailing_yield("inflation", "participation_rate", window=412),  # stateful Feature
    ],
    df=round_df,
    alpha=0.1,
    noise=GaussianNoise(0.01),
)
```

For each element:
1. **String**: the constructor resolves the column name against `df.columns`, slices the column from the numpy array, and wraps in an internal `BoundColumnFeature` (identity transform).
2. **`Feature` object** (e.g. from `trailing_yield(...)`): the constructor inspects its column dependencies, slices the relevant columns, and calls `feature.bind(data)`.

Column name resolution happens once in the constructor. After that, everything is numpy arrays and integer indices.

> **Note — future convenience methods:** It will be useful to allow certain lightweight transformations to be passed in the features list without pre-computing columns. For example, `lag("fear_greed", n)` would let the user iterate over different lag values without creating N differently-named columns in the DataFrame. These can be added as convenience factories that return Feature objects. Not needed now, but the `Feature` protocol supports this cleanly.

#### Concrete bound feature classes

**`BoundColumnFeature`** (internal) — Stateless. Holds a reference to column data and an optional numpy transform. `evaluate` returns `self.transform(self.data[..., :end])`. `step(t)` returns `self.transform(self.data[..., t])`. Cache is ignored (nothing to cache). Covers the vast majority of use cases.

**Stateful bound features** (e.g. trailing compounded yield) — For the rare case where a feature depends on simulation-generated quantities over a rolling window. `evaluate(end=t0, cache=True)` computes up to t0 and saves intermediate accumulators. `step(t)` updates the cached state incrementally. Reference the data array directly for lookback — no copying. For current needs, only trailing yield is anticipated; others are a future iteration.

Derived columns that don't depend on simulation output (ratios, rescaling, etc.) are computed in Polars during data preparation and added as new columns to the input DataFrame before model construction.

#### Design matrix assembly

`build_design_matrix` is retained as a utility for interactive use in the diagnostics notebook. The `ParticipationModel.fit()` method uses `BoundFeature.evaluate()` to assemble the design matrix internally.

No `train_test_split` in the library — that's a diagnostic notebook concern (see Notebook Structure below).

### Stage 3: Fit model (`model.py`)

`fit_ridge(dm, target_col, alpha) -> RidgeResult` — closed-form ridge regression with the intercept excluded from regularisation (Hastie et al., ESL Section 3.4.1). Returns `RidgeResult` with `.predict()`, `.coefficients`, `.residuals`, `.residual_std`, `.coefficient_std_errors`, `.effective_df`, `.aic`, `.bic`.

Alpha selection is not in the library — diagnostic notebook concern.

`fit_ridge` is a low-level function. In the simulation pipeline, the `ParticipationModel.fit()` method prepares features and target from `sample_path_historic`, then delegates to `fit_ridge`. The diagnostics notebook can also call `fit_ridge` directly for exploring configurations.

**Contract out:** `RidgeResult`. The simulation stage needs `.predict()` and `.residuals` only. The diagnostics notebook uses the full interface.

### Stage 4: Simulate (`simulation.py`, `emissions_schedule.py`, `exogenous.py`)

#### Sample path arrays

Two data arrays carry state through the pipeline:

- **`sample_path_historic`** — 2D numpy array `(T, n_cols)`. Converted from the round-indexed Polars DataFrame by the ParticipationModel constructor. Columns match the DataFrame column order. Used for fitting.

- **`sample_path_simulated`** — 3D numpy array `(n_paths, lookback + horizon + 1, n_cols)`. Allocated by the Simulator. The first `lookback` rows contain the tail of `sample_path_historic`, broadcast to all paths. `lookback` is the maximum lookback required by any feature (e.g. ~412 for 1-year trailing yield). The overhead of broadcasting is negligible at our data sizes (few hundred rows, <10 columns).

Columns are identified by index (resolved from DataFrame column names at model construction time):

| Index | Column | Filled by |
|-------|--------|-----------|
| 0 | participation_rate (P) | model.predict_next() |
| 1 | emissions_rate_per_round (I) | schedule.update() |
| 2.. | exogenous variables | Pre-filled before loop |

Exogenous columns are pre-filled for all time steps before the simulation loop: historical values from data, future values from the `ExogenousSampler`.

BoundFeatures hold a reference to whichever array they are bound to and read from it by column index. No copies — features index directly into the array for both current values and historical lookback.

#### State taxonomy

- **Chain state** (P, I): written by the model and schedule at each step.
- **Feature state**: internal to each BoundFeature. Accumulators for stateful features (e.g. running product for trailing yield). Not stored in the sample path array.
- **Exogenous worldstate**: pre-sampled forward paths for external variables. Read-only during the simulation loop.
- **Response**: model output at each step. Applied to chain state, not persisted separately.

Note: the on-chain smart contracts use the term `inflation` for the emissions rate. We use "emissions" everywhere in our own code and documentation, and only use `inflation` when referring directly to the contract field.

#### Emissions schedule state machine (`emissions_schedule.py`) — IMPLEMENTED

A pluggable interface for the rule that updates the emissions rate each round. This is the component to swap out when exploring alternative emissions schedules.

```python
class EmissionsSchedule(Protocol):
    def update(self, emissions_rate_per_round: ndarray, participation_rate: ndarray) -> ndarray:
        """Given current emissions rate and participation, return next-round emissions rate."""
        ...
```

Two implementations are provided:

- **`SignedStepSchedule`** — the current Livepeer protocol rule: step up or down by a fixed amount each round depending on whether participation is below or above the target. No bounds.
- **`ClampedSignedStepSchedule`** — same signed-step rule with the addition of a floor and ceiling.

Vectorised — operates on arrays of shape `(n_paths,)` for all Monte Carlo paths simultaneously.

#### Exogenous variable sampling (`exogenous.py`)

- `BootstrapSampler(block_size)` — block bootstrap from historical exogenous data
- `AR1Sampler()` — fit AR(1) to historical data, sample forward

Both implement:

```python
class ExogenousSampler(Protocol):
    def sample(self, historical: pl.DataFrame, n_paths: int, horizon: int,
               rng: Generator) -> ndarray:
        """Return array of shape (n_paths, horizon, n_features)."""
        ...
```

The Simulator pre-fills the exogenous columns of `sample_path_simulated` from the sampler output before entering the simulation loop.

#### Participation model (`simulation.py`)

The ParticipationModel is the central object in the pipeline. It:

1. Receives features as column names (strings) or `Feature` objects, resolves column dependencies against the DataFrame, converts to numpy, and binds each to produce BoundFeatures
2. Owns the ridge model — computes features and target from `sample_path_historic`, calls `fit_ridge`
3. At each simulation step: calls `step(t)` on each BoundFeature, assembles the feature vector, predicts, inverse-transforms → next P

**Three concrete classes**, differing only in how they handle the target and its inverse:

- **`RawParticipationModel`** — target is P. Prediction is the next P directly.
- **`LogitParticipationModel`** — target is logit(P). Prediction: expit(ŷ) → next P.
- **`DiffLogitParticipationModel`** — target is Δlogit(P). Prediction: expit(logit(P_t) + ŷ) → next P.

Each class knows how to:
1. Compute the target for fitting (identity, logit, or diff-logit of the P column)
2. Inverse-transform predictions back to [0,1]
3. Write the result into the sample path array

The target transform is internal to the model class — not a user-supplied callable, and not part of the Feature system. The model's output is a *response* — the participation response to current conditions. It is not itself state; it is an impulse applied to chain state.

The `ParticipationModel` Protocol is the interface the Simulator depends on:

```python
class ParticipationModel(Protocol):
    def fit(self) -> None:
        """Fit ridge model from sample_path_historic.

        Calls evaluate() on each BoundFeature to build the design
        matrix, computes the target (class-specific transform),
        calls fit_ridge.
        """
        ...

    def prepare(self, sample_path_simulated: NDArray, t0: int) -> None:
        """Rebind features to simulation array and initialise.

        Calls rebind on each BoundFeature to produce new BoundFeatures
        on sample_path_simulated, calls evaluate(end=t0, cache=True)
        on each.
        """
        ...

    def predict_next(self, t: int, rng: Generator) -> NDArray:
        """Predict next participation rate from state at time t.

        Calls step(t) on each BoundFeature, assembles feature vector
        with intercept, applies ridge predict, adds noise,
        inverse-transforms.  Returns P(t+1), shape (n_paths,).
        """
        ...
```

#### Noise models (`simulation.py`)

```python
class NoiseModel(Protocol):
    def __call__(self, n: int, rng: Generator) -> NDArray:
        """Draw n residual noise samples."""
        ...
```

- **`GaussianNoise(sigma)`** — `rng.normal(0, sigma, n)`
- **`BootstrapNoise(residuals)`** — resample from fitted residuals

#### Simulation loop

```python
sp = sample_path_simulated
for t in range(t0, t0 + horizon):
    sp[:, t + 1, P_COL] = model.predict_next(t, rng)
    sp[:, t + 1, I_COL] = schedule.update(sp[:, t, I_COL], sp[:, t, P_COL])
```

Each step:
1. `model.predict_next(t, rng)`: calls `step(t)` on each BoundFeature, assembles feature vector with intercept, applies ridge `.predict()`, adds noise, inverse-transforms → P(t+1).
2. `schedule.update(I(t), P(t))` → I(t+1).

Both read from time t and write to time t+1. The operations are independent — neither reads the other's output at t+1.

#### Simulator (`simulation.py`)

```python
@dataclass
class Simulator:
    schedule: EmissionsSchedule
    model: ParticipationModel

    def run(self, initial_state: SimulationState,
            n_paths: int, horizon: int,
            exog_sampler: ExogenousSampler,
            rng: Generator) -> SimulationResult:
        """Run Monte Carlo simulation."""
        ...
```

The Simulator:
1. Queries the model for `lookback` (max Feature lookback across all features)
2. Allocates `sample_path_simulated` with shape `(n_paths, lookback + horizon + 1, n_cols)`
3. Copies the tail of `sample_path_historic` into the first `lookback` rows, broadcast to all paths
4. Pre-fills exogenous columns using the `ExogenousSampler`
5. Calls `model.prepare(sample_path_simulated, t0=lookback)` — rebinds Features, initialises accumulators
6. Runs the simulation loop
7. Returns `SimulationResult` (slicing out the history prefix)

Construction in the inference notebook:

```python
model = LogitParticipationModel(
    features=[
        "logit_participation",   # pre-computed in data prep
        "inflation_annual",      # pre-computed in data prep
        "fear_greed",
    ],
    df=round_df,
    alpha=0.1,
    noise=GaussianNoise(sigma=0.01),
)
model.fit()

sim = Simulator(schedule=SignedStepSchedule(...), model=model)
result = sim.run(
    initial_state=SimulationState(participation_rate=0.45, emissions_rate_per_round=38),
    n_paths=1000, horizon=500,
    exog_sampler=BootstrapSampler(block_size=10),
    rng=default_rng(42),
)
```

**Contract out:** `SimulationResult` with `participation_rate_paths` in natural [0,1] units and `emissions_rate_per_round_paths` in parts per billion. Both arrays of shape `(n_paths, horizon+1)`.

### Stage 5: Derived quantities (`derived.py`)

Takes a `SimulationResult` (participation rate and emissions rate paths in natural units) and computes quantities that are derived from the simulated paths but do not feed back into the simulation loop.

- `compute_total_supply_paths(initial_supply: float, emissions_rate_paths: ndarray) -> ndarray` — cumulative product: `supply[t+1] = supply[t] * (1 + emissions_rate[t])`. Shape `(n_paths, horizon+1)`.
- `compute_yield_paths(emissions_rate_paths: ndarray, participation_rate_paths: ndarray) -> ndarray` — `yield[t] = emissions_rate[t] / participation_rate[t]` each round, compounded over the path. Not annualised.
- `compute_dilution_paths(emissions_rate_paths: ndarray) -> ndarray` — `dilution[t] = 1 - 1/(1 + emissions_rate[t])` per round.

These are used by the notebook for:
- Visualisation (fan charts of total supply, yield, dilution over time)
- Admissibility evaluation — computing quantiles and checking inequalities against thresholds

Admissibility itself is not formalised in the library. It's a predicate over the simulation results: the notebook computes statistics (quantiles, exceedance probabilities — one-liners in numpy), evaluates inequalities, and decides pass/fail. Different notebooks or analyses may define different admissibility criteria.

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
3. `features.py` — numpy transforms (logit, expit, annualise_ppb), build_design_matrix — DONE (Feature protocol to be added)
4. `data.py` — daily data loading and round reindexing — DONE
5. `model.py` — ridge fitting, closed-form solver — DONE
6. `emissions_schedule.py` — `EmissionsSchedule` protocol + `SignedStepSchedule`, `ClampedSignedStepSchedule` — DONE
7. `types.py` update — add `Feature` and `BoundFeature` protocols, revise `ParticipationModel` protocol (fit/prepare/predict_next)
8. `features.py` update — add `BoundColumnFeature` (internal, identity transform on column data)
9. `exogenous.py` — `ExogenousSampler` protocol + `BootstrapSampler`, `AR1Sampler`
10. `simulation.py` — `NoiseModel` implementations, three `ParticipationModel` classes, `Simulator` with Ω management
11. `derived.py` — total supply, yield, dilution path computations
12. Tests for each new/updated module
13. Build the inference and diagnostics notebooks using the library

## Verification

- Unit tests for each module using pytest (in `tests/` directory)
- Key things to test: forward/inverse transform roundtrip, BoundFeature evaluate/step consistency, BoundFeature rebind, design matrix shape, emissions schedule boundary cases (clipping at floor/ceiling), simulation reproducibility with fixed seed, derived quantity computations against hand-calculated examples
- Integration test: run full pipeline (load sample data → features → fit → simulate → derive) and check output types and shapes
- Regression test: compare library output against current notebook output for a known parameter set to ensure extraction didn't change behaviour

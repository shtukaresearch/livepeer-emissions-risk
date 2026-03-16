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
    features.py              # Polars expression plugins, presets, design matrix assembly
    model.py                 # Ridge regression fitting (statsmodels)
    emissions_schedule.py    # EmissionsSchedule protocol + SignedStepSchedule
    exogenous.py             # ExogenousSampler protocol + BootstrapSampler, AR1Sampler
    simulation.py            # ParticipationModel, NoiseModel, Simulator
    derived.py               # Post-simulation derived quantities (total supply, yield, dilution)
```

## Key Types

### Dataclasses

- **`SimulationState`** — simulation starting point: `participation_rate` (float, [0,1]), `emissions_rate_per_round` (float, parts per billion).
- **`SimulationResult`** — simulation output: `participation_rate_paths` (ndarray), `emissions_rate_per_round_paths` (ndarray). Both shape `(n_paths, horizon+1)`.
- **`SignedStepSchedule`** — the current Livepeer emissions rule: `target_participation_rate`, `emissions_change`, `emissions_floor`, `emissions_ceiling`.
- **`RidgeParticipationModel`** — wraps fitted ridge model, feature transforms, noise, and exogenous sampler.
- **`Simulator`** — composes an `EmissionsSchedule` and a `ParticipationModel`, exposes `.run()`.

### Protocols (pluggable interfaces)

- **`EmissionsSchedule`** — `update(emissions_rate_per_round, participation_rate) -> ndarray`. The state machine for emissions rate updates.
- **`ParticipationModel`** — `predict_next(participation_rate, emissions_rate, step, rng) -> ndarray`. Maps raw state to next participation rate.
- **`NoiseModel`** — `__call__(n: int, rng: Generator) -> ndarray`. Ship `GaussianNoise` and `BootstrapNoise`.
- **`ExogenousSampler`** — `sample(historical, n_paths, horizon, rng) -> ndarray`. Ship `BootstrapSampler` and `AR1Sampler`.

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

Feature engineering uses Polars expressions as the composition primitive. The library provides domain-specific expression plugins (custom transforms) and a dictionary of named presets for common patterns. The notebook user can mix presets with arbitrary Polars expressions.

#### Expression plugins (custom transforms)

These are functions that take and return `pl.Expr`, so they compose with `.pipe()`:

- `logit(expr) -> pl.Expr` — log(x / (1-x)), for participation rate
- `expit(expr) -> pl.Expr` — inverse logit, 1 / (1 + exp(-x))
- `rescale_around(expr, center) -> pl.Expr` — 2 * (x - center), e.g. F&G around 50
- `annualise_ppb(expr, rounds_per_year) -> pl.Expr` — (1 + x/1e9)^rounds_per_year - 1

The logit/expit pair is also used in the simulation stage (Stage 4) for inverse-transforming predicted values back to [0,1].

#### Presets

A dictionary of named `pl.Expr` for common column recipes:

```
PRESETS = {
    "P":              pl.col("bonded") / pl.col("total-supply"),
    "logit_P":        (pl.col("bonded") / pl.col("total-supply")).pipe(logit),
    "logit_P_diff":   (pl.col("bonded") / pl.col("total-supply")).pipe(logit).diff(),
    "I":              pl.col("inflation") / 1e9,
    "I_annual":       pl.col("inflation").pipe(annualise_ppb, ROUNDS_PER_YEAR),
    "fng_extreme":    pl.col("fear_greed_index").pipe(rescale_around, center=50),
    "fng_extreme_abs": pl.col("fear_greed_index").pipe(rescale_around, center=50).abs(),
    ...
}
```

Presets are convenience only. The notebook user can ignore them and write raw Polars expressions, or define their own.

#### Design matrix assembly

- `build_design_matrix(df, target: pl.Expr, features: list[pl.Expr]) -> pl.DataFrame` — evaluates the target and feature expressions against the round-indexed DataFrame, adds an intercept column, shifts target by -1 to get the next-step prediction target, drops resulting nulls. Returns a single DataFrame with columns `[intercept, feature_0, ..., feature_n, target]`.

**Contract out:** A Polars DataFrame with named columns. `target` column is the next-step value. Feature columns are whatever the user specified — the library doesn't prescribe their names or order beyond the intercept.

No `train_test_split` in the library — that's a diagnostic notebook concern (see Notebook Structure below).

### Stage 3: Fit model (`model.py`)

- `fit_ridge(df: pl.DataFrame, target_col: str, alpha: float) -> RegressionResult` — fit ridge regression on the design matrix. Returns a result object with `.predict()`, `.params`, `.resid`.

Alpha selection (cross-validation or otherwise) is not handled inside `fit_ridge`. The diagnostics notebook is responsible for evaluating different alphas and choosing one; the inference notebook passes the chosen alpha directly.

#### statsmodels vs scikit-learn for ridge

**statsmodels** (`OLS.fit_regularized(L1_wt=0)`): returns a rich result object with `.predict()`, `.params` (named coefficients), `.resid`, plus diagnostic statistics (confidence intervals, R², AIC/BIC). This is useful for the diagnostics notebook. Downside: no built-in cross-validated alpha selection — you write the CV loop yourself.

**scikit-learn** (`RidgeCV`): built-in efficient leave-one-out or K-fold CV for alpha selection. But the fitted model is opaque — you get `.coef_` and `.predict()` but no statistical diagnostics without computing them manually.

Recommendation: use **statsmodels** for the fit, since the diagnostics notebook benefits from the rich result object and we need to control the CV strategy anyway (the current notebook's CV is entangled with inference in ways we want to undo). Write a small alpha-selection utility in the diagnostics notebook using statsmodels fits across a grid.

**Contract out:** A statsmodels `RegressionResults`-like object. The simulation stage needs `.predict()` and `.resid` only. The diagnostics notebook uses the full interface.

### Stage 4: Simulate (`simulation.py`, `emissions_schedule.py`, `exogenous.py`)

The simulation loop recurses two steps each round:

1. `I' = schedule.update(I, P)` — emissions schedule state machine
2. `P' = model.predict_next(P, I', step, rng)` — participation model

Both operate in natural units (participation in [0,1], emissions rate in parts per billion). All transform logic is internal to the model.

Note: the on-chain smart contracts use the term `inflation` for the emissions rate. We use "emissions" everywhere in our own code and documentation, and only use `inflation` when referring directly to the contract field.

#### Emissions schedule state machine (`emissions_schedule.py`)

A pluggable interface for the rule that updates the emissions rate each round. This is the component to swap out when exploring alternative emissions schedules.

```python
class EmissionsSchedule(Protocol):
    def update(self, emissions_rate_per_round: ndarray, participation_rate: ndarray) -> ndarray:
        """Given current emissions rate and participation, return next-round emissions rate."""
        ...
```

The current Livepeer rule is one implementation:

```python
@dataclass
class SignedStepSchedule:
    target_participation_rate: float   # e.g. 0.5
    emissions_change: float            # per-round step size in parts per billion
    emissions_floor: float             # minimum emissions rate per round
    emissions_ceiling: float           # maximum emissions rate per round

    def update(self, emissions_rate_per_round, participation_rate):
        step = self.emissions_change * np.sign(
            self.target_participation_rate - participation_rate
        )
        return np.clip(
            emissions_rate_per_round + step, self.emissions_floor, self.emissions_ceiling
        )
```

Vectorized — operates on arrays of shape `(n_paths,)` for all Monte Carlo paths simultaneously.

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

#### Participation model (`simulation.py`)

Wraps the fitted ridge model, feature transforms, noise, and exogenous sampling into a single object that maps raw state to next-round participation rate. All forward transforms (e.g. logit on participation, log on emissions rate) and the inverse transform back to [0,1] are internal.

```python
class RidgeParticipationModel:
    def __init__(
        self,
        ridge_result,                          # statsmodels fitted result
        feature_transforms: dict[str, Callable],  # {"participation_rate": logit, "emissions_rate": identity}
        inverse_transform: Callable,           # e.g. expit, maps model output back to [0,1]
        noise: NoiseModel,
        exog_sampler: ExogenousSampler,
        historical_exog: pl.DataFrame,
    ): ...

    def predict_next(self, participation_rate, emissions_rate, step, rng):
        """Given current state in natural units, return next participation rate in [0,1].

        `step` is the time index into the pre-sampled exogenous paths (0..horizon-1).
        """
        transformed_p = self.feature_transforms["participation_rate"](participation_rate)
        transformed_i = self.feature_transforms["emissions_rate"](emissions_rate)
        exog = self.exog_paths[:, step, :]

        X = np.column_stack([np.ones(len(participation_rate)), transformed_p, transformed_i, exog])
        y_hat = self.ridge.predict(X) + self.noise(len(participation_rate), rng)
        return self.inverse_transform(y_hat)
```

The `ParticipationModel` Protocol is the interface the simulator depends on:

```python
class ParticipationModel(Protocol):
    def predict_next(self, participation_rate: ndarray, emissions_rate: ndarray,
                     step: int, rng: Generator) -> ndarray: ...
```

`RidgeParticipationModel` is one implementation. Alternative model types (e.g. mean-reverting OU process, neural net) would implement the same protocol.

Construction in the inference notebook:

```python
model = RidgeParticipationModel(
    ridge_result=fitted,
    feature_transforms={"participation_rate": logit, "emissions_rate": identity},
    inverse_transform=expit,
    noise=GaussianNoise(sigma=fitted.resid.std()),
    exog_sampler=BootstrapSampler(block_size=10),
    historical_exog=exog_df,
)

sim = Simulator(schedule=SignedStepSchedule(...), model=model)
result = sim.run(initial_state, n_paths=1000, horizon=500, rng=default_rng(42))
```

#### Simulator (`simulation.py`)

```python
@dataclass
class Simulator:
    schedule: EmissionsSchedule
    model: ParticipationModel

    def run(self, initial_state: SimulationState, n_paths: int, horizon: int,
            rng: Generator) -> SimulationResult:
        """Run Monte Carlo simulation.

        Internally calls model.prepare(n_paths, horizon, rng) to pre-sample
        exogenous paths before entering the loop. The caller does not need to
        manage this.
        """
        ...
```

```python
@dataclass
class SimulationState:
    participation_rate: float       # last observed, in [0,1]
    emissions_rate_per_round: float # last observed, in parts per billion
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

Exploratory. Iterate over candidate configurations (feature sets, transforms, differencing, alpha values) and compare them side-by-side. For each combination: fit on a training window, evaluate on held-out data, report AIC/RMSE/coefficient stability. Output is a comparison table. Train/test splitting lives here, not in the library.

### 2. Inference / forecasting notebook

Takes a fixed configuration as input: feature set (list of `pl.Expr`), alpha, noise model, protocol params, risk objectives. Fits on all available data, runs simulation, evaluates admissibility. This is the "production" notebook — no exploratory UI, just a clear pipeline from config to results.

### 3. Historical exploration notebook

`emissions-history.py` already serves this role — visualising raw data, trends, protocol behaviour over time.

## What's NOT Included

- **Loss function** (issue #8 breach-count loss) — deferred
- **Plotting/visualization** — stays in notebooks
- **Pandas support** — Polars only
- **Daily-resolution pipeline** — November notebook is deprecated

## Dependency Changes to `pyproject.toml`

Library core deps: `polars`, `numpy`, `statsmodels`, `requests`

Optional: `web3` (for on-chain fetching in `data.py`)

Move to dev/notebook deps: `marimo`, `altair`, `matplotlib`, `arrow`, `pytz`, `pyzmq`

Remove: `pandas` (replaced by `polars` throughout)

## Implementation Order

1. `constants.py` + `types.py` — foundational, no deps
2. `time.py` — estimated round↔datetime conversion, real timestamp→block lookup
3. `features.py` — expression plugins, presets, design matrix assembly
4. `data.py` — daily data loading and round reindexing (depends on `time.py`)
5. `model.py` — ridge fitting via statsmodels
6. `emissions_schedule.py` — `EmissionsSchedule` protocol + `SignedStepSchedule`
7. `exogenous.py` — `ExogenousSampler` protocol + `BootstrapSampler`, `AR1Sampler`
8. `simulation.py` — `NoiseModel`, `ParticipationModel`, `RidgeParticipationModel`, `Simulator`
9. `derived.py` — total supply, yield, dilution path computations
10. Tests for each module
11. Build the inference and diagnostics notebooks using the library

## Verification

- Unit tests for each module using pytest (in `tests/` directory)
- Key things to test: forward/inverse transform roundtrip, design matrix shape and column names, emissions schedule boundary cases (clipping at floor/ceiling), simulation reproducibility with fixed seed, derived quantity computations against hand-calculated examples
- Integration test: run full pipeline (load sample data → features → fit → simulate → derive) and check output types and shapes
- Regression test: compare library output against current notebook output for a known parameter set to ensure extraction didn't change behaviour

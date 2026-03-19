import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # Livepeer Emissions Risk — Inference

    Fits a participation model on historical data, runs Monte Carlo
    simulation, and evaluates the results.
    """)
    return (mo,)


@app.cell
def _():
    import altair as alt
    import numpy as np
    import polars as pl
    from numpy.random import default_rng

    from lpt_stake import (
        BootstrapSampler,
        GaussianNoise,
        LogitParticipationModel,
        RawParticipationModel,
        SignedStepSchedule,
        SimulationState,
        Simulator,
        compute_total_supply_paths,
    )
    from lpt_stake.features import annualise_ppb, logit
    from lpt_stake.constants import ROUNDS_PER_YEAR
    return (
        BootstrapSampler,
        GaussianNoise,
        LogitParticipationModel,
        ROUNDS_PER_YEAR,
        SignedStepSchedule,
        SimulationState,
        Simulator,
        alt,
        annualise_ppb,
        compute_total_supply_paths,
        default_rng,
        logit,
        np,
        pl,
    )


@app.cell
def _(ROUNDS_PER_YEAR, annualise_ppb, logit, pl):
    # Load and prepare data
    import os
    data_path = os.environ.get("LPT_DATA_SOURCE")
    raw = pl.read_csv(data_path, glob=False)

    df = raw.select(
        pl.col("round").cast(pl.Int64),
        pl.col("date").str.to_date(),
        (pl.col("bonded") / pl.col("total-supply")).alias("participation_rate"),
        pl.col("inflation").alias("emissions_rate"),
        pl.col("fear_greed_index"),
        pl.col("eth_price_usd"),
        pl.col("lpt_price_usd"),
    ).with_columns(
        # Derived columns for features
        pl.col("participation_rate").pipe(logit).alias("logit_participation"),
        pl.col("emissions_rate").pipe(annualise_ppb, ROUNDS_PER_YEAR).alias("inflation_annual"),
    ).drop_nulls()

    df
    return df, raw


@app.cell
def _(mo):
    mo.md("""
    ## Data overview
    """)
    return


@app.cell
def _(alt, df):
    # Participation rate over time
    chart_p = alt.Chart(df).mark_line().encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("participation_rate:Q", title="Participation Rate", scale=alt.Scale(zero=False)),
    ).properties(title="Historical Participation Rate", width=700, height=300)
    chart_p
    return


@app.cell
def _(alt, df):
    chart_i = alt.Chart(df).mark_line().encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("emissions_rate:Q", title="Emissions Rate (ppb)"),
    ).properties(title="Historical Emissions Rate", width=700, height=300)
    chart_i
    return


@app.cell
def _(mo):
    mo.md("""
    ## Configuration
    """)
    return


@app.cell
def _(mo):
    alpha_slider = mo.ui.slider(
        start=0.0, stop=1.0, step=0.01, value=0.1, label="Ridge alpha"
    )
    noise_slider = mo.ui.slider(
        start=0.0, stop=0.05, step=0.001, value=0.005, label="Noise sigma"
    )
    horizon_slider = mo.ui.slider(
        start=50, stop=1000, step=50, value=500, label="Horizon (rounds)"
    )
    n_paths_slider = mo.ui.slider(
        start=100, stop=5000, step=100, value=1000, label="Number of paths"
    )
    seed_input = mo.ui.number(start=0, stop=999999, value=42, label="Random seed")

    mo.vstack([
        mo.md("### Model parameters"),
        mo.hstack([alpha_slider, noise_slider]),
        mo.md("### Simulation parameters"),
        mo.hstack([horizon_slider, n_paths_slider, seed_input]),
    ])
    return (
        alpha_slider,
        horizon_slider,
        n_paths_slider,
        noise_slider,
        seed_input,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Fit model
    """)
    return


@app.cell
def _(
    GaussianNoise,
    LogitParticipationModel,
    alpha_slider,
    df,
    noise_slider,
    pl,
):
    model = LogitParticipationModel(
        features=["logit_participation", "inflation_annual", "fear_greed_index"],
        target="participation_rate", # shifted by 1 internally
        alpha=alpha_slider.value,
        noise=GaussianNoise(sigma=noise_slider.value),
    )
    model.fit(df)

    ridge = model.ridge_result
    coef_table = pl.DataFrame({
        "Feature": list(ridge.coefficients.keys()),
        "Coefficient": list(ridge.coefficients.values()),
        "Std Error": list(ridge.coefficient_std_errors.values()),
    })
    coef_table
    return model, ridge


@app.cell
def _(mo, ridge):
    mo.md(f"""
    **Fit diagnostics:**
    - Residual std: {ridge.residual_std:.6f}
    - Effective df: {ridge.effective_df:.2f}
    - AIC: {ridge.aic:.2f}
    - BIC: {ridge.bic:.2f}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Simulation
    """)
    return


@app.cell
def _(
    BootstrapSampler,
    SignedStepSchedule,
    SimulationState,
    Simulator,
    default_rng,
    df,
    horizon_slider,
    model,
    n_paths_slider,
    seed_input,
):
    # Protocol parameters (current Livepeer mainnet values)
    schedule = SignedStepSchedule(
        target_participation_rate=0.5,
        emissions_change=500,  # 500 ppb per round
    )

    sim = Simulator(
        schedule=schedule,
        model=model,
        exog_sampler=BootstrapSampler(block_size=10),
    )

    # Initial state from last row of data
    last = df.tail(1)
    initial = SimulationState(
        participation_rate=last["participation_rate"][0],
        emissions_rate_per_round=last["emissions_rate"][0],
    )

    result = sim.run(
        initial_state=initial,
        n_paths=n_paths_slider.value,
        horizon=horizon_slider.value,
        df=df,
        rng=default_rng(seed_input.value),
    )
    return (result,)


@app.cell
def _(mo):
    mo.md("""
    ## Results
    """)
    return


@app.cell
def _(alt, np, pl, result):
    # Fan chart for participation rate
    p_paths = result.participation_rate_paths
    rounds = np.arange(p_paths.shape[1])
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    p_q = np.quantile(p_paths, quantiles, axis=0)

    fan_data = pl.DataFrame({
        "round": np.tile(rounds, 3),
        "lower": np.concatenate([p_q[0], p_q[1], p_q[2]]),
        "upper": np.concatenate([p_q[4], p_q[3], p_q[2]]),
        "band": (
            ["5–95%"] * len(rounds)
            + ["25–75%"] * len(rounds)
            + ["median"] * len(rounds)
        ),
    })

    fan_chart_p = alt.Chart(fan_data).mark_area(opacity=0.4).encode(
        x=alt.X("round:Q", title="Round"),
        y=alt.Y("lower:Q", title="Participation Rate", scale=alt.Scale(zero=False)),
        y2="upper:Q",
        color=alt.Color("band:N", scale=alt.Scale(
            domain=["5–95%", "25–75%", "median"],
            range=["#ccc", "#888", "#333"],
        )),
    ).properties(title="Simulated Participation Rate", width=700, height=300)
    fan_chart_p
    return


@app.cell
def _(alt, np, pl, result):
    # Fan chart for emissions rate
    i_paths = result.emissions_rate_per_round_paths
    rounds_i = np.arange(i_paths.shape[1])
    i_q = np.quantile(i_paths, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)

    fan_data_i = pl.DataFrame({
        "round": np.tile(rounds_i, 3),
        "lower": np.concatenate([i_q[0], i_q[1], i_q[2]]),
        "upper": np.concatenate([i_q[4], i_q[3], i_q[2]]),
        "band": (
            ["5–95%"] * len(rounds_i)
            + ["25–75%"] * len(rounds_i)
            + ["median"] * len(rounds_i)
        ),
    })

    fan_chart_i = alt.Chart(fan_data_i).mark_area(opacity=0.4).encode(
        x=alt.X("round:Q", title="Round"),
        y=alt.Y("lower:Q", title="Emissions Rate (ppb)"),
        y2="upper:Q",
        color=alt.Color("band:N", scale=alt.Scale(
            domain=["5–95%", "25–75%", "median"],
            range=["#ccc", "#888", "#333"],
        )),
    ).properties(title="Simulated Emissions Rate", width=700, height=300)
    fan_chart_i
    return


@app.cell
def _(alt, compute_total_supply_paths, np, pl, raw, result):
    # Total supply projection
    last_supply = raw["total-supply"].to_list()[-1]
    supply_paths = compute_total_supply_paths(last_supply, result.emissions_rate_per_round_paths)

    rounds_s = np.arange(supply_paths.shape[1])
    s_q = np.quantile(supply_paths, [0.05, 0.5, 0.95], axis=0)

    supply_df = pl.DataFrame({
        "round": np.tile(rounds_s, 2),
        "lower": np.concatenate([s_q[0], s_q[1]]),
        "upper": np.concatenate([s_q[2], s_q[1]]),
        "band": ["5–95%"] * len(rounds_s) + ["median"] * len(rounds_s),
    })

    fan_chart_s = alt.Chart(supply_df).mark_area(opacity=0.4).encode(
        x=alt.X("round:Q", title="Round"),
        y=alt.Y("lower:Q", title="Total Supply", scale=alt.Scale(zero=False)),
        y2="upper:Q",
        color=alt.Color("band:N", scale=alt.Scale(
            domain=["5–95%", "median"],
            range=["#ccc", "#333"],
        )),
    ).properties(title="Projected Total Supply", width=700, height=300)
    fan_chart_s
    return


@app.cell
def _(horizon_slider, mo, np, result):
    # Summary statistics at horizon
    p_final = result.participation_rate_paths[:, -1]
    i_final = result.emissions_rate_per_round_paths[:, -1]

    mo.md(f"""
    ## Summary at horizon ({horizon_slider.value} rounds)

    | Statistic | Participation Rate | Emissions Rate (ppb) |
    |---|---|---|
    | Mean | {np.mean(p_final):.4f} | {np.mean(i_final):.0f} |
    | Median | {np.median(p_final):.4f} | {np.median(i_final):.0f} |
    | 5th pct | {np.quantile(p_final, 0.05):.4f} | {np.quantile(i_final, 0.05):.0f} |
    | 95th pct | {np.quantile(p_final, 0.95):.4f} | {np.quantile(i_final, 0.95):.0f} |
    | Std | {np.std(p_final):.4f} | {np.std(i_final):.0f} |
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

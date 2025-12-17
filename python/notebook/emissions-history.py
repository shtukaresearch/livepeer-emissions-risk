import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    import os, json
    return alt, json, mo, os, pl


@app.cell
def _():
    # Rounds calculations
    SECONDS_PER_ETH_SLOT = 12
    ETH_SLOTS_PER_ROUND = 6337
    ROUNDS_PER_YEAR = (3600 * 24 * 365) / (SECONDS_PER_ETH_SLOT * ETH_SLOTS_PER_ROUND)
    return (ROUNDS_PER_YEAR,)


@app.cell
def _(ROUNDS_PER_YEAR, json, os, pl):
    # Data loading
    DATA_PATH = os.getenv("LPT_DATA_PATH", "../data/lpt-daily-data-22-25.json")

    with open(DATA_PATH) as h:
        data_json = json.load(h)

    # Convert JSON data to DataFrame, coercing "total-supply" and "bonded" columns to float64 and "date" to DateTime objects
    def load_df_from_json(json):
        return pl.DataFrame(data_json).with_columns([
            pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d"),
            pl.col("total-supply").cast(pl.Float64),
            pl.col("bonded").cast(pl.Float64)
        ])

    def prepare_bonding_metrics(df):
        return df.with_columns([
            (pl.col("bonded") / pl.col("total-supply")).alias("bonding-rate"),
            (pl.col("bonded") / (pl.col("total-supply") - pl.col("bonded"))).alias("bonding-ratio")
        ]).with_columns([
            (pl.col("bonding-rate") * 100).alias("bonding-rate-%"),
            (pl.col("bonding-rate").diff() * 100 ).alias("bonding-rate-change-%")
        ])

    def prepare_emission_metrics(df):
        return df.with_columns([
            (pl.col("inflation") / 1_000_000_000).alias("emission-rate-round"),
            (((1 + (pl.col("inflation") / 1_000_000_000)) ** ROUNDS_PER_YEAR) - 1).alias("emission-rate-annualised")
        ]).with_columns([
            (1 - 1 / (1 + pl.col("emission-rate-round"))).alias("dilution-rate-round"),
            (1 - 1 / (1 + pl.col("emission-rate-annualised"))).alias("dilution-rate-annualised")
        ]).with_columns([
            (pl.col("dilution-rate-round") * 100).alias("dilution-rate-round-%"),
            (pl.col("dilution-rate-annualised") * 100).alias("dilution-rate-annualised-%")
        ])

    def prepare_yield_metrics(df):
        return df.with_columns([
            (pl.col("emission-rate-round") / pl.col("bonding-rate")).alias("yield-round"),
            (pl.col("emission-rate-annualised") / pl.col("bonding-rate")).alias("yield-annualised")
        ]).with_columns([
            ((1 + pl.col("yield-annualised")) * (1 - pl.col("dilution-rate-annualised")) - 1).alias("adjusted-yield-annualised"),
            (1 + pl.col("yield-round")).cum_prod().alias("yield-cumulative")
        ]).with_columns([
            (pl.col("yield-annualised") * 100).alias("yield-annualised-%"),
            (pl.col("adjusted-yield-annualised") * 100).alias("adjusted-yield-annualised-%")
        ])

    def prepare_adjusted_bonding_rate(df):
        return df.with_columns([
            (pl.col("bonded") / pl.col("yield-cumulative")).alias("adjusted-bonded")
        ]).with_columns([
            (100 * pl.col("adjusted-bonded") / df["total-supply"][0]).alias("adjusted-bonding-rate-%")
        ])

    # Trailing yields
    SHIFT = int(ROUNDS_PER_YEAR)

    def prepare_trailing_returns(df):
        return df.with_columns([
            (pl.col("yield-cumulative") / pl.col("yield-cumulative").shift(SHIFT)).alias("return-trailing-1y"),
            (pl.col("total-supply").shift(SHIFT) / pl.col("total-supply")).alias("dilution-trailing-1y")
        ]).with_columns([
            (pl.col("return-trailing-1y") * pl.col("dilution-trailing-1y")).alias("adjusted-return-trailing-1y")
        ]).with_columns([
            ((pl.col("return-trailing-1y") - 1) * 100).alias("return-trailing-1y-%"),
            ((pl.col("adjusted-return-trailing-1y") - 1) * 100).alias("adjusted-return-trailing-1y-%")
        ])

    PREPARATIONS = [
        load_df_from_json, 
        prepare_bonding_metrics, 
        prepare_emission_metrics, 
        prepare_yield_metrics, 
        prepare_adjusted_bonding_rate, 
        prepare_trailing_returns
    ]
    return PREPARATIONS, data_json


@app.cell
def _(PREPARATIONS, data_json):
    from functools import reduce
    df = reduce(lambda dfa, dfb: dfb(dfa), PREPARATIONS, data_json)
    df
    return (df,)


@app.cell
def _(alt, df):
    dilution_rate_chart = alt.Chart(df).mark_line(color="magenta").encode(
        x="date:T",
        y=alt.Y("dilution-rate-annualised-%:Q", axis=alt.Axis(title="Annualised dilution rate (%)"))
    )

    dilution_rate_chart
    return


@app.cell
def _(alt, df, pl):
    df_yield_flattened = df.with_columns([
        pl.col("yield-annualised-%").alias("Nominal"),
        pl.col("adjusted-yield-annualised-%").alias("Adjusted")
    ]).unpivot(
        index="date", 
        on=["Nominal", "Adjusted"], 
        variable_name="metric", 
        value_name="value"
    )

    alt.Chart(df_yield_flattened).mark_line().encode(
        x="date:T",
        y=alt.Y("value:Q", axis=alt.Axis(title="Yield (%)")),
        color="metric:N",
        tooltip=["date:T", "value:Q", "metric:N"]
    )
    return


@app.cell
def _(alt, df):
    bonding_rate_chart = alt.Chart(df).mark_line(color="green").encode(
        x="date:T",
        y="bonding-rate-%:Q"
    )

    bonding_rate_index_chart = alt.Chart(df).mark_line(color="blue").encode(
        x="date:T",
        y="adjusted-bonding-rate-%:Q"
    )

    bonding_rate_chart
    return


@app.cell
def _(alt, df):
    bonding_ratio_chart = alt.Chart(df).mark_line(color="purple").encode(
        x="date:T",
        y="bonding-ratio:Q"
    )
    bonding_ratio_chart
    return


@app.cell
def _(alt, df):
    alt.Chart(df).mark_line(color="orange").encode(
        x="date:T",
        y=alt.Y("bonding-rate-change-%:Q", title="Bonding rate change (%)")
    )
    return


@app.cell
def _(df, mo, pl):
    min_inflation_offset = df.select(pl.col("inflation").arg_min()).item()

    mo.md(f"""
    **Date of minimum emissions:** {df["date"][min_inflation_offset].strftime("%Y-%m-%d")}\n
    **Minimum annualised dilution rate:** {df.select(pl.col("dilution-rate-annualised-%").min()).item():.2f}%\n
    **Minimum annualised yield:** {df.select(pl.col("yield-annualised-%").min()).item():.2f}%\n
    **Minimum annualised adjusted yield:** {df.select(pl.col("adjusted-yield-annualised-%").min()).item():.2f}%
    """)
    return


@app.cell
def _(alt, df, pl):
    df_trailing_flattened = df.with_columns([
        pl.col("return-trailing-1y-%").alias("Nominal"),
        pl.col("adjusted-return-trailing-1y-%").alias("Adjusted")
    ]).unpivot(
        index="date", 
        on=["Nominal", "Adjusted"], 
        variable_name="metric", 
        value_name="value"
    )

    alt.Chart(df_trailing_flattened).mark_line().encode(
        x="date:T",
        y=alt.Y("value:Q", axis=alt.Axis(title="Trailing 1Y returns (%)")),
        color="metric:N",
        tooltip=["date:T", "value:Q", "metric:N"]
    )
    return


@app.cell
def _(alt, df):
    import numpy as np

    shuft = 1

    df_diddle = df.with_columns([
        (np.log(df["bonding-ratio"] / df["bonding-ratio"].shift(shuft)) / shuft).alias("log-bonding-ratio-diff")
    ])

    alt.Chart(df_diddle).transform_density(
        "log-bonding-ratio-diff",
        as_=["log-bonding-ratio-diff", "density"]
    ).mark_area(color="lightblue", opacity=0.7).encode(
        x=alt.X("log-bonding-ratio-diff:Q", axis=alt.Axis(title="Log Bonding Ratio Diff")),
        y=alt.Y("density:Q", axis=alt.Axis(title="Density"))
    )
    return df_diddle, np


@app.cell
def _(df_diddle, mo, np, pl):
    diffs = df_diddle["log-bonding-ratio-diff"].drop_nulls()

    def sim_random_walk(runs = 100):
        outcomes = []
        for _ in range(runs):
            outcomes.append(diffs.sample(30, with_replacement=True).sum())
        return outcomes

    simulated_180d = pl.Series(sim_random_walk(10000))
    p05_180d = simulated_180d.quantile(0.975)


    def logit_to_percent(logit):
        ratio = np.exp(logit)
        return 100 * ratio / (1 + ratio)

    mo.md(f"""
    *Lower 95% confidence interval for 180-day log bonding ratio diff:* {logit_to_percent(p05_180d):.6f}
    """)
    return


@app.cell
def _(df, pl):
    # DILUTION PLOTS
    from datetime import date

    dates = [date(2022,12,31), date(2023, 6, 30), date(2023, 12,31), date(2024,6,30), date(2024,12,31), date(2025,6,30), date(2025,11,20)]
    epochs = ["H2 2022", "H1 2023", "H2 2023", "H1 2024", "H2 2024", "H1 2025", "H2 2025"]

    df_2 = df.with_columns([
        (100*(1 - (pl.col("total-supply").shift(180) / pl.col("total-supply")) ) ).alias("dilution-180d-%")
    ])

    # select rows with given dates
    df_dilution_dates = df_2.filter(
        pl.col("date").is_in(dates)
    ).select(
        ["date", "dilution-180d-%"]
    ).with_columns([
        pl.Series(epochs).alias("epoch")
    ])

    # Add a specific row for H1 2026
    df_dilution_dates = df_dilution_dates.vstack(
        pl.DataFrame({
            "date": [date(2026, 6, 30)],
            "dilution-180d-%": [12.0],
            "epoch": ["H1 2026 (target)"]
        })
    )
    return dates, df_dilution_dates


@app.cell
def _(alt, dates, df_dilution_dates):
    # plot dilution on bar chart with bars ordered by date and labelled by epoch
    # set the opacity of the last bar to a lower value
    alt.Chart(df_dilution_dates).mark_bar(size=50).encode(
        x=alt.X("epoch:N", sort=dates, axis=alt.Axis(title="Epoch")),
        y=alt.Y("dilution-180d-%:Q", axis=alt.Axis(title="Semiannual Dilution (%)")),
        opacity=alt.condition(
            alt.datum.epoch == "H1 2026 (target)",
            alt.value(0.3),
            alt.value(1.0)
        ),
        tooltip=["epoch:N", "dilution-180d-%:Q"]
    ).properties(width=800)
    return


@app.cell
def _(np):
    rng= np.random.default_rng(42)
    arr = rng.normal(size=200)
    arr2 = rng.normal(-0.4, 0.3, size=200)

    # plot box and whisker plot of arr with whiskers at 90th and 10th percentiles using matplotlib
    # don't show outliers
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.boxplot(arr, whis=[5, 95], showfliers=False)
    ax.set_ylabel("Value")
    plt.show()
    return arr, arr2


@app.cell
def _(alt, arr, arr2, np, pl):
    # crop outliers from arr at 95th and 5th percentiles

    def plot_box_and_whiskers(data, label: str, significance=5):
        q_low = np.percentile(data, significance/2)
        q_high = np.percentile(data, 100 - significance/2)
        data_cropped = data[(data >= q_low) & (data <= q_high)]

        return alt.Chart(pl.DataFrame({"value": data_cropped})).mark_boxplot(extent="min-max").encode(
            y=alt.Y("value:Q", axis=alt.Axis(title=label))
        )

    def plot_box_and_whiskers_multiple(data: pl.DataFrame, label: str, significance=5):
        df = data.unpivot(
            index=None,
            on=data.columns,
            variable_name="Policy",
            value_name="value"
        )
        return alt.Chart(df).mark_boxplot(extent="min-max", size=60).encode(
            x=alt.X("Policy:N", axis=alt.Axis(title="Policy")),
            y=alt.Y("value:Q", axis=alt.Axis(title=label)),
            color="Policy:N"
        ).properties(width=400)

    # Show bloxplot for arr and arr2 side by side on the same axes
    # First, we're gonna unpivot arr and arr2 into a single DataFrame
    df_boxplot = pl.DataFrame({
        "No change": arr,
        "Proposed change": arr2
    })

    plot_box_and_whiskers_multiple(df_boxplot, "Forecast 1Y trailing yield on 2026-07-01")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import altair as alt
    import csv
    return alt, mo, pl


@app.cell
def _(mo, pl):
    BUDGET_DATA_PATH = "../survey/lpt-emissions-budget.csv"
    BONDING_DATA_PATH = "../survey/lpt-bonding-threshold.csv"

    # Load data into a Polars DataFrame
    budget_threshold_df_raw = pl.read_csv(BUDGET_DATA_PATH)
    bonding_threshold_df_raw = pl.read_csv(BONDING_DATA_PATH)

    # number of rows on bonding_threshold_df that are null in both columns
    bonding_threshold_partial_response_count = (bonding_threshold_df_raw
        .filter(pl.col("Cleaned low").is_not_null() & pl.col("Cleaned critical").is_not_null()).shape[0]
    )

    mo.md(f"""
    Number of responses to budget threshold questions: {len(budget_threshold_df_raw.drop_nulls())}\n
    Number of responses to bonding threshold questions (including partial responses): {bonding_threshold_partial_response_count}
    """)
    return bonding_threshold_df_raw, budget_threshold_df_raw


@app.cell
def _(pl):
    # Preprocessing budget threshold data

    def merge_count_responses(df: pl.DataFrame) -> pl.DataFrame:
        """Merge and count responses from budget threshold questions."""
        return (df
            .unpivot() # Convert from wide to long format by merging answers to both questions
            .group_by(["value", "variable"]) 
            .agg(pl.len().alias("count")) # Counts of all unique answers to each question
            .pivot(values="count", index="value", on="variable") # Pivot back to wide format
            .filter(pl.col("value").is_not_null())
            .sort("value")
            .fill_null(0)
        )

    def prepare_cumulative_count_cols(df: pl.DataFrame) -> pl.DataFrame:
        """
        Prepare cumulative count columns for budget threshold data.
        Sum "Adjusted low" in descending order.
        """
        return df.with_columns([
            pl.col("Adjusted low").reverse().cum_sum().reverse().alias("Descending low"),
            pl.col("Adjusted high").cum_sum().alias("Ascending high"),
        ])

    # Preprocessing bonding threshold data

    def prepare_cumulative_count_cols_bonding(df: pl.DataFrame) -> pl.DataFrame:
        """
        Prepare cumulative count columns for budget threshold data.
        Sum "Adjusted low" in descending order.
        """
        return df.with_columns([
            pl.col("Cleaned low").reverse().cum_sum().reverse().alias("Descending low"),
            pl.col("Cleaned critical").reverse().cum_sum().reverse().alias("Descending critical"),
        ])
    return (
        merge_count_responses,
        prepare_cumulative_count_cols,
        prepare_cumulative_count_cols_bonding,
    )


@app.cell
def _(
    bonding_threshold_df_raw,
    budget_threshold_df_raw,
    merge_count_responses,
    prepare_cumulative_count_cols,
    prepare_cumulative_count_cols_bonding,
):
    budget_df_prepared = prepare_cumulative_count_cols(merge_count_responses(budget_threshold_df_raw))
    bonding_df_prepared = prepare_cumulative_count_cols_bonding(merge_count_responses(bonding_threshold_df_raw.fill_null(0)))
    return bonding_df_prepared, budget_df_prepared


@app.cell
def _(alt, budget_df_prepared):
    # Create an overlaid area chart with stepped plots of the columns "Descending low" and "Ascending high" overlaid on the same axes

    base = alt.Chart(budget_df_prepared).encode(
        x=alt.X('value:Q', title='Annual emissions budget (% supply)'),
        tooltip=['value:Q']
    ).properties(title='Tolerance range for LPT emissions budget')

    alt.layer(
        base.mark_area(opacity=0.5, interpolate='step-before').encode(
            y=alt.Y('Descending low:Q', title='# objections'),
            color=alt.value('lightblue')
        ),
        base.mark_area(opacity=0.5, interpolate='step-after').encode(
            y=alt.Y('Ascending high:Q', title='# objections'),
            color=alt.value('pink')
        )
    )
    return


@app.cell
def _(alt, bonding_df_prepared):
    bonding_base = alt.Chart(bonding_df_prepared).encode( # x axis with logit scale
        x=alt.X('value:Q', title='Bonding rate (% supply)'),
        tooltip=['value:Q']
    ).properties(title='Low and critical thresholds for LPT bonding rate')

    alt.layer(
        bonding_base.mark_area(opacity=0.5, interpolate='step-before').encode(
            y=alt.Y('Descending low:Q', title='# objections'),
            color=alt.value('lightblue')
        ),
        bonding_base.mark_area(opacity=0.5, interpolate='step-before').encode(
            y=alt.Y('Descending critical:Q', title='# objections'),
            color=alt.value('pink')
        )
    )
    return


@app.cell
def _(budget_df_prepared):
    budget_df_prepared
    return


@app.cell
def _(alt, pl):
    control_dilution_responses = {
        "Support control + reduce": 18,
        "Support control": 5,
        "Somewhat support": 2,
        "Abstain": 5
    }

    # Plot pie chart of this data structure
    control_dilution_df = pl.DataFrame({
        "Response": list(control_dilution_responses.keys()),
        "count": list(control_dilution_responses.values())
    })
    control_dilution_chart = alt.Chart(control_dilution_df).mark_arc().encode(
        theta=alt.Theta(field="count", type="quantitative"),
        color=alt.Color(field="Response", type="nominal"),
        tooltip=['response:N', 'count:Q']
    ).properties(title='Support for controlling dilution of LPT', width=600, height=400)
    # Give legend more space so that labels are not cut off
    control_dilution_chart = control_dilution_chart.configure_legend(
        orient='right',
        titleFontSize=14,
        labelFontSize=12,
        symbolSize=200,
    )

    control_dilution_chart
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

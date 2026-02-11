import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import polars as pl
    return alt, pl


@app.cell
def _(alt, pl):
    # Example data matching your sketch structure
    # Replace this with your actual data
    df = pl.DataFrame({
        "x": [43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0] * 3,
        "y": [500, 500, 500, 700, 1000, float("NaN"), float("NaN")] + 
             [600, 600, 700, 1000, 1500, float("NaN"), float("NaN")] +
             [800, 800, 800, 1300, float("NaN"), float("NaN"), float("NaN")],
        "y2": [1500] * 7 * 3,
        "category": ["12%"] * 7 + ["11.5%"] * 7 + ["11%"] * 7,
    }, strict=False)

    df2 = pl.DataFrame({
        "x": [43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0],
        "12": [500, 500, 500, 700, 1000, float("NaN"), float("NaN")],
        "11_5": [600, 600, 700, 1000, 1500, float("NaN"), float("NaN")],
        "11": [800, 800, 800, 1300, float("NaN"), float("NaN"), float("NaN")],
        "y2": [1500,1500,1500,1500,1500,1500,1500]
    }, strict=False)

    axes = alt.Chart(df2).encode(
        x=alt.X("x:Q", title="targetBondingRate")
    )

    y_scale = alt.Scale(domain=[0, 1500])

    area_12 = axes.mark_area(opacity=0.3, color="blue").encode(y=alt.Y("12:Q", scale=y_scale))
    area_11_5 = axes.mark_area(opacity=0.3, color="green").encode(y=alt.Y("11_5:Q", scale=y_scale))
    area_11 = axes.mark_area(opacity=0.3, color="red").encode(y=alt.Y("11:Q", scale=y_scale))

    final_chart = (area_12 + area_11_5 + area_11).encode(
        y2=alt.Y2("y2:Q", scale=alt.Scale(domain=[0,1500]), title="inflationChange")
    ).properties(
        width=500,
        height=400
    )
    return df, y_scale


@app.cell
def _(alt, df, y_scale):
    alt.Chart(df).mark_area(opacity=0.9).encode(
        x=alt.X("x:Q", title="targetBondingRate"),
        y=alt.Y("y:Q", title="inflationChange", scale=y_scale),
        y2=alt.Y2("y2:Q", scale=y_scale),
        color=alt.Color("category:N", legend=alt.Legend(title="H1 2026 dilution target"), scale=alt.Scale(scheme="pastel2"))
    ).properties(
        width=500,
        height=400,
        title="Admissible parameter tunings"
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

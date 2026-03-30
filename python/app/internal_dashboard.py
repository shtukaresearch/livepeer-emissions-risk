import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    import pandas as pd

    APP_DIR = Path(__file__).resolve().parent
    PYTHON_DIR = APP_DIR.parent
    SRC_DIR = PYTHON_DIR / "src"

    for candidate in (PYTHON_DIR, SRC_DIR):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    import marimo as mo

    from app.pages.common import dashboard_theme, hero_header, info_panel
    from app.pages.cost_lenses import render_cost_lenses
    from app.pages.efficiency import render_efficiency
    from app.pages.overview import render_overview
    from lpt_stake.config import DAILY_METRICS_PATH
    from lpt_stake.data.io import read_parquet

    return (
        DAILY_METRICS_PATH,
        dashboard_theme,
        hero_header,
        info_panel,
        mo,
        pd,
        read_parquet,
        render_cost_lenses,
        render_efficiency,
        render_overview,
    )


@app.cell
def _(mo):
    section = mo.ui.radio(
        options=["Overview", "Expense vs Dilution", "Efficiency"],
        value="Overview",
        label="Dashboard section",
    )
    return (section,)


@app.cell
def _(DAILY_METRICS_PATH, read_parquet):
    df = read_parquet(DAILY_METRICS_PATH) if DAILY_METRICS_PATH.exists() else None
    return (df,)


@app.cell
def _(df, mo):
    if df is None:
        window = None
    else:
        window = mo.ui.dropdown(
            options=["7D", "14D", "30D", "90D", "180D", "365D", "YTD", "All", "Custom"],
            value="30D",
            label="Window",
        )
    return (window,)


@app.cell
def _(df, mo, pd, window):
    if df is None:
        date_range = None
    else:
        min_date = pd.Timestamp(df["date"].min()).date()
        max_date = pd.Timestamp(df["date"].max()).date()

        if window.value == "All":
            preset_start, preset_end = min_date, max_date
        elif window.value == "7D":
            preset_start, preset_end = max(min_date, max_date - pd.Timedelta(days=6)), max_date
        elif window.value == "14D":
            preset_start, preset_end = max(min_date, max_date - pd.Timedelta(days=13)), max_date
        elif window.value == "30D":
            preset_start, preset_end = max(min_date, max_date - pd.Timedelta(days=29)), max_date
        elif window.value == "90D":
            preset_start, preset_end = max(min_date, max_date - pd.Timedelta(days=89)), max_date
        elif window.value == "180D":
            preset_start, preset_end = max(min_date, max_date - pd.Timedelta(days=179)), max_date
        elif window.value == "365D":
            preset_start, preset_end = max(min_date, max_date - pd.Timedelta(days=364)), max_date
        elif window.value == "YTD":
            preset_start, preset_end = max(min_date, max_date.replace(month=1, day=1)), max_date
        else:
            preset_start, preset_end = min_date, max_date

        date_range = mo.ui.date_range(
            start=min_date,
            stop=max_date,
            value=(preset_start, preset_end),
            label="Custom range",
            disabled=window.value != "Custom",
        )
    return (date_range,)


@app.cell
def _(date_range, df, pd, window):
    if df is None:
        coverage_label = None
        filtered_df = None
        range_label = None
        row_count = None
        window_note = None
    else:
        start_value, end_value = date_range.value
        start_ts, end_ts = pd.Timestamp(start_value), pd.Timestamp(end_value)

        filtered_df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()
        coverage_label = (
            f"{pd.Timestamp(df['date'].min()).date().isoformat()} to "
            f"{pd.Timestamp(df['date'].max()).date().isoformat()}"
        )
        range_label = f"{start_ts.date().isoformat()} to {end_ts.date().isoformat()}"
        row_count = len(filtered_df)
        if window.value == "Custom":
            window_note = "Custom range is active."
        else:
            window_note = f"Preset `{window.value}` is active."
    return coverage_label, filtered_df, range_label, row_count, window_note


@app.cell
def _(
    coverage_label,
    dashboard_theme,
    date_range,
    df,
    filtered_df,
    hero_header,
    info_panel,
    mo,
    range_label,
    render_cost_lenses,
    render_efficiency,
    render_overview,
    row_count,
    section,
    window,
    window_note,
):
    help_panel = info_panel(
        "Reading Guide",
        [
            "The controls at the top filter every chart to the selected time window. Use presets for quick scans and Custom for exact research periods.",
            "Y-axes labeled with (%) are percentages, not raw ratios. Protocol Fees (USD) and TBC Cost are dollar charts.",
        ],
    )

    if df is None:
        result = mo.md(
            """
            # Internal TBC Dashboard

            Derived metrics have not been built yet.

            From the `python/` directory, run:

            ```bash
            uv run lpt-build-metrics
            uv run marimo run app/internal_dashboard.py
            ```
            """
        )
    elif filtered_df is None or filtered_df.empty:
        result = mo.md("No rows fall inside the selected date range.")
    else:
        header = hero_header(
            "Livepeer Emissions Research Dashboard",
            "A research-facing view of token-based compensation as cash-equivalent burden, passive-holder dilution, and network spend efficiency.",
            [
                ("Coverage", coverage_label),
                ("In View", range_label),
                ("Rows", str(row_count)),
                ("Mode", window.value),
            ],
        )
        control_note = info_panel(
            "Controls",
            [
                window_note.replace("`", ""),
                "Use the section switcher to move between the top-line burden view, the cost translation view, and the efficiency view.",
            ],
        )

        if section.value == "Overview":
            page = render_overview(filtered_df, range_label)
        elif section.value == "Expense vs Dilution":
            page = render_cost_lenses(filtered_df, range_label)
        else:
            page = render_efficiency(filtered_df, range_label)

        result = mo.vstack(
            [
                dashboard_theme(),
                header,
                mo.hstack([section, window], widths=[2, 1], gap=1),
                date_range,
                mo.hstack([control_note, help_panel], widths=[1, 1], gap=1),
                page,
            ]
        )

    result
    return


if __name__ == "__main__":
    app.run()

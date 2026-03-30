"""Shared Altair charts for the internal dashboard."""

from __future__ import annotations

import altair as alt
import pandas as pd


def _base_chart(
    df: pd.DataFrame,
    title: str,
    subtitle: str | list[str] | None = None,
) -> alt.Chart:
    if subtitle is None:
        chart_title: str | alt.TitleParams = title
    else:
        subtitles = [subtitle] if isinstance(subtitle, str) else subtitle
        chart_title = alt.TitleParams(text=title, subtitle=subtitles)
    return alt.Chart(df).properties(width=700, height=280, title=chart_title)


def kpi_line(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    y_title: str,
    y_format: str | None = None,
    subtitle: str | list[str] | None = None,
) -> alt.Chart:
    """Render a simple time-series line chart."""

    axis = alt.Axis(title=y_title, format=y_format) if y_format else alt.Axis(title=y_title)
    tooltip = [alt.Tooltip(f"{x}:T", title="Date")]
    if y_format:
        tooltip.append(alt.Tooltip(f"{y}:Q", title=y_title, format=y_format))
    else:
        tooltip.append(alt.Tooltip(f"{y}:Q", title=y_title))

    return _base_chart(df, title, subtitle).mark_line().encode(
        x=alt.X(f"{x}:T", title="Date"),
        y=alt.Y(f"{y}:Q", axis=axis),
        tooltip=tooltip,
    )


def cost_vs_dilution(df: pd.DataFrame) -> alt.Chart:
    """Render side-by-side cost and dilution charts."""

    cost = kpi_line(
        df,
        "date",
        "tbc_cost_usd",
        "TBC Cost (USD)",
        "USD",
        subtitle="USD value paid via token issuance.",
    )
    dilution = kpi_line(
        df,
        "date",
        "dilution_rate",
        "Dilution Rate",
        "Dilution (%)",
        ".2%",
        subtitle="Same issuance shown as ownership loss for passive holders.",
    )
    return alt.hconcat(cost, dilution)


def ratio_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    y_title: str,
    y_format: str | None = None,
    subtitle: str | list[str] | None = None,
) -> alt.Chart:
    """Render ratio-style metrics over time."""

    axis = alt.Axis(title=y_title, format=y_format) if y_format else alt.Axis(title=y_title)
    tooltip = [alt.Tooltip(f"{x}:T", title="Date")]
    if y_format:
        tooltip.append(alt.Tooltip(f"{y}:Q", title=y_title, format=y_format))
    else:
        tooltip.append(alt.Tooltip(f"{y}:Q", title=y_title))

    return _base_chart(df, title, subtitle).mark_area(opacity=0.35).encode(
        x=alt.X(f"{x}:T", title="Date"),
        y=alt.Y(f"{y}:Q", axis=axis),
        tooltip=tooltip,
    )

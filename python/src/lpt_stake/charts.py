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


def cost_vs_net_burden(df: pd.DataFrame) -> alt.Chart:
    """Render side-by-side gross issuance and net passive burden charts."""

    gross = kpi_line(
        df,
        "date",
        "tbc_cost_usd",
        "Gross Issuance (Protocol Cost)",
        "USD",
        subtitle="Total dollar value of tokens issued — includes bonded-holder share.",
    )
    net = kpi_line(
        df,
        "date",
        "net_burden_usd",
        "Net Passive Burden",
        "USD",
        subtitle="Gross × (1 − participation rate). Transfer from unbonded to bonded holders only.",
    )
    return alt.hconcat(gross, net)


def economic_earnings_chart(df: pd.DataFrame) -> alt.Chart:
    """Render economic earnings (fees minus net passive burden) with a zero baseline.

    Above zero: network generates more in fees than it extracts from passive holders.
    Below zero: network runs a dilution-funded deficit.
    """
    title = alt.TitleParams(
        text="Economic Earnings (Fees − Net Passive Burden)",
        subtitle=["Positive = self-sustaining. Negative = deficit funded by dilution of passive holders."],
    )
    base = alt.Chart(df).properties(width=700, height=280, title=title)

    line = base.mark_line().encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("economic_earnings_usd:Q", axis=alt.Axis(title="USD")),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("economic_earnings_usd:Q", title="Economic Earnings (USD)", format="$,.2f"),
        ],
    )

    zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color="gray", strokeDash=[4, 4], strokeWidth=1
    ).encode(y="y:Q")

    return line + zero_rule


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

"""Overview page for the internal TBC dashboard."""

from __future__ import annotations

import marimo as mo
import pandas as pd

from lpt_stake.charts import kpi_line
from app.pages.common import (
    cost_translation_row,
    cumulative_dilution,
    fmt_number,
    holder_impact_block,
    metric_grid,
    section_intro,
)


def render_overview(df: pd.DataFrame, range_label: str):
    """Render top-line KPI summaries and charts."""

    latest = df.sort_values("date").iloc[-1]
    total_tbc = df["tbc_cost_usd"].sum(min_count=1)
    range_dilution = cumulative_dilution(df)

    cards = metric_grid(
        [
            (
                "Total TBC in Range",
                fmt_number(total_tbc, "usd"),
                "Total cash-equivalent burden imposed by issuance in the selected window.",
                "accent",
            ),
            (
                "Cumulative Dilution",
                fmt_number(range_dilution, "pct"),
                "Total ownership dilution absorbed by passive holders in the selected window.",
                None,
            ),
            (
                "Ending Forward Inflation",
                fmt_number(latest["forward_inflation_rate"], "pct"),
                "Annualized supply growth implied by the ending per-round reward setting.",
                None,
            ),
            (
                "Ending Participation Rate",
                fmt_number(latest["participation_rate"], "pct"),
                "Share of supply that remained bonded at the end of the selected range.",
                None,
            ),
        ]
    )

    chart_cost = kpi_line(
        df,
        "date",
        "tbc_cost_usd",
        "TBC Cost",
        "USD",
        subtitle="Dollar value paid via issuance over time.",
    )
    chart_market = kpi_line(
        df,
        "date",
        "tbc_pct_market_cap",
        "TBC as % of Market Cap",
        "TBC / market cap (%)",
        ".2%",
        subtitle="How large issuance burden was relative to network value.",
    )

    return mo.vstack(
        [
            section_intro(
                "Overview",
                "The top line view of how much value issuance transferred, who paid for it, and how large the burden was relative to network value.",
                kicker="Headline Metrics",
            ),
            cards,
            cost_translation_row(df),
            holder_impact_block(df),
            chart_cost,
            chart_market,
        ]
    )

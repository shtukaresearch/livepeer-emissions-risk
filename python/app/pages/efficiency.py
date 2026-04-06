"""Efficiency page for the internal TBC dashboard."""

from __future__ import annotations

import marimo as mo
import pandas as pd

from lpt_stake.charts import economic_earnings_chart, kpi_line, ratio_chart
from app.pages.common import common_footing_table, fmt_number, info_panel, metric_grid, section_intro


def render_efficiency(df: pd.DataFrame, range_label: str):
    """Render efficiency and context metrics for spend quality."""

    has_earnings = (
        "economic_earnings_usd" in df.columns
        and df["economic_earnings_usd"].notna().any()
    )
    if has_earnings:
        total_earnings = df["economic_earnings_usd"].sum(min_count=1)
        earnings_tone = "accent" if total_earnings >= 0 else "warning"
        earnings_label = "Surplus" if total_earnings >= 0 else "Deficit"
        earnings_card = metric_grid([
            (
                f"Economic Earnings ({earnings_label})",
                fmt_number(total_earnings, "usd"),
                "Fees minus net passive burden over the selected window. "
                "Positive: network is self-sustaining. "
                "Negative: dilution-funded deficit.",
                earnings_tone,
            )
        ])
    else:
        earnings_card = info_panel(
            "Economic Earnings Unavailable",
            "Fee data is required to compute economic earnings. Run the fee refresh step to populate this metric.",
            tone="warning",
        )

    blocks = [
        section_intro(
            "Efficiency",
            "This page asks what the network bought with issuance and whether the burden looks reasonable against fees, usage, and participation.",
            kicker="Burden vs Outcome",
        ),
        earnings_card,
        common_footing_table(df),
        kpi_line(
            df,
            "date",
            "participation_rate",
            "Participation Rate",
            "Participation (%)",
            ".1%",
            subtitle="Security context: how much supply remained actively bonded.",
        ),
    ]

    if "tbc_pct_fees" in df.columns and df["tbc_pct_fees"].notna().any():
        blocks.append(
            ratio_chart(
                df,
                "date",
                "tbc_pct_fees",
                "TBC as % of Fees",
                "TBC / fees (%)",
                ".0%",
                subtitle="Issuance burden relative to protocol revenue.",
            )
        )
        if "fees_usd" in df.columns and df["fees_usd"].notna().any():
            blocks.append(
                kpi_line(
                    df,
                    "date",
                    "fees_usd",
                    "Protocol Fees (USD)",
                    "USD",
                    subtitle="Actual fee generation available to offset network costs.",
                )
            )
        if has_earnings:
            blocks.append(economic_earnings_chart(df))
    else:
        blocks.append(
            info_panel(
                "Protocol Fee Data Missing",
                "Run fetch-fee-data.py or lpt-refresh-fee-data to populate the revenue comparison lens.",
                tone="warning",
            )
        )

    return mo.vstack(blocks)

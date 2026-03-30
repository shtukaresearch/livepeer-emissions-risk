"""Efficiency page for the internal TBC dashboard."""

from __future__ import annotations

import marimo as mo
import pandas as pd

from lpt_stake.charts import kpi_line, ratio_chart
from app.pages.common import common_footing_table, info_panel, section_intro


def render_efficiency(df: pd.DataFrame, range_label: str):
    """Render efficiency and context metrics for spend quality."""

    blocks = [
        section_intro(
            "Efficiency",
            "This page asks what the network bought with issuance and whether the burden looks reasonable against fees, usage, and participation.",
            kicker="Burden vs Outcome",
        ),
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
    else:
        blocks.append(
            info_panel(
                "Protocol Fee Data Missing",
                "Run fetch-fee-data.py or lpt-refresh-fee-data to populate the revenue comparison lens.",
                tone="warning",
            )
        )

    return mo.vstack(blocks)

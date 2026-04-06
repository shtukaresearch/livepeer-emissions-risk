"""Expense and dilution framing page."""

from __future__ import annotations

import marimo as mo
import pandas as pd

from lpt_stake.charts import cost_vs_dilution, cost_vs_net_burden, kpi_line, ratio_chart
from lpt_stake.metrics.frames import dilution_frame, expense_frame
from app.pages.common import holder_impact_block, section_intro


def render_cost_lenses(df: pd.DataFrame, range_label: str):
    """Render complementary cost framings for the same issuance history."""

    expense_df = expense_frame(df)
    dilution_df = dilution_frame(df)

    blocks = [
        section_intro(
            "Expense vs Dilution",
            "The same issuance history translated into two units: cash-equivalent cost and passive-holder ownership loss. "
            "Gross issuance counts all tokens issued; net passive burden adjusts for the bonded share that flows back to stakers.",
            kicker="Cost Translation",
        ),
        holder_impact_block(df),
        cost_vs_net_burden(df),
        cost_vs_dilution(df),
    ]

    if "tbc_pct_fees" in expense_df.columns and expense_df["tbc_pct_fees"].notna().any():
        blocks.append(
            ratio_chart(
                expense_df,
                "date",
                "tbc_pct_fees",
                "TBC as % of Fees",
                "TBC / fees (%)",
                ".0%",
                subtitle="Burden of issuance compared with protocol revenue.",
            )
        )
    else:
        blocks.append(
            mo.md(
                "Protocol fee data are not available for this range yet. Run the fee "
                "refresh step to populate the fee lens."
            )
        )

    blocks.append(
        kpi_line(
            dilution_df,
            "date",
            "forward_inflation_rate",
            "Forward Inflation Rate",
            "Forward inflation (%)",
            ".2%",
            subtitle=(
                "If the current per-round inflation setting stayed unchanged for a full year, "
                "this is the annualized supply growth it implies."
            ),
        )
    )

    return mo.vstack(blocks)

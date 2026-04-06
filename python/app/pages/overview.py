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
    total_net_burden = df["net_burden_usd"].sum(min_count=1) if "net_burden_usd" in df.columns else float("nan")
    range_dilution = cumulative_dilution(df)

    cards = metric_grid(
        [
            (
                "Gross Issuance (Protocol Cost)",
                fmt_number(total_tbc, "usd"),
                "Total token-based compensation paid out by the protocol. Includes tokens received by bonded holders who are net beneficiaries, not payers.",
                "accent",
            ),
            (
                "Net Passive Burden",
                fmt_number(total_net_burden, "usd"),
                "Gross TBC × (1 − participation rate). The actual dollar transfer from unbonded to bonded holders in the selected window.",
                None,
            ),
            (
                "Cumulative Dilution",
                fmt_number(range_dilution, "pct"),
                "Total ownership dilution absorbed by unbonded (passive) holders in the selected window. Bonded holders are compensated by the same issuance.",
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
                "Share of supply that remained bonded at the end of the selected range. Higher participation reduces net passive burden.",
                None,
            ),
        ]
    )

    chart_cost = kpi_line(
        df,
        "date",
        "tbc_cost_usd",
        "Gross Issuance (Protocol Cost)",
        "USD",
        subtitle="Total dollar value of tokens issued — protocol expenditure before netting out the bonded-holder share.",
    )
    chart_net_burden = kpi_line(
        df,
        "date",
        "net_burden_usd",
        "Net Passive Burden",
        "USD",
        subtitle="Gross issuance × (1 − participation rate). Dollar transfer from unbonded to bonded holders. Approaches zero as participation → 100%.",
    )
    chart_market = kpi_line(
        df,
        "date",
        "tbc_pct_market_cap",
        "Gross Issuance as % of Market Cap",
        "Gross TBC / market cap (%)",
        ".2%",
        subtitle="Protocol expenditure relative to network value (gross, unadjusted for participation).",
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
            chart_net_burden,
            chart_market,
        ]
    )

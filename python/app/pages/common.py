"""Shared summary widgets for the internal dashboard pages."""

from __future__ import annotations

from html import escape

import marimo as mo
import pandas as pd


def fmt_number(value: float, kind: str = "number") -> str:
    if pd.isna(value):
        return "n/a"
    if kind == "usd":
        return f"${value:,.2f}"
    if kind == "pct":
        return f"{value * 100:.2f}%"
    if kind == "bps":
        return f"{value * 10_000:,.1f} bps"
    if kind == "lpt":
        return f"{value:,.2f} LPT"
    return f"{value:,.2f}"


def dashboard_theme():
    return mo.md(
        """
<style>
:root {
  --surface-0: #f4f6f8;
  --surface-1: #ffffff;
  --surface-2: #eef2f6;
  --surface-3: #d8e1ea;
  --ink-1: #102033;
  --ink-2: #41556b;
  --ink-3: #6d7f92;
  --accent: #0f7a6c;
  --accent-soft: #e4f4f2;
  --warning: #8b5e16;
  --warning-soft: #fcf4df;
  --shadow: 0 10px 28px rgba(16, 32, 51, 0.08);
  --radius-lg: 22px;
  --radius-md: 16px;
}

.research-hero {
  padding: 1.4rem 1.5rem;
  border-radius: var(--radius-lg);
  background:
    radial-gradient(circle at top right, rgba(15, 122, 108, 0.12), transparent 32%),
    linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
  border: 1px solid var(--surface-3);
  box-shadow: var(--shadow);
  margin-bottom: 1rem;
}

.research-kicker {
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.72rem;
  color: var(--ink-3);
  margin-bottom: 0.45rem;
  font-weight: 700;
}

.research-title {
  font-size: 2rem;
  line-height: 1.1;
  color: var(--ink-1);
  margin: 0;
  font-weight: 800;
}

.research-subtitle {
  margin: 0.55rem 0 0;
  color: var(--ink-2);
  font-size: 1rem;
  max-width: 60rem;
}

.research-badges {
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
  margin-top: 1rem;
}

.research-badge {
  background: var(--surface-2);
  color: var(--ink-2);
  border: 1px solid var(--surface-3);
  border-radius: 999px;
  padding: 0.42rem 0.8rem;
  font-size: 0.82rem;
}

.research-section {
  margin: 0.2rem 0 0.8rem;
}

.research-section h1,
.research-section h2 {
  margin: 0;
  color: var(--ink-1);
}

.research-section p {
  margin: 0.45rem 0 0;
  color: var(--ink-2);
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.9rem;
  margin: 0.5rem 0 1rem;
}

.metric-card {
  background: var(--surface-1);
  border: 1px solid var(--surface-3);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow);
  padding: 1rem 1.05rem;
}

.metric-card.accent {
  background: linear-gradient(180deg, #ffffff 0%, var(--accent-soft) 100%);
  border-color: rgba(15, 122, 108, 0.24);
}

.metric-card.warning {
  background: linear-gradient(180deg, #ffffff 0%, var(--warning-soft) 100%);
  border-color: rgba(139, 94, 22, 0.24);
}

.metric-label {
  color: var(--ink-3);
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
  margin-bottom: 0.45rem;
}

.metric-value {
  color: var(--ink-1);
  font-size: 1.45rem;
  line-height: 1.15;
  font-weight: 800;
  margin-bottom: 0.35rem;
}

.metric-detail {
  color: var(--ink-2);
  font-size: 0.9rem;
  line-height: 1.45;
}

.research-panel {
  background: var(--surface-1);
  border: 1px solid var(--surface-3);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow);
  padding: 1rem 1.1rem;
  margin: 0.5rem 0 1rem;
}

.research-panel.accent {
  background: linear-gradient(180deg, #ffffff 0%, var(--accent-soft) 100%);
  border-color: rgba(15, 122, 108, 0.24);
}

.research-panel.warning {
  background: linear-gradient(180deg, #ffffff 0%, var(--warning-soft) 100%);
  border-color: rgba(139, 94, 22, 0.24);
}

.research-panel-title {
  color: var(--ink-1);
  font-weight: 800;
  margin: 0 0 0.35rem;
  font-size: 1rem;
}

.research-panel p {
  color: var(--ink-2);
  margin: 0.35rem 0 0;
}

.comparison-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 0.5rem;
}

.comparison-table th,
.comparison-table td {
  padding: 0.68rem 0.72rem;
  border-bottom: 1px solid var(--surface-3);
  text-align: left;
  font-size: 0.92rem;
}

.comparison-table th {
  color: var(--ink-3);
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.comparison-table td {
  color: var(--ink-1);
}
</style>
        """
    )


def _panel_class(tone: str = "neutral") -> str:
    if tone == "accent":
        return "research-panel accent"
    if tone == "warning":
        return "research-panel warning"
    return "research-panel"


def section_intro(title: str, description: str, kicker: str | None = None):
    kicker_html = (
        f'<div class="research-kicker">{escape(kicker)}</div>' if kicker else ""
    )
    return mo.md(
        f"""
<div class="research-section">
  {kicker_html}
  <h1>{escape(title)}</h1>
  <p>{escape(description)}</p>
</div>
        """
    )


def hero_header(title: str, subtitle: str, badges: list[tuple[str, str]]):
    badge_html = "".join(
        f'<div class="research-badge"><strong>{escape(label)}:</strong> {escape(value)}</div>'
        for label, value in badges
    )
    return mo.md(
        f"""
<div class="research-hero">
  <div class="research-kicker">Internal Research Dashboard</div>
  <h1 class="research-title">{escape(title)}</h1>
  <p class="research-subtitle">{escape(subtitle)}</p>
  <div class="research-badges">{badge_html}</div>
</div>
        """
    )


def metric_grid(cards: list[tuple[str, str, str, str | None]]):
    card_html = []
    for label, value, detail, tone in cards:
        extra_class = f" {tone}" if tone else ""
        card_html.append(
            f"""
<div class="metric-card{extra_class}">
  <div class="metric-label">{escape(label)}</div>
  <div class="metric-value">{escape(value)}</div>
  <div class="metric-detail">{escape(detail)}</div>
</div>
            """
        )
    return mo.md(f'<div class="metric-grid">{"".join(card_html)}</div>')


def info_panel(title: str, body: list[str] | str, tone: str = "neutral"):
    paragraphs = body if isinstance(body, list) else [body]
    body_html = "".join(f"<p>{escape(paragraph)}</p>" for paragraph in paragraphs)
    return mo.md(
        f"""
<div class="{_panel_class(tone)}">
  <div class="research-panel-title">{escape(title)}</div>
  {body_html}
</div>
        """
    )


def cumulative_dilution(df: pd.DataFrame) -> float:
    return 1 - (1 - df["dilution_rate"].fillna(0)).prod()


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def holder_impact_block(df: pd.DataFrame, holder_lpt: float = 10_000) -> mo.Html:
    latest = df.sort_values("date").iloc[-1]
    range_dilution = cumulative_dilution(df)
    lost_lpt = holder_lpt * range_dilution
    lost_usd = lost_lpt * latest["lpt_price_usd"] if pd.notna(latest["lpt_price_usd"]) else float("nan")

    return info_panel(
        "Passive (Unbonded) Holder Impact",
        [
            f"A passive holder with {holder_lpt:,.0f} LPT would lose about {fmt_number(lost_lpt, 'lpt')} of ownership over this range.",
            f"At the ending token price, that is roughly {fmt_number(lost_usd, 'usd')}.",
            "This impact falls only on unbonded holders. Bonded holders receive emissions proportional to their stake and are net beneficiaries, not payers.",
        ],
        tone="accent",
    )


def cost_translation_row(df: pd.DataFrame) -> mo.Html:
    avg_market_cap = pd.to_numeric(df["market_cap_usd"], errors="coerce").mean()
    total_fees = _numeric_series(df, "fees_usd").sum(min_count=1)
    total_tbc = pd.to_numeric(df["tbc_cost_usd"], errors="coerce").sum(min_count=1)
    burden_vs_market_cap = (
        total_tbc / avg_market_cap
        if pd.notna(avg_market_cap) and avg_market_cap
        else float("nan")
    )
    burden_vs_fees = (
        total_tbc / total_fees
        if pd.notna(total_fees) and total_fees
        else float("nan")
    )
    avg_daily_cost = pd.to_numeric(df["tbc_cost_usd"], errors="coerce").mean()
    total_net_burden = _numeric_series(df, "net_burden_usd").sum(min_count=1)

    return metric_grid(
        [
            (
                "Gross Cost as % of Avg Market Cap",
                fmt_number(burden_vs_market_cap, "pct"),
                "Protocol expenditure (gross issuance) normalized by average network value.",
                None,
            ),
            (
                "Gross Cost as % of Fees",
                fmt_number(burden_vs_fees, "pct"),
                "Protocol expenditure normalized by protocol revenue.",
                None,
            ),
            (
                "Average Daily Gross Cost",
                fmt_number(avg_daily_cost, "usd"),
                "Average gross issuance burden per day across the selected range.",
                None,
            ),
            (
                "Net Passive Burden",
                fmt_number(total_net_burden, "usd"),
                "Gross TBC adjusted for participation rate — the actual dollar transfer from unbonded to bonded holders.",
                None,
            ),
        ]
    )


def common_footing_table(df: pd.DataFrame) -> mo.Html:
    total_tbc = pd.to_numeric(df["tbc_cost_usd"], errors="coerce").sum(min_count=1)
    total_fees = _numeric_series(df, "fees_usd").sum(min_count=1)
    avg_market_cap = pd.to_numeric(df["market_cap_usd"], errors="coerce").mean()
    cumulative = cumulative_dilution(df)

    rows = [
        (
            "Token-based compensation",
            fmt_number(total_tbc, "usd"),
            fmt_number(total_tbc / avg_market_cap if pd.notna(avg_market_cap) and avg_market_cap else float("nan"), "pct"),
            fmt_number(total_tbc / total_fees if pd.notna(total_fees) and total_fees else float("nan"), "pct"),
            fmt_number(cumulative, "bps"),
        ),
    ]

    if pd.notna(total_fees):
        rows.append(
            (
                "Protocol fees",
                fmt_number(total_fees, "usd"),
                fmt_number(total_fees / avg_market_cap if pd.notna(avg_market_cap) and avg_market_cap else float("nan"), "pct"),
                fmt_number(1.0, "pct"),
                "n/a",
            )
        )

    row_html = "".join(
        f"""
<tr>
  <td>{escape(name)}</td>
  <td>{escape(usd_value)}</td>
  <td>{escape(market_cap_value)}</td>
  <td>{escape(fees_value)}</td>
  <td>{escape(dilution_value)}</td>
</tr>
        """
        for name, usd_value, market_cap_value, fees_value, dilution_value in rows
    )
    return mo.md(
        f"""
<div class="{_panel_class()}">
  <div class="research-panel-title">Common Footing Comparison</div>
  <p>Any future grant, subsidy, or operating cost should be added in these same units.</p>
  <table class="comparison-table">
    <thead>
      <tr>
        <th>Stream</th>
        <th>USD in Range</th>
        <th>% of Avg Market Cap</th>
        <th>Share of Fees</th>
        <th>Dilution-Equivalent Burden</th>
      </tr>
    </thead>
    <tbody>
      {row_html}
    </tbody>
  </table>
</div>
        """
    )


import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    from datetime import datetime
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import requests
    import yfinance as yf

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"

    state_path = data_dir / "lpt-daily-data.json"
    daily_output_path = data_dir / "Data2022-2025.csv"
    round_output_path = data_dir / "Data2022-2025[perRound].csv"

    return (
        data_dir,
        daily_output_path,
        datetime,
        mo,
        np,
        pd,
        requests,
        round_output_path,
        state_path,
        yf,
    )


@app.cell
def _(mo, state_path):
    mo.md(
        f"""
        # Data Fetching

        This notebook reads historic Livepeer state from `{state_path}` and enriches it
        with Yahoo Finance market data plus the Fear & Greed index.
        """
    )
    return


@app.cell
def _(pd, state_path):
    state_df = pd.read_json(state_path)
    state_df["date"] = pd.to_datetime(state_df["date"]).dt.date
    state_df
    return (state_df,)


@app.cell
def _(state_df):
    tickers = ["LPT-USD", "BTC-USD", "ETH-USD"]
    start = state_df["date"].iloc[0].strftime("%Y-%m-%d")
    end = state_df["date"].iloc[-1].strftime("%Y-%m-%d")
    return end, start, tickers


@app.cell
def _(end, pd, start, tickers, yf):
    market_raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1d",
        progress=False,
    )

    market_close = market_raw["Close"].reset_index()
    market_volume = market_raw["Volume"].reset_index()

    market_close.rename(
        columns={
            "Date": "date",
            "LPT-USD": "lpt_price_usd",
            "BTC-USD": "btc_price_usd",
            "ETH-USD": "eth_price_usd",
        },
        inplace=True,
    )
    market_volume.rename(
        columns={
            "Date": "date",
            "LPT-USD": "lpt_volume",
            "BTC-USD": "btc_volume",
            "ETH-USD": "eth_volume",
        },
        inplace=True,
    )

    market_df = pd.merge(market_close, market_volume, on="date", how="inner")
    market_df["date"] = pd.to_datetime(market_df["date"]).dt.date
    market_df
    return (market_df,)


@app.cell
def _(datetime, end, pd, requests, start):
    def fetch_fear_greed_index(start_date, end_date):
        response = requests.get("https://api.alternative.me/fng/?limit=0", timeout=20)
        response.raise_for_status()

        rows = response.json().get("data", [])
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["date", "fear_greed_index"])

        numeric_values = pd.to_numeric(df["timestamp"], errors="coerce")
        parsed_from_unix = pd.to_datetime(numeric_values, unit="s", errors="coerce", utc=True)
        parsed_from_text = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["date"] = parsed_from_unix.fillna(parsed_from_text).dt.date
        df = df[["date", "value"]].rename(columns={"value": "fear_greed_index"})
        df["fear_greed_index"] = pd.to_numeric(df["fear_greed_index"], errors="coerce")
        return df[(df["date"] >= start_date) & (df["date"] <= end_date)].sort_values("date")

    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()
    fear_greed_df = fetch_fear_greed_index(start_date, end_date).reset_index(drop=True)
    fear_greed_df
    return (fear_greed_df,)


@app.cell
def _(daily_output_path, fear_greed_df, market_df, state_df):
    final_df = fear_greed_df.merge(market_df, on="date", how="inner")
    final_df = final_df.merge(state_df, on="date", how="inner")
    final_df.to_csv(daily_output_path, index=False)
    final_df
    return (final_df,)


@app.cell
def _(daily_output_path, mo):
    mo.md(f"Saved merged daily dataset to `{daily_output_path}`.")
    return


@app.cell
def _(np, pd, round_output_path, final_df):
    rounds_per_day = 24 / 21

    round_source_df = final_df.copy()
    round_source_df = round_source_df.interpolate(method="linear")
    round_source_df["day_idx"] = range(1, len(round_source_df) + 1)

    value_columns = [column for column in round_source_df.columns if column != "date"]
    round_source_df[value_columns] = round_source_df[value_columns].apply(pd.to_numeric, errors="coerce")

    round_positions = np.linspace(
        round_source_df["day_idx"].min(),
        round_source_df["day_idx"].max(),
        int(rounds_per_day * len(round_source_df)),
    )
    nearest_idx = np.abs(round_source_df["day_idx"].values[:, None] - round_positions).argmin(axis=0)

    round_data = {
        column: round_source_df[column].iloc[nearest_idx].values for column in value_columns
    }
    round_data["date"] = round_source_df["date"].iloc[nearest_idx].values

    round_df = pd.DataFrame(round_data)
    round_df.insert(0, "round", range(1, len(round_df) + 1))
    round_df.to_csv(round_output_path, index=False)
    round_df
    return (round_df,)


@app.cell
def _(mo, round_output_path):
    mo.md(f"Saved per-round dataset to `{round_output_path}`.")
    return


if __name__ == "__main__":
    app.run()

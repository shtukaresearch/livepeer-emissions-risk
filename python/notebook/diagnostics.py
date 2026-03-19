import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("""
    # Livepeer Emissions Risk — Diagnostics

    Compare participation model variants and feature sets.  Fit on a
    training window, evaluate on held-out data, inspect coefficients
    and residuals.
    """)
    return (mo,)


@app.cell
def _():
    import altair as alt
    import numpy as np
    import polars as pl

    from lpt_stake import (
        DiffLogitParticipationModel,
        GaussianNoise,
        LogitParticipationModel,
        RawParticipationModel,
    )
    from lpt_stake.features import annualise_ppb, logit
    from lpt_stake.constants import ROUNDS_PER_YEAR
    return (
        DiffLogitParticipationModel,
        GaussianNoise,
        LogitParticipationModel,
        ROUNDS_PER_YEAR,
        RawParticipationModel,
        alt,
        annualise_ppb,
        logit,
        np,
        pl,
    )


@app.cell
def _(ROUNDS_PER_YEAR, annualise_ppb, logit, pl):
    import os
    data_path = os.environ.get("LPT_DATA_SOURCE")
    raw = pl.read_csv(data_path, glob=False)

    df = raw.select(
        pl.col("round").cast(pl.Int64),
        pl.col("date").str.to_date(),
        (pl.col("bonded") / pl.col("total-supply")).alias("participation_rate"),
        pl.col("inflation").alias("emissions_rate"),
        pl.col("fear_greed_index"),
        pl.col("eth_price_usd"),
        pl.col("lpt_price_usd"),
    ).with_columns(
        pl.col("participation_rate").pipe(logit).alias("logit_participation"),
        pl.col("emissions_rate").pipe(annualise_ppb, ROUNDS_PER_YEAR).alias("inflation_annual"),
    ).drop_nulls()

    df
    return (df,)


@app.cell
def _(GaussianNoise):
    FEATURE_COLS = ["logit_participation", "inflation_annual", "fear_greed_index"]
    ZERO_NOISE = GaussianNoise(sigma=0.0)
    return FEATURE_COLS, ZERO_NOISE


@app.function
def evaluate_on_test(model, df_test, feature_cols, np):
    """One-step-ahead prediction on test data. Returns (predictions, actuals, rmse, mae)."""
    p_test = df_test["participation_rate"].to_numpy()
    preds = []
    for i in range(len(df_test) - 1):
        fv = np.array([[df_test[col][i] for col in feature_cols]])
        pred = model.predict(fv, np.array([p_test[i]]), np.random.default_rng(0))
        preds.append(pred[0])
    preds = np.array(preds)
    actuals = p_test[1:]
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    mae = float(np.mean(np.abs(preds - actuals)))
    return preds, actuals, rmse, mae


@app.cell
def _(mo):
    mo.md("""
    ## Train / test split
    """)
    return


@app.cell
def _(df, mo):
    n_total = len(df)
    split_slider = mo.ui.slider(
        start=100, stop=n_total - 50, step=10,
        value=int(n_total * 0.8),
        label=f"Training set size (of {n_total})",
    )
    split_slider
    return (split_slider,)


@app.cell
def _(df, mo, split_slider):
    n_train = split_slider.value
    df_train = df.head(n_train)
    df_test = df.slice(n_train)
    mo.md(f"**Train:** {len(df_train)} rows  |  **Test:** {len(df_test)} rows")
    return df_test, df_train


@app.cell
def _(mo):
    mo.md("""
    ## Model comparison
    """)
    return


@app.cell
def _(mo):
    alpha_slider = mo.ui.slider(
        start=0.0, stop=2.0, step=0.01, value=0.1, label="Ridge alpha"
    )
    alpha_slider
    return (alpha_slider,)


@app.cell
def _(
    DiffLogitParticipationModel,
    FEATURE_COLS,
    LogitParticipationModel,
    RawParticipationModel,
    ZERO_NOISE,
    alpha_slider,
    df_test,
    df_train,
    np,
    pl,
):
    alpha = alpha_slider.value

    models = {
        "Raw": RawParticipationModel(features=FEATURE_COLS, target="participation_rate", alpha=alpha, noise=ZERO_NOISE),
        "Logit": LogitParticipationModel(features=FEATURE_COLS, target="participation_rate", alpha=alpha, noise=ZERO_NOISE),
        "DiffLogit": DiffLogitParticipationModel(features=FEATURE_COLS, target="participation_rate", alpha=alpha, noise=ZERO_NOISE),
    }

    for _m in models.values():
        _m.fit(df_train)

    results = {}
    for _name, _m in models.items():
        _preds, _actuals, _rmse, _mae = evaluate_on_test(_m, df_test, FEATURE_COLS, np)
        results[_name] = {
            "RMSE": _rmse,
            "MAE": _mae,
            "AIC": _m.ridge_result.aic,
            "BIC": _m.ridge_result.bic,
            "Eff. df": _m.ridge_result.effective_df,
            "Resid. std": _m.ridge_result.residual_std,
            "predictions": _preds,
            "actuals": _actuals,
        }

    comparison = pl.DataFrame({
        "Model": list(results.keys()),
        "RMSE (test)": [r["RMSE"] for r in results.values()],
        "MAE (test)": [r["MAE"] for r in results.values()],
        "AIC (train)": [r["AIC"] for r in results.values()],
        "BIC (train)": [r["BIC"] for r in results.values()],
        "Eff. df": [r["Eff. df"] for r in results.values()],
        "Resid. std (train)": [r["Resid. std"] for r in results.values()],
    })
    comparison
    return models, results


@app.cell
def _(mo):
    mo.md("""
    ## Coefficients
    """)
    return


@app.cell
def _(models, pl):
    _rows = []
    for _name, _m in models.items():
        for _feat, _coef in _m.ridge_result.coefficients.items():
            _se = _m.ridge_result.coefficient_std_errors[_feat]
            _rows.append({"Model": _name, "Feature": _feat, "Coefficient": _coef, "Std Error": _se})
    coef_table = pl.DataFrame(_rows)
    coef_table
    return


@app.cell
def _(mo):
    mo.md("""
    ## Residual diagnostics
    """)
    return


@app.cell
def _(mo):
    model_picker = mo.ui.dropdown(
        options=["Raw", "Logit", "DiffLogit"],
        value="Logit",
        label="Model to inspect",
    )
    model_picker
    return (model_picker,)


@app.cell
def _(alt, model_picker, models, np, pl):
    _m = models[model_picker.value]
    resid = _m.ridge_result.residuals

    resid_df = pl.DataFrame({
        "index": np.arange(len(resid)),
        "residual": resid,
    })

    resid_time = alt.Chart(resid_df).mark_point(size=10, opacity=0.5).encode(
        x=alt.X("index:Q", title="Observation"),
        y=alt.Y("residual:Q", title="Residual"),
    ).properties(title=f"{model_picker.value} — Residuals over time", width=700, height=250)

    resid_hist = alt.Chart(resid_df).mark_bar().encode(
        x=alt.X("residual:Q", bin=alt.Bin(maxbins=40), title="Residual"),
        y=alt.Y("count():Q", title="Count"),
    ).properties(title=f"{model_picker.value} — Residual distribution", width=700, height=250)

    resid_time & resid_hist
    return


@app.cell
def _(mo):
    mo.md("""
    ## One-step-ahead predictions vs actuals (test set)
    """)
    return


@app.cell
def _(alt, model_picker, np, pl, results):
    r = results[model_picker.value]
    pred_df = pl.DataFrame({
        "index": np.arange(len(r["actuals"])),
        "actual": r["actuals"],
        "predicted": r["predictions"],
    }).unpivot(
        index="index",
        on=["actual", "predicted"],
        variable_name="series",
        value_name="participation_rate",
    )

    pred_chart = alt.Chart(pred_df).mark_line().encode(
        x=alt.X("index:Q", title="Test observation"),
        y=alt.Y("participation_rate:Q", title="Participation Rate", scale=alt.Scale(zero=False)),
        color="series:N",
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(
            domain=["actual", "predicted"],
            range=[[0], [5, 3]],
        )),
    ).properties(
        title=f"{model_picker.value} — Predicted vs Actual (test set)",
        width=700, height=300,
    )
    pred_chart
    return


@app.cell
def _(mo):
    mo.md("""
    ## Alpha sweep
    """)
    return


@app.cell
def _(
    FEATURE_COLS,
    LogitParticipationModel,
    ZERO_NOISE,
    alt,
    df_test,
    df_train,
    np,
    pl,
):
    alphas = np.logspace(-3, 1, 20)
    sweep_rows = []

    for _a in alphas:
        _m = LogitParticipationModel(
            features=FEATURE_COLS, target="participation_rate",
            alpha=float(_a), noise=ZERO_NOISE,
        )
        _m.fit(df_train)
        _, _, _rmse, _ = evaluate_on_test(_m, df_test, FEATURE_COLS, np)
        sweep_rows.append({
            "alpha": float(_a),
            "RMSE": _rmse,
            "AIC": _m.ridge_result.aic,
            "BIC": _m.ridge_result.bic,
            "eff_df": _m.ridge_result.effective_df,
        })

    sweep_df = pl.DataFrame(sweep_rows)

    sweep_rmse = alt.Chart(sweep_df).mark_line(point=True).encode(
        x=alt.X("alpha:Q", scale=alt.Scale(type="log"), title="Alpha"),
        y=alt.Y("RMSE:Q", title="Test RMSE", scale=alt.Scale(zero=False)),
    ).properties(title="Alpha sweep — Test RMSE", width=350, height=250)

    sweep_ic = alt.Chart(
        sweep_df.select("alpha", "AIC", "BIC")
        .unpivot(index="alpha", variable_name="criterion", value_name="value")
    ).mark_line(point=True).encode(
        x=alt.X("alpha:Q", scale=alt.Scale(type="log"), title="Alpha"),
        y=alt.Y("value:Q", title="Information Criterion"),
        color="criterion:N",
    ).properties(title="Alpha sweep — AIC / BIC", width=350, height=250)

    sweep_rmse | sweep_ic
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

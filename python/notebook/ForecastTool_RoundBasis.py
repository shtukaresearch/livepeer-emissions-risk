# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta

    from sklearn.linear_model import RidgeCV
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    return KFold, Ridge, RidgeCV, mo, np, pd, plt


@app.cell
def _(np, pd):
    def params():
        # Policy / simulation params
        sigma = 0.0000005 # inflation change step per round
        P_star = 0.5 # target participation rate P^*
        gamma_min = 0.0001 # lower bound of inflation per round
        gamma_max = 0.01 # upper bound of inflation per round

        # Simulation
        horizon_blocks = 180
        n_sims = 20
        random_seed = 42

        # Risk band for participation (D0 = [Plow, Phigh])
        Plow = 0.40
        Phigh = 0.60

        # Admissibility thresholds (from framework)
        T_star = 10 # expected number of days outside D0 allowed over horizon
        Ttail = 20 # tail threshold: unacceptable if time-outside > Ttail
        eps_tail = 0.05 # allowed probability of exceeding Ttail across sims
        gamma_star = 0.25 # target emission rate
        yield_star = 0.4 # target yield rate

        return dict(sigma=sigma, P_star=P_star, gamma_min=gamma_min, gamma_max=gamma_max,
                    horizon_blocks=horizon_blocks, n_sims=n_sims,
                    random_seed=random_seed, Plow=Plow, Phigh=Phigh,
                    T_star=T_star, Ttail=Ttail, eps_tail=eps_tail)


    # ------------------------------------------------------------
    # Load and prepare data
    # ------------------------------------------------------------
    import os
    def load_data():
        path = os.getenv("LPT_DATA_SOURCE")    # adjust path if needed
        return pd.read_csv(path)
    df_raw = load_data()


    df_raw['participation-rate'] = df_raw['bonded']/df_raw['total-supply']
    # Fix the inflation calculation:
    df_raw["inflation_per_round"] = df_raw["inflation"]/1e9  # inflation per round
    df_raw["annual_inflation_rate"] = (1 + df_raw["inflation_per_round"]) ** 417 - 1 # annualized issuance rate
    df_raw["fng_extreme"] = 2 * (df_raw["fear_greed_index"] - 50)
    df_raw["fng_extreme_absolute"] = 2 * np.abs(df_raw["fear_greed_index"] - 50)

    df = df_raw.set_index("round")

    # Identify key columns
    p_col = "participation-rate"
    g_col = "inflation_per_round"

    # Convert numerics
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[p_col, g_col])
    #mo.ui.data_explorer(df)
    return df, df_raw, params


@app.cell
def _(df, mo):
    window_size_training = mo.ui.slider(100, len(df), step=10, value=828, label='training set size')
    window_size_test = mo.ui.slider(0, 200, step=10, value=0, label='test set size')

    features_list = ["fear_greed_index", "fng_extreme", "fng_extreme_absolute", "btc_price_usd", "eth_price_usd", "lpt_price_usd", 
                    'btc_volume', 'eth_volume', 'lpt_volume']
    feature_selector = mo.ui.multiselect(options=features_list, value=["fear_greed_index", "btc_price_usd"], label="Select Exogeneous Variables")

    switch_differencing = mo.ui.switch(label="Differencing the target", value=True)
    return (
        feature_selector,
        switch_differencing,
        window_size_test,
        window_size_training,
    )


@app.cell
def _(
    df,
    mo,
    np,
    pd,
    plt,
    switch_differencing,
    window_size_test,
    window_size_training,
):
    def prepare_data(df, target_col, features, cutoff_round):
        # We'll construct the regression to predict Y_{t+1} from [1, Y_t, gamma_t, x_t]
        df2 = df.copy()

        # Find rows with any NaN
        rows_with_nan = df2[df2.isnull().any(axis=1)]

        # Fill NaN in those columns with their respective mean
        #df2['fear-greed-index'] = df2['fear_greed_index'].fillna(df2['fear_greed_index'].mean())

        df2.rename(columns={'participation-rate': 'P'}, inplace=True)
        df2.rename(columns={'inflation_per_round': 'I'}, inplace=True)
        df2['Y'] = np.log(df2['P']/(1-df2['P']))
        df2['logP'] = np.log(df2['P'])
        # target y = Y_{t+1}
        df2['Y_next'] = df2['Y'].shift(-1)
        df2['logP_next'] = df2['logP'].shift(-1)
        df2['P_next'] = df2['P'].shift(-1)
        # drop last row with NaN target
        df2 = df2.iloc[:-1]

        # exogenous features: 
        exog_cols = [c for c in df2.columns if c in features]
        # design matrix columns: intercept, Y_t, gamma_t, exog...
        X = pd.DataFrame(index=df2.index)
        X['intercept'] = 1.0
        if target_col == 'logit':
            X['Y_t'] = df2['Y']
            y = df2['Y_next'].values
        elif target_col == 'log':
            X['logP_t'] = df2['logP']
            y = df2['logP_next'].values
        else:
            X['P_t'] = df2['P']
            y = df2['P_next'].values

        X['I_t'] = df2['I']
        for c in exog_cols:
            if c not in ['intercept', 'Y_t', 'logP_t', 'P_t','I_t']:
                X[c] = df2[c]        

        # Split to Training Set
        cutoff = cutoff_round
        X_train = X.loc[:cutoff]
        X_test = X.loc[cutoff:]
        cutoff_loc = X.index.get_loc(cutoff)
        y_train = y[:cutoff_loc + 1]
        y_test = y[cutoff_loc + 1:]

        return X_train, y_train, X_test, y_test


    def prepare_data_updated(df, target_col, features):
        # We'll construct the regression to predict Y_{t+1} from [1, Y_t, gamma_t, x_t]
        df2 = df.copy()

        # Find rows with any NaN
        rows_with_nan = df2[df2.isnull().any(axis=1)]

        # Fill NaN in those columns with their respective mean
        #df2['fear-greed-index'] = df2['fear_greed_index'].fillna(df2['fear_greed_index'].mean())

        df2.rename(columns={'participation-rate': 'P'}, inplace=True)
        df2.rename(columns={'inflation_per_round': 'I'}, inplace=True)
        df2['Y'] = np.log(df2['P']/(1-df2['P']))
        df2['logP'] = np.log(df2['P'])
        # target y = Y_{t+1}
        df2['Y_next'] = df2['Y'].shift(-1)
        df2['logP_next'] = df2['logP'].shift(-1)
        df2['P_next'] = df2['P'].shift(-1)

        # add first-order differencing
        df2['Y_diff'] = df2['Y'] - df2['Y'].shift(1)
        df2['logP_diff'] = df2['logP'] - df2['logP'].shift(1)
        df2['P_diff'] = df2['P'] - df2['P'].shift(1)

        df2['Y_diff_next'] = df2['Y_diff'].shift(-1)
        df2['logP_diff_next'] = df2['logP_diff'].shift(-1)
        df2['P_diff_next'] = df2['P_diff'].shift(-1)

        # drop first and last row with NaN difference and target
        df2 = df2.iloc[1:-1]

        # exogenous features: 
        exog_cols = [c for c in df2.columns if c in features]
        # design matrix columns: intercept, Y_t, gamma_t, exog...
        X = pd.DataFrame(index=df2.index)
        X['intercept'] = 1.0

        # Split to Training Set
        train_idx_end = min(len(X), start_idx_training.value + window_size_training.value)
        test_idx_end = min(len(X), train_idx_end + window_size_test.value)

        if switch_differencing.value:
            if target_col == 'logit':
                X['Y_t_diff'] = df2['Y_diff']
                y = df2['Y_diff_next'].values
                P0 = df2['Y'].iloc[train_idx_end - 1] # last training Participation rate
                y0 = df2['Y_next'].iloc[train_idx_end - 1] # last training Participation rate target
            elif target_col == 'log':
                X['logP_t_diff'] = df2['logP_diff']
                y = df2['logP_diff_next'].values
                P0 = df2['logP'].iloc[train_idx_end - 1] # last training Participation rate
                y0 = df2['logP_next'].iloc[train_idx_end - 1] # last training Participation rate target
            else:
                X['P_t_diff'] = df2['P_diff']
                y = df2['P_diff_next'].values
                P0 = df2['P'].iloc[train_idx_end - 1] # last training Participation rate
                y0 = df2['P_next'].iloc[train_idx_end - 1] # last training Participation rate target
        else:
            if target_col == 'logit':
                X['Y_t'] = df2['Y']
                y = df2['Y_next'].values
                P0 = df2['Y'].iloc[train_idx_end - 1] # last training Participation rate
                y0 = df2['Y_next'].iloc[train_idx_end - 1] # last training Participation rate target
            elif target_col == 'log':
                X['logP_t'] = df2['logP']
                y = df2['logP_next'].values
                P0 = df2['logP'].iloc[train_idx_end - 1] # last training Participation rate
                y0 = df2['logP_next'].iloc[train_idx_end - 1] # last training Participation rate target
            else:
                X['P_t'] = df2['P']
                y = df2['P_next'].values
                P0 = df2['P'].iloc[train_idx_end - 1] # last training Participation rate
                y0 = df2['P_next'].iloc[train_idx_end - 1] # last training Participation rate target

        X['I_t'] = df2['I']
        for c in exog_cols:
            if c not in ['intercept', 'Y_t', 'logP_t', 'P_t','I_t', 'Y_t_diff', 'logP_t_diff', 'P_t_diff']:
                X[c] = df2[c]        

        X_train = X[start_idx_training.value: train_idx_end]
        X_test = X[train_idx_end: test_idx_end]
        y_train = y[start_idx_training.value: train_idx_end]
        y_test = y[train_idx_end: test_idx_end]

        return X_train, y_train, X_test, y_test, X, y, P0, y0

    # Reactive cell: plot and split
    def plot_and_split(cutoff_round):
        cutoff = cutoff_round
        train_data = df[:cutoff]
        test_data = df[cutoff:]

        # Plot DataFrame
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df.index, df["participation-rate"], label="Participation Rate", color="blue")
        ax.axvline(cutoff, color="red", linestyle="--", label="Cutoff Round")
        #ax.set_title("Data over Time")
        ax.set_xlabel("Round")
        ax.set_ylabel("P")
        ax.legend()

        return mo.vstack([
            fig,
            mo.md(f"**Cutoff Round for training & test sets:** {cutoff_round}"),
            mo.md(f"Training Size: {len(train_data)} days"),
            mo.md(f"Test Size: {len(test_data)} days")
        ])

    def split_plot():
        train_idx_end = min(len(df), start_idx_training.value + window_size_training.value)
        train_data = df[start_idx_training.value : train_idx_end]
        test_data = df[train_idx_end: train_idx_end + window_size_test.value]

        fig, ax = plt.subplots()
        ax.plot(df.index, df["participation-rate"], label="Participation Rate", color="blue")
        ax.axvspan(start_idx_training.value, train_idx_end, color="green", alpha=0.3, label="Train")
        ax.axvspan(train_idx_end, train_idx_end + window_size_test.value, color="orange", alpha=0.3, label="Test")
        ax.legend()
        return mo.vstack([
            fig,
            mo.md(f"Training Size: {len(train_data)} rounds"),
            mo.md(f"Test Size: {len(test_data)} rounds")
        ])



    # CV selection UI
    round_picker = mo.ui.slider(start = 10, stop = len(df), label="Training and Test sets cutoff", value=600)

    start_idx_training = mo.ui.slider(0, len(df) - window_size_training.value-1, step=2, value=len(df) - window_size_training.value-1, label='training start-point')

    #start_idx_test= mo.ui.slider(100, len(df) - window_size_test.value, step=2, value=len(df)- window_size_test.value, label='test start-point')

    kfold_dropdown = mo.ui.dropdown(options=range(2, 11), value=5, label="Number of Folds (k)")
    return (
        kfold_dropdown,
        plot_and_split,
        prepare_data_updated,
        round_picker,
        split_plot,
        start_idx_training,
    )


@app.cell
def _(
    KFold,
    Ridge,
    RidgeCV,
    df,
    feature_selector,
    kfold_dropdown,
    mo,
    np,
    plt,
    prepare_data_updated,
    radio_transform,
):
    def fit_linear(X, y):
        dP = df['P'].diff().dropna()
        P_lag = df['P'].shift(1).dropna()
        # regress dP on (P_tau - P_lag)
        X = (0.5 - P_lag).loc[dP.index].values.reshape(-1,1)  # using 0.5 as example
        beta = np.linalg.lstsq(X, dP.values, rcond=None)[0][0]
        sigma_eps = np.std(dP - beta * X.ravel())
        return dict(beta=float(beta), sigma_eps=float(sigma_eps))


    def fit_ridge(X, y, exog_cols):
        model = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 2, 4, 10, 20, 50], cv=kfold_dropdown.value)
        model.fit(X, y)
        best_lambda = model.alpha_

        n_features = X.shape[1]
        # ridge closed form: theta = (X^T X + lambda * I)^{-1} X^T y
        I = np.eye(n_features)
        I[0,0] = 0 # don't regularize intercept

        beta = np.linalg.inv(X.T @ X + best_lambda * I) @ X.T @ y

        # split coefficients
        coef = dict()
        for i, c in enumerate(exog_cols):
            coef[c] = float(beta[i])


        # residual variance
        y_pred = np.dot(X, beta)
        resid = y - y_pred
        sigma_eps = float(np.std(resid, ddof=1))
        sigma_boot = resid - resid.mean()

        # Use residuals as shocks
        shocks = y[1:] - y[:-1] #resid - resid.mean()  # center
        sigma_hat = np.std(shocks, ddof=1)
        # Threshold: e.g. 3 standard deviations = “jump”
        threshold = 3 * sigma_hat
        jump_mask = np.abs(shocks) > threshold
        jumps = shocks[jump_mask]
        N_jump = jump_mask.sum()
        p_jump = N_jump / len(shocks)   # probability of a jump at any time step


        return coef, beta, sigma_eps, sigma_boot, jumps, p_jump

    def ridge_cv(X, y):
        model = RidgeCV(alphas=[0.01, 0.1, 0.5, 1, 2, 4, 10, 20, 50], cv=kfold_dropdown.value)
        model.fit(X, y)
        best_lambda = model.alpha_

        kf = KFold(n_splits=kfold_dropdown.value, shuffle=True, random_state=42)

        coefs = []
        for train_idx, test_idx in kf.split(X):
            model = Ridge(alpha=best_lambda)
            model.fit(X.iloc[train_idx], y[train_idx])
            coefs.append(model.coef_)

        coefs = np.array(coefs)
        #print("Coefficient mean:", coefs.mean(axis=0))
        #print("Coefficient std:", coefs.std(axis=0))

        fig, axes = plt.subplots(coefs.shape[1], 1, figsize=(6, 6), sharex=True)

        for i, ax in enumerate(axes):
            ax.plot(coefs[:, i], color='blue')
            ax.set_ylabel(f'Feature {X.columns[i]}')
            # Each subplot has its own y-scale (default behavior)
        ax.set_xlabel("Fold")
        '''fig, ax = plt.subplots()
        for j in range(coefs.shape[1]):
            ax.plot(coefs[:, j], marker='o', label=f'Feature {X.columns[j]}')

        ax.set_xlabel("Fold")
        ax.set_ylabel("Coefficient value")
        ax.set_title("Coefficient Stability Across Folds")
        ax.legend()'''

        return fig, coefs.mean(axis=0), coefs.std(axis=0)


    def show_fit():
        target_col = radio_transform.value
        exog_cols = feature_selector.value
        X_train, y_train, X_test, y_test, X, y, P0, y0 = prepare_data_updated(df, target_col, exog_cols)
        opt_coef, opt_beta, sigma_eps, sigma_boot, jumps, p_jump = fit_ridge(X_train, y_train, X_train.columns.to_list())
        y_pred = np.dot(X_test, opt_beta)
        RMSE = np.mean((y_test - y_pred)**2)**0.5

        # AIC
        rss = np.sum((y_test - y_pred)**2)
        n = len(y_test)
        k = X.shape[1]  # parameters + intercept
        sigma2 = rss / n
        logL = -n/2 * (np.log(2*np.pi*sigma2) + 1)
        aic = 2*k - 2*logL


        fig_cv, coefs_mean, coefs_std = ridge_cv(X, y)

        return mo.vstack([
                            mo.md(f"### Ridge Coefficients\n{opt_coef}"),
                            mo.md(f"### RMSE on Test Set: **{RMSE:.4f}**"),
                            mo.md(f"### AIC: **{aic:.4f}**"),
                            mo.md(f"### Validation of coefficients\naverage across folds: {coefs_mean}"),
                            mo.md(f"variance across folds: {coefs_std}"), 
                            fig_cv
                        ])
    return fit_ridge, show_fit


@app.cell
def _(
    feature_selector,
    kfold_dropdown,
    mo,
    plot_and_split,
    round_picker,
    show_fit,
    split_plot,
    start_idx_training,
    window_size_test,
    window_size_training,
):
    # Display UI and plot
    data_prep_control = mo.vstack([
        round_picker,
        plot_and_split(round_picker.value)
    ], align='center')

    data_prep_control2 = mo.vstack([
        mo.hstack([window_size_training, mo.md(f"{window_size_training.value}")]),
        mo.hstack([start_idx_training, mo.md(f"{start_idx_training.value}")]), 
        mo.hstack([window_size_test, mo.md(f"{window_size_test.value}")]), 
        #mo.hstack([start_idx_test, mo.md(f"{start_idx_test.value}")]),
        mo.hstack([feature_selector, kfold_dropdown]),
        mo.hstack([split_plot(), show_fit()])
    ], align='center')

    #data_prep_control2
    return (data_prep_control2,)


@app.cell
def _(
    df,
    dropdown_block_size,
    feature_selector,
    fit_ridge,
    np,
    params,
    plt,
    prepare_data_updated,
    radio_horizon,
    radio_paths,
    radio_residual,
    radio_sampling,
    radio_transform,
    slider_P_star,
    slider_fan,
    slider_gamma_max,
    slider_gamma_min,
    slider_sigma,
    switch_differencing,
):

    def sample_exog(X, n_paths, horizon, method="bootstrap", block_size=None, random_state=None):
        """
        Parameters:
        -----------
        X : pd.DataFrame
            Historical data for exogenous variables (columns = features).
        n_paths : int
            Number of simulated paths.
        horizon : int
            Forecast horizon (number of future steps).
        method : str
            "bootstrap" for historical block bootstrapping, "ar1" for AR(1) simulation.
        block_size : int or None
            Block size for bootstrap (required if method="bootstrap").
        random_state : int or None
            Seed for reproducibility.

        Returns:
        --------
        np.ndarray
            Array of shape (n_paths, horizon, n_features) with simulated future samples.
        """

        if random_state is not None:
            np.random.seed(random_state)

        n_features = X.shape[1]
        samples = np.zeros((n_paths, horizon, n_features))

        for j, col in enumerate(X.columns):
            series = X[col].values
            n_obs = len(series)

            if method == "bootstrap":
                if block_size is None:
                    raise ValueError("block_size must be provided for bootstrap method.")
                num_blocks = int(np.ceil(horizon / block_size))

                for path in range(n_paths):
                    blocks = []
                    for _ in range(num_blocks):
                        start = np.random.randint(0, n_obs - block_size + 1)
                        block = series[start:start + block_size]
                        blocks.append(block)
                    sample = np.concatenate(blocks)[:horizon]
                    samples[path, :, j] = sample

            elif method == "AR1":
                y_lag = series[:-1]
                y_curr = series[1:]
                phi = np.dot(y_lag, y_curr) / np.dot(y_lag, y_lag)
                sigma = np.std(y_curr - phi * y_lag)
                last_val = series[-1]

                for path in range(n_paths):
                    sim = [last_val]
                    for _ in range(horizon - 1):
                        next_val = phi * sim[-1] + np.random.normal(0, sigma)
                        sim.append(next_val)
                    samples[path, :, j] = sim

            else:
                raise ValueError("method must be either 'bootstrap' or 'ar1'.")

        return samples


    def simulate(df, exog_variables, params):
        np.random.seed(params['random_seed'])
        n = params['n_sims']
        H = params['horizon_blocks']
        sigma = params['sigma']
        P_star = params['P_star']
        gamma_min = params['gamma_min']
        gamma_max = params['gamma_max']
        target_col = radio_transform.value

        X, y, X_test, y_test, X_whole, y_whole, P_initial, y_test_initial = prepare_data_updated(df, target_col, exog_variables)
        # unpack fit
        coef, beta, ridge_eps, ridge_boot, jumps, p_jump = fit_ridge(X.values, y, X.columns.to_list())

        # initial states
        if switch_differencing.value:
            if target_col == 'logit':
                P0 = X['Y_t_diff'].iloc[-1]
                beta_P = coef['Y_t_diff']
            elif target_col == 'log':
                P0 = X['logP_t_diff'].iloc[-1]
                beta_P = coef['logP_t_diff']
            else: 
                P0 = X['P_t_diff'].iloc[-1]
                beta_P = coef['P_t_diff']
        else:
            if target_col == 'logit':
                P0 = X['Y_t'].iloc[-1]
                beta_P = coef['Y_t']
            elif target_col == 'log':
                P0 = X['logP_t'].iloc[-1]
                beta_P = coef['logP_t']
            else: 
                P0 = X['P_t'].iloc[-1]
                beta_P = coef['P_t']

        I0 = X['I_t'].iloc[-1]

        # For exogenous regressors we'll bootstrap (with replacement) historical rows
        X_future = sample_exog(X[exog_variables], n, H, method=radio_sampling.value, block_size=dropdown_block_size.value,  random_state=42)

        # storage
        P_paths = np.zeros((n, H+1))
        I_paths = np.zeros((n, H+1))
        P_paths[:,0] = P0
        I_paths[:,0] = I0


        # precompute which indices in theta correspond to which features
        # cols order: intercept, Y_t, gamma_t, (exog...)
        beta_intercept = coef['intercept']
        beta_I = coef['I_t']
        beta_exog = np.array([coef[c] for c in exog_variables if c not in ['intercept','Y_t', 'logP_t', 'P_t','I_t']]).reshape(-1,1)

        P_0 = P_initial
        for t in range(H):
            # sample exogenous for this step for all sims
            exog_vals = X_future[:, t, :]

            # compute Y_t, gamma_t arrays
            P_t = P_paths[:, t]
            I_t = I_paths[:, t]

            # dynamics for Y_{t+1}
            Y_mean = beta_intercept + beta_P * P_t + beta_I * I_t + (exog_vals @ beta_exog).ravel()

            if radio_residual.value == 'gaussian noise':
                eps = np.random.randn(n) * ridge_eps
            elif radio_residual.value == 'student noise':
                nu = 11
                T = np.random.standard_t(df=nu, size=(n,))
                scale_factor = np.std(ridge_boot, ddof=1) * np.sqrt((nu - 2) / nu)
                eps = T * scale_factor 
            elif radio_residual.value == 'poisson noise':
                eps = np.random.randn(n) * ridge_eps
                if np.random.rand() < p_jump:
                    eps += np.random.choice(jumps, size=n, replace=True)
            else:
                eps = np.random.choice(ridge_boot, size=(n, ), replace=True)

            Y_next = Y_mean + eps

            if switch_differencing.value:
                if target_col == 'logit':
                    P_0 += P_t
                    P_curr = 1/(1+np.exp(-P_0))
                elif target_col == 'log':
                    P_0 += P_t
                    P_curr = np.exp(P_0)
                else:
                    P_0 += P_t
                    P_curr = P_0
            else:
                if target_col == 'logit':
                    P_curr = 1/(1+np.exp(-P_t))
                elif target_col == 'log':
                    P_curr = np.exp(P_t)
                else:
                    P_curr = P_t

            # apply protocol policy to update gamma: gamma_{t+1} = clip(gamma_t + sigma * sign(P_star - P_t), [gamma_min, gamma_max])
            control = sigma * np.sign(P_star - P_curr)
            #I_t_per_round = (I_t + 1) ** (1/417) - 1
            #I_next_per_round = np.clip(I_t_per_round + control, gamma_min, gamma_max)
            #I_next = (1 + I_next_per_round) ** 417 - 1 # adjusted annualized issuance rate
            I_next = np.clip(I_t + control, gamma_min, gamma_max)

            P_paths[:,t+1] = Y_next
            I_paths[:,t+1] = I_next

        if switch_differencing.value:
            if target_col == 'logit':
                P_paths = P_initial + np.cumsum(P_paths, axis = 1)
                P_paths = 1/(1+np.exp(-P_paths))
                y_test = y_test_initial + np.cumsum(y_test)
                y_test = 1/(1+np.exp(-y_test))
            elif target_col == 'log':
                P_paths = P_initial + np.cumsum(P_paths, axis = 1)
                P_paths = np.exp(P_paths)
                y_test = y_test_initial + np.cumsum(y_test)
                y_test = np.exp(y_test)
            else:
                P_paths = P_initial + np.cumsum(P_paths, axis = 1)
                y_test = y_test_initial + np.cumsum(y_test)
        else:
            if target_col == 'logit':
                P_paths = 1/(1+np.exp(-P_paths))
                y_test = 1/(1+np.exp(-y_test))
            elif target_col == 'log':
                P_paths = np.exp(P_paths)
                y_test = np.exp(y_test)

        return P_paths, I_paths, X_future, X_test, y_test, coef


    def simulation_plots():
        exog_cols = feature_selector.value
        parameters = params()

        # Adjusted Parameters:
        parameters['P_star'] = slider_P_star.value /100 # adjust % value
        parameters['gamma_max'] = slider_gamma_max.value /100 # adjust % value
        parameters['gamma_min'] = slider_gamma_min.value /100 # adjust % value
        parameters['sigma'] = slider_sigma.value * 1e-9   # adjust to the correct value (ppb)
        parameters['n_sims'] = int(radio_paths.value)
        parameters['horizon_blocks'] = int(radio_horizon.value)

        P_paths, I_paths, X_paths, X_test, y_test, optimal_beta = simulate(df, exog_cols, parameters)

        # Create subplots with shared x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        horizon = parameters['horizon_blocks']
        percentile = slider_fan.value
        p10 = np.percentile(P_paths, 50 - percentile/2, axis=0)
        p25 = np.percentile(P_paths, 50 - percentile/4, axis=0)
        p50 = np.percentile(P_paths, 50, axis=0)
        p75 = np.percentile(P_paths, 50 + percentile/4, axis=0)
        p90 = np.percentile(P_paths, 50 + percentile/2, axis=0)

        ax1.fill_between(range(0, horizon + 1), p10, p90, color='skyblue', alpha=0.4, label=f'{percentile}% interval')
        ax1.fill_between(range(0, horizon + 1), p25, p75, color='dodgerblue', alpha=0.6, label=f'{percentile/2}% interval')
        #ax1.plot(range(0, horizon + 1), p50, color='blue', linewidth=2, label='Median')
        ax1.plot(range(0, len(y_test)), y_test, color='red', linewidth=2, label='True Value')

        ax1.set_ylabel('Participation Rate')
        #ax1.set_title('Forecast Fan Charts')

        # Plot on second axis
        p2 = np.percentile(I_paths, 50 - percentile/2, axis=0)
        p25 = np.percentile(I_paths, 50 - percentile/4, axis=0)
        p50 = np.percentile(I_paths, 50, axis=0)
        p75 = np.percentile(I_paths, 50 + percentile/4, axis=0)
        p97 = np.percentile(I_paths, 50 + percentile/2, axis=0)

        ax2.fill_between(range(0, horizon + 1), p2, p97, color='skyblue', alpha=0.4, label=f'{percentile}% interval')
        ax2.fill_between(range(0, horizon + 1), p25, p75, color='dodgerblue', alpha=0.6, label=f'{percentile/2}% interval')
        #ax2.plot(range(0, horizon + 1), p50, color='blue', linewidth=2, label='Median')
        ax2.plot(range(0, len(X_test['I_t'])), X_test['I_t'], color='red', linewidth=2, label='True Value')

        ax2.set_ylabel('Issuance Rate')
        ax2.set_xlabel('Horizon Rounds')

        plt.legend()
        plt.tight_layout()
        plt.show()

        return optimal_beta, fig #mo.vstack([mo.md(f"$\\hat{{\\beta}}$: {optimal_beta}"), fig ])
    return simulate, simulation_plots


@app.cell
def _(mo):

    # Create sliders for different parameters
    slider_gamma_max = mo.ui.slider(start=0.01, stop=0.2, step=0.002, value=0.1, label="$\\gamma_+$ (%)")
    slider_gamma_min = mo.ui.slider(start=0.00, stop=0.1, step=0.002, value=0.02, label="$\\gamma_-$ (%)")
    slider_sigma = mo.ui.slider(start=100, stop=1500, step=100, value=500, label="$\\sigma$ (ppb)")
    slider_P_star = mo.ui.slider(start=30, stop=60, step=1, value=50, label="$P^*$ (%)")
    radio_paths = mo.ui.radio(
        options=['5', '20', '200', '1000'],
        value='20',  # default selection
        label="Number of simulations"
    )
    radio_horizon = mo.ui.radio(
        options=['30', '90', '180', '262'],
        value='262',  # default selection
        label="Horizon Rounds"
    )
    radio_sampling = mo.ui.radio(
        options=['bootstrap', 'AR1'],
        value='bootstrap',  # default selection
        label="Sampling of Exogeneous Variables"
    )
    radio_transform = mo.ui.radio(
        options=['logit', 'log', 'none'],
        value='logit',  # default selection
        label="Participation rate domain transformation"
    )

    dropdown_block_size = mo.ui.dropdown(options=[3,7,14,30], value=7, label="Bootstrap Block Size")

    radio_residual = mo.ui.radio(
        options=['gaussian noise', 'bootstraped residuals', 'student noise', 'poisson noise'],
        value='bootstraped residuals',  # default selection
        label="Forecast Diffusion"
    )
    return (
        dropdown_block_size,
        radio_horizon,
        radio_paths,
        radio_residual,
        radio_sampling,
        radio_transform,
        slider_P_star,
        slider_gamma_max,
        slider_gamma_min,
        slider_sigma,
    )


@app.cell
def _(
    dropdown_block_size,
    mo,
    radio_horizon,
    radio_paths,
    radio_sampling,
    radio_transform,
    slider_P_star,
    slider_gamma_max,
    slider_gamma_min,
    slider_sigma,
    switch_differencing,
):
    # Display parameters and their current values
    simulation_parameters_control = mo.vstack([
        mo.hstack([slider_gamma_max, slider_gamma_max.value]),
        mo.hstack([slider_gamma_min, slider_gamma_min.value]),
        mo.hstack([slider_sigma, slider_sigma.value]),
        mo.hstack([slider_P_star, slider_P_star.value]),
        mo.hstack([radio_paths, radio_horizon, radio_sampling, radio_transform]),
        mo.hstack([dropdown_block_size, switch_differencing])
    ], gap = 'xs', align='start')
    return (simulation_parameters_control,)


@app.cell
def _(
    df,
    df_raw,
    feature_selector,
    mo,
    np,
    params,
    plt,
    radio_horizon,
    radio_paths,
    simulate,
    slider_E_end,
    slider_E_start,
    slider_P_star,
    slider_gamma_max,
    slider_gamma_min,
    slider_gamma_star,
    slider_sigma,
    slider_yield_star,
    start_idx_training,
    window_size_training,
):
    def risk_admissibility():
        exog_cols = feature_selector.value
        parameters = params()

        # Adjusted Parameters:
        parameters['P_star'] = slider_P_star.value /100
        parameters['gamma_max'] = slider_gamma_max.value /100
        parameters['gamma_min'] = slider_gamma_min.value /100
        parameters['sigma'] = slider_sigma.value * 1e-9
        parameters['n_sims'] = int(radio_paths.value)
        parameters['horizon_blocks'] = int(radio_horizon.value)

        P_paths, I_paths, X_paths, X_test, y_test, optimal_beta = simulate(df, exog_cols, parameters)

        P = P_paths
        H = parameters['horizon_blocks']
        #Plow = slider_Plow.value
        #Phigh = slider_Phigh.value
        #T_star = slider_Tstar.value
        #Ttail = slider_Ttail.value
        #eps_tail = slider_Teps.value
        gamma_star = slider_gamma_star.value
        yield_star = slider_yield_star.value

        # insert the dates

        # Dilution Objective
        train_ind_end = min(len(df_raw) , start_idx_training.value + window_size_training.value)
        total_supply = df['total-supply']/1e18 # 18 decimals 
        total_supply_paths = np.zeros((parameters["n_sims"], parameters["horizon_blocks"] + 1))
        total_supply_paths[:, 0] = total_supply.iloc[train_ind_end]
        for i in range(parameters["horizon_blocks"]):
            inflation_token = total_supply_paths[:, i]*I_paths[:, i]
            total_supply_paths[:, i+1] = total_supply_paths[:, i] + inflation_token

        '''start_date = pd.Timestamp(df_raw['date'].iloc[train_ind_end])
        ratio_days = 21
        ratio_blocks = 24
        block_quantity = parameters['horizon_blocks']  # Example: user specifies 50 blocks

        # Compute total days based on ratio
        days_per_block = ratio_days / ratio_blocks  # 21 days per 24 blocks
        total_days = block_quantity * days_per_block

        # Generate date array with rounding to whole days
        end_date = start_date + pd.Timedelta(days=total_days)
        date_array = pd.date_range(start=start_date, end=end_date, periods=block_quantity)

        # Round each timestamp to the nearest day
        date_array = date_array.normalize()  # strips time, keeps date only

        dates = pd.date_range(start=start_date, end=date_array[-1], freq=dropdown_interval.value)'''

        # compute time outside D0 for each sim (count of days where P not in [Plow,Phigh])
        #outside = ((P < Plow) | (P > Phigh)).sum(axis=1) # includes t=0..H

        #expected_outside = float(np.mean(outside))
        #prob_exceed_tail = float((outside > Ttail).mean())

        admissible = True #(expected_outside <= T_star) and (prob_exceed_tail <= eps_tail)

        # Emission and Yield Rate target acceptance
        emissions_semiannual = (
            (total_supply_paths[:, slider_E_end.value] - total_supply_paths[:, slider_E_start.value]) / 
            total_supply_paths[:,  slider_E_end.value]
        )
        ET = np.quantile(emissions_semiannual, .95)
        YT = ((1+I_paths)**417 - 1)/P_paths
        YT = np.quantile(YT[:,-1], .95)

        admissible = (admissible) and (ET <= gamma_star) and (YT <= yield_star)

        # also compute percentiles for plotting
        q10 = np.percentile(total_supply_paths, 10, axis=0)
        q50 = np.percentile(total_supply_paths, 50, axis=0)
        q90 = np.percentile(total_supply_paths, 90, axis=0)


        # plot for emissions 
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(range(parameters["horizon_blocks"] + 1), np.mean(total_supply_paths, axis=0), label="Mean", color="orange")
        ax.plot(range(parameters["horizon_blocks"] + 1), q50, label="Median", alpha=0.6, color="orange")
        ax.fill_between(range(0, parameters["horizon_blocks"] + 1), q10, q90, color='orange', alpha=0.4, label='80% interval')
        ax.axvspan(slider_E_start.value, slider_E_end.value, color="green", alpha=0.3, label="$I$")
        ax.set_title("Total Supply fan chart")
        ax.legend()

        #return result
        return mo.vstack([
                            fig,
                          mo.md(f"E[I]: {ET:.3f} (acceptance target={gamma_star})"),
                          mo.md(f"Yield: {YT:.3f} (acceptance target={yield_star})"),
                          mo.md(f"RISK-ADMISSIBLE: {'✅ YES' if admissible else '❌ NO'}")
                         ])


    #result_risk = risk_admissibility()
    return (risk_admissibility,)


@app.cell
def _(mo, radio_horizon):
    # UI for risk admissibility parameters
    #slider_Plow = mo.ui.slider(start=0.0, stop=0.5, step=0.01, value=0.4, label="$P_{{low}}$")
    #slider_Phigh = mo.ui.slider(start=0.5, stop=1.0, step=0.01, value=0.6, label="$P_{{high}}$")
    #slider_Tstar = mo.ui.slider(start=1, stop=100, step=1, value=10, label="$T^*$")
    #slider_Ttail = mo.ui.slider(start=1, stop=100, step=1, value=20, label="$T_{{tail}}$")
    #slider_Teps = mo.ui.slider(start=0.01, stop=0.3, step=0.01, value=0.05, label="$\\epsilon_T$")
    slider_fan = mo.ui.slider(start=10, stop=100, step=1, value=80, label="Distribution interval")
    slider_gamma_star = mo.ui.slider(start=0.01, stop=0.2, step=0.005, value=0.12, label="$\\tau_E$")
    slider_yield_star = mo.ui.slider(start=0.1, stop=1.0, step=0.01, value=1.0, label="yield")

    slider_E_start = mo.ui.slider(start=0, stop=int(radio_horizon.value), step=1, value=58, label="$t_-$")
    slider_E_end = mo.ui.slider(start=0, stop=int(radio_horizon.value), step=1, value=int(radio_horizon.value), label="$t_+$")
    return (
        slider_E_end,
        slider_E_start,
        slider_fan,
        slider_gamma_star,
        slider_yield_star,
    )


@app.cell
def _(mo, slider_E_end, slider_E_start, slider_gamma_star, slider_yield_star):
    risk_admissibility_parameters_control = mo.vstack([
        mo.hstack([slider_E_start, slider_E_start.value]),
        mo.hstack([slider_E_end, slider_E_end.value]),
        mo.hstack([slider_gamma_star, slider_gamma_star.value]),
        mo.hstack([slider_yield_star, slider_yield_star.value])
        ], align='start', justify='start')
    return (risk_admissibility_parameters_control,)


@app.cell
def _(mo):
    # Maintenance Objectives

    '''# Define allowed range
    min_date = pd.Timestamp("2025-01-01")
    max_date = pd.Timestamp("2025-12-31")

    # UI controls with restricted range
    button_start_date = mo.ui.date(label="Start Date", value=min_date, min=min_date, max=max_date)
    button_end_date = mo.ui.date(label="End Date", value=max_date, min=min_date, max=max_date)

    start_date = mo.ui.date(label="Start Date", value=pd.Timestamp("2025-01-01"))
    end_date = mo.ui.date(label="End Date", value=pd.Timestamp("2025-12-31"))'''

    dropdown_interval = mo.ui.dropdown(
        label="Measurement Interval",
        options=["1D", "1W", "1M"],  # daily, weekly, monthly
        value="1M"
    )
    return


@app.cell
def _(
    data_prep_control2,
    mo,
    radio_residual,
    risk_admissibility,
    risk_admissibility_parameters_control,
    simulation_parameters_control,
    simulation_plots,
    slider_fan,
):
    data_prep_header = mo.md("## Model Estimation")
    simulation_header = mo.md("## Simulation Parameters")
    risk_admissibility_header = mo.md("## Risk-Admissibility Parameters")

    layout1 = mo.hstack([
                                mo.vstack([simulation_header, simulation_parameters_control]),
                                mo.vstack([risk_admissibility_header, risk_admissibility_parameters_control])
    ], gap='lg', align='start', justify='center')

    layout2 = mo.vstack([
        data_prep_header, data_prep_control2
    ])

    optimal_beta, simulation_fig = simulation_plots()

    layout_fanzones = mo.hstack([ slider_fan, mo.md(f'{slider_fan.value}%'), radio_residual], align='start', justify='start')

    display_layout = mo.vstack([
        layout2, layout1, 
        mo.hstack([mo.md("## Risk-Admissibility: "), risk_admissibility()]), 
        mo.md(f'**Fitted coefficients $\\hat{{\\beta}}$** : {optimal_beta}'),
        mo.md("## Simulation Dynamics"),
        layout_fanzones,
        simulation_fig
    ])

    display_layout
    return (simulation_fig,)


@app.cell
def _(mo, simulation_fig, slider_P_star, slider_sigma):
    def save(_):
        simulation_fig.savefig(f"simulation_fan_chart-{slider_P_star.value}-{slider_sigma.value}.svg")

    mo.ui.button(label="Save Figure", on_click=save)
    return


@app.cell
def _(
    df,
    feature_selector,
    np,
    params,
    radio_horizon,
    radio_paths,
    simulate,
    slider_gamma_max,
    slider_gamma_min,
):
    def simulate_total_supply(start, I_paths):
        return start * np.cumprod(1 + I_paths, axis=1)

    def trailing_dilution(supply, window=207):
        dilution = 1-supply[:, :-window] / supply[:, window:]
        return dilution * 100

    def trailing_yield(supply, bonding_rate, window=415):
        # per round yield is change in supply divided by bonding rate
        per_round_yield = (supply[:,1:] - supply[:,:-1]) / (supply[:,:-1] * bonding_rate[:, :-1])
        # cumulative yield
        cum_yield_factor = (1 + per_round_yield).cumprod(axis=1)
        # trailing cumulative yield over window
        return 100*(cum_yield_factor[:,window:] / cum_yield_factor[:,:-window] - 1)
    
    
    

    def simulate_paths(target_bonding_rate = 50, inflation_change = 500):
        exog_cols = feature_selector.value
        parameters = params()

        # Adjusted Parameters:
        parameters['P_star'] = target_bonding_rate /100 # adjust % value
        parameters['gamma_max'] = slider_gamma_max.value /100 # adjust % value
        parameters['gamma_min'] = slider_gamma_min.value /100 # adjust % value
        parameters['sigma'] = inflation_change * 1e-9   # adjust to the correct value (ppb)
        parameters['n_sims'] = int(radio_paths.value)
        parameters['horizon_blocks'] = int(radio_horizon.value)

        P_paths, I_paths, X_paths, X_test, y_test, o_beta = simulate(df, exog_cols, parameters)

        total_supply_paths = simulate_total_supply(start=df['total-supply'].iloc[-1]/1e18, I_paths=I_paths)
        # concatenate total_supply and P_paths with historic data
        total_supply_full = np.zeros((total_supply_paths.shape[0], total_supply_paths.shape[1] + len(df)))
        total_supply_full[:, :len(df)] = df['total-supply'].iloc[:].values / 1e18
        total_supply_full[:, len(df):] = total_supply_paths
    
        P_full = np.zeros((P_paths.shape[0], P_paths.shape[1] + len(df)))
        P_full[:, :len(df)] = df['participation-rate'].iloc[:].values
        P_full[:, len(df):] = P_paths

        return total_supply_full, P_full
    return simulate_paths, trailing_dilution, trailing_yield


@app.cell
def _(np, simulate_paths):
    import altair as alt
    import polars as pl

    def crop_to_significance(data, significance):
        q_low = np.percentile(data, significance/2)
        q_high = np.percentile(data, 100 - significance/2)
        return data[(data >= q_low) & (data <= q_high)]

    def plot_box_and_whiskers(data, label: str, significance=5):
        data_cropped = crop_to_significance(data, significance)
    
        return alt.Chart(pl.DataFrame({"value": data_cropped})).mark_boxplot(extent="min-max").encode(
            y=alt.Y("value:Q", axis=alt.Axis(title=label)).scale(zero=False)
        ).properties(width=400)

    def plot_box_and_whiskers_multiple(data: pl.DataFrame, label: str, significance=5):
        df = data.unpivot(
            index=None,
            on=data.columns,
            variable_name="Policy",
            value_name="value"
        )
        return alt.Chart(df).mark_boxplot(extent="min-max", size=60).encode(
            x=alt.X("Policy:N", axis=alt.Axis(title="Policy")),
            y=alt.Y("value:Q", axis=alt.Axis(title=label)).scale(zero=False),
            color="Policy:N"
        ).properties(width=400)

    # compute trailing dilution of simulations with parameters (50, 500) and (46, 700)
    simulations = [
        simulate_paths(target_bonding_rate = 50, inflation_change = 500),
        simulate_paths(target_bonding_rate = 46, inflation_change = 700)
    ]

    SIGNIFICANCE=10
    return (
        SIGNIFICANCE,
        alt,
        crop_to_significance,
        pl,
        plot_box_and_whiskers_multiple,
        simulations,
    )


@app.cell
def _(
    SIGNIFICANCE,
    crop_to_significance,
    pl,
    plot_box_and_whiskers_multiple,
    simulations,
    trailing_dilution,
):
    data_dilution = pl.DataFrame({
        "No change": crop_to_significance(trailing_dilution(simulations[0][0])[:,-1], significance=SIGNIFICANCE),
        "Proposed change": crop_to_significance(trailing_dilution(simulations[1][0])[:,-1], significance=SIGNIFICANCE)
    })
    plot_box_and_whiskers_multiple(data_dilution, label="6-month Dilution")
    return


@app.cell
def _(
    SIGNIFICANCE,
    alt,
    crop_to_significance,
    pd,
    pl,
    plot_box_and_whiskers_multiple,
    simulations,
    trailing_yield,
):
    data_trailing_yield = pl.DataFrame({
        "No change": crop_to_significance(trailing_yield(*simulations[0])[:,-1], significance=SIGNIFICANCE),
        "Proposed change": crop_to_significance(trailing_yield(*simulations[1])[:,-1], significance=SIGNIFICANCE)
    })
    plotz = plot_box_and_whiskers_multiple(data_trailing_yield, label="Predicted 1Y Trailing Yield on 2026-07-01")


    threshold: str = "1Y trailing yield on 2025-11-20"

    base = alt.Chart(pd.DataFrame({
        'y': [73.38791078],
        'label': [threshold]
    }))
    
    line = base.mark_rule(strokeDash=[4,4], color='red').encode(
        y='y:Q'
    )

    # add the following string as a text mark on line
    text = base.mark_text(
        align='center',
        baseline='bottom',
        dy=-5 
    ).encode(
        y='y:Q',
        text='label:N'
    )

    yield_box_plot = (text + line + plotz).properties(width=400)

    # save as svg
    #yield_box_plot.save(f"yield_box_plot.png")
    yield_box_plot
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

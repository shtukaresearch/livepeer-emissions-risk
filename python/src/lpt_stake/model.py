"""Ridge regression fitting for the lpt_stake library.

Provides a closed-form ridge regression solver that excludes the intercept
from regularisation, and a result container with diagnostic statistics.

The intercept is not penalised because penalising it would make the
procedure depend on the origin chosen for Y (Hastie et al., *Elements of
Statistical Learning*, 2nd ed., Section 3.4.1).

The closed-form solution is:

.. math::

    \\hat{\\beta} = (X^T X + \\alpha D)^{-1} X^T y

where *D* is the identity matrix with ``D[0, 0] = 0`` so that the intercept
(assumed to be the first column) is not regularised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray


@dataclass(frozen=True)
class RidgeResult:
    """Result of a ridge regression fit.

    Fields
    ------
    coefficients : dict[str, float]
        Mapping from column name to fitted coefficient value.  Keys are
        ordered to match the feature columns of the input design matrix.
    residuals : NDArray
        Training residuals, ``y - X @ beta``, shape ``(n,)``.
    residual_std : float
        Sample standard deviation of the residuals (``ddof=1``).
    coefficient_std_errors : dict[str, float]
        Estimated standard errors for each coefficient, assuming i.i.d.
        normal residuals with constant variance.  Computed from the
        sandwich covariance ``sigma^2 W X'X W`` where
        ``W = (X'X + alpha*D)^{-1}``.
    effective_df : float
        Effective degrees of freedom: ``tr(H)`` where
        ``H = X (X'X + alpha*D)^{-1} X'`` is the hat matrix.  Computed as
        ``tr(W @ X'X)`` by the cyclic trace property.

        For uniform penalisation (``D = I``), this reduces to
        ``sum(d_j^2 / (d_j^2 + alpha))`` where ``d_j`` are the singular
        values of X (Hastie et al., ESL 2nd ed., formula 3.50,
        Section 3.4.1).  Here ``D`` has ``D[0,0] = 0`` because the
        intercept is excluded from regularisation, so the trace is computed
        directly.
    aic : float
        Akaike information criterion:
        ``n * log(RSS / n) + 2 * effective_df``.
    bic : float
        Bayesian information criterion:
        ``n * log(RSS / n) + log(n) * effective_df``.
    """

    coefficients: dict[str, float]
    residuals: NDArray[np.floating]
    residual_std: float
    coefficient_std_errors: dict[str, float]
    effective_df: float
    aic: float
    bic: float

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Predict target values from a feature matrix.

        Parameters
        ----------
        X
            Feature matrix of shape ``(n, p)`` with the same column order
            as the design matrix used for fitting (including intercept).

        Returns
        -------
        NDArray
            Predicted values, shape ``(n,)``.
        """
        beta = np.array(list(self.coefficients.values()))
        return X @ beta


def fit_ridge(
    dm: pl.DataFrame,
    target_col: str,
    alpha: float,
) -> RidgeResult:
    """Fit ridge regression on a design matrix.

    The design matrix is expected to come from
    :func:`~lpt_stake.features.build_design_matrix`, with an intercept column
    as the first column and the target as the last column.

    The intercept is excluded from regularisation (see module docstring).

    Parameters
    ----------
    dm
        Design matrix as a Polars DataFrame.  Must contain *target_col* and
        at least one feature column.
    target_col
        Name of the target column in *dm*.
    alpha
        Regularisation strength (non-negative).

    Returns
    -------
    RidgeResult
        Fitted model with coefficients, residuals, and diagnostics.

    Raises
    ------
    ValueError
        If *target_col* is not in *dm* or *alpha* is negative.
    """
    if target_col not in dm.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame. "
            f"Available columns: {dm.columns}"
        )
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")

    feature_cols = [c for c in dm.columns if c != target_col]
    X = dm.select(feature_cols).to_numpy()
    y = dm[target_col].to_numpy()
    n, p = X.shape

    # Penalty matrix: identity with D[0,0] = 0 (don't regularise intercept)
    D = np.eye(p)
    D[0, 0] = 0.0

    # Closed-form ridge solution
    XtX = X.T @ X
    W = np.linalg.inv(XtX + alpha * D)
    beta = W @ X.T @ y

    # Residuals
    y_hat = X @ beta
    residuals = y - y_hat
    residual_std = float(np.std(residuals, ddof=1))

    # Effective degrees of freedom (ESL formula 3.50, Section 3.4.1):
    # df(alpha) = tr(H) where H = X (X'X + alpha*D)^{-1} X' is the hat matrix.
    # By the cyclic trace property: tr(H) = tr(W @ X'X).
    effective_df = float(np.trace(W @ XtX))

    # Coefficient standard errors (sandwich covariance)
    # Cov(beta_hat) = sigma^2 * W @ X'X @ W
    # Assumes i.i.d. normal residuals with constant variance.
    sigma2 = float(np.sum(residuals ** 2) / (n - effective_df))
    cov_beta = sigma2 * W @ XtX @ W
    std_errors = np.sqrt(np.diag(cov_beta))

    # AIC and BIC
    rss = float(np.sum(residuals ** 2))
    aic = n * np.log(rss / n) + 2 * effective_df
    bic = n * np.log(rss / n) + np.log(n) * effective_df

    coefficients = {col: float(beta[i]) for i, col in enumerate(feature_cols)}
    coefficient_std_errors = {
        col: float(std_errors[i]) for i, col in enumerate(feature_cols)
    }

    return RidgeResult(
        coefficients=coefficients,
        residuals=residuals,
        residual_std=residual_std,
        coefficient_std_errors=coefficient_std_errors,
        effective_df=float(effective_df),
        aic=float(aic),
        bic=float(bic),
    )

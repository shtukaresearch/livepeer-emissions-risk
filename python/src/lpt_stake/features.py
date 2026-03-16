"""Feature engineering utilities for the lpt_stake library.

Provides Polars expression plugins (composable transforms) and a design-matrix
assembly function.  Expression plugins take and return ``pl.Expr``, so they
compose with ``.pipe()``.
"""

from __future__ import annotations

import polars as pl


# ---------------------------------------------------------------------------
# Expression plugins
# ---------------------------------------------------------------------------


def logit(expr: pl.Expr) -> pl.Expr:
    """Log-odds transform: ``log(x / (1 - x))``.

    Maps (0, 1) → (-∞, +∞).  The inverse is :func:`expit`.

    Parameters
    ----------
    expr
        A Polars expression whose values are in (0, 1).
    """
    return (expr / (1 - expr)).log()


def expit(expr: pl.Expr) -> pl.Expr:
    """Inverse logit (logistic sigmoid): ``1 / (1 + exp(-x))``.

    Maps (-∞, +∞) → (0, 1).  The inverse is :func:`logit`.

    Parameters
    ----------
    expr
        A Polars expression with real-valued inputs.
    """
    return 1 / (1 + (-expr).exp())


def annualise_ppb(expr: pl.Expr, rounds_per_year: float) -> pl.Expr:
    """Annualise a per-round parts-per-billion rate by compounding.

    Computes ``(1 + ppb / 1e9) ^ rounds_per_year - 1``.

    The exact algebraic property is::

        log(1 + annualised) == rounds_per_year * log(1 + ppb / 1e9)

    For small ``ppb``, the result is approximately
    ``rounds_per_year * ppb / 1e9`` (linear approximation), with relative
    error bounded by approximately ``rounds_per_year * ppb / 2e9``.

    Parameters
    ----------
    expr
        A Polars expression containing per-round rates in parts per billion.
    rounds_per_year
        Number of rounds per year (see ``constants.ROUNDS_PER_YEAR``).
    """
    return (1 + expr / 1e9).pow(rounds_per_year) - 1


# ---------------------------------------------------------------------------
# Design matrix assembly
# ---------------------------------------------------------------------------


def build_design_matrix(
    df: pl.DataFrame,
    target: pl.Expr,
    features: list[pl.Expr],
) -> pl.DataFrame:
    """Assemble a design matrix for regression.

    Evaluates *target* and *features* against *df*, adds an intercept column,
    shifts the target by −1 to create a next-step prediction target, and drops
    rows where nulls were introduced by the shift or by feature expressions
    (e.g. ``diff()``).

    The input DataFrame must be null-free in the columns referenced by
    *target* and *features*.  If nulls are present in the input, a
    ``ValueError`` is raised.

    Parameters
    ----------
    df
        Round-indexed DataFrame.  Must not contain nulls in referenced columns.
    target
        Polars expression for the prediction target (e.g.
        ``pl.col("participation")``).
    features
        List of Polars expressions for the feature columns.

    Returns
    -------
    pl.DataFrame
        Columns are ``[intercept, feature_0, ..., feature_n, target]``.
        The intercept is a column of ones.  The target is the next-step
        value.  Rows with nulls (from shifting or feature transforms) are
        dropped.

    Raises
    ------
    ValueError
        If *df* contains nulls in the columns referenced by *target* or
        *features*.
    """
    # Check for nulls in input
    null_counts = df.null_count()
    total_nulls = sum(null_counts.row(0))
    if total_nulls > 0:
        null_cols = [
            col for col in df.columns if null_counts[col][0] > 0
        ]
        raise ValueError(
            f"Input DataFrame contains nulls in columns: {null_cols}. "
            "Clean the data before calling build_design_matrix."
        )

    # Evaluate features and target
    feature_exprs = [
        feat.alias(f"feature_{i}") for i, feat in enumerate(features)
    ]
    target_expr = target.shift(-1).alias("target")

    assembled = df.select(
        pl.lit(1.0).alias("intercept"),
        *feature_exprs,
        target_expr,
    )

    return assembled.drop_nulls()

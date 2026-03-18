"""Feature engineering utilities for the lpt_stake library.

Provides Polars expression plugins (composable transforms) and a
design-matrix assembly function.

Expression plugins take and return ``pl.Expr``, so they compose with
``.pipe()``.  These are used during data preparation in notebooks.
"""

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)


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
    and drops rows where nulls were introduced by feature or target
    expressions (e.g. ``diff()``, ``shift(-1)``).  Dropped rows are logged
    at DEBUG level with their index and the null column(s).

    This function does **not** shift the target.  If you want next-step
    prediction (i.e. features at time *t* predict target at time *t* + 1),
    apply the shift in the target expression yourself::

        build_design_matrix(
            df,
            target=pl.col("participation").shift(-1),
            features=[...],
        )

    Parameters
    ----------
    df
        Round-indexed DataFrame.
    target
        Polars expression for the prediction target.  Applied as-is — no
        shifting is performed by this function.
    features
        List of Polars expressions for the feature columns.

    Returns
    -------
    pl.DataFrame
        Columns are ``[intercept, feature_0, ..., feature_n, target]``.
        The intercept is a column of ones.  Rows with nulls (from feature
        or target expressions) are dropped.
    """
    # Evaluate features and target
    feature_exprs = [
        feat.alias(f"feature_{i}") for i, feat in enumerate(features)
    ]
    target_expr = target.alias("target")

    assembled = df.select(
        pl.lit(1.0).alias("intercept"),
        *feature_exprs,
        target_expr,
    ).with_row_index("__row_idx__")

    # Identify and log rows with nulls before dropping
    null_mask = assembled.select(
        pl.any_horizontal(
            pl.col(c).is_null() for c in assembled.columns if c != "__row_idx__"
        ).alias("has_null")
    )["has_null"]

    if null_mask.any():
        null_rows = assembled.filter(null_mask)
        for row in null_rows.iter_rows(named=True):
            idx = row["__row_idx__"]
            null_cols = [
                c for c in row
                if c != "__row_idx__" and row[c] is None
            ]
            logger.debug(
                "build_design_matrix: dropping row %d (null in %s)",
                idx,
                null_cols,
            )
        n_dropped = null_rows.height
        logger.info(
            "build_design_matrix: dropped %d of %d rows due to nulls",
            n_dropped,
            assembled.height,
        )

    return assembled.filter(~null_mask).drop("__row_idx__")

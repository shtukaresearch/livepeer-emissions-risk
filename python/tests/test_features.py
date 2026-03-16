"""Tests for lpt_stake.features."""

import math

import polars as pl
import pytest

from lpt_stake.constants import ROUNDS_PER_YEAR
from lpt_stake.features import annualise_ppb, build_design_matrix, expit, logit


# ---------------------------------------------------------------------------
# logit / expit
# ---------------------------------------------------------------------------


class TestLogit:
    """Test logit expression plugin: log(x / (1 - x))."""

    def test_midpoint_is_zero(self):
        df = pl.DataFrame({"x": [0.5]})
        result = df.select(pl.col("x").pipe(logit).alias("y"))["y"][0]
        assert result == pytest.approx(0.0)

    def test_antisymmetry(self):
        """logit(0.5 + d) == -logit(0.5 - d)."""
        deltas = [0.1, 0.2, 0.3, 0.4]
        above = [0.5 + d for d in deltas]
        below = [0.5 - d for d in deltas]
        df = pl.DataFrame({"above": above, "below": below})
        result = df.select(
            pl.col("above").pipe(logit).alias("logit_above"),
            pl.col("below").pipe(logit).alias("logit_below"),
        )
        for a, b in zip(
            result["logit_above"].to_list(), result["logit_below"].to_list()
        ):
            assert a == pytest.approx(-b)

    def test_below_half_is_negative(self):
        df = pl.DataFrame({"x": [0.1, 0.2, 0.3, 0.4]})
        result = df.select(pl.col("x").pipe(logit).alias("y"))["y"]
        assert all(v < 0 for v in result.to_list())

    def test_above_half_is_positive(self):
        df = pl.DataFrame({"x": [0.6, 0.7, 0.8, 0.9]})
        result = df.select(pl.col("x").pipe(logit).alias("y"))["y"]
        assert all(v > 0 for v in result.to_list())


class TestExpit:
    """Test expit expression plugin: 1 / (1 + exp(-x))."""

    def test_zero_maps_to_half(self):
        df = pl.DataFrame({"x": [0.0]})
        result = df.select(pl.col("x").pipe(expit).alias("y"))["y"][0]
        assert result == pytest.approx(0.5)

    def test_monotonically_increasing(self):
        xs = [-3.0, -1.0, 0.0, 1.0, 3.0]
        df = pl.DataFrame({"x": xs})
        result = df.select(pl.col("x").pipe(expit).alias("y"))["y"].to_list()
        for i in range(len(result) - 1):
            assert result[i] < result[i + 1]


class TestLogitExpitInverse:
    """Test that logit and expit are inverses."""

    @pytest.mark.parametrize("x", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_expit_of_logit_is_identity(self, x):
        df = pl.DataFrame({"x": [x]})
        result = df.select(
            pl.col("x").pipe(logit).pipe(expit).alias("y")
        )["y"][0]
        assert result == pytest.approx(x)

    @pytest.mark.parametrize("x", [-3.0, -1.0, 0.0, 1.0, 3.0])
    def test_logit_of_expit_is_identity(self, x):
        df = pl.DataFrame({"x": [x]})
        result = df.select(
            pl.col("x").pipe(expit).pipe(logit).alias("y")
        )["y"][0]
        assert result == pytest.approx(x)


# ---------------------------------------------------------------------------
# annualise_ppb
# ---------------------------------------------------------------------------


class TestAnnualisePpb:
    """Test annualise_ppb expression plugin.

    The exact relationship is:
        log(1 + annualised) == rounds_per_year * log(1 + ppb / 1e9)
    """

    @pytest.mark.parametrize("ppb", [100_000, 200_000, 300_000, 500_000])
    def test_log_relationship_exact(self, ppb):
        """The defining algebraic property of compound annualisation."""
        df = pl.DataFrame({"ppb": [ppb]})
        annualised = df.select(
            pl.col("ppb").pipe(annualise_ppb, ROUNDS_PER_YEAR).alias("a")
        )["a"][0]

        lhs = math.log(1 + annualised)
        rhs = ROUNDS_PER_YEAR * math.log(1 + ppb / 1e9)
        assert lhs == pytest.approx(rhs, rel=1e-12)

    @pytest.mark.parametrize("ppb", [100_000, 200_000, 300_000])
    def test_linear_approximation_bound(self, ppb):
        """The linear approximation (n * r) should underestimate the true
        annualised rate, and the relative error should be approximately
        n * r / 2 for small r."""
        df = pl.DataFrame({"ppb": [ppb]})
        annualised = df.select(
            pl.col("ppb").pipe(annualise_ppb, ROUNDS_PER_YEAR).alias("a")
        )["a"][0]

        r = ppb / 1e9
        n = ROUNDS_PER_YEAR
        linear = n * r

        # Compounding always exceeds the linear approximation
        assert annualised > linear

        # Relative error of linear approximation bounded by n*r/2
        relative_error = (annualised - linear) / annualised
        assert relative_error < n * r / 2

    def test_zero_ppb_gives_zero(self):
        df = pl.DataFrame({"ppb": [0]})
        result = df.select(
            pl.col("ppb").pipe(annualise_ppb, ROUNDS_PER_YEAR).alias("a")
        )["a"][0]
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# build_design_matrix
# ---------------------------------------------------------------------------


class TestBuildDesignMatrix:
    """Test build_design_matrix assembly."""

    @pytest.fixture()
    def sample_df(self):
        """A small round-indexed DataFrame with no nulls."""
        return pl.DataFrame(
            {
                "round": list(range(10)),
                "participation": [0.4, 0.42, 0.45, 0.43, 0.44, 0.46, 0.48, 0.47, 0.49, 0.50],
                "inflation": [200_000] * 10,
            }
        )

    def test_output_has_intercept(self, sample_df):
        result = build_design_matrix(
            sample_df,
            target=pl.col("participation"),
            features=[pl.col("inflation")],
        )
        assert "intercept" in result.columns
        assert all(v == 1.0 for v in result["intercept"].to_list())

    def test_output_columns(self, sample_df):
        features = [pl.col("inflation"), pl.col("participation")]
        result = build_design_matrix(
            sample_df,
            target=pl.col("participation"),
            features=features,
        )
        # intercept + features + target
        assert len(result.columns) == 1 + len(features) + 1
        assert result.columns[0] == "intercept"
        assert result.columns[-1] == "target"

    def test_target_evaluated_as_is(self, sample_df):
        """The target column should be the expression evaluated directly,
        with no implicit shifting."""
        result = build_design_matrix(
            sample_df,
            target=pl.col("participation"),
            features=[pl.col("inflation")],
        )
        original_participation = sample_df["participation"].to_list()
        target_values = result["target"].to_list()
        for orig, target in zip(original_participation, target_values):
            assert target == pytest.approx(orig)

    def test_caller_can_shift_target(self, sample_df):
        """Next-step prediction is the caller's responsibility via shift(-1)."""
        result = build_design_matrix(
            sample_df,
            target=pl.col("participation").shift(-1),
            features=[pl.col("inflation")],
        )
        # shift(-1) loses the last row
        assert len(result) == len(sample_df) - 1
        original_participation = sample_df["participation"].to_list()
        target_values = result["target"].to_list()
        for i, t in enumerate(target_values):
            assert t == pytest.approx(original_participation[i + 1])

    def test_no_row_loss_with_plain_target(self, sample_df):
        """A plain column reference as target loses no rows."""
        result = build_design_matrix(
            sample_df,
            target=pl.col("participation"),
            features=[pl.col("inflation")],
        )
        assert len(result) == len(sample_df)

    def test_diff_feature_loses_one_row(self, sample_df):
        """A diff() feature introduces one null row."""
        result = build_design_matrix(
            sample_df,
            target=pl.col("participation"),
            features=[pl.col("participation").diff()],
        )
        assert len(result) == len(sample_df) - 1

    def test_null_inputs_dropped_and_logged(self, caplog):
        """Rows with nulls in the input are dropped and logged."""
        df = pl.DataFrame(
            {
                "participation": [0.4, None, 0.45, 0.43],
                "inflation": [200_000] * 4,
            }
        )
        import logging

        with caplog.at_level(logging.DEBUG, logger="lpt_stake.features"):
            result = build_design_matrix(
                df,
                target=pl.col("participation"),
                features=[pl.col("inflation")],
            )
        assert len(result) == 3
        # Row index and null column name appear in debug log
        assert "1" in caplog.text
        assert "target" in caplog.text

    def test_no_nulls_in_output(self, sample_df):
        """Nulls introduced by expressions (diff, shift) are dropped."""
        result = build_design_matrix(
            sample_df,
            target=pl.col("participation").shift(-1),
            features=[pl.col("participation").diff()],
        )
        assert result.null_count().row(0) == tuple(0 for _ in result.columns)

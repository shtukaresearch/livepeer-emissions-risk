"""Tests for lpt_stake.model."""

import numpy as np
import polars as pl
import pytest

from lpt_stake.features import build_design_matrix
from lpt_stake.model import RidgeResult, fit_ridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_design_matrix(
    n: int = 50, *, true_beta: np.ndarray | None = None, seed: int = 0
) -> tuple[pl.DataFrame, np.ndarray]:
    """Build a design matrix from synthetic data with known coefficients.

    Returns the design matrix DataFrame (from build_design_matrix) and the
    true coefficient vector including intercept.
    """
    rng = np.random.default_rng(seed)
    if true_beta is None:
        true_beta = np.array([2.0, 0.5, -0.3])  # intercept, feat0, feat1

    n_features = len(true_beta) - 1
    x = rng.standard_normal((n, n_features))
    noise = rng.normal(0, 0.01, size=n)
    y = true_beta[0] + x @ true_beta[1:] + noise

    cols = {f"feat_{i}": x[:, i].tolist() for i in range(n_features)}
    cols["target_raw"] = y.tolist()
    df = pl.DataFrame(cols)

    dm = build_design_matrix(
        df,
        target=pl.col("target_raw"),
        features=[pl.col(f"feat_{i}") for i in range(n_features)],
    )
    return dm, true_beta


# ---------------------------------------------------------------------------
# RidgeResult
# ---------------------------------------------------------------------------


class TestRidgeResult:
    """Test the RidgeResult dataclass."""

    def test_predict_equals_matrix_multiply(self):
        """predict(X) must equal X @ coefficients."""
        dm, _ = _make_design_matrix()
        result = fit_ridge(dm, "target", alpha=0.0)

        X = dm.drop("target").to_numpy()
        beta = np.array([result.coefficients[c] for c in dm.columns if c != "target"])
        expected = X @ beta
        np.testing.assert_allclose(result.predict(X), expected)

    def test_residuals_identity(self):
        """residuals must equal y - predict(X)."""
        dm, _ = _make_design_matrix()
        result = fit_ridge(dm, "target", alpha=0.0)

        X = dm.drop("target").to_numpy()
        y = dm["target"].to_numpy()
        expected = y - result.predict(X)
        np.testing.assert_allclose(result.residuals, expected)

    def test_residual_std_uses_ddof_1(self):
        """residual_std should use ddof=1 (sample standard deviation)."""
        dm, _ = _make_design_matrix()
        result = fit_ridge(dm, "target", alpha=0.0)
        expected = np.std(result.residuals, ddof=1)
        assert result.residual_std == pytest.approx(expected)

    def test_coefficients_are_named(self):
        """coefficients dict keys should match the feature column names."""
        dm, _ = _make_design_matrix()
        result = fit_ridge(dm, "target", alpha=0.1)
        feature_cols = [c for c in dm.columns if c != "target"]
        assert list(result.coefficients.keys()) == feature_cols


# ---------------------------------------------------------------------------
# fit_ridge: alpha=0 recovers OLS
# ---------------------------------------------------------------------------


class TestFitRidgeUnregularised:
    """With alpha=0, fit_ridge should recover the OLS solution."""

    def test_matches_lstsq(self):
        dm, _ = _make_design_matrix(n=50, seed=1)
        result = fit_ridge(dm, "target", alpha=0.0)

        X = dm.drop("target").to_numpy()
        y = dm["target"].to_numpy()
        beta_lstsq, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        beta_ridge = np.array(
            [result.coefficients[c] for c in dm.columns if c != "target"]
        )
        np.testing.assert_allclose(beta_ridge, beta_lstsq, atol=1e-10)


# ---------------------------------------------------------------------------
# fit_ridge: coefficient recovery
# ---------------------------------------------------------------------------


class TestFitRidgeCoefficientRecovery:
    """With small alpha and low noise, fit_ridge recovers the true coefficients."""

    def test_recovers_true_beta(self):
        true_beta = np.array([3.0, 1.5, -0.8])
        dm, _ = _make_design_matrix(n=200, true_beta=true_beta, seed=2)
        result = fit_ridge(dm, "target", alpha=1e-6)

        beta_hat = np.array(
            [result.coefficients[c] for c in dm.columns if c != "target"]
        )
        np.testing.assert_allclose(beta_hat, true_beta, atol=0.05)


# ---------------------------------------------------------------------------
# fit_ridge: intercept not regularised
# ---------------------------------------------------------------------------


class TestInterceptNotRegularised:
    """The intercept should be approximately invariant to alpha when features
    are centred.

    Rationale: penalising the intercept would make the procedure depend on
    the origin chosen for Y (Hastie et al., ESL 2nd ed., Section 3.4.1).
    """

    def test_intercept_stable_across_alphas(self):
        rng = np.random.default_rng(10)
        n = 100
        x = rng.standard_normal((n, 2))
        x -= x.mean(axis=0)  # Centre features
        true_beta = np.array([5.0, 1.0, -1.0])
        y = true_beta[0] + x @ true_beta[1:] + rng.normal(0, 0.01, n)

        df = pl.DataFrame(
            {"feat_0": x[:, 0].tolist(), "feat_1": x[:, 1].tolist(), "y": y.tolist()}
        )
        dm = build_design_matrix(
            df,
            target=pl.col("y"),
            features=[pl.col("feat_0"), pl.col("feat_1")],
        )

        alphas = [0.0, 0.1, 1.0, 10.0, 100.0]
        intercepts = []
        for alpha in alphas:
            result = fit_ridge(dm, "target", alpha=alpha)
            intercepts.append(result.coefficients["intercept"])

        for ic in intercepts:
            assert ic == pytest.approx(intercepts[0], abs=0.05)


# ---------------------------------------------------------------------------
# fit_ridge: shrinkage
# ---------------------------------------------------------------------------


class TestShrinkage:
    """As alpha increases, non-intercept coefficients shrink toward zero.

    Follows from the SVD representation of ridge coefficients
    (Hastie et al., ESL 2nd ed., equation 3.47): each component is scaled
    by d_j^2 / (d_j^2 + alpha), which is monotonically decreasing in alpha.
    """

    def test_coefficients_shrink_with_alpha(self):
        true_beta = np.array([2.0, 3.0, -2.0])
        dm, _ = _make_design_matrix(n=100, true_beta=true_beta, seed=3)

        alphas = [0.0, 0.1, 1.0, 10.0, 100.0]
        norms = []
        for alpha in alphas:
            result = fit_ridge(dm, "target", alpha=alpha)
            non_intercept = [
                v
                for k, v in result.coefficients.items()
                if k != "intercept"
            ]
            norms.append(np.linalg.norm(non_intercept))

        for i in range(len(norms) - 1):
            assert norms[i] >= norms[i + 1] - 1e-10

    def test_large_alpha_coefficients_near_zero(self):
        true_beta = np.array([2.0, 3.0, -2.0])
        dm, _ = _make_design_matrix(n=100, true_beta=true_beta, seed=4)
        result = fit_ridge(dm, "target", alpha=1e6)

        for k, v in result.coefficients.items():
            if k != "intercept":
                assert abs(v) < 0.01


# ---------------------------------------------------------------------------
# Diagnostics: effective degrees of freedom
# ---------------------------------------------------------------------------


class TestEffectiveDf:
    """Test effective degrees of freedom.

    Defined as df(alpha) = tr(H) where H = X (X'X + alpha*D)^{-1} X' is the
    hat matrix.  For uniform penalty this reduces to sum_j d_j^2 / (d_j^2 +
    alpha) (Hastie et al., ESL 2nd ed., formula 3.50, Section 3.4.1).  Here
    the intercept is excluded from D, so we compute the trace directly.
    """

    def test_alpha_zero_equals_num_columns(self):
        """With no regularisation, effective df = number of columns (incl. intercept)."""
        dm, _ = _make_design_matrix(n=50, seed=5)
        result = fit_ridge(dm, "target", alpha=0.0)
        p = len(dm.columns) - 1  # exclude target
        assert result.effective_df == pytest.approx(p)

    def test_decreases_with_alpha(self):
        dm, _ = _make_design_matrix(n=50, seed=6)
        alphas = [0.0, 0.1, 1.0, 10.0, 100.0]
        dfs = [fit_ridge(dm, "target", alpha=a).effective_df for a in alphas]

        for i in range(len(dfs) - 1):
            assert dfs[i] >= dfs[i + 1] - 1e-10

    def test_large_alpha_approaches_one(self):
        """Very large alpha shrinks effective df.  The unregularised intercept
        column (all ones) still contributes ~1; non-intercept columns
        contribute nearly nothing."""
        dm, _ = _make_design_matrix(n=50, seed=7)
        result = fit_ridge(dm, "target", alpha=1e8)
        assert result.effective_df < 2.0
        assert result.effective_df >= 1.0 - 1e-6


# ---------------------------------------------------------------------------
# Diagnostics: coefficient standard errors
# ---------------------------------------------------------------------------


class TestCoefficientStdErrors:
    """Test coefficient standard errors.

    Computed from the sandwich covariance sigma^2 W X'X W where
    W = (X'X + alpha*D)^-1, assuming i.i.d. normal residuals with constant
    variance.
    """

    def test_alpha_zero_matches_ols(self):
        """At alpha=0, standard errors should match the classical OLS formula
        sqrt(diag(sigma^2 (X'X)^-1))."""
        dm, _ = _make_design_matrix(n=100, seed=8)
        result = fit_ridge(dm, "target", alpha=0.0)

        X = dm.drop("target").to_numpy()
        n, p = X.shape
        sigma2 = np.sum(result.residuals ** 2) / (n - p)
        ols_cov = sigma2 * np.linalg.inv(X.T @ X)
        ols_se = np.sqrt(np.diag(ols_cov))

        feature_cols = [c for c in dm.columns if c != "target"]
        for col, expected_se in zip(feature_cols, ols_se):
            assert result.coefficient_std_errors[col] == pytest.approx(
                expected_se, rel=1e-6
            )

    def test_keys_match_coefficients(self):
        dm, _ = _make_design_matrix()
        result = fit_ridge(dm, "target", alpha=0.1)
        assert list(result.coefficient_std_errors.keys()) == list(
            result.coefficients.keys()
        )

    def test_all_positive(self):
        dm, _ = _make_design_matrix()
        result = fit_ridge(dm, "target", alpha=0.1)
        for se in result.coefficient_std_errors.values():
            assert se > 0.0


# ---------------------------------------------------------------------------
# Diagnostics: AIC and BIC
# ---------------------------------------------------------------------------


class TestAicBic:
    """Test AIC = n*log(RSS/n) + 2*df_eff and BIC = n*log(RSS/n) + log(n)*df_eff."""

    def test_aic_bic_finite(self):
        dm, _ = _make_design_matrix()
        result = fit_ridge(dm, "target", alpha=0.1)
        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)

    def test_bic_exceeds_aic_for_large_n(self):
        """For n > e^2 ~ 7.4, log(n) > 2 so the BIC penalty exceeds the AIC
        penalty, giving BIC >= AIC when effective_df > 0."""
        dm, _ = _make_design_matrix(n=50, seed=9)
        result = fit_ridge(dm, "target", alpha=0.1)
        assert result.bic >= result.aic


# ---------------------------------------------------------------------------
# fit_ridge: input validation
# ---------------------------------------------------------------------------


class TestFitRidgeValidation:
    """Test error handling for bad inputs."""

    def test_missing_target_column(self):
        dm, _ = _make_design_matrix()
        with pytest.raises(ValueError, match="not_a_column"):
            fit_ridge(dm, "not_a_column", alpha=0.1)

    def test_negative_alpha_raises(self):
        dm, _ = _make_design_matrix()
        with pytest.raises(ValueError, match="alpha"):
            fit_ridge(dm, "target", alpha=-0.1)

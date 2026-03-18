"""Tests for ParticipationModel classes in lpt_stake.simulation."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from lpt_stake.features import np_expit, np_logit
from lpt_stake.simulation import (
    DiffLogitParticipationModel,
    GaussianNoise,
    LogitParticipationModel,
    RawParticipationModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


ZERO_NOISE = GaussianNoise(sigma=0.0)


def _make_raw_linear_df(n: int = 50, seed: int = 42) -> pl.DataFrame:
    """Data where P(n+1) = a + b * x(n).  Linear in raw space."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.2, 0.8, size=n)
    p = np.empty(n)
    p[0] = 0.5
    for i in range(n - 1):
        p[i + 1] = np.clip(0.3 + 0.4 * x[i], 0.01, 0.99)
    return pl.DataFrame({"x": x, "participation_rate": p})


def _make_logit_linear_df(n: int = 50, seed: int = 42) -> pl.DataFrame:
    """Data where logit(P(n+1)) = a + b * x(n).  Linear in logit space."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n)
    p = np.empty(n)
    p[0] = 0.5
    for i in range(n - 1):
        # logit(P(n+1)) = -0.2 + 0.5 * x(n)
        p[i + 1] = float(np_expit(np.array(-0.2 + 0.5 * x[i])))
    return pl.DataFrame({"x": x, "participation_rate": p})


def _make_difflogit_linear_df(n: int = 50, seed: int = 42) -> pl.DataFrame:
    """Data where logit(P(n+1)) - logit(P(n)) = a + b * x(n).
    Linear in diff-logit space.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.5, 0.5, size=n)
    p = np.empty(n)
    p[0] = 0.5
    for i in range(n - 1):
        # delta_logit = 0.01 + 0.1 * x(n)
        delta = 0.01 + 0.1 * x[i]
        p[i + 1] = float(np_expit(np_logit(np.array(p[i])) + delta))
    return pl.DataFrame({"x": x, "participation_rate": p})


class _FakeFeature:
    """A Feature object that returns a pre-computed column for testing."""

    def __init__(self, col: str) -> None:
        self.columns = [col]
        self._col = col

    def evaluate(self, df: pl.DataFrame) -> np.ndarray:
        return df[self._col].to_numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Roundtrip tests — the primary correctness check
# ---------------------------------------------------------------------------


class TestRawRoundtrip:
    """Fit RawParticipationModel on data linear in raw space."""

    def test_roundtrip(self):
        df = _make_raw_linear_df()
        model = RawParticipationModel(
            features=["x"], target="participation_rate",
            alpha=0.0, noise=ZERO_NOISE,
        )
        model.fit(df)

        x = df["x"].to_numpy()
        p = df["participation_rate"].to_numpy()
        rng = np.random.default_rng(0)
        for i in range(len(df) - 1):
            fv = x[i].reshape(1, 1)
            pred = model.predict(fv, p[i:i + 1], rng)
            assert pred[0] == pytest.approx(p[i + 1], abs=1e-6)


class TestLogitRoundtrip:
    """Fit LogitParticipationModel on data linear in logit space."""

    def test_roundtrip(self):
        df = _make_logit_linear_df()
        model = LogitParticipationModel(
            features=["x"], target="participation_rate",
            alpha=0.0, noise=ZERO_NOISE,
        )
        model.fit(df)

        x = df["x"].to_numpy()
        p = df["participation_rate"].to_numpy()
        rng = np.random.default_rng(0)
        for i in range(len(df) - 1):
            fv = x[i].reshape(1, 1)
            pred = model.predict(fv, p[i:i + 1], rng)
            assert pred[0] == pytest.approx(p[i + 1], abs=1e-6)


class TestDiffLogitRoundtrip:
    """Fit DiffLogitParticipationModel on data linear in diff-logit space."""

    def test_roundtrip(self):
        df = _make_difflogit_linear_df()
        model = DiffLogitParticipationModel(
            features=["x"], target="participation_rate",
            alpha=0.0, noise=ZERO_NOISE,
        )
        model.fit(df)

        x = df["x"].to_numpy()
        p = df["participation_rate"].to_numpy()
        rng = np.random.default_rng(0)
        for i in range(len(df) - 1):
            fv = x[i].reshape(1, 1)
            pred = model.predict(fv, p[i:i + 1], rng)
            assert pred[0] == pytest.approx(p[i + 1], abs=1e-6)


# ---------------------------------------------------------------------------
# Feature object (vs string) path
# ---------------------------------------------------------------------------


class TestFeatureObject:
    """Verify that Feature objects work the same as string column names."""

    def test_feature_object_same_as_string(self):
        df = _make_raw_linear_df()

        model_str = RawParticipationModel(
            features=["x"], target="participation_rate",
            alpha=0.1, noise=ZERO_NOISE,
        )
        model_str.fit(df)

        model_feat = RawParticipationModel(
            features=[_FakeFeature("x")], target="participation_rate",
            alpha=0.1, noise=ZERO_NOISE,
        )
        model_feat.fit(df)

        str_coefs = list(model_str.ridge_result.coefficients.values())
        feat_coefs = list(model_feat.ridge_result.coefficients.values())
        for s, f in zip(str_coefs, feat_coefs):
            assert s == pytest.approx(f)


# ---------------------------------------------------------------------------
# Predict shape and range
# ---------------------------------------------------------------------------


class TestPredictShape:
    """predict() must return (n_paths,) with values in [0, 1]."""

    @pytest.mark.parametrize("model_cls,df_fn", [
        (RawParticipationModel, _make_raw_linear_df),
        (LogitParticipationModel, _make_logit_linear_df),
        (DiffLogitParticipationModel, _make_difflogit_linear_df),
    ])
    def test_shape_and_range(self, model_cls, df_fn):
        df = df_fn()
        model = model_cls(
            features=["x"], target="participation_rate",
            alpha=0.1, noise=GaussianNoise(sigma=0.001),
        )
        model.fit(df)

        n_paths = 100
        rng = np.random.default_rng(42)
        fv = rng.uniform(-0.5, 0.5, size=(n_paths, 1))
        current_p = np.full(n_paths, 0.5)
        pred = model.predict(fv, current_p, rng)

        assert pred.shape == (n_paths,)
        assert np.all(pred >= 0.0)
        assert np.all(pred <= 1.0)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestPredictBeforeFit:
    """predict() before fit() should raise RuntimeError."""

    def test_raises(self):
        model = RawParticipationModel(
            features=["x"], target="participation_rate",
            alpha=0.1, noise=ZERO_NOISE,
        )
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(np.ones((1, 1)), np.ones(1), np.random.default_rng(0))

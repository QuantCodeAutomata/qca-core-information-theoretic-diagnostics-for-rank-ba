"""Unit tests for information-theoretic diagnostics and chain estimators."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.entropy import (
    bucketize_covariate,
    pooled_transition_counts,
    entropy_production,
    conditional_transition_counts,
    conditional_entropy_production,
    transfer_entropy_X_to_A,
    transfer_entropy_A_to_X,
)


def _make_state_panel(n_stocks: int = 8, n_dates: int = 60, k: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-31", periods=n_dates, freq="ME")
    cols = [f"S{i}" for i in range(n_stocks)]
    data = rng.integers(1, k + 1, size=(n_dates, n_stocks)).astype(float)
    return pd.DataFrame(data, index=idx, columns=cols)


# ---------- Bucketization ----------
def test_bucketize_range():
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(20, 50)))
    b = bucketize_covariate(x, kf=10)
    assert b.min().min() >= 1
    assert b.max().max() <= 10


def test_bucketize_empty_row_ok():
    x = pd.DataFrame(np.full((3, 5), np.nan))
    b = bucketize_covariate(x, kf=10)
    assert b.isna().all().all()


# ---------- Pooled diagnostics ----------
def test_pooled_transition_counts_shape():
    s = _make_state_panel()
    N = pooled_transition_counts(s.iloc[:-1], s.shift(-1).iloc[:-1], k=10)
    assert N.shape == (10, 10)
    assert N.sum() > 0


def test_entropy_production_symmetric_zero():
    N = np.ones((5, 5)) * 100.0  # perfectly symmetric
    sigma = entropy_production(N)
    assert abs(sigma) < 1e-9


def test_entropy_production_asymmetric_positive():
    # A strongly directed one-way flow yields positive sigma.
    N = np.array([[10.0, 100.0], [1.0, 10.0]])
    sigma = entropy_production(N)
    assert sigma > 0.1


# ---------- Conditional diagnostics ----------
def test_conditional_counts_shape():
    s = _make_state_panel()
    x = _make_state_panel(seed=1)
    C = conditional_transition_counts(s.iloc[:-1], s.shift(-1).iloc[:-1], x.iloc[:-1], k=10, kf=10)
    assert C.shape == (10, 10, 10)


def test_conditional_entropy_production_geq_pooled_up_to_noise():
    s = _make_state_panel(seed=2)
    x = _make_state_panel(seed=3)
    N_pool = pooled_transition_counts(s.iloc[:-1], s.shift(-1).iloc[:-1], k=10)
    sigma_pool = entropy_production(N_pool)
    C = conditional_transition_counts(s.iloc[:-1], s.shift(-1).iloc[:-1], x.iloc[:-1], k=10, kf=10)
    sigma_cond, _, _ = conditional_entropy_production(C)
    # Should hold up to smoothing noise; use loose tolerance for random data.
    assert sigma_cond >= sigma_pool - 5e-3


# ---------- Transfer entropy ----------
def test_te_nonneg_random():
    s = _make_state_panel(seed=4)
    x = _make_state_panel(seed=5)
    te = transfer_entropy_X_to_A(
        s.iloc[:-1], s.shift(-1).iloc[:-1], x.iloc[:-1], k=10, kf=10
    )
    assert te >= -1e-9


def test_te_axtox_nonneg_random():
    s = _make_state_panel(seed=6)
    x = _make_state_panel(seed=7)
    te = transfer_entropy_A_to_X(
        s.iloc[:-1], s.shift(-1).iloc[:-1], x.iloc[:-1], x.shift(-1).iloc[:-1], k=10, kf=10
    )
    assert te >= -1e-9


def test_te_stronger_when_x_drives_a():
    # Construct dataset where next-a is a deterministic function of current x.
    rng = np.random.default_rng(11)
    n_dates, n_stocks = 200, 20
    idx = pd.date_range("2015-01-31", periods=n_dates, freq="ME")
    cols = [f"S{i}" for i in range(n_stocks)]
    x_vals = rng.integers(1, 11, size=(n_dates, n_stocks))
    # a_{t+1} = x_t (perfect information flow x -> a); a_t is independent noise
    a_vals = rng.integers(1, 11, size=(n_dates, n_stocks))
    a_next_vals = x_vals.copy()
    a = pd.DataFrame(a_vals, index=idx, columns=cols).astype(float)
    a_next = pd.DataFrame(a_next_vals, index=idx, columns=cols).astype(float)
    x = pd.DataFrame(x_vals, index=idx, columns=cols).astype(float)
    te_xa = transfer_entropy_X_to_A(a, a_next, x, k=10, kf=10)
    # Should be substantial (much larger than zero)
    assert te_xa > 0.5


# ---------- Chain estimator ----------
def test_unconditional_row_stochastic():
    from src.chain_estimator import fit_unconditional
    s = _make_state_panel(seed=8)
    chain = fit_unconditional(s.iloc[:-1], s.shift(-1).iloc[:-1], k=10)
    P = chain.P
    assert P.shape == (10, 10)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-8)
    assert (P >= 0).all()


def test_conditional_generator_produces_valid_transitions():
    from src.chain_estimator import fit_conditional, centered_decile_score
    s = _make_state_panel(seed=9)
    x_panel = _make_state_panel(seed=10)
    fit = fit_conditional(
        s.iloc[:-1],
        s.shift(-1).iloc[:-1],
        covariates={"x": x_panel.iloc[:-1]},
        k=10,
        kf=10,
        epsilon=0.1,
    )
    # Evaluate the transition matrix at midpoint covariate bucket.
    phi = np.array([centered_decile_score(5, 10)])
    P = fit.transition_matrix(phi)
    assert P.shape == (10, 10)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
    assert (P >= -1e-9).all()

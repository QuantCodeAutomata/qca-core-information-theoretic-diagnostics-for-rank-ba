"""Unit tests for features, distance matrix, portfolio helpers, and TE network."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import (
    cross_sectional_decile,
    trailing_mean_return,
    trailing_realized_vol,
    arccos_distance_matrix,
    leading_eigenvector_loading,
    distance_centrality,
)
from src.portfolio import (
    regime_and_blend,
    stationary_block_bootstrap,
)
from src.te_network import hits_scores


def test_cross_sectional_decile_range():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(30, 40)))
    d = cross_sectional_decile(df, k=10)
    assert d.min().min() >= 1
    assert d.max().max() <= 10


def test_trailing_mean_return_lookback():
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(scale=0.01, size=(300, 5)))
    m = trailing_mean_return(r, window=126)
    # First 125 rows must be NaN (insufficient lookback)
    assert m.iloc[124].isna().all()
    assert not m.iloc[126].isna().all()


def test_trailing_vol_positive():
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(scale=0.01, size=(100, 5)))
    v = trailing_realized_vol(r, window=21)
    v_valid = v.dropna(how="all").dropna(axis=1, how="all")
    assert (v_valid.to_numpy() >= 0).all()


def test_arccos_distance_matrix_properties():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 10))
    corr = np.corrcoef(X, rowvar=False)
    M = arccos_distance_matrix(corr)
    # Symmetry
    assert np.allclose(M, M.T, atol=1e-10)
    # Zero diagonal
    assert np.allclose(np.diag(M), 0.0, atol=1e-10)
    # Range in [0, pi]
    assert M.min() >= 0 - 1e-9
    assert M.max() <= np.pi + 1e-9


def test_leading_eigenvector_loading_bounded():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 8))
    corr = np.corrcoef(X, rowvar=False)
    v = leading_eigenvector_loading(corr)
    assert v.shape == (8,)
    assert (v >= 0).all()  # absolute values


def test_distance_centrality_shape():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 8))
    corr = np.corrcoef(X, rowvar=False)
    M = arccos_distance_matrix(corr)
    c = distance_centrality(M)
    assert c.shape == (8,)
    assert (c >= 0).all()


def test_regime_and_blend_rising_vs_falling():
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    r_rising = pd.Series([0.02] * 6, index=idx)  # positive trailing return
    lam, theta = regime_and_blend(r_rising, idx[-1])
    assert lam == 0.0 and theta == 1.0

    r_falling = pd.Series([-0.02] * 6, index=idx)
    lam2, theta2 = regime_and_blend(r_falling, idx[-1])
    assert lam2 == 0.75 and theta2 == 0.4


def test_stationary_block_bootstrap_summary_dict_and_seed():
    rng = np.random.default_rng(42)
    x = pd.Series(rng.normal(scale=0.01, size=500))
    b1 = stationary_block_bootstrap(x, n_boot=200, expected_block=21, seed=0)
    b2 = stationary_block_bootstrap(x, n_boot=200, expected_block=21, seed=0)
    assert set(["p_one_sided", "ci_low", "ci_high", "mean"]).issubset(b1.keys())
    # Reproducibility with same seed
    assert b1 == b2
    # CI bounds ordered
    assert b1["ci_low"] <= b1["ci_high"]


def test_hits_scores_basic():
    # A 3-node directed graph where node 0 sends edges out (hub) and node 2 receives (authority).
    A = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    hubs, auths = hits_scores(A, iters=200)
    assert hubs.shape == (3,)
    assert auths.shape == (3,)
    # node 0 should be the top hub; node 2 should be the top authority.
    assert np.argmax(hubs) == 0
    assert np.argmax(auths) == 2

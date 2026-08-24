"""Stock-to-stock transfer-entropy network and HITS hub scores."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .entropy import LAPLACE, _smooth_counts


def pairwise_te_matrix(
    states: pd.DataFrame, k: int, alpha: float = LAPLACE
) -> tuple[np.ndarray, list[str]]:
    """Compute TE(i -> j) for all pairs from bucketed monthly state paths.

    States must be a DataFrame indexed by date with tickers as columns and
    integer bucket values in 1..k.
    """
    tickers = list(states.columns)
    N = len(tickers)
    A = np.zeros((N, N))
    # For efficiency, precompute per-ticker sequences and next-values
    arrs = {tk: states[tk].dropna().astype(int).values - 1 for tk in tickers}
    dates = {tk: states[tk].dropna().index for tk in tickers}

    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            if i == j:
                continue
            common = dates[ti].intersection(dates[tj])
            if len(common) < 5:
                continue
            a_i = states.loc[common, ti].astype(int).values - 1
            a_j = states.loc[common, tj].astype(int).values - 1
            # a_j_{t}, a_j_{t+1}, a_i_{t}
            if len(common) < 3:
                continue
            aj_t = a_j[:-1]
            aj_next = a_j[1:]
            ai_t = a_i[:-1]
            # Build 3-way histogram
            H = np.zeros((k, k, k))
            np.add.at(H, (ai_t, aj_t, aj_next), 1)
            H = _smooth_counts(H, alpha)
            p = H / H.sum()
            # p(a_j_{t+1} | a_j_t, a_i_t)
            p_ij = p.sum(axis=2, keepdims=True)  # p(a_i, a_j)
            p_bp_given_ij = p / p_ij
            # p(a_j_{t+1} | a_j_t)
            p_j = p.sum(axis=0)  # (k,k)
            p_jt = p_j.sum(axis=1, keepdims=True)
            p_bp_given_j = p_j / p_jt
            with np.errstate(divide="ignore", invalid="ignore"):
                log_ratio = np.log(p_bp_given_ij) - np.log(p_bp_given_j[None, :, :])
                term = p * log_ratio
            term = np.where(np.isfinite(term), term, 0.0)
            A[i, j] = float(term.sum())
    A = np.clip(A, 0.0, None)
    return A, tickers


def hits_scores(A: np.ndarray, iters: int = 100, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Power-iteration HITS on a nonnegative weighted directed adjacency A."""
    n = A.shape[0]
    if n == 0:
        return np.zeros(n), np.zeros(n)
    h = np.ones(n) / n
    a = np.ones(n) / n
    for _ in range(iters):
        a_new = A.T @ h
        na = np.linalg.norm(a_new)
        if na == 0:
            break
        a_new /= na
        h_new = A @ a_new
        nh = np.linalg.norm(h_new)
        if nh == 0:
            break
        h_new /= nh
        if np.linalg.norm(h_new - h) + np.linalg.norm(a_new - a) < tol:
            h, a = h_new, a_new
            break
        h, a = h_new, a_new
    return h, a

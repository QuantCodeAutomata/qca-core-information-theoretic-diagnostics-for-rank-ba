"""Information-theoretic diagnostics.

Custom — Context7 confirmed no library implements exactly the paper's pooled
entropy production, conditional entropy production, and plug-in transfer
entropy over cross-sectional rank chains with pseudocount smoothing.

Formulas
--------
Pooled entropy production:
    sigma = sum_{a,b} mu_ab log(mu_ab / mu_ba)
Conditional:
    sigma(x) = sum_{a,b} mu_ab|x log(mu_ab|x / mu_ba|x)
    sigma_cond = sum_x p(x) sigma(x); Delta sigma = sigma_cond - sigma.
Transfer entropy (plug-in, discrete):
    TE(X->A) = sum p(a_{t+1},a_t,x_t) log[ p(a_{t+1}|a_t,x_t) / p(a_{t+1}|a_t) ]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


LAPLACE = 1e-4  # pseudocount smoothing default


def bucketize_covariate(x: pd.DataFrame, kf: int = 10) -> pd.DataFrame:
    """Cross-sectional decile buckets per row (values 1..kf).

    Deterministic tie handling: ranks use method='first'.
    """
    ranks = x.rank(axis=1, method="first")
    n_valid = x.notna().sum(axis=1)
    # ceil(kf * rank / n_valid), clipped to [1, kf]
    out = np.ceil(ranks.div(n_valid, axis=0) * kf)
    return out.clip(lower=1, upper=kf).astype(float).where(x.notna())


def _smooth_counts(counts: np.ndarray, alpha: float = LAPLACE) -> np.ndarray:
    return counts.astype(float) + alpha


def pooled_transition_counts(
    state_current: pd.DataFrame, state_next: pd.DataFrame, k: int
) -> np.ndarray:
    """Pool (a_t, a_{t+1}) counts over all stocks and dates.

    Parameters
    ----------
    state_current : DataFrame of current decile states (values in 1..k)
    state_next    : DataFrame of next-month decile states (same shape/index)
    k : number of buckets
    """
    a = state_current.to_numpy().flatten()
    b = state_next.to_numpy().flatten()
    mask = (~pd.isna(a)) & (~pd.isna(b))
    a = a[mask].astype(int) - 1
    b = b[mask].astype(int) - 1
    N = np.zeros((k, k))
    np.add.at(N, (a, b), 1)
    return N


def entropy_production(counts: np.ndarray, alpha: float = LAPLACE) -> float:
    """sigma = sum_ab mu_ab log(mu_ab / mu_ba).

    Uses pseudocount smoothing on the joint counts then normalizes to a joint
    probability distribution.
    """
    C = _smooth_counts(counts, alpha)
    mu = C / C.sum()
    mu_ba = mu.T
    with np.errstate(divide="ignore", invalid="ignore"):
        term = mu * (np.log(mu) - np.log(mu_ba))
    return float(term.sum())


def conditional_transition_counts(
    state_current: pd.DataFrame,
    state_next: pd.DataFrame,
    covariate: pd.DataFrame,
    k: int,
    kf: int,
) -> np.ndarray:
    """Joint counts over (x, a_t, a_{t+1}) with covariate buckets x in 1..kf.

    Returns array of shape (kf, k, k).
    """
    a = state_current.to_numpy().flatten()
    b = state_next.to_numpy().flatten()
    x = covariate.to_numpy().flatten()
    mask = (~pd.isna(a)) & (~pd.isna(b)) & (~pd.isna(x))
    a = a[mask].astype(int) - 1
    b = b[mask].astype(int) - 1
    x = x[mask].astype(int) - 1
    N = np.zeros((kf, k, k))
    np.add.at(N, (x, a, b), 1)
    return N


def conditional_entropy_production(
    cond_counts: np.ndarray, alpha: float = LAPLACE
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (sigma_cond, sigma_x_vector, p_x_vector)."""
    kf = cond_counts.shape[0]
    sigmas = np.zeros(kf)
    weights = np.zeros(kf)
    total = cond_counts.sum()
    for xi in range(kf):
        n_x = cond_counts[xi].sum()
        weights[xi] = n_x / total if total > 0 else 0.0
        sigmas[xi] = entropy_production(cond_counts[xi], alpha=alpha)
    sigma_cond = float((weights * sigmas).sum())
    return sigma_cond, sigmas, weights


def transfer_entropy_X_to_A(
    state_current: pd.DataFrame,
    state_next: pd.DataFrame,
    covariate: pd.DataFrame,
    k: int,
    kf: int,
    alpha: float = LAPLACE,
) -> float:
    """Discrete plug-in transfer entropy TE(X -> A).

    TE(X->A) = sum p(a',a,x) log[p(a'|a,x) / p(a'|a)].
    """
    C = conditional_transition_counts(state_current, state_next, covariate, k, kf)
    C = _smooth_counts(C, alpha)
    p_axb = C / C.sum()  # p(x, a, a')

    # Marginal over covariate x: p(a, a')
    p_ab = p_axb.sum(axis=0)  # (k, k)
    p_a = p_ab.sum(axis=1, keepdims=True)  # (k,1)
    p_bp_given_a = p_ab / p_a  # p(a' | a)

    # p(a', a | x) -> p(a' | a, x)
    p_ax = p_axb.sum(axis=2, keepdims=True)  # (kf,k,1) -- p(a,x)? actually p(x,a)
    p_bp_given_ax = p_axb / p_ax  # p(a' | a, x)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(p_bp_given_ax) - np.log(p_bp_given_a[None, :, :])
        term = p_axb * log_ratio
    # Zero out non-finite (from 0*log0)
    term = np.where(np.isfinite(term), term, 0.0)
    return float(term.sum())


def transfer_entropy_A_to_X(
    state_current: pd.DataFrame,
    state_next: pd.DataFrame,
    covariate_current: pd.DataFrame,
    covariate_next: pd.DataFrame,
    k: int,
    kf: int,
    alpha: float = LAPLACE,
) -> float:
    """TE(A -> X) using next-period covariate bucket."""
    # Build joint counts (x_t, x_{t+1}, a_t)
    a = covariate_current.to_numpy().flatten()  # x_t (reuse machinery)
    b = covariate_next.to_numpy().flatten()  # x_{t+1}
    x = state_current.to_numpy().flatten()  # a_t
    mask = (~pd.isna(a)) & (~pd.isna(b)) & (~pd.isna(x))
    a = a[mask].astype(int) - 1
    b = b[mask].astype(int) - 1
    x = x[mask].astype(int) - 1
    N = np.zeros((k, kf, kf))
    np.add.at(N, (x, a, b), 1)
    N = _smooth_counts(N, alpha)
    p = N / N.sum()  # p(a_t, x_t, x_{t+1})

    p_xx = p.sum(axis=0)  # p(x_t, x_{t+1})
    p_xt = p_xx.sum(axis=1, keepdims=True)
    p_xp_given_x = p_xx / p_xt

    p_xa = p.sum(axis=2, keepdims=True)  # p(a_t, x_t)
    p_xp_given_xa = p / p_xa

    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(p_xp_given_xa) - np.log(p_xp_given_x[None, :, :])
        term = p * log_ratio
    term = np.where(np.isfinite(term), term, 0.0)
    return float(term.sum())


# Convenient alias so callers may use a symmetric name
transfer_entropy = transfer_entropy_X_to_A


@dataclass
class DiagnosticRow:
    chain: str
    covariate: str
    sigma: float
    sigma_cond: float
    delta_sigma: float
    te_x_to_a: float
    te_a_to_x: float
    net_te: float


def diagnose_chain(
    state_now: pd.DataFrame,
    state_next: pd.DataFrame,
    covariates: dict[str, pd.DataFrame],
    k: int,
    kf: int,
    chain_label: str,
    alpha: float = LAPLACE,
) -> pd.DataFrame:
    """Compute full diagnostic table for one chain and a set of covariates."""
    pooled = pooled_transition_counts(state_now, state_next, k)
    sigma = entropy_production(pooled, alpha=alpha)

    rows: list[DiagnosticRow] = []
    for name, cov in covariates.items():
        cov_aligned = cov.reindex(state_now.index)
        # Align next-covariate for TE(A->X)
        cov_next = cov.shift(-1).reindex(state_now.index)
        cnt = conditional_transition_counts(state_now, state_next, cov_aligned, k, kf)
        sigma_cond, _, _ = conditional_entropy_production(cnt, alpha=alpha)
        te_x_a = transfer_entropy_X_to_A(state_now, state_next, cov_aligned, k, kf, alpha)
        te_a_x = transfer_entropy_A_to_X(
            state_now, state_next, cov_aligned, cov_next, k, kf, alpha
        )
        rows.append(
            DiagnosticRow(
                chain=chain_label,
                covariate=name,
                sigma=sigma,
                sigma_cond=sigma_cond,
                delta_sigma=sigma_cond - sigma,
                te_x_to_a=te_x_a,
                te_a_to_x=te_a_x,
                net_te=te_x_a - te_a_x,
            )
        )
    return pd.DataFrame([r.__dict__ for r in rows])

"""Unconditional and conditional continuous-time rank Markov chain estimators.

Custom — Context7 confirmed no library implements the paper's per-pair convex
generator MLE with ridge penalty over discrete covariate buckets.

The conditional CTMC uses log-linear generator:
    q_ab(x) = exp(eta_ab + gamma_ab . phi(x))   for a != b
    q_aa(x) = -sum_{b != a} q_ab(x)
The one-month transition matrix is P(x) = expm(Q(x)).
Because we work directly on monthly transitions, we absorb the 1-month
holding time into Q by treating each monthly transition as a single unit-time
observation and estimating Q * (1 month) in one shot; equivalently we
parametrize the log-transition rate directly. For each pair (a,b) we minimize:
    -log L_ab = Delta N_ab log(sum_x R_a(x) e^{gamma_ab . phi(x)})
               - gamma_ab . sum_x Delta N_ab(x) phi(x)
               + epsilon * ||gamma_ab||^2
where Delta N_ab(x) is the number of a->b transitions observed with covariate
bucket x, and R_a(x) = sum_b Delta N_ab(x) (a-row totals, i.e., holding-time
proxy = number of monthly observations starting in state a with covariate x).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize


@dataclass
class UnconditionalChain:
    """Pooled row-stochastic monthly transition matrix."""

    P: np.ndarray  # (k,k)
    counts: np.ndarray

    def predict_row(self, a: int) -> np.ndarray:
        return self.P[a - 1]


def fit_unconditional(
    state_now: pd.DataFrame, state_next: pd.DataFrame, k: int, alpha: float = 1e-4
) -> UnconditionalChain:
    """Fit pooled monthly transition matrix with Laplace smoothing."""
    a = state_now.to_numpy().flatten()
    b = state_next.to_numpy().flatten()
    mask = (~pd.isna(a)) & (~pd.isna(b))
    a = a[mask].astype(int) - 1
    b = b[mask].astype(int) - 1
    N = np.zeros((k, k))
    np.add.at(N, (a, b), 1)
    N_s = N + alpha
    P = N_s / N_s.sum(axis=1, keepdims=True)
    return UnconditionalChain(P=P, counts=N)


@dataclass
class ConditionalChain:
    """Log-linear conditional generator model.

    Attributes
    ----------
    eta : (k,k) baseline log-intensities (diagonal set to zero)
    gamma : (k,k,d) per-pair covariate coefficients
    k : number of states
    d : feature dimension of phi
    """

    eta: np.ndarray
    gamma: np.ndarray
    k: int
    d: int
    feature_names: list[str] = field(default_factory=list)

    def transition_matrix(self, phi_x: np.ndarray) -> np.ndarray:
        """Compute P = expm(Q(x)) with Q parameterized log-linearly.

        Since our per-pair NLL is written directly on monthly transitions,
        Q(x) here has entries q_ab = exp(eta + gamma . phi) and we treat the
        resulting matrix as the direct log-monthly-rate. We then take expm to
        obtain a valid stochastic matrix.
        """
        # off-diagonal log-rates
        log_q = self.eta + self.gamma @ phi_x  # (k,k)
        Q = np.exp(log_q)
        np.fill_diagonal(Q, 0.0)
        Q[np.arange(self.k), np.arange(self.k)] = -Q.sum(axis=1)
        P = expm(Q)
        # numerical cleanup
        P = np.clip(P, 1e-12, None)
        P = P / P.sum(axis=1, keepdims=True)
        return P


def centered_decile_score(bucket: int, kf: int = 10) -> float:
    """phi(x) = (x - (kf+1)/2)/kf  for x in 1..kf."""
    return (bucket - (kf + 1) / 2.0) / kf


def encode_covariates(cov_row: dict[str, int], feature_names: list[str], kf: int = 10) -> np.ndarray:
    """Encode a per-observation covariate bucket dict to a phi vector."""
    return np.array(
        [centered_decile_score(cov_row[name], kf) for name in feature_names],
        dtype=float,
    )


def _build_sufficient_stats(
    state_now: pd.DataFrame,
    state_next: pd.DataFrame,
    covariates: dict[str, pd.DataFrame],
    k: int,
    kf: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Aggregate transitions into per-pair and per-row phi-weighted stats.

    Returns
    -------
    trans_counts : (k, k, n_obs) is replaced by more compact structures:
        we return `S_ab` (sum of phi over observations in a->b) and
        `phi_by_a` (sum of phi over row observations with holding state a)
        and `n_ab` (raw counts).
    """
    feature_names = list(covariates.keys())
    d = len(feature_names)
    idx = state_now.index

    # Flatten everything
    a_all: list[int] = []
    b_all: list[int] = []
    phi_all: list[np.ndarray] = []
    for date in idx:
        row_a = state_now.loc[date]
        row_b = state_next.loc[date]
        cov_rows = {name: covariates[name].loc[date] for name in feature_names}
        for tk in row_a.index:
            va = row_a.get(tk)
            vb = row_b.get(tk)
            if pd.isna(va) or pd.isna(vb):
                continue
            missing = False
            phi = np.zeros(d)
            for j, name in enumerate(feature_names):
                v = cov_rows[name].get(tk)
                if pd.isna(v):
                    missing = True
                    break
                phi[j] = centered_decile_score(int(v), kf)
            if missing:
                continue
            a_all.append(int(va) - 1)
            b_all.append(int(vb) - 1)
            phi_all.append(phi)

    if not a_all:
        return (
            np.zeros((k, k)),
            np.zeros((k, k, d)),
            feature_names,
        )
    a_arr = np.array(a_all)
    b_arr = np.array(b_all)
    phi_arr = np.stack(phi_all)  # (M, d)

    n_ab = np.zeros((k, k))
    S_ab = np.zeros((k, k, d))
    for m in range(len(a_arr)):
        n_ab[a_arr[m], b_arr[m]] += 1
        S_ab[a_arr[m], b_arr[m]] += phi_arr[m]

    # We also need row-level phi lists for the partition-function log-sum-exp
    # in the NLL. Store per-row (a) the raw phi matrix compactly.
    # To keep memory bounded, we approximate log-sum-exp over observations
    # by bucketed grouping: since phi entries take only kf distinct values
    # per feature, we can group by unique phi vectors within each starting
    # state a.
    row_phi: dict[int, np.ndarray] = {}
    row_weight: dict[int, np.ndarray] = {}
    for a in range(k):
        mask = a_arr == a
        if mask.sum() == 0:
            row_phi[a] = np.zeros((0, d))
            row_weight[a] = np.zeros(0)
            continue
        phi_a = phi_arr[mask]
        # unique rows
        unique, inv, counts = np.unique(phi_a, axis=0, return_inverse=True, return_counts=True)
        row_phi[a] = unique
        row_weight[a] = counts.astype(float)
    return n_ab, S_ab, feature_names, row_phi, row_weight  # type: ignore[return-value]


def fit_conditional(
    state_now: pd.DataFrame,
    state_next: pd.DataFrame,
    covariates: dict[str, pd.DataFrame],
    k: int,
    kf: int,
    epsilon: float = 0.1,
) -> ConditionalChain:
    """Fit the per-pair convex log-linear generator MLE.

    Solves for each ordered pair (a,b) with a != b independently.
    """
    stats = _build_sufficient_stats(state_now, state_next, covariates, k, kf)
    n_ab, S_ab, feature_names, row_phi, row_weight = stats
    d = len(feature_names)

    eta = np.zeros((k, k))
    gamma = np.zeros((k, k, d))

    for a in range(k):
        phi_a = row_phi[a]  # (U_a, d)
        w_a = row_weight[a]  # (U_a,)
        R_a = w_a.sum()
        for b in range(k):
            if a == b:
                continue
            n = n_ab[a, b]
            if R_a <= 0:
                # No observations at all; leave zero
                continue

            s_ab = S_ab[a, b]  # (d,)

            def nll(g: np.ndarray) -> tuple[float, np.ndarray]:
                # log Z(g) = log sum_x w(x) exp(g . phi(x)) over row observations
                scores = phi_a @ g  # (U_a,)
                max_s = scores.max() if scores.size else 0.0
                exp_s = np.exp(scores - max_s)
                Z = float((w_a * exp_s).sum())
                logZ = max_s + np.log(Z) if Z > 0 else max_s
                val = n * logZ - g @ s_ab + epsilon * (g @ g)
                # gradient
                weights = w_a * exp_s
                weights /= weights.sum() if weights.sum() > 0 else 1.0
                grad_logZ = weights @ phi_a  # (d,)
                grad = n * grad_logZ - s_ab + 2 * epsilon * g
                return val, grad

            g0 = np.zeros(d)
            res = minimize(
                nll,
                g0,
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 100, "ftol": 1e-8},
            )
            g_hat = res.x
            gamma[a, b] = g_hat
            # closed-form eta from baseline rate: expected rate matches empirical
            # Choose eta so that sum_x w(x) exp(eta + g.phi(x)) = n_ab
            scores = phi_a @ g_hat
            max_s = scores.max() if scores.size else 0.0
            Z = float((w_a * np.exp(scores - max_s)).sum())
            # exp(eta) * (max_s exp offset) * Z = n_ab  =>
            # eta = log(n_ab / Z) - max_s
            eta_val = np.log(max(n, 1e-6) / max(Z, 1e-12)) - max_s
            eta[a, b] = eta_val

    return ConditionalChain(eta=eta, gamma=gamma, k=k, d=d, feature_names=feature_names)


def predict_row_prob(
    chain: ConditionalChain,
    a_state: int,
    cov_bucket: dict[str, int],
    kf: int = 10,
) -> np.ndarray:
    """Predict the k-vector of next-state probabilities given (a_state, x)."""
    phi = np.array(
        [centered_decile_score(cov_bucket[n], kf) for n in chain.feature_names],
        dtype=float,
    )
    P = chain.transition_matrix(phi)
    return P[a_state - 1]

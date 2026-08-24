"""Walk-forward forecasting evaluation.

Fits conditional and unconditional chain models on an expanding history and
evaluates predictive log-likelihood, calibration, and MAE against realized
next-month states.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .chain_estimator import (
    ConditionalChain,
    UnconditionalChain,
    fit_conditional,
    fit_unconditional,
    predict_row_prob,
    centered_decile_score,
)


@dataclass
class ForecastResult:
    chain_label: str
    model_label: str
    mean_log_lik: float
    mae: float
    n_obs: int


def _iter_stock_month_obs(
    state_now: pd.DataFrame, state_next: pd.DataFrame
):
    for date in state_now.index:
        row_a = state_now.loc[date]
        row_b = state_next.loc[date]
        for tk in row_a.index:
            va = row_a.get(tk)
            vb = row_b.get(tk)
            if pd.isna(va) or pd.isna(vb):
                continue
            yield date, tk, int(va), int(vb)


def evaluate_unconditional(
    chain: UnconditionalChain,
    state_now: pd.DataFrame,
    state_next: pd.DataFrame,
) -> ForecastResult:
    logs: list[float] = []
    maes: list[float] = []
    P = chain.P
    for _, _, a, b in _iter_stock_month_obs(state_now, state_next):
        p = P[a - 1]
        logs.append(float(np.log(max(p[b - 1], 1e-12))))
        exp_next = float(np.arange(1, chain.P.shape[0] + 1) @ p)
        maes.append(abs(exp_next - b))
    return ForecastResult(
        chain_label="",
        model_label="unconditional",
        mean_log_lik=float(np.mean(logs)) if logs else 0.0,
        mae=float(np.mean(maes)) if maes else 0.0,
        n_obs=len(logs),
    )


def evaluate_conditional(
    chain: ConditionalChain,
    state_now: pd.DataFrame,
    state_next: pd.DataFrame,
    covariates: dict[str, pd.DataFrame],
    kf: int = 10,
) -> tuple[ForecastResult, pd.DataFrame]:
    """Evaluate conditional chain; also return per-obs prediction records."""
    logs: list[float] = []
    maes: list[float] = []
    records: list[dict] = []
    k = chain.k
    feat_names = chain.feature_names

    # cache transition matrices by covariate signature
    cache: dict[tuple, np.ndarray] = {}

    for date in state_now.index:
        row_a = state_now.loc[date]
        row_b = state_next.loc[date]
        cov_rows = {n: covariates[n].loc[date] for n in feat_names}
        for tk in row_a.index:
            va = row_a.get(tk)
            vb = row_b.get(tk)
            if pd.isna(va) or pd.isna(vb):
                continue
            phi = np.zeros(len(feat_names))
            missing = False
            key_parts = []
            for j, name in enumerate(feat_names):
                v = cov_rows[name].get(tk)
                if pd.isna(v):
                    missing = True
                    break
                bucket = int(v)
                phi[j] = centered_decile_score(bucket, kf)
                key_parts.append(bucket)
            if missing:
                continue
            key = tuple(key_parts)
            P = cache.get(key)
            if P is None:
                P = chain.transition_matrix(phi)
                cache[key] = P
            a = int(va)
            b = int(vb)
            p = P[a - 1]
            logs.append(float(np.log(max(p[b - 1], 1e-12))))
            exp_next = float(np.arange(1, k + 1) @ p)
            maes.append(abs(exp_next - b))
            records.append(
                {"date": date, "ticker": tk, "a": a, "b": b, "exp_next": exp_next}
            )
    fr = ForecastResult(
        chain_label="",
        model_label="conditional",
        mean_log_lik=float(np.mean(logs)) if logs else 0.0,
        mae=float(np.mean(maes)) if maes else 0.0,
        n_obs=len(logs),
    )
    return fr, pd.DataFrame(records)


def walk_forward_forecast(
    state_now: pd.DataFrame,
    state_next: pd.DataFrame,
    covariates_by_set: dict[str, dict[str, pd.DataFrame]],
    train_end: str,
    test_start: str,
    test_end: str,
    k: int,
    kf: int,
    epsilon: float = 0.1,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fit models on data up to train_end, evaluate over [test_start,test_end].

    Returns
    -------
    results_df : rows per (predictor_set, model) with mean log-lik and MAE
    preds : dict of per-set prediction record DataFrames
    """
    train_mask = state_now.index <= pd.Timestamp(train_end)
    test_mask = (state_now.index >= pd.Timestamp(test_start)) & (
        state_now.index <= pd.Timestamp(test_end)
    )

    tr_now, tr_next = state_now.loc[train_mask], state_next.loc[train_mask]
    te_now, te_next = state_now.loc[test_mask], state_next.loc[test_mask]

    # Unconditional
    uncond = fit_unconditional(tr_now, tr_next, k=k)
    uncond_res = evaluate_unconditional(uncond, te_now, te_next)
    rows: list[dict] = []
    rows.append(
        {
            "predictor_set": "unconditional",
            "model": "unconditional",
            "mean_log_lik": uncond_res.mean_log_lik,
            "mae": uncond_res.mae,
            "n_obs": uncond_res.n_obs,
        }
    )

    preds: dict[str, pd.DataFrame] = {}
    for set_name, covs in covariates_by_set.items():
        # Reindex covariates to the state observation index, then slice
        covs_aligned = {n: df.reindex(state_now.index) for n, df in covs.items()}
        cov_train = {n: df.loc[train_mask] for n, df in covs_aligned.items()}
        chain = fit_conditional(
            tr_now, tr_next, cov_train, k=k, kf=kf, epsilon=epsilon
        )
        cov_test = {n: df.loc[test_mask] for n, df in covs_aligned.items()}
        cond_res, records = evaluate_conditional(
            chain, te_now, te_next, cov_test, kf=kf
        )
        rows.append(
            {
                "predictor_set": set_name,
                "model": "conditional",
                "mean_log_lik": cond_res.mean_log_lik,
                "mae": cond_res.mae,
                "n_obs": cond_res.n_obs,
            }
        )
        preds[set_name] = records

    df = pd.DataFrame(rows)
    # gains vs. unconditional
    baseline = df.loc[df.predictor_set == "unconditional", "mean_log_lik"].iloc[0]
    df["oos_ll_gain"] = df["mean_log_lik"] - baseline
    return df, preds

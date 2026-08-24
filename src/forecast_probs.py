"""Utility to produce pi_R and pi_V forecast probabilities per stock-month.

pi_i(t) = P(next-month decile in {1, 2} | current state, covariates).
For return chain decile 1 = best return; for volatility chain decile 1 = calmest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .chain_estimator import ConditionalChain, centered_decile_score


def top2_probability(
    chain: ConditionalChain,
    state_now: pd.DataFrame,
    covariates: dict[str, pd.DataFrame],
    kf: int = 10,
) -> pd.DataFrame:
    """Compute P(next state in {1,2}) for each stock-month.

    Returns a DataFrame aligned to state_now with the same shape.
    """
    feat_names = chain.feature_names
    out = pd.DataFrame(index=state_now.index, columns=state_now.columns, dtype=float)
    cache: dict[tuple, np.ndarray] = {}
    for date in state_now.index:
        row_a = state_now.loc[date]
        cov_rows = {n: covariates[n].loc[date] for n in feat_names}
        for tk in row_a.index:
            va = row_a.get(tk)
            if pd.isna(va):
                continue
            missing = False
            phi = np.zeros(len(feat_names))
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
            p_row = P[int(va) - 1]
            out.loc[date, tk] = float(p_row[:2].sum())
    return out

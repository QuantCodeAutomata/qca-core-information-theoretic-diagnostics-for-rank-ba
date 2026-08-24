"""Experiment 8: Regime-aware fixed-threshold chains and TE leader studies."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from exp.common import RESULTS_DIR, load_universe_and_features
from src.entropy import LAPLACE, pooled_transition_counts, entropy_production, transfer_entropy
from src.te_network import pairwise_te_matrix, hits_scores


K = 10


def _fixed_thresholds(x: pd.DataFrame, train_end: pd.Timestamp) -> np.ndarray:
    """Compute nine decile cutpoints from pooled training-sample excess values."""
    vals = x.loc[x.index <= train_end].stack().dropna().to_numpy()
    return np.quantile(vals, np.linspace(0.1, 0.9, 9))


def _apply_thresholds(x: pd.DataFrame, cuts: np.ndarray) -> pd.DataFrame:
    arr = x.to_numpy()
    out = np.digitize(arr, cuts) + 1  # buckets 1..10
    return pd.DataFrame(out, index=x.index, columns=x.columns, dtype=float).where(x.notna())


def main():
    prices, rets, fp, market = load_universe_and_features()
    train_end = pd.Timestamp("2022-12-31")

    # Monthly frequency
    monthly_r = (1 + rets).resample("ME").prod() - 1
    monthly_market = (1 + market).resample("ME").prod() - 1
    excess_r = monthly_r.sub(monthly_market, axis=0)
    cuts_r = _fixed_thresholds(excess_r, train_end)
    st_r = _apply_thresholds(excess_r, cuts_r)

    daily_v = rets.rolling(21).std()
    monthly_v = daily_v.resample("ME").last()
    market_v = market.rolling(21).std().resample("ME").last()
    excess_v = monthly_v.sub(market_v, axis=0)
    cuts_v = _fixed_thresholds(excess_v.dropna(how="all"), train_end)
    st_v = _apply_thresholds(excess_v, cuts_v)

    # Regime-aware pooled sigma
    out = {}
    for lab, s in [("regime_R", st_r), ("regime_V", st_v)]:
        now = s.iloc[:-1]
        nxt = s.shift(-1).iloc[:-1]
        cnt = pooled_transition_counts(now, nxt, K)
        sigma = entropy_production(cnt, alpha=LAPLACE)
        # boundary term
        marg_now = cnt.sum(axis=1) + LAPLACE
        marg_next = cnt.sum(axis=0) + LAPLACE
        marg_now /= marg_now.sum()
        marg_next /= marg_next.sum()
        boundary = 0.0
        for a in range(K):
            for b in range(K):
                p_ab = (cnt[a, b] + LAPLACE) / (cnt.sum() + LAPLACE * K * K)
                boundary += p_ab * np.log(marg_now[a] / marg_next[b])
        out[lab] = {"sigma_transition": float(sigma),
                    "sigma_boundary": float(boundary),
                    "sigma_total": float(sigma + boundary)}

    # TE leader study: TE between stock states and market state chain
    market_state = pd.qcut(monthly_market.rank(method="first"), 10, labels=False) + 1
    # Broadcast market state across universe as covariate
    market_state_df = pd.DataFrame(
        np.repeat(market_state.values[:, None], st_r.shape[1], axis=1),
        index=market_state.index, columns=st_r.columns, dtype=float
    ).reindex(st_r.index)

    te_rows = []
    for horizon_label, lag in [("21d_equiv", 1), ("30d_equiv", 1)]:  # monthly = ~21td
        for lab, s in [("R", st_r), ("V", st_v)]:
            now = s.iloc[:-lag]
            nxt = s.shift(-lag).iloc[:-lag]
            mkt_now = market_state_df.reindex(now.index)
            mkt_nxt = market_state_df.shift(-lag).reindex(now.index)
            te_xa = transfer_entropy(now, nxt, mkt_now, k=K, kf=K)
            # TE(A -> X): market is the target; stock state is the covariate
            te_ax = transfer_entropy(mkt_now, mkt_nxt, now, k=K, kf=K)
            te_rows.append(dict(chain=lab, horizon=horizon_label,
                                 te_market_to_stock=te_xa,
                                 te_stock_to_market=te_ax,
                                 net_te=te_xa - te_ax))
    te_df = pd.DataFrame(te_rows)
    te_df.to_csv(RESULTS_DIR / "exp8_te_leader.csv", index=False)

    # Full pairwise TE (subset for speed) and HITS
    stock_subset = st_r.columns[:60]
    A, tks = pairwise_te_matrix(st_r[stock_subset], K)
    hubs, auths = hits_scores(A)
    hub_df = pd.DataFrame({"ticker": tks, "hub": hubs, "authority": auths})
    hub_df = hub_df.sort_values("hub", ascending=False)
    hub_df.to_csv(RESULTS_DIR / "exp8_hub_scores.csv", index=False)

    with open(RESULTS_DIR / "exp8_regime_entropy.json", "w") as f:
        json.dump(out, f, indent=2)
    return out, te_df, hub_df


if __name__ == "__main__":
    out, te, hub = main()
    print(out)
    print(te)

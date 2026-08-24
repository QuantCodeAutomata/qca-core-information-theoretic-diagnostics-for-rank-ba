"""Experiment 6: Independent-source replay (uses Massive with a different universe)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from exp.common import RESULTS_DIR
from src.data_loader import SECONDARY_UNIVERSE, download_daily_prices, daily_returns
from src.features import build_features
from src.portfolio import compute_scores_from_forecasts, backtest_combined, _annualized_metrics
from src.chain_estimator import fit_conditional
from src.forecast_probs import top2_probability
from src.entropy import bucketize_covariate

K = 10
KF = 10


def stationary_block_bootstrap(x: np.ndarray, block_len: float, n_boot: int = 2000,
                                rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(x)
    p = 1.0 / block_len
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i0 = rng.integers(0, n)
        idx[0] = i0
        for t in range(1, n):
            if rng.random() < p:
                idx[t] = rng.integers(0, n)
            else:
                idx[t] = (idx[t - 1] + 1) % n
        sample = x[idx]
        out[b] = sample.mean() * 252  # annualized
    return out


def main():
    tag = "yahoo_repl"
    prices = download_daily_prices(SECONDARY_UNIVERSE, start="2018-01-01", end="2024-12-31", tag=tag)
    rets = daily_returns(prices)
    fp = build_features(prices)
    market = rets.mean(axis=1)

    sR = fp.state_R.astype(float)
    sV = fp.state_V.astype(float)
    covs = {n: fp.covariates[n].astype(float) for n in ["size", "beta", "mom", "illiq", "rv63",
                                                           "eig_load", "dist_cent"]}
    train_end = pd.Timestamp("2021-12-31")
    train_R_now = sR.loc[sR.index <= train_end].iloc[:-1]
    train_R_next = sR.loc[sR.index <= train_end].shift(-1).iloc[:-1]
    train_V_now = sV.loc[sV.index <= train_end].iloc[:-1]
    train_V_next = sV.loc[sV.index <= train_end].shift(-1).iloc[:-1]

    chain_R = fit_conditional(train_R_now, train_R_next,
                               {n: c.reindex(train_R_now.index) for n, c in covs.items()},
                               k=K, kf=KF, epsilon=0.1)
    chain_V = fit_conditional(train_V_now, train_V_next,
                               {n: c.reindex(train_V_now.index) for n, c in covs.items()},
                               k=K, kf=KF, epsilon=0.1)
    pi_R = top2_probability(chain_R, sR.iloc[:-1], covs, kf=KF)
    pi_V = top2_probability(chain_V, sV.iloc[:-1], covs, kf=KF)

    S_LS, S_LO, _ = compute_scores_from_forecasts(pi_R, pi_V, market)
    S_LS = S_LS.dropna(how="all")
    S_LO = S_LO.dropna(how="all")
    res = backtest_combined(S_LS, S_LO, rets, market)

    # Bootstrap excess returns vs equal-weight
    ew = rets.mean(axis=1).reindex(res.daily_returns.index).fillna(0.0)
    excess = (res.daily_returns - ew).values
    boot_ann_excess = stationary_block_bootstrap(excess, block_len=21.0, n_boot=1000)
    p_val = float((boot_ann_excess <= 0).mean())
    ci = [float(np.quantile(boot_ann_excess, 0.025)),
          float(np.quantile(boot_ann_excess, 0.975))]

    perf = {
        "yahoo_replay_2018_2024": _annualized_metrics(res.daily_returns),
        "equal_weight_benchmark": _annualized_metrics(ew),
        "bootstrap_one_sided_p_value_excess_positive": p_val,
        "bootstrap_95ci_annualized_excess": ci,
        "mean_ann_excess": float(boot_ann_excess.mean()),
    }
    with open(RESULTS_DIR / "exp6_yahoo_performance.json", "w") as f:
        json.dump(perf, f, indent=2)
    res.daily_returns.to_csv(RESULTS_DIR / "exp6_yahoo_daily.csv")
    return perf


if __name__ == "__main__":
    print(main())

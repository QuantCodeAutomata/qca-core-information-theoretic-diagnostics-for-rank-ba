"""Experiment 3: Main tradable portfolio replication (combined book)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from exp.common import (
    RESULTS_DIR,
    load_universe_and_features,
    build_state_pairs,
    market_covariate_names,
    m_matrix_covariate_names,
)
from src.chain_estimator import fit_conditional
from src.forecast_probs import top2_probability
from src.portfolio import (
    compute_scores_from_forecasts,
    backtest_combined,
    _annualized_metrics,
)

K = 10
KF = 10


def build_forecast_scores(fp, train_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit R and V chains and generate pi_R, pi_V per stock-month."""
    pairs = build_state_pairs(fp)
    sR_now, sR_next = pairs["R"]
    sV_now, sV_next = pairs["V"]

    train_mask = sR_now.index <= pd.Timestamp(train_end)
    cov_names = market_covariate_names() + m_matrix_covariate_names()
    covs_R = {
        n: fp.covariates[n].astype(float).reindex(sR_now.index)
        for n in cov_names
    }
    covs_V = {
        n: fp.covariates[n].astype(float).reindex(sV_now.index)
        for n in cov_names
    }

    chain_R = fit_conditional(
        sR_now.loc[train_mask],
        sR_next.loc[train_mask],
        {n: df.loc[train_mask] for n, df in covs_R.items()},
        k=K,
        kf=KF,
        epsilon=0.1,
    )
    chain_V = fit_conditional(
        sV_now.loc[train_mask],
        sV_next.loc[train_mask],
        {n: df.loc[train_mask] for n, df in covs_V.items()},
        k=K,
        kf=KF,
        epsilon=0.1,
    )

    pi_R = top2_probability(chain_R, sR_now, covs_R, kf=KF)
    pi_V = top2_probability(chain_V, sV_now, covs_V, kf=KF)
    return pi_R, pi_V


def main():
    prices, rets, fp, market = load_universe_and_features()
    pi_R, pi_V = build_forecast_scores(fp, train_end="2020-12-31")

    S_LS, S_LO, lam_series = compute_scores_from_forecasts(pi_R, pi_V, market)
    # Restrict to dates present in daily returns
    S_LS = S_LS.dropna(how="all")
    S_LO = S_LO.dropna(how="all")

    result = backtest_combined(
        S_LS,
        S_LO,
        rets,
        market,
        k_book=15,
        tol=0.08,
        tc_bps=5.0,
    )
    result_10bps = backtest_combined(
        S_LS, S_LO, rets, market, k_book=15, tol=0.08, tc_bps=10.0
    )

    # Benchmarks: equal-weight and (approx) cap-weight = equal weight (no cap data)
    eq_daily = rets.mean(axis=1)
    ew_perf = _annualized_metrics(eq_daily)

    perf = {
        "combined_5bps": result.performance,
        "combined_10bps": result_10bps.performance,
        "equal_weight_market": ew_perf,
    }
    with open(RESULTS_DIR / "exp3_performance.json", "w") as f:
        json.dump(perf, f, indent=2)
    result.daily_returns.to_csv(RESULTS_DIR / "exp3_daily_returns.csv")
    result.holdings.to_csv(RESULTS_DIR / "exp3_holdings.csv", index=False)
    result.turnover_series.to_csv(RESULTS_DIR / "exp3_turnover.csv")
    return result


if __name__ == "__main__":
    res = main()
    print(res.performance)

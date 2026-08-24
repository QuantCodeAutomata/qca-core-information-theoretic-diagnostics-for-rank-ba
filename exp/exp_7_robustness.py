"""Experiment 7: Robustness — no-trade band, TC stress, cross-sectional bootstrap."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from exp.common import RESULTS_DIR, load_universe_and_features
from exp.exp_3_portfolio import build_forecast_scores
from src.portfolio import (
    compute_scores_from_forecasts,
    backtest_combined,
    _annualized_metrics,
)
from exp.exp_6_yahoo import stationary_block_bootstrap


def main():
    prices, rets, fp, market = load_universe_and_features()
    pi_R, pi_V = build_forecast_scores(fp, train_end="2020-12-31")
    S_LS, S_LO, _ = compute_scores_from_forecasts(pi_R, pi_V, market)
    S_LS = S_LS.dropna(how="all")
    S_LO = S_LO.dropna(how="all")

    grid_rows = []
    for tol in [0.05, 0.08, 0.12, 0.18, 0.25]:
        for tc in [5.0, 10.0]:
            res = backtest_combined(S_LS, S_LO, rets, market, tol=tol, tc_bps=tc)
            grid_rows.append(dict(tol=tol, tc_bps=tc, **res.performance))
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(RESULTS_DIR / "exp7_robustness_grid.csv", index=False)

    # Book size sensitivity
    book_rows = []
    for k_book in [10, 15, 20]:
        res = backtest_combined(S_LS, S_LO, rets, market, k_book=k_book)
        book_rows.append(dict(k_book=k_book, **res.performance))
    book_df = pd.DataFrame(book_rows)
    book_df.to_csv(RESULTS_DIR / "exp7_book_size.csv", index=False)

    # Bootstrap
    base = backtest_combined(S_LS, S_LO, rets, market)
    ew = rets.mean(axis=1).reindex(base.daily_returns.index).fillna(0.0)
    excess = (base.daily_returns - ew).values
    boot = stationary_block_bootstrap(excess, block_len=21.0, n_boot=2000)
    boot_summary = {
        "mean_ann_excess": float(boot.mean()),
        "one_sided_p_excess_positive": float((boot <= 0).mean()),
        "ci_95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
    }
    with open(RESULTS_DIR / "exp7_bootstrap.json", "w") as f:
        json.dump(boot_summary, f, indent=2)

    # Cross-sectional bootstrap (small n_draws for speed)
    rng = np.random.default_rng(7)
    tickers = list(rets.columns)
    xs_rows = []
    for d in range(5):
        subset = rng.choice(tickers, size=int(len(tickers) * 0.7), replace=False)
        sub_rets = rets[subset]
        # Reduce scores to subset
        S_LS_s = S_LS[[c for c in S_LS.columns if c in subset]]
        S_LO_s = S_LO[[c for c in S_LO.columns if c in subset]]
        r = backtest_combined(S_LS_s, S_LO_s, sub_rets, sub_rets.mean(axis=1))
        xs_rows.append(dict(draw=d, **r.performance))
    xs_df = pd.DataFrame(xs_rows)
    xs_df.to_csv(RESULTS_DIR / "exp7_cross_sectional_bootstrap.csv", index=False)

    return {"grid": grid, "book": book_df, "bootstrap": boot_summary, "xs": xs_df}


if __name__ == "__main__":
    out = main()
    print(out["grid"])

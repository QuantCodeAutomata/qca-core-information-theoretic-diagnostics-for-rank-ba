"""Experiment 9: Fundamentals ablation (proxy).

Because we lack real Compustat/point-in-time fundamentals, we approximate
fundamentals with slow-moving price-derived proxies that mimic accounting
signals: (i) 12-month trailing log-drawdown as a leverage/quality proxy,
(ii) trailing 252d realized vol as a risk proxy, (iii) trailing 3y momentum
as a value/quality proxy. This preserves the methodology (adding an
alternative predictor set and measuring incremental predictive gain) while
being reproducible from the primary price panel. The limitation is
explicitly documented in the RESULTS.md.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from exp.common import RESULTS_DIR, load_universe_and_features, build_state_pairs
from src.forecaster import walk_forward_forecast
from src.entropy import bucketize_covariate


K = 10
KF = 10


def build_fundamental_proxies(rets: pd.DataFrame, month_ends: pd.DatetimeIndex) -> dict:
    """Raw slow-moving proxies, bucketed to deciles at each month-end."""
    cum = (1 + rets).cumprod()
    dd = cum / cum.cummax() - 1
    trail_dd = dd.rolling(252).min().reindex(month_ends)
    long_mom = (cum / cum.shift(756) - 1).reindex(month_ends)
    long_rv = rets.rolling(252).std().reindex(month_ends)
    return {
        "dd12m": bucketize_covariate(trail_dd, KF).astype(float),
        "mom3y": bucketize_covariate(long_mom, KF).astype(float),
        "rv252": bucketize_covariate(long_rv, KF).astype(float),
    }


def main():
    prices, rets, fp, market = load_universe_and_features()
    pairs = build_state_pairs(fp)

    fund = build_fundamental_proxies(rets, fp.state_R.index)
    market_covs = {n: fp.covariates[n].astype(float) for n in ["size", "beta", "mom", "illiq"]}
    predictor_sets = {
        "market_only": market_covs,
        "fundamentals_only": fund,
        "market_plus_fund": {**market_covs, **fund},
    }

    all_results = []
    for lab, (now, nxt) in pairs.items():
        df, _ = walk_forward_forecast(
            now, nxt, predictor_sets,
            train_end="2020-12-31", test_start="2021-01-01", test_end="2024-12-31",
            k=K, kf=KF, epsilon=0.1,
        )
        df["chain"] = lab
        all_results.append(df)
    df = pd.concat(all_results, ignore_index=True)
    df.to_csv(RESULTS_DIR / "exp9_fundamentals_forecast.csv", index=False)

    summary = {}
    for lab in ["R", "V"]:
        sub = df[df.chain == lab]
        for ps in predictor_sets.keys():
            row = sub[sub.predictor_set == ps]
            if not row.empty:
                summary[f"{lab}_{ps}_gain"] = float(row.oos_ll_gain.iloc[0])
    with open(RESULTS_DIR / "exp9_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    print(main())

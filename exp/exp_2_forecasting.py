"""Experiment 2: Out-of-sample forecasting."""
from __future__ import annotations

import json

import pandas as pd

from exp.common import (
    RESULTS_DIR,
    load_universe_and_features,
    build_state_pairs,
    market_covariate_names,
    m_matrix_covariate_names,
)
from src.forecaster import walk_forward_forecast

K = 10
KF = 10


def main():
    prices, rets, fp, market = load_universe_and_features()
    pairs = build_state_pairs(fp)

    train_end = "2020-12-31"
    test_start = "2021-01-01"
    test_end = "2024-12-31"

    all_results = []
    for lab, (now, nxt) in pairs.items():
        covs_market = {n: fp.covariates[n].astype(float) for n in market_covariate_names()}
        covs_market_m = {
            n: fp.covariates[n].astype(float)
            for n in market_covariate_names() + m_matrix_covariate_names()
        }
        covs_rv63 = {"rv63": fp.covariates["rv63"].astype(float)}

        predictor_sets = {
            "market": covs_market,
            "market_plus_M": covs_market_m,
            "rv63_only": covs_rv63,
        }

        df, _ = walk_forward_forecast(
            now,
            nxt,
            predictor_sets,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            k=K,
            kf=KF,
            epsilon=0.1,  # paper's default ridge
        )
        df["chain"] = lab
        all_results.append(df)

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(RESULTS_DIR / "exp2_oos_loglik.csv", index=False)

    summary = {}
    for lab in ["R", "V"]:
        sub = results[results.chain == lab]
        summary[f"{lab}_baseline_loglik"] = float(
            sub[sub.predictor_set == "unconditional"].mean_log_lik.iloc[0]
        )
        for ps in ["market", "market_plus_M", "rv63_only"]:
            row = sub[sub.predictor_set == ps]
            if not row.empty:
                summary[f"{lab}_{ps}_gain"] = float(row.oos_ll_gain.iloc[0])
                summary[f"{lab}_{ps}_mae"] = float(row.mae.iloc[0])
    with open(RESULTS_DIR / "exp2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return results


if __name__ == "__main__":
    df = main()
    print(df.to_string(index=False))

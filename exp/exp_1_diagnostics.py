"""Experiment 1: Core information-theoretic diagnostics."""
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
from src.entropy import (
    LAPLACE,
    diagnose_chain,
    pooled_transition_counts,
    entropy_production,
)

K = 10
KF = 10


def main() -> pd.DataFrame:
    prices, rets, fp, market = load_universe_and_features()
    pairs = build_state_pairs(fp)

    # Pooled sigma per chain
    pooled_rows = []
    for lab, (now, nxt) in pairs.items():
        cnt = pooled_transition_counts(now, nxt, K)
        sigma = entropy_production(cnt, alpha=LAPLACE)
        pooled_rows.append(dict(chain=lab, sigma=sigma, n_transitions=int(cnt.sum())))
    pooled_df = pd.DataFrame(pooled_rows)

    # Diagnostic tables per chain
    tables: list[pd.DataFrame] = []
    for lab, (now, nxt) in pairs.items():
        # Align covariates to now's index
        covs = {
            name: fp.covariates[name].astype(float).reindex(now.index)
            for name in market_covariate_names() + m_matrix_covariate_names()
        }
        tbl = diagnose_chain(now, nxt, covs, k=K, kf=KF, chain_label=lab)
        tables.append(tbl)
    diag = pd.concat(tables, ignore_index=True)

    # Save
    pooled_df.to_csv(RESULTS_DIR / "exp1_pooled_sigma.csv", index=False)
    diag.to_csv(RESULTS_DIR / "exp1_diagnostic_table.csv", index=False)

    # Summary: verify sigma_cond >= sigma numerically
    diag["cond_ge_pooled"] = diag["sigma_cond"] >= diag["sigma"] - 1e-9
    summary = {
        "pooled_sigma_R": float(pooled_df[pooled_df.chain == "R"].sigma.iloc[0]),
        "pooled_sigma_V": float(pooled_df[pooled_df.chain == "V"].sigma.iloc[0]),
        "avg_delta_sigma_R": float(diag[diag.chain == "R"].delta_sigma.mean()),
        "avg_delta_sigma_V": float(diag[diag.chain == "V"].delta_sigma.mean()),
        "avg_net_te_R": float(diag[diag.chain == "R"].net_te.mean()),
        "avg_net_te_V": float(diag[diag.chain == "V"].net_te.mean()),
        "cond_ge_pooled_share": float(diag["cond_ge_pooled"].mean()),
    }
    with open(RESULTS_DIR / "exp1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return diag


if __name__ == "__main__":
    df = main()
    print(df.to_string(index=False))

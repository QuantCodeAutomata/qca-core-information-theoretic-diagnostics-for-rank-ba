"""Experiment 4: Residual-distance diversified long-only sleeve and benchmarks."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from exp.common import RESULTS_DIR, load_universe_and_features, build_state_pairs
from exp.exp_3_portfolio import build_forecast_scores
from src.portfolio import (
    residual_distance_matrix,
    diversified_selection,
    _hysteresis_select_top,
    _annualized_metrics,
    regime_and_blend,
)

K = 10


def _run_lo_sleeve(
    S_LO: pd.DataFrame,
    rets: pd.DataFrame,
    market: pd.Series,
    gamma_div: float,
    n_book: int = 30,
    tol: float = 0.08,
) -> tuple[pd.Series, list[dict], list[float]]:
    prev: set[str] = set()
    current_w = pd.Series(0.0, index=rets.columns)
    daily_dates: list[pd.Timestamp] = []
    daily_r: list[float] = []
    holdings: list[dict] = []
    div_scores: list[float] = []
    rebalance_dates = list(S_LO.index)
    for i, rb in enumerate(rebalance_dates):
        scores = S_LO.loc[rb]

        # Residual distance using trailing year of returns up to rb
        window = rets.loc[rets.index <= rb].tail(252)
        M_res, tk_idx = residual_distance_matrix(window, market)
        selected = diversified_selection(scores, M_res, tk_idx, prev, n_book, tol, gamma_div)

        # measure realized diversification of selection
        if selected and tk_idx and M_res.size > 0:
            idx_map = {tk: j for j, tk in enumerate(tk_idx)}
            sel_ids = [idx_map[t] for t in selected if t in idx_map]
            if len(sel_ids) > 1:
                sub = M_res[np.ix_(sel_ids, sel_ids)]
                iu = np.triu_indices(len(sub), k=1)
                div_scores.append(float(sub[iu].mean()))

        target = pd.Series(0.0, index=rets.columns)
        for tk in selected:
            target[tk] = 1.0 / n_book
        current_w = target.copy()
        prev = set(selected)
        for tk in selected:
            holdings.append(dict(date=rb, ticker=tk, weight=1.0 / n_book))

        end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else rets.index[-1]
        w = rets.loc[(rets.index > rb) & (rets.index <= end)]
        for d, row in w.iterrows():
            r = row.reindex(current_w.index).fillna(0.0)
            daily_r.append(float((current_w * r).sum()))
            daily_dates.append(d)
            new_w = current_w * (1 + r)
            g = new_w.abs().sum()
            if g > 0:
                new_w *= float(current_w.abs().sum() / g)
            current_w = new_w
    return pd.Series(daily_r, index=pd.DatetimeIndex(daily_dates)), holdings, div_scores


def main():
    prices, rets, fp, market = load_universe_and_features()
    pi_R, pi_V = build_forecast_scores(fp, train_end="2020-12-31")
    # long-only score
    S_LO = pd.DataFrame(index=pi_R.index, columns=pi_R.columns, dtype=float)
    for d in pi_R.index:
        lam, _ = regime_and_blend(market, d)
        S_LO.loc[d] = (1 - lam) * pi_R.loc[d] + lam * pi_V.loc[d]

    S_LO = S_LO.dropna(how="all")

    daily_current, _, div_current = _run_lo_sleeve(S_LO, rets, market, gamma_div=0.0)
    daily_div, _, div_div = _run_lo_sleeve(S_LO, rets, market, gamma_div=0.5)

    perf = {
        "current_sleeve": _annualized_metrics(daily_current)
        | {"mean_diversification": float(np.mean(div_current)) if div_current else 0.0},
        "diversified_sleeve_gamma05": _annualized_metrics(daily_div)
        | {"mean_diversification": float(np.mean(div_div)) if div_div else 0.0},
    }
    with open(RESULTS_DIR / "exp4_diversified_performance.json", "w") as f:
        json.dump(perf, f, indent=2)
    daily_current.to_csv(RESULTS_DIR / "exp4_current_lo_daily.csv")
    daily_div.to_csv(RESULTS_DIR / "exp4_diversified_lo_daily.csv")
    return perf


if __name__ == "__main__":
    print(main())

"""Experiment 5: TE-hub insurance overlay."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from exp.common import RESULTS_DIR, load_universe_and_features, build_state_pairs
from exp.exp_3_portfolio import build_forecast_scores
from src.portfolio import compute_scores_from_forecasts, backtest_combined, _annualized_metrics
from src.te_network import pairwise_te_matrix, hits_scores


K = 10


def build_overlay(states_R: pd.DataFrame, rets: pd.DataFrame, rb_dates: pd.DatetimeIndex,
                  top_frac: float = 0.10) -> pd.DataFrame:
    """Monthly market-neutral overlay: long high-hub, short low-hub."""
    tickers = list(rets.columns)
    overlay = pd.DataFrame(0.0, index=rb_dates, columns=tickers)
    for rb in rb_dates:
        # Use trailing 24 months of monthly states
        recent = states_R.loc[states_R.index <= rb].tail(24)
        if recent.shape[0] < 6:
            continue
        A, tks = pairwise_te_matrix(recent, K)
        h, _ = hits_scores(A)
        ser = pd.Series(h, index=tks).sort_values(ascending=False)
        n_side = max(1, int(len(ser) * top_frac))
        longs = ser.head(n_side).index
        shorts = ser.tail(n_side).index
        w = pd.Series(0.0, index=tickers)
        for tk in longs:
            w[tk] = 0.5 / n_side
        for tk in shorts:
            w[tk] = -0.5 / n_side
        overlay.loc[rb] = w
    return overlay


def apply_overlay(
    base_daily: pd.Series,
    overlay_weights: pd.DataFrame,
    rets: pd.DataFrame,
    alpha: float = 0.25,
) -> pd.Series:
    """Combine base daily returns with overlay daily returns."""
    rb = list(overlay_weights.index)
    dates = base_daily.index
    port = base_daily.copy()
    ov_daily = pd.Series(0.0, index=dates)
    current_w = pd.Series(0.0, index=rets.columns)
    for i, r_ in enumerate(rb):
        target = overlay_weights.loc[r_]
        current_w = target.copy()
        end = rb[i + 1] if i + 1 < len(rb) else dates[-1]
        w = rets.loc[(rets.index > r_) & (rets.index <= end)]
        for d, row in w.iterrows():
            if d not in dates:
                continue
            r_row = row.reindex(current_w.index).fillna(0.0)
            ov_daily.loc[d] = float((current_w * r_row).sum())
            new_w = current_w * (1 + r_row)
            g = new_w.abs().sum()
            if g > 0:
                new_w *= float(current_w.abs().sum() / g)
            current_w = new_w
    return (1 - alpha) * port + alpha * ov_daily


def main():
    prices, rets, fp, market = load_universe_and_features()
    pi_R, pi_V = build_forecast_scores(fp, train_end="2020-12-31")
    S_LS, S_LO, _ = compute_scores_from_forecasts(pi_R, pi_V, market)
    S_LS = S_LS.dropna(how="all")
    S_LO = S_LO.dropna(how="all")
    base = backtest_combined(S_LS, S_LO, rets, market)
    states_R = fp.state_R.astype(float)
    overlay = build_overlay(states_R, rets, S_LO.index, top_frac=0.10)

    combined = apply_overlay(base.daily_returns, overlay, rets, alpha=0.25)

    perf = {
        "base": _annualized_metrics(base.daily_returns),
        "with_overlay": _annualized_metrics(combined),
    }
    with open(RESULTS_DIR / "exp5_overlay_performance.json", "w") as f:
        json.dump(perf, f, indent=2)
    combined.to_csv(RESULTS_DIR / "exp5_overlay_daily.csv")
    return perf


if __name__ == "__main__":
    print(main())

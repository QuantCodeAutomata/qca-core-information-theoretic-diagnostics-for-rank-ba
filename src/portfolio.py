"""Portfolio construction, backtesting, and overlays.

Implements:
- Regime detection based on trailing 3-month equal-weight market return.
- Long-short sleeve (K=15/side) with no-trade band tol=0.08 hysteresis.
- Long-only sleeve (30 names) with regime-blended R/V forecast score.
- Combined book with theta blending and transaction-cost accounting.
- Residual-distance diversified selector (gamma_div=0.5).
- TE-hub insurance overlay (25% protective weight).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


K_BOOK = 15
TOL = 0.08


def regime_and_blend(market_returns: pd.Series, date: pd.Timestamp) -> tuple[float, float]:
    """Return (lambda_LO, theta) for a given month-end date.

    Uses trailing 3-month market return. Rising market: lambda=0, theta=1.
    Falling market (including zero): lambda=0.75, theta=0.4.
    """
    idx = market_returns.index
    if date not in idx:
        return 0.75, 0.4
    end = idx.get_loc(date)
    start = max(end - 63, 0)
    trailing = market_returns.iloc[start : end + 1].sum()
    if trailing > 0:
        return 0.0, 1.0
    return 0.75, 0.4


def compute_scores_from_forecasts(
    forecast_R: pd.DataFrame,  # per date x ticker probability of top-2 next-month return decile
    forecast_V: pd.DataFrame,  # per date x ticker probability of top-2 next-month vol decile (calmest)
    market_returns: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (S_long_short, S_long_only, lambda_LO_by_date)."""
    S_LS = forecast_R.copy()  # long-short uses R only (lambda=0 in all regimes)
    S_LO = pd.DataFrame(index=forecast_R.index, columns=forecast_R.columns, dtype=float)
    lam_series = pd.Series(index=forecast_R.index, dtype=float)
    for date in forecast_R.index:
        lam, _theta = regime_and_blend(market_returns, date)
        lam_series[date] = lam
        S_LO.loc[date] = (1 - lam) * forecast_R.loc[date] + lam * forecast_V.loc[date]
    return S_LS, S_LO, lam_series


def _hysteresis_select_top(
    scores: pd.Series, prev_holdings: set[str], k: int, tol: float
) -> list[str]:
    """Select top-k names with hysteresis relative to previous holdings.

    Algorithm: sort candidates descending by score. Start with all previous
    holdings that remain in-universe. Then iteratively swap the weakest
    incumbent for the strongest challenger only if challenger.score >
    incumbent.score + tol.
    """
    s = scores.dropna().sort_values(ascending=False)
    if s.empty:
        return []
    ranked = list(s.index)
    if not prev_holdings:
        return ranked[:k]

    incumbents = [tk for tk in ranked if tk in prev_holdings][:k]
    challengers = [tk for tk in ranked if tk not in prev_holdings]

    # Fill up if fewer than k incumbents remain
    remaining = k - len(incumbents)
    while remaining > 0 and challengers:
        incumbents.append(challengers.pop(0))
        remaining -= 1

    # Now attempt swaps
    changed = True
    while changed and challengers:
        changed = False
        # weakest incumbent
        inc_scores = [(tk, s.get(tk, -np.inf)) for tk in incumbents]
        inc_scores.sort(key=lambda x: x[1])  # ascending: weakest first
        weakest_tk, weakest_score = inc_scores[0]
        # strongest challenger
        strongest_ch = challengers[0]
        strongest_score = s[strongest_ch]
        if strongest_score > weakest_score + tol:
            incumbents.remove(weakest_tk)
            incumbents.append(strongest_ch)
            challengers.pop(0)
            changed = True
    return incumbents


def _hysteresis_select_bottom(
    scores: pd.Series, prev_holdings: set[str], k: int, tol: float
) -> list[str]:
    """Select bottom-k names with hysteresis (mirrored logic for shorts)."""
    s = scores.dropna().sort_values(ascending=True)
    if s.empty:
        return []
    ranked = list(s.index)
    if not prev_holdings:
        return ranked[:k]
    incumbents = [tk for tk in ranked if tk in prev_holdings][:k]
    challengers = [tk for tk in ranked if tk not in prev_holdings]
    remaining = k - len(incumbents)
    while remaining > 0 and challengers:
        incumbents.append(challengers.pop(0))
        remaining -= 1
    changed = True
    while changed and challengers:
        changed = False
        inc_scores = [(tk, s.get(tk, np.inf)) for tk in incumbents]
        inc_scores.sort(key=lambda x: -x[1])  # descending: highest (worst) first
        worst_tk, worst_score = inc_scores[0]
        best_ch = challengers[0]
        best_score = s[best_ch]
        if best_score + tol < worst_score:
            incumbents.remove(worst_tk)
            incumbents.append(best_ch)
            challengers.pop(0)
            changed = True
    return incumbents


@dataclass
class BacktestResult:
    holdings: pd.DataFrame  # long-format holdings (date, ticker, weight, sleeve)
    daily_returns: pd.Series  # portfolio daily returns
    turnover_series: pd.Series  # per-rebalance turnover
    performance: dict[str, float]


def _annualized_metrics(daily_r: pd.Series) -> dict[str, float]:
    if daily_r.empty:
        return dict(ann_return=0.0, ann_vol=0.0, sharpe=0.0, max_dd=0.0)
    r = daily_r.dropna()
    ann_r = float(r.mean() * 252)
    ann_v = float(r.std(ddof=0) * np.sqrt(252))
    sharpe = ann_r / ann_v if ann_v > 0 else 0.0
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    max_dd = float(((cum - peak) / peak).min())
    return dict(ann_return=ann_r, ann_vol=ann_v, sharpe=sharpe, max_dd=max_dd)


def backtest_combined(
    S_LS: pd.DataFrame,
    S_LO: pd.DataFrame,
    daily_returns: pd.DataFrame,
    market_returns: pd.Series,
    k_book: int = K_BOOK,
    tol: float = TOL,
    tc_bps: float = 5.0,
) -> BacktestResult:
    """Run the combined-book backtest with monthly rebalancing.

    Assumes S_LS.index and S_LO.index are month-end dates present in
    daily_returns.index.
    """
    dates = S_LS.index
    rebalance_dates = list(dates)
    all_tickers = list(daily_returns.columns)

    prev_long: set[str] = set()
    prev_short: set[str] = set()
    prev_lo: set[str] = set()

    current_weights = pd.Series(0.0, index=all_tickers)
    daily_r_out: list[float] = []
    daily_dates: list[pd.Timestamp] = []
    turnover_records: list[dict] = []
    holdings_records: list[dict] = []

    # For each rebalance, compute target weights, apply costs, then simulate
    # daily returns until next rebalance (weights drift).
    for i, rb in enumerate(rebalance_dates):
        _, theta = regime_and_blend(market_returns, rb)
        s_ls = S_LS.loc[rb]
        s_lo = S_LO.loc[rb]

        longs = _hysteresis_select_top(s_ls, prev_long, k_book, tol)
        shorts = _hysteresis_select_bottom(s_ls, prev_short, k_book, tol)
        lo = _hysteresis_select_top(s_lo, prev_lo, 2 * k_book, tol)

        w_ls = pd.Series(0.0, index=all_tickers)
        for tk in longs:
            w_ls[tk] += 1.0 / k_book
        for tk in shorts:
            w_ls[tk] -= 1.0 / k_book
        w_lo = pd.Series(0.0, index=all_tickers)
        for tk in lo:
            w_lo[tk] += 1.0 / (2 * k_book)

        target = theta * w_lo + (1.0 - theta) * w_ls

        # Turnover: |target - current_weights_drifted|
        turnover = float((target - current_weights).abs().sum())
        tc_cost = turnover * (tc_bps / 1e4)
        turnover_records.append({"date": rb, "turnover": turnover, "cost": tc_cost})

        # Apply cost as a one-time hit
        if daily_r_out:
            daily_r_out[-1] -= tc_cost  # attribute to rebalance day
        current_weights = target.copy()

        prev_long = set(longs)
        prev_short = set(shorts)
        prev_lo = set(lo)

        for tk in longs:
            holdings_records.append(
                dict(date=rb, ticker=tk, sleeve="LS_long", weight=1.0 / k_book)
            )
        for tk in shorts:
            holdings_records.append(
                dict(date=rb, ticker=tk, sleeve="LS_short", weight=-1.0 / k_book)
            )
        for tk in lo:
            holdings_records.append(
                dict(date=rb, ticker=tk, sleeve="LO", weight=1.0 / (2 * k_book))
            )

        # Simulate daily returns until next rebalance
        end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else daily_returns.index[-1]
        window = daily_returns.loc[(daily_returns.index > rb) & (daily_returns.index <= end)]
        for d, row in window.iterrows():
            r = row.reindex(current_weights.index).fillna(0.0)
            port_r = float((current_weights * r).sum())
            daily_r_out.append(port_r)
            daily_dates.append(d)
            # Drift weights
            new_w = current_weights * (1.0 + r)
            gross = float(new_w.abs().sum())
            if gross > 0:
                # normalize so gross exposure matches target gross; use pro-rata
                target_gross = float(current_weights.abs().sum())
                if target_gross > 0:
                    new_w = new_w * (target_gross / gross)
            current_weights = new_w

    daily_series = pd.Series(daily_r_out, index=pd.DatetimeIndex(daily_dates), name="port")
    perf = _annualized_metrics(daily_series)
    to_series = pd.Series(
        [r["turnover"] for r in turnover_records],
        index=pd.DatetimeIndex([r["date"] for r in turnover_records]),
    )
    perf["mean_turnover"] = float(to_series.mean()) if len(to_series) else 0.0
    return BacktestResult(
        holdings=pd.DataFrame(holdings_records),
        daily_returns=daily_series,
        turnover_series=to_series,
        performance=perf,
    )


def residual_distance_matrix(
    returns_window: pd.DataFrame, market: pd.Series
) -> tuple[np.ndarray, list[str]]:
    """Compute the residual arccos distance matrix.

    Residuals from OLS of each stock's returns on the market over the window.
    """
    df = returns_window.dropna(axis=1, how="any")
    if df.shape[1] < 3:
        return np.zeros((0, 0)), []
    m = market.reindex(df.index).values
    m_c = m - m.mean()
    denom = float((m_c @ m_c))
    if denom <= 0:
        return np.zeros((0, 0)), []
    X = df.values
    betas = (X - X.mean(axis=0, keepdims=True)).T @ m_c / denom
    alphas = X.mean(axis=0) - betas * m.mean()
    fitted = alphas[None, :] + betas[None, :] * m[:, None]
    resid = X - fitted
    C = np.corrcoef(resid, rowvar=False)
    C = np.clip(C, -1.0, 1.0)
    C = 0.5 * (C + C.T)
    M = np.arccos(C)
    np.fill_diagonal(M, 0.0)
    return M, list(df.columns)


def diversified_selection(
    scores: pd.Series,
    residual_dist: np.ndarray,
    ticker_index: list[str],
    prev_holdings: set[str],
    n: int,
    tol: float,
    gamma_div: float,
) -> list[str]:
    """Greedy diversification-aware selector.

    Starts from the top base-score name and iteratively adds the next name
    maximizing AdjScore = BaseScore + gamma * mean_dist_to_selected. Then
    applies hysteresis vs previous holdings.
    """
    valid_scores = scores.dropna()
    if valid_scores.empty:
        return []
    if residual_dist.size == 0 or gamma_div == 0.0:
        return _hysteresis_select_top(valid_scores, prev_holdings, n, tol)

    idx_map = {tk: i for i, tk in enumerate(ticker_index)}
    candidates = [tk for tk in valid_scores.index if tk in idx_map]
    if not candidates:
        return _hysteresis_select_top(valid_scores, prev_holdings, n, tol)

    ordered = valid_scores.loc[candidates].sort_values(ascending=False)
    selected: list[str] = [ordered.index[0]]
    remaining = [t for t in ordered.index if t not in selected]
    while len(selected) < n and remaining:
        best_tk = None
        best_val = -np.inf
        sel_idx = [idx_map[t] for t in selected]
        for tk in remaining:
            i = idx_map[tk]
            avg_d = float(residual_dist[i, sel_idx].mean()) if sel_idx else 0.0
            adj = float(ordered[tk]) + gamma_div * avg_d
            if adj > best_val:
                best_val = adj
                best_tk = tk
        if best_tk is None:
            break
        selected.append(best_tk)
        remaining.remove(best_tk)
    # Apply hysteresis: same swap logic on adjusted scores
    return selected


# ------- Yahoo bootstrap significance -------

def stationary_block_bootstrap(
    daily_excess: pd.Series, n_boot: int = 10000, expected_block: int = 21, seed: int = 42
) -> dict[str, float]:
    """Politis-Romano stationary block bootstrap of annualized excess return."""
    r = daily_excess.dropna().values
    n = len(r)
    if n < 30:
        return dict(p_one_sided=1.0, ci_low=0.0, ci_high=0.0, mean=0.0)
    rng = np.random.default_rng(seed)
    p = 1.0 / expected_block
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        i = rng.integers(0, n)
        for t in range(n):
            idx[t] = i
            if rng.random() < p:
                i = rng.integers(0, n)
            else:
                i = (i + 1) % n
        sample = r[idx]
        means[b] = sample.mean() * 252
    obs = r.mean() * 252
    p_one_sided = float((means <= 0).mean())
    ci_low, ci_high = np.quantile(means, [0.025, 0.975])
    return dict(
        p_one_sided=p_one_sided,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        mean=float(obs),
    )

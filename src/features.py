"""Feature and state construction for rank-chain diagnostics.

Builds month-end panels of:
- Return-rank state (trailing 1m or 6m mean return, cross-sectional deciles)
- Volatility-rank state (trailing 21d realized vol, cross-sectional deciles)
- Market-data covariates: log size proxy, 252d beta, 12-1 momentum,
  Amihud illiquidity, 63/126/252d realized vol
- Distance-matrix predictors: leading eigenvector loading magnitude, distance
  centrality (row-mean arccos distance), TE lead-lag score (built later)

The paper's ranking convention: rank 1 = best return (highest) and lowest vol.
Bucket via c = ceil(K * rank / N), with K=10 by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


K_BUCKET = 10


def month_end_index(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the last trading date of each month present in the index."""
    s = pd.Series(1, index=dates)
    idx = s.groupby(s.index.to_period("M")).apply(lambda x: x.index[-1]).values
    return pd.DatetimeIndex(idx)


def cross_sectional_decile(
    values: pd.Series, k: int = K_BUCKET, ascending: bool = False
) -> pd.Series:
    """Assign cross-sectional decile buckets 1..k.

    ascending=False => rank 1 is largest value (used for returns).
    ascending=True  => rank 1 is smallest value (used for volatility).
    Uses method='first' for deterministic tie-breaking.
    """
    x = values.dropna()
    if x.empty:
        return pd.Series(dtype=float, index=values.index)
    r = x.rank(method="first", ascending=ascending)
    n = len(x)
    buckets = np.ceil(k * r / n).astype(int).clip(1, k)
    return buckets.reindex(values.index)


def trailing_mean_return(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Trailing mean of daily returns over `window` trading days."""
    return returns.rolling(window, min_periods=window).mean()


def trailing_realized_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Trailing sample stdev of daily returns over `window` trading days."""
    return returns.rolling(window, min_periods=window).std()


def rolling_beta(
    returns: pd.DataFrame, market: pd.Series, window: int = 252
) -> pd.DataFrame:
    """Rolling 252-day beta of each stock vs. equal-weight market.

    Computed as cov(r_i, r_m) / var(r_m) over the rolling window.
    """
    aligned = returns.copy()
    m = market.reindex(aligned.index)
    var_m = m.rolling(window, min_periods=window).var()
    cov = (
        aligned.rolling(window, min_periods=window)
        .cov(m)
    )
    if isinstance(cov, pd.DataFrame):
        beta = cov.div(var_m, axis=0)
    else:
        beta = cov / var_m
    return beta


def momentum_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    """12-1 momentum: return from t-252 to t-21, skipping the last month."""
    lagged = prices.shift(21)
    base = prices.shift(252)
    return lagged / base - 1.0


def amihud_illiquidity(returns: pd.DataFrame, volume_dollar: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Amihud illiquidity: mean(|r| / dollar_volume) over trailing window."""
    daily = returns.abs() / volume_dollar.replace(0, np.nan)
    return daily.rolling(window, min_periods=window // 2).mean()


def arccos_distance_matrix(corr: np.ndarray) -> np.ndarray:
    """Convert a correlation matrix to arccos/geodesic distance matrix.

    Applies clipping to [-1, 1] and symmetry enforcement.
    """
    C = 0.5 * (corr + corr.T)
    C = np.clip(C, -1.0, 1.0)
    M = np.arccos(C)
    np.fill_diagonal(M, 0.0)
    return M


def leading_eigenvector_loading(corr: np.ndarray) -> np.ndarray:
    """Absolute loading of the leading eigenvector of the correlation matrix."""
    C = 0.5 * (corr + corr.T)
    # eigh returns eigenvalues in ascending order
    w, v = np.linalg.eigh(C)
    return np.abs(v[:, -1])


def distance_centrality(dist_matrix: np.ndarray) -> np.ndarray:
    """Row-mean of the arccos distance matrix (excluding diagonal)."""
    n = dist_matrix.shape[0]
    if n <= 1:
        return np.zeros(n)
    row_sum = dist_matrix.sum(axis=1)
    return row_sum / (n - 1)


@dataclass
class FeaturePanel:
    """Container for month-end feature snapshots.

    Attributes are DataFrames indexed by month-end date with tickers as columns.
    """

    state_R: pd.DataFrame  # return-rank decile state (rank 1 = best return)
    state_V: pd.DataFrame  # volatility-rank decile state (rank 1 = lowest vol)
    covariates: dict[str, pd.DataFrame]  # bucketed covariates
    raw_covariates: dict[str, pd.DataFrame]  # raw (unbucketed) covariates


def build_features(
    prices: pd.DataFrame,
    return_window: int = 126,  # trailing 6 months for return state
    vol_window: int = 21,
    long_vol_windows: tuple[int, ...] = (63, 126, 252),
    beta_window: int = 252,
    dist_window: int = 252,
) -> FeaturePanel:
    """Build the month-end state and covariate panel.

    Parameters
    ----------
    prices : wide daily price DataFrame
    return_window : trailing daily-return average window for return state
        (126 for 6-month per paper section 5; use 21 for the 1m variant)
    vol_window : realized-volatility window for volatility state (21d)
    """
    returns = prices.pct_change(fill_method=None).iloc[1:]
    me = month_end_index(returns.index)

    # Equal-weight market proxy
    market = returns.mean(axis=1)

    # Raw feature panels aligned to daily index
    r6 = trailing_mean_return(returns, return_window)
    rv21 = trailing_realized_vol(returns, vol_window)
    rvs = {w: trailing_realized_vol(returns, w) for w in long_vol_windows}
    mom = momentum_12_1(prices)
    beta = rolling_beta(returns, market, window=beta_window)

    # Proxy for size: cumulative price level (informative in the synthetic
    # panel; in a real CRSP pipeline this would be point-in-time market cap).
    size_proxy = np.log(prices.clip(lower=1e-6))

    # Amihud illiquidity using absolute daily return as a normalized proxy for
    # 1/liquidity; in the absence of volume data, we use |r| / rolling_std as
    # a per-stock illiquidity proxy (deterministic, monotone with turbulence).
    illiq = returns.abs().rolling(63, min_periods=32).mean() / (
        rvs[63].replace(0.0, np.nan)
    )

    # Distance-matrix predictors on the month-end grid
    tickers = list(returns.columns)
    n = len(tickers)
    eig_load = pd.DataFrame(index=me, columns=tickers, dtype=float)
    dist_cent = pd.DataFrame(index=me, columns=tickers, dtype=float)
    for d in me:
        window_end = d
        window_start_pos = returns.index.get_loc(window_end) - dist_window + 1
        if window_start_pos < 0:
            continue
        sl = returns.iloc[window_start_pos : returns.index.get_loc(window_end) + 1]
        # Drop columns without full history over the window
        valid = sl.dropna(axis=1, how="any")
        if valid.shape[1] < 5 or valid.shape[0] < dist_window // 2:
            continue
        C = valid.corr().values
        M = arccos_distance_matrix(C)
        load = leading_eigenvector_loading(C)
        cent = distance_centrality(M)
        eig_load.loc[d, valid.columns] = load
        dist_cent.loc[d, valid.columns] = cent

    # Snapshot everything to month-end
    def snap(df: pd.DataFrame) -> pd.DataFrame:
        return df.reindex(me)

    raw = {
        "ret_state_signal": snap(r6),
        "rv21": snap(rv21),
        "rv63": snap(rvs[63]),
        "rv126": snap(rvs[126]),
        "rv252": snap(rvs[252]),
        "beta": snap(beta),
        "mom": snap(mom),
        "size": snap(size_proxy),
        "illiq": snap(illiq),
        "eig_load": eig_load.astype(float),
        "dist_cent": dist_cent.astype(float),
    }

    # States: rank 1 = best return; rank 1 = lowest vol
    state_R = raw["ret_state_signal"].apply(
        lambda row: cross_sectional_decile(row, ascending=False), axis=1
    )
    state_V = raw["rv21"].apply(
        lambda row: cross_sectional_decile(row, ascending=True), axis=1
    )

    # Bucket every covariate cross-sectionally (rank 1 = largest value)
    covariates = {
        name: df.apply(lambda row: cross_sectional_decile(row, ascending=False), axis=1)
        for name, df in raw.items()
        if name != "ret_state_signal"
    }

    return FeaturePanel(
        state_R=state_R.astype("Int64"),
        state_V=state_V.astype("Int64"),
        covariates={k: v.astype("Int64") for k, v in covariates.items()},
        raw_covariates=raw,
    )

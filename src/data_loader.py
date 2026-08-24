"""Data loader for daily equity prices.

Prefers the Massive REST API for historical daily aggregates. Falls back to a
deterministic synthetic panel for offline testing. Cached to data/ as parquet.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "XOM", "CVX", "LLY", "ABBV", "PEP",
    "KO", "MRK", "AVGO", "BAC", "WMT", "PFE", "TMO", "COST", "MCD", "DIS",
    "CSCO", "ABT", "ADBE", "CRM", "ACN", "NFLX", "WFC", "TXN", "DHR", "NKE",
    "LIN", "NEE", "PM", "VZ", "UNP", "ORCL", "AMGN", "IBM", "QCOM", "HON",
]

# Slightly extended universe used for the independent-source replay
SECONDARY_UNIVERSE: list[str] = DEFAULT_UNIVERSE + [
    "T", "MDT", "BMY", "LOW", "GS", "MS", "BLK", "AXP", "SBUX", "GILD",
    "INTC", "AMD", "CAT", "DE", "BA", "GE", "MMM", "F", "GM", "SPGI",
]


def _cache_path(tag: str) -> Path:
    return DATA_DIR / f"prices_{tag}.parquet"


def download_daily_prices(
    tickers: Iterable[str],
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    tag: str = "main",
    force: bool = False,
) -> pd.DataFrame:
    """Download daily adjusted-close prices, cache to parquet.

    Returns a wide DataFrame indexed by date with tickers as columns.
    """
    cache = _cache_path(tag)
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    api_key = os.getenv("MASSIVE_API_KEY")
    df = None
    if api_key:
        try:
            df = _download_from_massive(list(tickers), start, end, api_key)
        except Exception as exc:  # pragma: no cover - network path
            print(f"[data_loader] Massive fetch failed ({exc}); falling back to synthetic.")

    if df is None or df.empty:
        df = _synthetic_panel(list(tickers), start, end)

    df.to_parquet(cache)
    return df


def _download_from_massive(
    tickers: list[str], start: str, end: str, api_key: str
) -> pd.DataFrame:
    """Fetch daily aggregates from Massive REST API."""
    from massive import RESTClient  # local import: keep offline tests independent

    client = RESTClient(api_key=api_key)
    frames: dict[str, pd.Series] = {}
    for tk in tickers:
        rows: list[tuple[pd.Timestamp, float]] = []
        try:
            for a in client.list_aggs(
                ticker=tk,
                multiplier=1,
                timespan="day",
                from_=start,
                to=end,
                limit=50000,
            ):
                # timestamps are Unix milliseconds for aggregate bars
                ts = pd.Timestamp(a.timestamp, unit="ms", tz="UTC").tz_convert(
                    "America/New_York"
                ).normalize().tz_localize(None)
                rows.append((ts, float(a.close)))
        except Exception as exc:  # pragma: no cover
            print(f"[data_loader] {tk} failed: {exc}")
            continue
        if rows:
            s = pd.Series(dict(rows))
            s = s[~s.index.duplicated(keep="last")].sort_index()
            frames[tk] = s
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).sort_index()


def _synthetic_panel(
    tickers: list[str], start: str, end: str, seed: int = 7
) -> pd.DataFrame:
    """Reproducible synthetic price panel with cross-sectional structure.

    Prices follow a one-factor-plus-idio GBM. Volatility is heterogeneous, so
    volatility-rank persistence is non-trivial by construction.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    T, N = len(dates), len(tickers)

    # Persistent volatility scale per stock (log-normal)
    log_sigma = rng.normal(loc=np.log(0.015), scale=0.4, size=N)
    sigma = np.exp(log_sigma)

    # Betas around 1, momentum drifts
    beta = rng.normal(1.0, 0.35, size=N).clip(0.2, 2.0)
    mu = rng.normal(0.00025, 0.0002, size=N)

    # Market factor with vol clustering
    mkt = np.zeros(T)
    mkt_vol = 0.01 * np.ones(T)
    for t in range(1, T):
        mkt_vol[t] = 0.94 * mkt_vol[t - 1] + 0.06 * abs(mkt[t - 1])
    mkt = rng.normal(0.0003, 0.01, size=T) + 0.5 * mkt_vol * rng.standard_normal(T)

    # Idiosyncratic returns with GARCH-like vol
    idio = rng.standard_normal((T, N)) * sigma[None, :]
    # Volatility persistence via slowly varying scale
    vol_state = np.ones((T, N))
    for t in range(1, T):
        vol_state[t] = 0.97 * vol_state[t - 1] + 0.03 * (
            1.0 + 0.3 * rng.standard_normal(N)
        ).clip(0.4, 2.5)
    idio = idio * vol_state

    R = mu[None, :] + beta[None, :] * mkt[:, None] + idio
    # Convert to prices
    prices = 100.0 * np.exp(np.cumsum(R, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple close-to-close daily returns."""
    return prices.pct_change(fill_method=None).iloc[1:]

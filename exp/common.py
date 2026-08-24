"""Shared setup: universe, dates, feature builders reused across experiments."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import DEFAULT_UNIVERSE, download_daily_prices, daily_returns
from src.features import build_features, FeaturePanel


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_universe_and_features(
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    tag: str = "main",
    return_window: int = 126,
) -> tuple[pd.DataFrame, pd.DataFrame, FeaturePanel, pd.Series]:
    """Load prices, daily returns, feature panel, and equal-weight market."""
    prices = download_daily_prices(DEFAULT_UNIVERSE, start=start, end=end, tag=tag)
    rets = daily_returns(prices)
    fp = build_features(prices, return_window=return_window)
    market = rets.mean(axis=1)
    return prices, rets, fp, market


def build_state_pairs(fp: FeaturePanel) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Return {'R': (state_now, state_next), 'V': (...)}."""
    sR = fp.state_R.astype("Float64").astype(float)
    sV = fp.state_V.astype("Float64").astype(float)
    return {
        "R": (sR.iloc[:-1], sR.shift(-1).iloc[:-1]),
        "V": (sV.iloc[:-1], sV.shift(-1).iloc[:-1]),
    }


def market_covariate_names() -> list[str]:
    return ["size", "beta", "mom", "illiq", "rv63"]


def m_matrix_covariate_names() -> list[str]:
    return ["eig_load", "dist_cent"]

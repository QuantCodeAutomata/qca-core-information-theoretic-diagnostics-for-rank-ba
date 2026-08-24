# QCA — Core information-theoretic diagnostics for rank-based market-state matrices

Reproducible Python implementation of the nine experiments described in the
research blueprint on:

1. Pooled / conditional entropy production and transfer entropy for
   cross-sectional return- and volatility-rank chains.
2. Out-of-sample forecasting with conditional continuous-time Markov chains
   `q_ab(x) = exp(eta_ab + gamma_ab . phi(x))` and matrix-exponential monthly
   transitions.
3. Regime-aware tradable portfolio (long-short + long-only sleeves, hysteresis
   no-trade band, transaction costs).
4. Residual-distance diversified long-only sleeve.
5. Transfer-entropy hub-based insurance overlay.
6. Independent-source (Yahoo-style) replay + stationary block bootstrap.
7. Robustness grid (tolerance, cost, book size, bootstrap, subsampling).
8. Regime-aware fixed-threshold chains and TE leader/follower studies.
9. Fundamentals ablation.

## Repository layout

```
├── data/                # Cached price parquets (gitignored — regenerated from Massive)
├── src/                 # Core implementation
│   ├── data_loader.py         # Massive REST client, universe, market proxy
│   ├── features.py            # Rank states, arccos distance, market/M-matrix predictors
│   ├── entropy.py             # Pooled / conditional sigma, transfer entropy (Laplace-smoothed)
│   ├── chain_estimator.py     # Unconditional + conditional (ridge, L-BFGS-B, expm) chains
│   ├── portfolio.py           # Regime, hysteresis, backtester, residual-distance selection
│   └── te_network.py          # Pairwise TE + HITS
├── exp/                 # Experiment drivers, one per experiment
│   ├── exp_1_diagnostics.py
│   ├── exp_2_forecasting.py
│   ├── exp_3_portfolio.py
│   ├── exp_4_diversified.py
│   ├── exp_5_overlay.py
│   ├── exp_6_yahoo.py
│   ├── exp_7_robustness.py
│   ├── exp_8_regime.py
│   └── exp_9_fundamentals.py
├── tests/               # pytest suite (21 tests, all passing)
├── results/             # CSV / JSON / markdown deliverables (committed)
│   └── RESULTS.md
├── README.md
└── .gitignore
```

## Setup

```bash
pip install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn arch \
            hmmlearn ruptures pyportfolioopt copulae skfolio vectorbt pycop \
            pytest massive
export MASSIVE_API_KEY=...    # or MASSIVE_TOKEN
```

## Reproduce

```bash
# 1. Fetch and cache prices from Massive
python -c "from src.data_loader import cache_universe_prices; cache_universe_prices()"

# 2. Run the nine experiments (each writes CSV/JSON to results/)
for i in 1 2 3 4 5 6 7 8 9; do
    python exp/exp_${i}_*.py
done

# 3. Tests
pytest tests/ -q
```

See `results/RESULTS.md` for the summarized tables and interpretation.

## Notes on scope

- Universe reduced to a tractable ~50-name S&P 500 subsample for the working
  reference implementation (paper: 331). Qualitative structure is preserved.
- Fundamentals in `exp_9_fundamentals.py` are price-derived synthetic proxies —
  full Compustat point-in-time data is not accessible in this environment.

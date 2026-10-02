# Stock Market Future Prediction

Research framework for leakage-aware next-period stock-return forecasting and simple backtesting.

The repository retains the original research notes and notebook, but the executable workflow now lives in the `stock_forecast` package.

## Research protocol

The framework predicts **next-period return**, rather than fitting directly to future prices:

```text
OHLCV history
    ↓
features available at time t
    ↓
target = Close[t+1] / Close[t] - 1
    ↓
chronological train/test split
    ↓
naive / Ridge / Random Forest / HistGradientBoosting
    ↓
forecast metrics + simple sign-strategy diagnostics
```

No random train/test split is used. Any preprocessing that learns parameters (for example StandardScaler in the Ridge pipeline) is fitted only on the training segment.

## Metrics

Forecasting:
- MAE
- RMSE
- R²
- directional accuracy

Research backtest diagnostics:
- cumulative strategy return
- annualized Sharpe ratio
- maximum drawdown
- turnover

The backtest includes a configurable transaction-cost assumption and is intentionally simple. It is for research comparison, **not investment advice or a production trading strategy**.

## Install

Core framework:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

To download market data with yfinance:

```bash
pip install -e ".[data,dev]"
```

## Use the existing AAPL data

The repository contains a historical AAPL sample in `Code/stock_data.csv`.

```bash
python -m stock_forecast.benchmark \
  --data Code/stock_data.csv \
  --output-dir artifacts
```

Outputs:

```text
artifacts/
  leaderboard.csv
  predictions.csv
  config.json
  ridge.joblib
  random_forest.joblib
  hist_gradient_boosting.joblib
```

## Download another ticker

```bash
python -m stock_forecast.download \
  --ticker MSFT \
  --start 2015-01-01 \
  --end 2026-01-01 \
  --output data/msft.csv
```

Then:

```bash
python -m stock_forecast.benchmark --data data/msft.csv
```

## Leakage controls

The older LSTM experiment scaled the complete dataset before splitting it. That lets test-period extrema influence training transformations. The new framework avoids this by:

1. constructing only backward-looking features;
2. using next-period return as the target;
3. splitting chronologically;
4. fitting trainable preprocessing and models only on the training segment.

## Historical material

The root Markdown files and `stockdata.ipynb` are retained as research notes/history. The old scratch scripts under `Code/` are removed except for the historical CSV dataset.

## Tests

```bash
pytest
```

GitHub Actions validates the feature pipeline and runs the benchmark against the committed sample dataset.

## Research caution

Financial time series are non-stationary and noisy. Good performance on one historical split is not evidence of persistent out-of-sample profitability. For serious studies, add rolling/walk-forward evaluation, multiple assets/regimes, robust cost/slippage assumptions, and statistical comparison against simple benchmarks.

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def forecast_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": float(
            np.mean(np.sign(y_true) == np.sign(y_pred))
        ),
    }


def strategy_metrics(
    actual_returns,
    predicted_returns,
    transaction_cost_bps: float = 5.0,
) -> dict[str, float]:
    """Simple sign strategy for research diagnostics, not trading advice."""
    actual = np.asarray(actual_returns, dtype=float)
    pred = np.asarray(predicted_returns, dtype=float)
    positions = np.sign(pred)

    turnover = np.abs(np.diff(np.r_[0.0, positions]))
    costs = turnover * transaction_cost_bps / 10_000.0
    strategy = positions * actual - costs

    equity = np.cumprod(1.0 + strategy)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0

    std = strategy.std(ddof=1)
    sharpe = (
        float(strategy.mean() / std * np.sqrt(252))
        if std > 0
        else 0.0
    )

    return {
        "cumulative_return": float(equity[-1] - 1.0),
        "annualized_sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "turnover": float(turnover.sum()),
    }

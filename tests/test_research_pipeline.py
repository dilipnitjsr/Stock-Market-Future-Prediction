import numpy as np
import pandas as pd

from stock_forecast.features import build_features, chronological_split
from stock_forecast.metrics import forecast_metrics


def sample_ohlcv(rows=180):
    idx = pd.date_range("2025-01-01", periods=rows, freq="B")
    rng = np.random.default_rng(42)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, rows))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.002, rows)),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(100_000, 500_000, rows),
        },
        index=idx,
    )


def test_target_is_next_period_return():
    frame = sample_ohlcv()
    features = build_features(frame)
    date = features.index[0]
    pos = frame.index.get_loc(date)
    expected = frame.iloc[pos + 1]["Close"] / frame.iloc[pos]["Close"] - 1
    assert abs(features.loc[date, "target_return"] - expected) < 1e-12


def test_chronological_split_preserves_time():
    featured = build_features(sample_ohlcv())
    train, test = chronological_split(featured)
    assert train.index.max() < test.index.min()


def test_metrics_directional_accuracy():
    metrics = forecast_metrics([0.01, -0.02], [0.005, -0.01])
    assert metrics["directional_accuracy"] == 1.0

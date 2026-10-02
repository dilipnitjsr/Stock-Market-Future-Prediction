from __future__ import annotations

import pandas as pd


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a clean single-ticker OHLCV frame indexed by date.

    Supports ordinary CSV files and the two-row column header emitted by
    recent yfinance downloads saved with DataFrame.to_csv().
    """
    df = frame.copy()

    if "Date" not in df.columns and df.index.name != "Date":
        # yfinance's multi-index CSV can arrive with metadata rows already read
        # as records; callers should use load_csv() for that layout.
        raise ValueError("Expected a Date column")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="raise")
        df = df.set_index("Date")

    wanted = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    df = df[wanted].apply(pd.to_numeric, errors="raise")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna()


def load_csv(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)

    # Detect the yfinance multi-index CSV format:
    # Price,Close,High,Low,Open,Volume
    # Ticker,AAPL,AAPL,...
    # Date,,,,,
    if len(raw) >= 2 and str(raw.iloc[0, 0]) not in {"Date", "date"}:
        first_column = raw.columns[0]
        if first_column in {"Price", "Ticker"}:
            data = pd.read_csv(path, skiprows=2)
            data = data.rename(columns={data.columns[0]: "Date"})
            return normalize_ohlcv(data)

    return normalize_ohlcv(raw)


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Create features at time t and next-period return target.

    Every predictor uses information available at or before t. The target is
    Close[t+1] / Close[t] - 1, so no future observation enters a feature.
    """
    df = ohlcv.copy()
    close = df["Close"]

    out = pd.DataFrame(index=df.index)
    out["ret_1"] = close.pct_change(1)
    out["ret_2"] = close.pct_change(2)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)

    for window in (5, 10, 20, 60):
        out[f"ma_ratio_{window}"] = close / close.rolling(window).mean() - 1
        out[f"volatility_{window}"] = close.pct_change().rolling(window).std()

    out["range_pct"] = (df["High"] - df["Low"]) / close
    out["open_gap"] = df["Open"] / close.shift(1) - 1
    out["volume_change"] = df["Volume"].pct_change()
    out["target_return"] = close.shift(-1) / close - 1

    return out.replace([float("inf"), float("-inf")], pd.NA).dropna()


def chronological_split(
    frame: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if len(frame) < 50:
        raise ValueError("At least 50 feature rows are required")

    split = max(1, min(int(len(frame) * (1 - test_fraction)), len(frame) - 1))
    return frame.iloc[:split].copy(), frame.iloc[split:].copy()

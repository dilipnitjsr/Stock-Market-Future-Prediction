from __future__ import annotations

import argparse
from pathlib import Path

import yfinance as yf


def main() -> None:
    parser = argparse.ArgumentParser(description="Download historical OHLCV data")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", default="data/stock.csv")
    args = parser.parse_args()

    data = yf.download(
        args.ticker,
        start=args.start,
        end=args.end,
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise SystemExit("No data returned")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output)
    print(f"Wrote {len(data)} rows to {output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import build_features, chronological_split, load_csv
from .metrics import forecast_metrics, strategy_metrics
from .models import make_model

MODELS = ["naive_zero", "ridge", "random_forest", "hist_gradient_boosting"]


def run_benchmark(
    data_path: str,
    output_dir: str = "artifacts",
    test_fraction: float = 0.2,
    transaction_cost_bps: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    ohlcv = load_csv(data_path)
    featured = build_features(ohlcv)
    train, test = chronological_split(featured, test_fraction=test_fraction)

    feature_cols = [c for c in featured.columns if c != "target_return"]
    X_train, y_train = train[feature_cols], train["target_return"]
    X_test, y_test = test[feature_cols], test["target_return"]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    pred_table = pd.DataFrame(index=test.index)
    pred_table["actual_return"] = y_test

    for name in MODELS:
        if name == "naive_zero":
            pred = np.zeros(len(test), dtype=float)
        else:
            model = make_model(name, seed=seed)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            joblib.dump(
                {"model": model, "feature_columns": feature_cols},
                output / f"{name}.joblib",
            )

        pred_table[name] = pred
        row = {"model": name}
        row.update(forecast_metrics(y_test, pred))
        row.update(
            {
                f"strategy_{k}": v
                for k, v in strategy_metrics(
                    y_test,
                    pred,
                    transaction_cost_bps=transaction_cost_bps,
                ).items()
            }
        )
        rows.append(row)

    leaderboard = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    pred_table.to_csv(output / "predictions.csv")

    config = {
        "data_path": data_path,
        "test_fraction": test_fraction,
        "transaction_cost_bps": transaction_cost_bps,
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_columns": feature_cols,
        "models": MODELS,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return leaderboard


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-aware stock-return forecasting benchmark"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run_benchmark(
        data_path=args.data,
        output_dir=args.output_dir,
        test_fraction=args.test_fraction,
        transaction_cost_bps=args.transaction_cost_bps,
        seed=args.seed,
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

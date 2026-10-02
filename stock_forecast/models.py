from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(name: str, seed: int = 42):
    key = name.lower()

    if key == "ridge":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )

    if key == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )

    if key == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        )

    raise ValueError(f"Unknown model: {name}")

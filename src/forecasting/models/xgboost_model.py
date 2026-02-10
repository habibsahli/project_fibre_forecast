"""XGBoost model helpers."""

from __future__ import annotations


def train_xgboost(X_train, y_train, params: dict | None = None):
    from xgboost import XGBRegressor

    params = params or {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": 0,
        "tree_method": "hist",
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    return model

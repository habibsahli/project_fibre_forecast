"""Exponential smoothing helpers."""

from __future__ import annotations


def train_exponential_smoothing(train_values, seasonal_period: int = 7):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    candidates = {
        "SES": ExponentialSmoothing(train_values, trend=None, seasonal=None),
        "Holt": ExponentialSmoothing(train_values, trend="add", seasonal=None),
        "Holt-Winters": ExponentialSmoothing(
            train_values, trend="add", seasonal="add", seasonal_periods=seasonal_period
        ),
    }

    best_name = None
    best_model = None

    for name, model in candidates.items():
        try:
            fitted = model.fit(optimized=True)
            best_name = name
            best_model = fitted
            break  # Return first successful fit
        except Exception:
            continue

    return best_name, best_model

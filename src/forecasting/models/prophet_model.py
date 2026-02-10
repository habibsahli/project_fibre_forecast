"""Prophet model helpers."""

from __future__ import annotations

import pandas as pd


def train_prophet(
    train_df: pd.DataFrame,
    holidays: pd.DataFrame | None = None,
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
    seasonality_mode: str = "additive",
):
    from prophet import Prophet

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        seasonality_mode=seasonality_mode,
        interval_width=0.95,
    )
    model.fit(train_df)
    return model


def predict_prophet(model, periods: int = 0, future_df: pd.DataFrame | None = None):
    if future_df is None:
        future_df = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future_df)
    return forecast

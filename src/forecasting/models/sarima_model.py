"""SARIMA model helpers."""

from __future__ import annotations


def train_sarima(train_values, seasonal_period: int = 30):
    """
    Train SARIMA model with optimized hyperparameters.
    
    Args:
        train_values: Time series training data
        seasonal_period: Seasonal period (default: 30 - monthly pattern found best)
        
    Returns:
        Fitted SARIMA model
        
    Note: Tuning results showed 30-day seasonality outperforms 7-day by 5.5%
    """
    from pmdarima import auto_arima

    model = auto_arima(
        train_values,
        seasonal=True,
        m=seasonal_period,
        stepwise=True,
        trace=False,
        max_p=5,
        max_q=5,
        max_d=2,
        max_P=2,
        max_Q=2,
        max_D=1,
        information_criterion="aic",
    )
    return model

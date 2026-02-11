"""
Statistical Generative Model for Fibre Subscription Time Series

A lightweight generative model using statistical methods and time series analysis.
Generates synthetic fibre subscription data that preserves temporal patterns.

Advantages:
- No heavy dependencies (works with existing scikit-learn/numpy)
- Fast training and generation
- Interpretable parameters
- Good for bootstrapping generative AI concepts
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class StatisticalTimeSeriesGenerator:
    """
    Statistical generative model for fibre subscription forecasting.

    Uses time series decomposition and statistical modeling to generate
    realistic synthetic data that preserves:
    - Trend patterns
    - Seasonal cycles (weekly/monthly)
    - Autocorrelation structure
    - Distribution properties
    """

    def __init__(self, seasonal_periods=[7, 30], noise_factor=0.1):
        """
        Initialize the statistical generator.

        Args:
            seasonal_periods: List of seasonal periods (e.g., [7, 30] for weekly/monthly)
            noise_factor: Amount of random noise to add (0.1 = 10% noise)
        """
        self.seasonal_periods = seasonal_periods
        self.noise_factor = noise_factor
        self.scaler = StandardScaler()

        # Learned parameters
        self.trend_model = None
        self.seasonal_components = {}
        self.residual_stats = {}
        self.autocorr_coeffs = {}

    def decompose_series(self, series):
        """Decompose time series into trend, seasonal, and residual components."""
        series = np.array(series)

        # Detrend using linear regression on time index
        time_idx = np.arange(len(series)).reshape(-1, 1)
        trend_model = LinearRegression()
        trend_model.fit(time_idx, series)
        trend = trend_model.predict(time_idx)

        # Remove trend
        detrended = series - trend

        # Extract seasonal components
        seasonal_components = {}
        residual = detrended.copy()

        for period in self.seasonal_periods:
            if len(series) >= period * 2:  # Need at least 2 full cycles
                # Calculate seasonal averages
                seasonal = np.zeros_like(detrended)
                for i in range(period):
                    indices = np.arange(i, len(detrended), period)
                    seasonal[indices] = np.mean(detrended[indices])
                seasonal_components[period] = seasonal
                residual = residual - seasonal

        return trend, seasonal_components, residual

    def learn_autocorrelation(self, residual, max_lag=7):
        """Learn autocorrelation structure from residuals."""
        autocorr = {}
        for lag in range(1, min(max_lag + 1, len(residual))):
            autocorr[lag] = np.corrcoef(residual[:-lag], residual[lag:])[0, 1]
        return autocorr

    def fit(self, df_daily, feature_col='nb_abonnements'):
        """
        Train the statistical generator on fibre subscription data.

        Args:
            df_daily: DataFrame with date and subscription count columns
            feature_col: Column name for subscription counts
        """
        print("🔬 Training Statistical Time Series Generator...")

        series = df_daily[feature_col].values
        series_scaled = self.scaler.fit_transform(series.reshape(-1, 1)).flatten()

        # Decompose series
        trend, seasonal_components, residual = self.decompose_series(series_scaled)

        # Learn autocorrelation
        autocorr = self.learn_autocorrelation(residual)

        # Store learned parameters
        self.trend_model = LinearRegression()
        time_idx = np.arange(len(series)).reshape(-1, 1)
        self.trend_model.fit(time_idx, series_scaled)

        self.seasonal_components = seasonal_components
        self.residual_stats = {
            'mean': np.mean(residual),
            'std': np.std(residual)
        }
        self.autocorr_coeffs = autocorr

        print("✅ Model trained successfully!")
        print(f"   - Trend: {'✅ Detected' if abs(trend[-1] - trend[0]) > 0.1 else '❌ Flat'}")
        print(f"   - Seasonal periods: {list(seasonal_components.keys())}")
        print(f"   - Residual std: {self.residual_stats['std']:.3f}")
        print(f"   - Autocorrelation lags: {len(autocorr)}")

    def generate_sequence(self, length, start_idx=0):
        """
        Generate a synthetic time series sequence.

        Args:
            length: Length of sequence to generate
            start_idx: Starting time index for trend continuation

        Returns:
            Synthetic sequence (scaled)
        """
        # Generate trend
        time_idx = np.arange(start_idx, start_idx + length).reshape(-1, 1)
        trend = self.trend_model.predict(time_idx)

        # Generate seasonal components
        seasonal = np.zeros(length)
        for period, component in self.seasonal_components.items():
            # Repeat seasonal pattern
            pattern = component[:period]
            n_repeats = int(np.ceil(length / period))
            full_pattern = np.tile(pattern, n_repeats)[:length]
            seasonal += full_pattern

        # Generate residuals with autocorrelation
        residuals = np.random.normal(
            self.residual_stats['mean'],
            self.residual_stats['std'],
            length
        )

        # Apply autocorrelation
        for lag, coeff in self.autocorr_coeffs.items():
            if lag < length:
                residuals[lag:] += coeff * residuals[:-lag]

        # Add noise
        noise = np.random.normal(0, self.noise_factor, length)

        # Combine components
        synthetic_scaled = trend + seasonal + residuals + noise

        return synthetic_scaled

    def generate(self, n_samples=1, sequence_length=None, start_from_end=True):
        """
        Generate synthetic fibre subscription sequences.

        Args:
            n_samples: Number of synthetic sequences
            sequence_length: Length of each sequence
            start_from_end: Start generation from end of training data

        Returns:
            Array of synthetic sequences (original scale)
        """
        if sequence_length is None:
            sequence_length = 30  # Default 30 days

        synthetic_sequences = []

        for _ in range(n_samples):
            if start_from_end:
                # Continue from end of training data
                start_idx = len(self.scaler.mean_)  # Approximate
            else:
                start_idx = np.random.randint(0, 100)  # Random start

            synthetic_scaled = self.generate_sequence(sequence_length, start_idx)
            synthetic_original = self.scaler.inverse_transform(
                synthetic_scaled.reshape(-1, 1)
            ).flatten()

            synthetic_sequences.append(synthetic_original)

        return np.array(synthetic_sequences)

    def generate_scenarios(self, n_scenarios=10, forecast_horizon=30):
        """
        Generate multiple forecast scenarios.

        Args:
            n_scenarios: Number of scenarios to generate
            forecast_horizon: Days to forecast

        Returns:
            DataFrame with scenario forecasts
        """
        scenarios = []

        for i in range(n_scenarios):
            synthetic = self.generate(n_samples=1, sequence_length=forecast_horizon)
            scenarios.append({
                'scenario': f'scenario_{i+1}',
                'forecast': synthetic[0]
            })

        return pd.DataFrame(scenarios)

    def evaluate_synthetic_quality(self, df_real, n_samples=100):
        """
        Evaluate quality of synthetic data by comparing statistics.

        Args:
            df_real: Real data DataFrame
            n_samples: Number of synthetic samples to generate

        Returns:
            Dictionary with quality metrics
        """
        real_data = df_real['nb_abonnements'].values

        # Generate synthetic data
        synthetic_data = self.generate(n_samples=n_samples, sequence_length=len(real_data))

        # Calculate statistics
        real_stats = {
            'mean': np.mean(real_data),
            'std': np.std(real_data),
            'min': np.min(real_data),
            'max': np.max(real_data),
            'trend': real_data[-1] - real_data[0]
        }

        synthetic_stats = {
            'mean': np.mean(synthetic_data),
            'std': np.std(synthetic_data),
            'min': np.min(synthetic_data),
            'max': np.max(synthetic_data),
            'trend': np.mean(synthetic_data[:, -1] - synthetic_data[:, 0])
        }

        # Calculate differences
        quality_metrics = {}
        for key in real_stats.keys():
            diff = abs(real_stats[key] - synthetic_stats[key])
            rel_diff = diff / abs(real_stats[key]) if real_stats[key] != 0 else diff
            quality_metrics[f'{key}_diff'] = diff
            quality_metrics[f'{key}_rel_diff'] = rel_diff

        quality_metrics.update({
            'real_stats': real_stats,
            'synthetic_stats': synthetic_stats
        })

        return quality_metrics


def generate_synthetic_scenarios(df_daily, n_scenarios=50, forecast_horizon=30):
    """
    Convenience function to generate synthetic scenarios.

    Args:
        df_daily: DataFrame with daily subscription data
        n_scenarios: Number of scenarios to generate
        forecast_horizon: Days to forecast

    Returns:
        DataFrame with synthetic scenarios
    """
    model = StatisticalTimeSeriesGenerator()
    model.fit(df_daily)
    scenarios = model.generate_scenarios(n_scenarios=n_scenarios, forecast_horizon=forecast_horizon)
    return scenarios


if __name__ == "__main__":
    print("Statistical Time Series Generator for Fibre Subscriptions")
    print("This model uses classical statistics instead of deep learning")
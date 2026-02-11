"""
Generative Models for Fibre Subscription Forecasting

This module contains generative AI models for:
- Synthetic data generation for model training augmentation
- Scenario generation for uncertainty modeling
- Time series imputation and anomaly detection
"""

from .timegan_model import TimeGANModel, generate_synthetic_scenarios as timegan_scenarios
from .statistical_generator import StatisticalTimeSeriesGenerator, generate_synthetic_scenarios as stat_scenarios

__all__ = [
    'TimeGANModel',
    'StatisticalTimeSeriesGenerator',
    'timegan_scenarios',
    'stat_scenarios'
]
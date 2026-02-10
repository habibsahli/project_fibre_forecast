#!/usr/bin/env python3
"""
Direct Comparison: SARIMA Old vs New Configuration
Shows side-by-side improvement from tuning
"""

import warnings
warnings.filterwarnings('ignore')

import time
import pandas as pd
import numpy as np
from pathlib import Path

from src.forecasting.data_loader import get_engine, load_daily_data
from src.forecasting.metrics import calculate_metrics

def train_sarima_fixed(train_values, order=(1, 1, 1), seasonal_order=(1, 1, 1, 30)):
    """Train SARIMA with fixed parameters"""
    try:
        from pmdarima import ARIMA
        model = ARIMA(order=order, seasonal_order=seasonal_order, suppress_warnings=True)
        model.fit(train_values)
        return model
    except Exception as e:
        print(f"  Error: {e}")
        return None

# Load data
print("\n" + "="*80)
print("📊 SARIMA CONFIGURATION COMPARISON")
print("="*80 + "\n")

engine = get_engine()
df_daily = load_daily_data(engine)
print(f"✅ Loaded {len(df_daily)} days of data")

# Train/test split
split_idx = int(len(df_daily) * 0.8)
train_values = df_daily.iloc[:split_idx]['nb_abonnements'].values
test_values = df_daily.iloc[split_idx:]['nb_abonnements'].values

print(f"📊 Train: {len(train_values)} days | Test: {len(test_values)} days\n")

# Configuration 1: Original (Weekly)
print("1️⃣  ORIGINAL CONFIGURATION (Weekly)")
print("   Order: (1, 1, 1) × Seasonal: (1, 1, 1, 7)")
start = time.time()
model1 = train_sarima_fixed(train_values, (1, 1, 1), (1, 1, 1, 7))
time1 = time.time() - start
if model1:
    y_pred1 = model1.predict(n_periods=len(test_values))
    metrics1 = calculate_metrics(test_values, y_pred1)
    print(f"   ✅ MAPE: {metrics1['MAPE']:.2f}% | MAE: {metrics1['MAE']:.2f} | RMSE: {metrics1['RMSE']:.2f}")
    print(f"   ⏱️  Training: {time1:.2f}s\n")

# Configuration 2: Optimized (Monthly)
print("2️⃣  OPTIMIZED CONFIGURATION (Monthly)")
print("   Order: (1, 1, 1) × Seasonal: (1, 1, 1, 30)")
start = time.time()
model2 = train_sarima_fixed(train_values, (1, 1, 1), (1, 1, 1, 30))
time2 = time.time() - start
if model2:
    y_pred2 = model2.predict(n_periods=len(test_values))
    metrics2 = calculate_metrics(test_values, y_pred2)
    print(f"   ✅ MAPE: {metrics2['MAPE']:.2f}% | MAE: {metrics2['MAE']:.2f} | RMSE: {metrics2['RMSE']:.2f}")
    print(f"   ⏱️  Training: {time2:.2f}s\n")

# Comparison
print("="*80)
print("📈 COMPARISON")
print("="*80 + "\n")

improvement_mape = ((metrics1['MAPE'] - metrics2['MAPE']) / metrics1['MAPE']) * 100
improvement_mae = ((metrics1['MAE'] - metrics2['MAE']) / metrics1['MAE']) * 100

print(f"MAPE improvement:  {improvement_mape:+.1f}%")
print(f"MAE improvement:   {improvement_mae:+.1f}%")
print(f"RMSE difference:   {metrics2['RMSE'] - metrics1['RMSE']:+.2f}")

if improvement_mape > 0:
    print(f"\n✅ Optimized config is better! +{improvement_mape:.1f}% accuracy gain")
else:
    print(f"\n⚠️  Original config is better. Monthly seasonality may not be optimal for this data.")

print("\n" + "="*80)

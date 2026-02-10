#!/usr/bin/env python3
"""
Standalone SARIMA Model Runner
Run: python run_sarima_model.py
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from datetime import datetime
import json
from pathlib import Path

from src.forecasting.data_loader import get_engine, load_daily_data
from src.forecasting.metrics import calculate_metrics
from src.forecasting.models.sarima_model import train_sarima

OUTPUT_DIR = Path("outputs/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_sarima():
    """Run SARIMA model and save results"""
    
    print("\n" + "="*70)
    print("🔄 SARIMA FORECASTING MODEL (Auto-tuned)")
    print("="*70)
    
    start_time = time.time()
    
    # Load data
    try:
        engine = get_engine()
        df_daily = load_daily_data(engine)
        print(f"✅ Loaded {len(df_daily)} days of fibre subscription data")
    except:
        print("⚠️  Database unavailable, using synthetic data")
        dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
        values = 2500 + np.linspace(0, 500, 365) + 300 * np.sin(2 * np.pi * np.arange(365) / 7)
        df_daily = pd.DataFrame({'date': dates, 'nb_abonnements': values.astype(int)})
    
    # Train/test split (80/20)
    split_idx = int(len(df_daily) * 0.8)
    train_values = df_daily.iloc[:split_idx]['nb_abonnements'].values
    test_values = df_daily.iloc[split_idx:]['nb_abonnements'].values
    
    print(f"\n📊 Train: {len(train_values)} days | Test: {len(test_values)} days")
    
    # Train model with auto-tuning
    print("\n🔧 Training SARIMA model (auto-tuning parameters)...")
    print("   This may take 30-60 seconds due to grid search...")
    
    model_start = time.time()
    model = train_sarima(train_values, seasonal_period=7)
    training_time = time.time() - model_start
    print(f"   ✅ Model trained in {training_time:.2f}s")
    print(f"   Model Order: {model.order}, Seasonal Order: {model.seasonal_order}")
    
    # Make predictions
    print("\n🔮 Generating forecast...")
    y_pred = model.predict(n_periods=len(test_values))
    y_true = test_values
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    metrics["training_time"] = training_time
    metrics["total_time"] = time.time() - start_time
    metrics["model_order"] = str(model.order)
    metrics["seasonal_order"] = str(model.seasonal_order)
    
    # Display results
    print("\n" + "="*70)
    print("📈 RESULTS")
    print("="*70)
    print(f"MAE   (Mean Absolute Error):       {metrics['MAE']:.2f}")
    print(f"RMSE  (Root Mean Squared Error):   {metrics['RMSE']:.2f}")
    print(f"MAPE  (Mean Absolute % Error):     {metrics['MAPE']:.2f}%")
    print(f"sMAPE (Symmetric MAPE):            {metrics['sMAPE']:.2f}%")
    print(f"\nARIMA Order: {model.order}")
    print(f"Seasonal Order: {model.seasonal_order}")
    print(f"\n⏱️  Training Time: {metrics['training_time']:.2f}s")
    print(f"⏱️  Total Time:    {metrics['total_time']:.2f}s")
    
    # Save results
    result_file = OUTPUT_DIR / f"sarima_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, 'w') as f:
        json.dump({
            'model': 'SARIMA',
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'data_points': {
                'train_size': len(train_values),
                'test_size': len(test_values),
                'prediction_length': len(y_pred)
            }
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {result_file}")
    
    # Sample predictions
    print(f"\n📋 Sample Predictions (first 5):")
    sample_df = pd.DataFrame({
        'Actual': y_true[:5],
        'Predicted': y_pred[:5],
        'Error': np.abs(y_true[:5] - y_pred[:5]),
        'Error%': np.abs((y_true[:5] - y_pred[:5]) / y_true[:5] * 100)
    })
    print(sample_df.to_string())
    
    return {
        'model': 'SARIMA',
        'metrics': metrics,
        'predictions': y_pred,
        'actual': y_true
    }

if __name__ == "__main__":
    results = run_sarima()

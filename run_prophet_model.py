#!/usr/bin/env python3
"""
Standalone Prophet Model Runner
Run: python run_prophet_model.py
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
from src.forecasting.models.prophet_model import train_prophet, predict_prophet

# Output directory
OUTPUT_DIR = Path("outputs/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_prophet():
    """Run Prophet model and save results"""
    
    print("\n" + "="*70)
    print("🔮 PROPHET FORECASTING MODEL")
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
    train_df = pd.DataFrame({'ds': df_daily.iloc[:split_idx]['date'], 
                            'y': df_daily.iloc[:split_idx]['nb_abonnements']})
    test_df = pd.DataFrame({'ds': df_daily.iloc[split_idx:]['date'], 
                           'y': df_daily.iloc[split_idx:]['nb_abonnements']})
    
    print(f"\n📊 Train: {len(train_df)} days | Test: {len(test_df)} days")
    
    # Train model
    print("\n🔧 Training Prophet model...")
    model_start = time.time()
    model = train_prophet(train_df, holidays=None)
    print(f"   ✅ Model trained in {time.time() - model_start:.2f}s")
    
    # Make predictions
    print("\n🔮 Generating forecast...")
    forecast = predict_prophet(model, future_df=test_df[["ds"]])
    y_pred = forecast["yhat"].values
    y_true = test_df["y"].values
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    metrics["training_time"] = time.time() - model_start
    metrics["total_time"] = time.time() - start_time
    
    # Display results
    print("\n" + "="*70)
    print("📈 RESULTS")
    print("="*70)
    print(f"MAE   (Mean Absolute Error):       {metrics['MAE']:.2f}")
    print(f"RMSE  (Root Mean Squared Error):   {metrics['RMSE']:.2f}")
    print(f"MAPE  (Mean Absolute % Error):     {metrics['MAPE']:.2f}%")
    print(f"sMAPE (Symmetric MAPE):            {metrics['sMAPE']:.2f}%")
    print(f"\n⏱️  Training Time: {metrics['training_time']:.2f}s")
    print(f"⏱️  Total Time:    {metrics['total_time']:.2f}s")
    
    # Save results
    result_file = OUTPUT_DIR / f"prophet_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, 'w') as f:
        json.dump({
            'model': 'Prophet',
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'data_points': {
                'train_size': len(train_df),
                'test_size': len(test_df),
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
        'model': 'Prophet',
        'metrics': metrics,
        'predictions': y_pred,
        'actual': y_true
    }

if __name__ == "__main__":
    results = run_prophet()

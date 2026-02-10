#!/usr/bin/env python3
"""
Standalone XGBoost Model Runner
Run: python run_xgboost_model.py
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
from src.forecasting.models.xgboost_model import train_xgboost
from src.forecasting.feature_engineering import build_feature_frame

OUTPUT_DIR = Path("outputs/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_xgboost():
    """Run XGBoost model and save results"""
    
    print("\n" + "="*70)
    print("🌲 XGBOOST GRADIENT BOOSTING MODEL")
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
    
    # Build features
    print("\n🔧 Engineering features (lag, rolling, temporal)...")
    feature_start = time.time()
    df_features, _, _ = build_feature_frame(df_daily)
    feature_cols = [c for c in df_features.columns if c not in ['date', 'nb_abonnements']]
    print(f"   ✅ Created {len(feature_cols)} features in {time.time() - feature_start:.2f}s")
    
    # Train/test split (80/20)
    split_idx = int(len(df_features) * 0.8)
    train_df = df_features.iloc[:split_idx]
    test_df = df_features.iloc[split_idx:]
    
    X_train = train_df[feature_cols].values
    y_train = train_df["nb_abonnements"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["nb_abonnements"].values
    
    print(f"\n📊 Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"   Features: {X_train.shape[1]}")
    
    # Train model
    print("\n🔧 Training XGBoost model...")
    model_start = time.time()
    model = train_xgboost(X_train, y_train)
    training_time = time.time() - model_start
    print(f"   ✅ Model trained in {training_time:.2f}s")
    
    # Make predictions
    print("\n🌲 Generating predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    metrics["training_time"] = training_time
    metrics["total_time"] = time.time() - start_time
    metrics["n_features"] = X_train.shape[1]
    metrics["n_trees"] = 200
    
    # Display results
    print("\n" + "="*70)
    print("📈 RESULTS")
    print("="*70)
    print(f"MAE   (Mean Absolute Error):       {metrics['MAE']:.2f}")
    print(f"RMSE  (Root Mean Squared Error):   {metrics['RMSE']:.2f}")
    print(f"MAPE  (Mean Absolute % Error):     {metrics['MAPE']:.2f}%")
    print(f"sMAPE (Symmetric MAPE):            {metrics['sMAPE']:.2f}%")
    print(f"\n📊 Model Config:")
    print(f"   Trees: 200 | Max Depth: 5 | Learning Rate: 0.05")
    print(f"   Features Used: {len(feature_cols)}")
    print(f"\n⏱️  Training Time: {metrics['training_time']:.2f}s")
    print(f"⏱️  Total Time:    {metrics['total_time']:.2f}s")
    
    # Feature importance
    print(f"\n📊 Top 5 Most Important Features:")
    feature_importance = model.feature_importances_
    top_indices = np.argsort(feature_importance)[-5:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        print(f"   {rank}. {feature_cols[idx]:25s} - Importance: {feature_importance[idx]:.4f}")
    
    # Save results
    result_file = OUTPUT_DIR / f"xgboost_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, 'w') as f:
        json.dump({
            'model': 'XGBoost',
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'data_points': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'n_features': len(feature_cols),
                'prediction_length': len(y_pred)
            }
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {result_file}")
    
    # Sample predictions
    print(f"\n📋 Sample Predictions (first 5):")
    sample_df = pd.DataFrame({
        'Actual': y_test[:5],
        'Predicted': y_pred[:5],
        'Error': np.abs(y_test[:5] - y_pred[:5]),
        'Error%': np.abs((y_test[:5] - y_pred[:5]) / y_test[:5] * 100)
    })
    print(sample_df.to_string())
    
    return {
        'model': 'XGBoost',
        'metrics': metrics,
        'predictions': y_pred,
        'actual': y_test
    }

if __name__ == "__main__":
    results = run_xgboost()

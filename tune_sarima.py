#!/usr/bin/env python3
"""
SARIMA Model Fine-Tuning & Optimization
Tests different configurations to find the best SARIMA parameters
"""

import warnings
warnings.filterwarnings('ignore')

import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from src.forecasting.data_loader import get_engine, load_daily_data
from src.forecasting.metrics import calculate_metrics

# Output directory
TUNING_DIR = Path("outputs/sarima_tuning")
TUNING_DIR.mkdir(parents=True, exist_ok=True)

def train_sarima_custom(train_values, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    """Train SARIMA with custom order"""
    try:
        from pmdarima import ARIMA
        model = ARIMA(order=order, seasonal_order=seasonal_order, suppress_warnings=True)
        model.fit(train_values)
        return model
    except Exception as e:
        return None

def test_configuration(name, train_values, test_values, order, seasonal_order):
    """Test a single SARIMA configuration"""
    try:
        start = time.time()
        model = train_sarima_custom(train_values, order, seasonal_order)
        
        if model is None:
            return {
                'config': name,
                'order': order,
                'seasonal_order': seasonal_order,
                'status': 'FAILED',
                'mape': np.nan,
                'mae': np.nan,
                'rmse': np.nan,
                'time': time.time() - start
            }
        
        # Forecast
        y_pred = model.predict(n_periods=len(test_values))
        
        # Metrics
        metrics = calculate_metrics(test_values, y_pred)
        
        return {
            'config': name,
            'order': order,
            'seasonal_order': seasonal_order,
            'status': 'SUCCESS',
            'mape': metrics['MAPE'],
            'mae': metrics['MAE'],
            'rmse': metrics['RMSE'],
            'time': time.time() - start
        }
    except Exception as e:
        return {
            'config': name,
            'order': order,
            'seasonal_order': seasonal_order,
            'status': f'ERROR: {str(e)[:50]}',
            'mape': np.nan,
            'mae': np.nan,
            'rmse': np.nan,
            'time': time.time() - start
        }

def main():
    print("\n" + "="*80)
    print("🔧 SARIMA FINE-TUNING & OPTIMIZATION")
    print("="*80)
    
    # Load data
    try:
        engine = get_engine()
        df_daily = load_daily_data(engine)
        print(f"✅ Loaded {len(df_daily)} days of data from clean_data table")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Train/test split
    split_idx = int(len(df_daily) * 0.8)
    train_values = df_daily.iloc[:split_idx]['nb_abonnements'].values
    test_values = df_daily.iloc[split_idx:]['nb_abonnements'].values
    
    print(f"📊 Train: {len(train_values)} days | Test: {len(test_values)} days")
    
    # Define configurations to test
    print("\n🔍 Testing different SARIMA configurations...")
    print("   (This may take 2-3 minutes)\n")
    
    configs = [
        # Baseline (current)
        ("Baseline (Current)", (1, 1, 1), (1, 1, 1, 7)),
        
        # Seasonal period variations
        ("Seasonal: 7 days", (1, 1, 1), (1, 1, 1, 7)),
        ("Seasonal: 14 days", (1, 1, 1), (1, 1, 1, 14)),
        ("Seasonal: 30 days", (1, 1, 1), (1, 1, 1, 30)),
        
        # Non-seasonal order variations (p,d,q)
        ("Low Order (0,1,1)", (0, 1, 1), (1, 1, 1, 7)),
        ("Low Order (1,0,1)", (1, 0, 1), (1, 1, 1, 7)),
        ("Low Order (1,1,0)", (1, 1, 0), (1, 1, 1, 7)),
        
        # Higher orders
        ("Higher Order (2,1,2)", (2, 1, 2), (1, 1, 1, 7)),
        ("Higher Order (2,2,2)", (2, 2, 2), (1, 1, 1, 7)),
        ("Higher Order (3,1,2)", (3, 1, 2), (1, 1, 1, 7)),
        
        # Seasonal order variations (P,D,Q,s)
        ("Seasonal Order (1,0,1,7)", (1, 1, 1), (1, 0, 1, 7)),
        ("Seasonal Order (2,1,1,7)", (1, 1, 1), (2, 1, 1, 7)),
        ("Seasonal Order (1,1,2,7)", (1, 1, 1), (1, 1, 2, 7)),
        ("Seasonal Order (2,1,2,7)", (1, 1, 1), (2, 1, 2, 7)),
        
        # Combined optimizations
        ("Optimized 1 (2,1,1)(1,1,1,7)", (2, 1, 1), (1, 1, 1, 7)),
        ("Optimized 2 (1,1,2)(1,1,1,7)", (1, 1, 2), (1, 1, 1, 7)),
        ("Optimized 3 (2,1,2)(1,1,1,7)", (2, 1, 2), (1, 1, 1, 7)),
        ("Optimized 4 (1,1,1)(2,1,1,7)", (1, 1, 1), (2, 1, 1, 7)),
        ("Optimized 5 (2,1,1)(2,1,1,7)", (2, 1, 1), (2, 1, 1, 7)),
    ]
    
    results = []
    for name, order, seasonal_order in tqdm(configs, desc="Testing configs"):
        result = test_configuration(name, train_values, test_values, order, seasonal_order)
        results.append(result)
    
    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results)
    results_df_success = results_df[results_df['status'] == 'SUCCESS'].copy()
    
    if len(results_df_success) == 0:
        print("❌ No successful configurations found!")
        return
    
    results_df_success = results_df_success.sort_values('mape')
    
    # Display results
    print("\n" + "="*80)
    print("📊 TOP 10 CONFIGURATIONS (By MAPE)")
    print("="*80 + "\n")
    
    for idx, row in results_df_success.head(10).iterrows():
        print(f"#{idx+1}. {row['config']}")
        print(f"   Order: {row['order']} × Seasonal: {row['seasonal_order']}")
        print(f"   MAPE: {row['mape']:.2f}% | MAE: {row['mae']:.2f} | RMSE: {row['rmse']:.2f}")
        print(f"   Training Time: {row['time']:.2f}s\n")
    
    # Best configuration
    best = results_df_success.iloc[0]
    print("="*80)
    print("🥇 BEST CONFIGURATION")
    print("="*80)
    print(f"\nConfiguration: {best['config']}")
    print(f"Order (p,d,q):        {best['order']}")
    print(f"Seasonal (P,D,Q,s):   {best['seasonal_order']}")
    print(f"\n✅ Results:")
    print(f"   MAPE:  {best['mape']:.2f}%")
    print(f"   MAE:   {best['mae']:.2f}")
    print(f"   RMSE:  {best['rmse']:.2f}")
    print(f"   Time:  {best['time']:.2f}s")
    
    # Comparison with baseline
    baseline = results_df_success[results_df_success['config'] == 'Baseline (Current)'].iloc[0]
    improvement = ((baseline['mape'] - best['mape']) / baseline['mape']) * 100
    print(f"\n📈 Improvement over baseline:")
    print(f"   Baseline MAPE: {baseline['mape']:.2f}%")
    print(f"   Best MAPE:     {best['mape']:.2f}%")
    print(f"   Improvement:   {improvement:+.1f}%")
    
    # Save all results
    csv_file = TUNING_DIR / f"sarima_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(csv_file, index=False)
    
    json_file = TUNING_DIR / f"sarima_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_configs': len(configs),
            'successful_configs': len(results_df_success),
            'data_points': {
                'train_size': len(train_values),
                'test_size': len(test_values)
            },
            'best_configuration': {
                'name': best['config'],
                'order': best['order'],
                'seasonal_order': best['seasonal_order'],
                'metrics': {
                    'mape': float(best['mape']),
                    'mae': float(best['mae']),
                    'rmse': float(best['rmse']),
                    'training_time': float(best['time'])
                }
            },
            'improvement_percent': float(improvement)
        }, f, indent=2)
    
    print(f"\n💾 Results saved to:")
    print(f"   CSV:  {csv_file}")
    print(f"   JSON: {json_file}")
    
    # Recommendation
    print("\n" + "="*80)
    print("📝 RECOMMENDATION")
    print("="*80)
    if improvement > 0:
        print(f"\n✅ Found a better configuration!")
        print(f"   Use Order: {best['order']}")
        print(f"   Use Seasonal: {best['seasonal_order']}")
        print(f"   This will improve accuracy by {improvement:.1f}%")
    else:
        print(f"\n⚠️  Current baseline is already near-optimal.")
        print(f"   Consider testing even more configurations or ensemble methods.")

if __name__ == "__main__":
    main()

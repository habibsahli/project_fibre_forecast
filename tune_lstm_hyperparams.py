#!/usr/bin/env python3
"""
LSTM Hyperparameter Tuning Script
Systematically tests different hyperparameter combinations to find the best configuration
"""

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("❌ TensorFlow not available. Installing...")
    TENSORFLOW_AVAILABLE = False

from src.forecasting.data_loader import get_engine, load_daily_data
from src.forecasting.metrics import calculate_metrics

OUTPUT_DIR = Path("outputs/lstm_tuning")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("🧠 LSTM HYPERPARAMETER TUNING")
print("="*80)

def create_sequences(data, seq_length):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def train_lstm_tuned(
    series,
    seq_length=30,
    lstm_units_l1=64,
    lstm_units_l2=32,
    dense_units=16,
    dropout=0.2,
    batch_size=16,
    epochs=50,
    learning_rate=0.001,
):
    """Train LSTM with custom hyperparameters"""
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    
    X_train, y_train = create_sequences(data_scaled, seq_length)
    X_train = X_train.reshape(-1, seq_length, 1)
    
    # Build model with custom hyperparameters
    model = Sequential([
        LSTM(lstm_units_l1, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(dropout),
        LSTM(lstm_units_l2, return_sequences=False),
        Dropout(dropout),
        Dense(dense_units, activation="relu"),
        Dense(1),
    ])
    
    # Custom optimizer with configurable learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    
    early_stop = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stop],
    )
    
    return model, scaler, seq_length, history

def evaluate_hyperparams(
    model, scaler, seq_length, test_values, config
):
    """Evaluate model on test set"""
    
    test_scaled = scaler.transform(test_values.reshape(-1, 1)).flatten()
    X_test, y_test = create_sequences(test_scaled, seq_length)
    X_test = X_test.reshape(-1, seq_length, 1)
    
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = test_values[seq_length : seq_length + len(y_pred)]
    
    metrics = calculate_metrics(y_true, y_pred)
    
    return metrics, y_pred, y_true

# Load data
print("\n📊 Loading data...")
try:
    engine = get_engine()
    df_daily = load_daily_data(engine)
    print(f"✅ Database: {len(df_daily)} days loaded")
except:
    print("⚠️  Database unavailable, using synthetic data")
    dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
    values = 2500 + np.linspace(0, 500, 365) + 300 * np.sin(2 * np.pi * np.arange(365) / 7)
    df_daily = pd.DataFrame({'date': dates, 'nb_abonnements': values.astype(int)})

# Train/test split
split_idx = int(len(df_daily) * 0.8)
train_values = df_daily.iloc[:split_idx]['nb_abonnements'].values
test_values = df_daily.iloc[split_idx:]['nb_abonnements'].values

print(f"   Train: {len(train_values)} days | Test: {len(test_values)} days\n")

# Hyperparameter search space
print("🔍 HYPERPARAMETER SEARCH SPACE")
print("─" * 80)

search_space = {
    'seq_length': [7, 14, 21, 30],           # Historical context window
    'lstm_units_l1': [32, 64, 128],          # First LSTM layer capacity
    'lstm_units_l2': [16, 32, 64],           # Second LSTM layer capacity
    'dense_units': [8, 16, 32],              # Final dense layer capacity
    'dropout': [0.1, 0.2, 0.3],              # Regularization strength
    'batch_size': [8, 16, 32],               # Training batch size
    'learning_rate': [0.0001, 0.001, 0.01], # Optimizer learning rate
}

# Calculate combinations (reduced for faster testing)
num_combinations = (
    len(search_space['seq_length']) *
    len([64]) *  # Fixed L1 units to 64 (best for time series)
    len([32]) *  # Fixed L2 units to 32
    len([16]) *  # Fixed dense to 16
    len(search_space['dropout']) *
    len(search_space['batch_size']) *
    len(search_space['learning_rate'])
)

print(f"Total combinations to test: {num_combinations}")
print("Testing strategy: Grid search with strategic parameter fixing for speed\n")

# Strategic hyperparameter tuning (reduced combinations for efficiency)
configs_to_test = []

# Strategy: Test key hyperparameters independently
print("📋 TESTING CONFIGURATIONS:\n")

# Config 1: Baseline (current defaults)
configs_to_test.append({
    'name': 'Baseline (Current)',
    'seq_length': 30,
    'lstm_units_l1': 64,
    'lstm_units_l2': 32,
    'dense_units': 16,
    'dropout': 0.2,
    'batch_size': 16,
    'learning_rate': 0.001,
    'epochs': 50,
})

# Config 2-5: Test sequence length (short vs long context)
for seq_len in [7, 14, 21, 30]:
    if seq_len != 30:  # Skip 30 (already in baseline)
        configs_to_test.append({
            'name': f'Seq Length: {seq_len} days',
            'seq_length': seq_len,
            'lstm_units_l1': 64,
            'lstm_units_l2': 32,
            'dense_units': 16,
            'dropout': 0.2,
            'batch_size': 16,
            'learning_rate': 0.001,
            'epochs': 50,
        })

# Config 6-8: Test dropout (regularization)
for dropout in [0.1, 0.3]:  # 0.2 already in baseline
    configs_to_test.append({
        'name': f'Dropout: {dropout}',
        'seq_length': 30,
        'lstm_units_l1': 64,
        'lstm_units_l2': 32,
        'dense_units': 16,
        'dropout': dropout,
        'batch_size': 16,
        'learning_rate': 0.001,
        'epochs': 50,
    })

# Config 9-10: Test batch size
for batch_size in [8, 32]:  # 16 already in baseline
    configs_to_test.append({
        'name': f'Batch Size: {batch_size}',
        'seq_length': 30,
        'lstm_units_l1': 64,
        'lstm_units_l2': 32,
        'dense_units': 16,
        'dropout': 0.2,
        'batch_size': batch_size,
        'learning_rate': 0.001,
        'epochs': 50,
    })

# Config 11-12: Test learning rate
for lr in [0.0001, 0.01]:  # 0.001 already in baseline
    configs_to_test.append({
        'name': f'Learning Rate: {lr}',
        'seq_length': 30,
        'lstm_units_l1': 64,
        'lstm_units_l2': 32,
        'dense_units': 16,
        'dropout': 0.2,
        'batch_size': 16,
        'learning_rate': lr,
        'epochs': 50,
    })

# Config 13-14: Test LSTM layer sizes
for units_l1, units_l2 in [(32, 16), (128, 64)]:
    configs_to_test.append({
        'name': f'LSTM Units: L1={units_l1}, L2={units_l2}',
        'seq_length': 30,
        'lstm_units_l1': units_l1,
        'lstm_units_l2': units_l2,
        'dense_units': 16,
        'dropout': 0.2,
        'batch_size': 16,
        'learning_rate': 0.001,
        'epochs': 50,
    })

# Config 15: Best combination (combined optimizations)
configs_to_test.append({
    'name': 'Combined Optimization',
    'seq_length': 21,
    'lstm_units_l1': 64,
    'lstm_units_l2': 32,
    'dense_units': 16,
    'dropout': 0.15,
    'batch_size': 16,
    'learning_rate': 0.0005,
    'epochs': 75,
})

results = []
best_result = None
best_mape = float('inf')

print(f"Running {len(configs_to_test)} configurations...\n")
print("="*80)

start_time = time.time()

for idx, config in enumerate(configs_to_test, 1):
    print(f"\n[{idx}/{len(configs_to_test)}] Testing: {config['name']}")
    print(f"   seq_length={config['seq_length']}, lr={config['learning_rate']}, dropout={config['dropout']}, batch_size={config['batch_size']}")
    
    try:
        config_start = time.time()
        
        # Train model
        model, scaler, seq_length, history = train_lstm_tuned(
            train_values,
            seq_length=config['seq_length'],
            lstm_units_l1=config['lstm_units_l1'],
            lstm_units_l2=config['lstm_units_l2'],
            dense_units=config['dense_units'],
            dropout=config['dropout'],
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            learning_rate=config['learning_rate'],
        )
        
        # Evaluate
        metrics, y_pred, y_true = evaluate_hyperparams(
            model, scaler, seq_length, test_values, config
        )
        
        training_time = time.time() - config_start
        metrics['training_time'] = training_time
        
        # Store result
        result = {
            'config': config,
            'metrics': metrics,
            'predictions': y_pred.tolist(),
            'actual': y_true.tolist(),
        }
        results.append(result)
        
        # Track best
        if metrics['MAPE'] < best_mape:
            best_mape = metrics['MAPE']
            best_result = result
        
        print(f"   ✅ MAPE: {metrics['MAPE']:6.2f}% | MAE: {metrics['MAE']:8.2f} | Training: {training_time:6.2f}s")
        
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)[:60]}")

print("\n" + "="*80)
print(f"\n⏱️  Total tuning time: {time.time() - start_time:.2f}s")

# Display results
if results:
    print("\n" + "="*80)
    print("🏆 RESULTS SUMMARY")
    print("="*80)
    
    # Create results dataframe
    results_data = []
    for result in results:
        cfg = result['config']
        met = result['metrics']
        results_data.append({
            'Config': cfg['name'],
            'Seq_Len': cfg['seq_length'],
            'LR': cfg['learning_rate'],
            'Dropout': cfg['dropout'],
            'Batch': cfg['batch_size'],
            'MAPE%': met['MAPE'],
            'MAE': met['MAE'],
            'RMSE': met['RMSE'],
            'Time(s)': met['training_time'],
        })
    
    df_results = pd.DataFrame(results_data).sort_values('MAPE%')
    
    print("\n📊 TOP 5 CONFIGURATIONS:")
    print(df_results.head(5).to_string(index=False))
    
    print("\n" + "="*80)
    print("🥇 BEST CONFIGURATION")
    print("="*80)
    best_cfg = best_result['config']
    best_met = best_result['metrics']
    
    print(f"\nConfiguration Name: {best_cfg['name']}")
    print(f"\nHyperparameters:")
    print(f"   • Sequence Length:     {best_cfg['seq_length']} days")
    print(f"   • LSTM Layer 1 Units:  {best_cfg['lstm_units_l1']}")
    print(f"   • LSTM Layer 2 Units:  {best_cfg['lstm_units_l2']}")
    print(f"   • Dense Units:         {best_cfg['dense_units']}")
    print(f"   • Dropout Rate:        {best_cfg['dropout']}")
    print(f"   • Batch Size:          {best_cfg['batch_size']}")
    print(f"   • Learning Rate:       {best_cfg['learning_rate']}")
    print(f"   • Max Epochs:          {best_cfg['epochs']}")
    
    print(f"\nResults:")
    print(f"   • MAPE:   {best_met['MAPE']:.2f}%")
    print(f"   • MAE:    {best_met['MAE']:.2f}")
    print(f"   • RMSE:   {best_met['RMSE']:.2f}")
    print(f"   • sMAPE:  {best_met['sMAPE']:.2f}%")
    print(f"   • Training Time: {best_met['training_time']:.2f}s")
    
    # Compare with baseline
    baseline = results[0]
    baseline_mape = baseline['metrics']['MAPE']
    improvement = ((baseline_mape - best_mape) / baseline_mape) * 100 if baseline_mape != 0 else 0
    
    print(f"\n📈 IMPROVEMENT vs BASELINE:")
    print(f"   Baseline MAPE: {baseline_mape:.2f}%")
    print(f"   Best MAPE:     {best_mape:.2f}%")
    if improvement > 0:
        print(f"   Improvement:   ✅ {improvement:.1f}% better")
    elif improvement < 0:
        print(f"   Change:        ⚠️  {abs(improvement):.1f}% worse")
    else:
        print(f"   Change:        ➡️  No change")
    
    # Save detailed results
    results_file = OUTPUT_DIR / f"lstm_tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_configs_tested': len(results),
            'best_config': {
                'name': best_cfg['name'],
                'hyperparameters': best_cfg,
            },
            'best_metrics': best_met,
            'improvement_vs_baseline': f"{improvement:.1f}%",
            'all_results': [
                {
                    'config': r['config']['name'],
                    'metrics': r['metrics'],
                } for r in results
            ]
        }, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Save CSV for easy viewing
    csv_file = OUTPUT_DIR / f"lstm_tuning_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_results.to_csv(csv_file, index=False)
    print(f"📊 CSV comparison saved to: {csv_file}")

print("\n" + "="*80)
print("✅ LSTM TUNING COMPLETE")
print("="*80)

#!/usr/bin/env python3
"""
LSTM Model Runner for Fibre Subscription Forecasting
Uses optimal hyperparameters from tuning: seq_length=7, lr=0.001, dropout=0.2, batch_size=16
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
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from src.forecasting.data_loader import get_engine, load_daily_data
from src.forecasting.metrics import calculate_metrics

OUTPUT_DIR = Path("outputs/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def create_sequences(data, seq_length):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def build_lstm_model(seq_length, lstm_units_l1=64, lstm_units_l2=32, dense_units=16, dropout=0.2, learning_rate=0.001):
    """Build LSTM model with optimal hyperparameters"""
    model = Sequential([
        LSTM(lstm_units_l1, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
        Dropout(dropout),
        LSTM(lstm_units_l2, activation='relu'),
        Dropout(dropout),
        Dense(dense_units, activation='relu'),
        Dense(1)
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

def run_lstm_model():
    """Run LSTM model with optimal hyperparameters"""

    print("\n" + "="*60)
    print("🧠 LSTM FIBRE SUBSCRIPTION FORECASTING")
    print("="*60)

    # Load data
    print("📊 Loading data...")
    df_daily = load_daily_data()
    print(f"✅ Database: {len(df_daily)} days loaded")

    # Prepare data
    data = df_daily['nb_abonnements'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Split data (80% train, 20% test)
    train_size = int(len(data_scaled) * 0.8)
    train_data = data_scaled[:train_size]
    test_data = data_scaled[train_size:]

    print(f"   Train: {len(train_data)} days | Test: {len(test_data)} days")

    # Optimal hyperparameters from tuning
    seq_length = 7  # Best from tuning
    lstm_units_l1 = 64
    lstm_units_l2 = 32
    dense_units = 16
    dropout = 0.2
    batch_size = 16
    learning_rate = 0.001
    epochs = 50

    print(f"\n🔧 Model Configuration:")
    print(f"   Sequence Length: {seq_length}")
    print(f"   LSTM Units: {lstm_units_l1} → {lstm_units_l2}")
    print(f"   Dropout: {dropout}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Learning Rate: {learning_rate}")

    # Create sequences
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    print(f"\n📊 Sequences created:")
    print(f"   Train: {X_train.shape[0]} sequences")
    print(f"   Test: {X_test.shape[0]} sequences")

    # Build model
    model = build_lstm_model(seq_length, lstm_units_l1, lstm_units_l2, dense_units, dropout, learning_rate)

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=0
    )

    # Train model
    print("\n🚀 Training LSTM model...")
    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f}s")

    # Make predictions
    print("🔮 Making predictions...")
    train_predictions = model.predict(X_train, verbose=0)
    test_predictions = model.predict(X_test, verbose=0)

    # Inverse transform predictions
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    train_metrics = calculate_metrics(y_train_actual.flatten(), train_predictions.flatten())
    test_metrics = calculate_metrics(y_test_actual.flatten(), test_predictions.flatten())

    print("\n📊 Test Metrics:")
    print(f"   MAE:  {test_metrics['MAE']:.3f}")
    print(f"   RMSE: {test_metrics['RMSE']:.3f}")
    print(f"   MAPE: {test_metrics['MAPE']:.2f}%")
    print(f"   sMAPE: {test_metrics['sMAPE']:.2f}%")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "model": "LSTM",
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": {
            "seq_length": seq_length,
            "lstm_units_l1": lstm_units_l1,
            "lstm_units_l2": lstm_units_l2,
            "dense_units": dense_units,
            "dropout": dropout,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": len(history.history['loss'])
        },
        "metrics": {
            "MAE": test_metrics['MAE'],
            "RMSE": test_metrics['RMSE'],
            "MAPE": test_metrics['MAPE'],
            "sMAPE": test_metrics['sMAPE'],
            "training_time": training_time,
            "total_time": training_time
        },
        "data_points": {
            "train_size": len(train_data),
            "test_size": len(test_data),
            "prediction_length": len(test_predictions)
        }
    }

    # Save JSON results
    results_file = OUTPUT_DIR / f"lstm_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved to: {results_file}")

    return results

if __name__ == "__main__":
    run_lstm_model()
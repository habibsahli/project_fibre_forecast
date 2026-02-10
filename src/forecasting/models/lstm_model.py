"""LSTM model helpers."""

from __future__ import annotations

import numpy as np


def create_sequences(data, seq_length: int = 30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def train_lstm(series, seq_length: int = 30, epochs: int = 50, batch_size: int = 16):
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    X_train, y_train = create_sequences(data_scaled, seq_length)
    X_train = X_train.reshape(-1, seq_length, 1)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early_stop = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stop],
    )

    return model, scaler, seq_length

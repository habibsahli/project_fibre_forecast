"""
TimeGAN Implementation for Fibre Subscription Time Series Generation

TimeGAN (Time-series Generative Adversarial Networks) generates realistic
synthetic time series data that preserves temporal dependencies and patterns.

Used for:
- Data augmentation (generate more training data)
- Scenario generation (multiple possible futures)
- Uncertainty modeling
- Missing data imputation
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

class TimeGANModel:
    """
    Time-series Generative Adversarial Network for fibre subscription forecasting.

    Architecture:
    - Autoencoder: Learns compressed representations
    - Generator: Creates synthetic time series
    - Discriminator: Distinguishes real vs synthetic
    - Recovery Network: Reconstructs original data
    """

    def __init__(self, seq_len=30, latent_dim=10, batch_size=32, epochs=100):
        """
        Initialize TimeGAN model.

        Args:
            seq_len: Length of time series sequences
            latent_dim: Dimension of latent space
            batch_size: Training batch size
            epochs: Number of training epochs
        """
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs

        # Models to be built
        self.autoencoder = None
        self.generator = None
        self.discriminator = None
        self.recovery = None

        # Scalers
        self.scaler = MinMaxScaler()

        # Training history
        self.history = {}

    def build_autoencoder(self, input_dim):
        """Build autoencoder for representation learning."""
        # Encoder
        encoder_input = layers.Input(shape=(self.seq_len, input_dim))
        x = layers.LSTM(64, return_sequences=True)(encoder_input)
        x = layers.LSTM(self.latent_dim)(x)
        encoder_output = layers.Dense(self.latent_dim)(x)

        # Decoder (Recovery Network)
        decoder_input = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(self.latent_dim)(decoder_input)
        x = layers.RepeatVector(self.seq_len)(x)
        x = layers.LSTM(self.latent_dim, return_sequences=True)(x)
        decoder_output = layers.TimeDistributed(layers.Dense(input_dim))(x)

        # Models
        self.encoder = keras.Model(encoder_input, encoder_output, name='encoder')
        self.recovery = keras.Model(decoder_input, decoder_output, name='recovery')
        self.autoencoder = keras.Model(encoder_input, self.recovery(self.encoder(encoder_input)), name='autoencoder')

        # Compile
        self.autoencoder.compile(optimizer='adam', loss='mse')

    def build_generator(self, input_dim):
        """Build generator network."""
        generator_input = layers.Input(shape=(self.seq_len, self.latent_dim))
        x = layers.LSTM(64, return_sequences=True)(generator_input)
        x = layers.LSTM(32, return_sequences=True)(x)
        generator_output = layers.TimeDistributed(layers.Dense(input_dim))(x)

        self.generator = keras.Model(generator_input, generator_output, name='generator')

    def build_discriminator(self, input_dim):
        """Build discriminator network."""
        discriminator_input = layers.Input(shape=(self.seq_len, input_dim))
        x = layers.LSTM(64, return_sequences=True)(discriminator_input)
        x = layers.LSTM(32)(x)
        x = layers.Dense(16, activation='relu')(x)
        discriminator_output = layers.Dense(1, activation='sigmoid')(x)

        self.discriminator = keras.Model(discriminator_input, discriminator_output, name='discriminator')
        self.discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def create_sequences(self, data, seq_len):
        """Create rolling sequences from time series data."""
        sequences = []
        for i in range(len(data) - seq_len + 1):
            sequences.append(data[i:i + seq_len])
        return np.array(sequences)

    def fit(self, df_daily, feature_col='nb_abonnements'):
        """
        Train TimeGAN on fibre subscription data.

        Args:
            df_daily: DataFrame with date and subscription count columns
            feature_col: Column name for subscription counts
        """
        print("🚀 Training TimeGAN on fibre subscription data...")

        # Prepare data
        data = df_daily[feature_col].values.reshape(-1, 1)
        data_scaled = self.scaler.fit_transform(data)

        # Create sequences
        sequences = self.create_sequences(data_scaled, self.seq_len)
        print(f"📊 Created {len(sequences)} sequences of length {self.seq_len}")

        # Build models
        input_dim = sequences.shape[2]
        self.build_autoencoder(input_dim)
        self.build_generator(input_dim)
        self.build_discriminator(input_dim)

        # Phase 1: Train Autoencoder
        print("🔧 Phase 1: Training Autoencoder...")
        self.autoencoder.fit(sequences, sequences,
                           epochs=self.epochs//3,
                           batch_size=self.batch_size,
                           verbose=0)

        # Phase 2: Train GAN
        print("🎭 Phase 2: Training GAN...")
        gan_input = layers.Input(shape=(self.seq_len, self.latent_dim))
        gan_output = self.discriminator(self.generator(gan_input))
        self.gan = keras.Model(gan_input, gan_output)
        self.gan.compile(optimizer='adam', loss='binary_crossentropy')

        # Training loop
        d_losses = []
        g_losses = []

        for epoch in range(self.epochs):
            # Train Discriminator
            # Real data
            idx = np.random.randint(0, sequences.shape[0], self.batch_size)
            real_sequences = sequences[idx]

            # Fake data
            noise = np.random.normal(0, 1, (self.batch_size, self.seq_len, self.latent_dim))
            fake_sequences = self.generator.predict(noise, verbose=0)

            # Labels
            real_labels = np.ones((self.batch_size, 1))
            fake_labels = np.zeros((self.batch_size, 1))

            # Train discriminator
            d_loss_real = self.discriminator.train_on_batch(real_sequences, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_sequences, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator
            noise = np.random.normal(0, 1, (self.batch_size, self.seq_len, self.latent_dim))
            valid_labels = np.ones((self.batch_size, 1))
            g_loss = self.gan.train_on_batch(noise, valid_labels)

            d_losses.append(d_loss[0])
            g_losses.append(g_loss)

            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{self.epochs} - D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}")

        self.history = {'d_losses': d_losses, 'g_losses': g_losses}
        print("✅ TimeGAN training completed!")

    def generate(self, n_samples=100, sequence_length=None):
        """
        Generate synthetic fibre subscription sequences.

        Args:
            n_samples: Number of synthetic sequences to generate
            sequence_length: Length of each sequence (default: training seq_len)

        Returns:
            Array of synthetic sequences
        """
        if sequence_length is None:
            sequence_length = self.seq_len

        # Generate noise
        noise = np.random.normal(0, 1, (n_samples, sequence_length, self.latent_dim))

        # Generate synthetic data
        synthetic_scaled = self.generator.predict(noise, verbose=0)

        # Inverse transform to original scale
        synthetic_original = self.scaler.inverse_transform(
            synthetic_scaled.reshape(-1, 1)
        ).reshape(n_samples, sequence_length, 1)

        return synthetic_original.squeeze()

    def generate_scenarios(self, n_scenarios=10, forecast_horizon=30):
        """
        Generate multiple forecast scenarios.

        Args:
            n_scenarios: Number of different scenarios
            forecast_horizon: Days to forecast

        Returns:
            DataFrame with scenario forecasts
        """
        scenarios = []

        for i in range(n_scenarios):
            # Generate synthetic sequence
            synthetic = self.generate(n_samples=1, sequence_length=forecast_horizon)
            scenarios.append({
                'scenario': f'scenario_{i+1}',
                'forecast': synthetic[0] if len(synthetic.shape) > 1 else synthetic
            })

        return pd.DataFrame(scenarios)

    def plot_training_history(self):
        """Plot training losses."""
        if not self.history:
            print("No training history available")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.history['d_losses'], label='Discriminator Loss')
        plt.plot(self.history['g_losses'], label='Generator Loss')
        plt.title('TimeGAN Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def save_model(self, path):
        """Save trained models."""
        if self.generator:
            self.generator.save(path / 'timegan_generator.h5')
        if self.discriminator:
            self.discriminator.save(path / 'timegan_discriminator.h5')
        if self.autoencoder:
            self.autoencoder.save(path / 'timegan_autoencoder.h5')

    def load_model(self, path):
        """Load trained models."""
        try:
            self.generator = keras.models.load_model(path / 'timegan_generator.h5')
            self.discriminator = keras.models.load_model(path / 'timegan_discriminator.h5')
            self.autoencoder = keras.models.load_model(path / 'timegan_autoencoder.h5')
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {e}")


def generate_synthetic_scenarios(df_daily, n_scenarios=50, seq_len=30):
    """
    Convenience function to generate synthetic scenarios.

    Args:
        df_daily: DataFrame with daily subscription data
        n_scenarios: Number of scenarios to generate
        seq_len: Sequence length for training

    Returns:
        DataFrame with synthetic scenarios
    """
    model = TimeGANModel(seq_len=seq_len, epochs=50)  # Reduced epochs for demo
    model.fit(df_daily)
    scenarios = model.generate_scenarios(n_scenarios=n_scenarios)
    return scenarios


if __name__ == "__main__":
    # Example usage
    print("TimeGAN Model for Fibre Subscription Forecasting")
    print("Run this module to test the implementation")
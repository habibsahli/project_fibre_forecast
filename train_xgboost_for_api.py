#!/usr/bin/env python3
"""
Train and Save XGBoost Model for Flask API
Trains the best XGBoost model and saves it for web deployment
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

from src.forecasting.data_loader import get_engine, load_daily_data, load_detailed_data
from src.forecasting.feature_engineering import build_feature_frame
from src.forecasting.models.xgboost_model import train_xgboost

OUTPUT_DIR = Path("outputs")
MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_and_save_xgboost():
    """Train XGBoost model and save for Flask API"""

    print("🚀 Training XGBoost Model for Flask API")
    print("="*50)

    # Load data
    print("📊 Loading data...")
    engine = get_engine()
    df_daily = load_daily_data(engine)
    df_detailed = load_detailed_data(engine)
    print(f"✅ Loaded {len(df_daily)} daily records")

    # Feature engineering
    print("🔧 Engineering features...")
    df_features, top_govs, top_offres = build_feature_frame(df_daily, df_detailed)
    feature_cols = [c for c in df_features.columns if c not in ["date", "nb_abonnements"]]
    print(f"✅ Created {len(feature_cols)} features")

    # Train/test split
    split_idx = int(len(df_features) * 0.8)
    train_df = df_features.iloc[:split_idx].copy()
    test_df = df_features.iloc[split_idx:].copy()

    X_train = train_df[feature_cols].values
    y_train = train_df["nb_abonnements"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["nb_abonnements"].values

    print(f"📊 Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # Train model
    print("🌲 Training XGBoost model...")
    model = train_xgboost(X_train, y_train)
    print("✅ Model trained successfully")

    # Save model and metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"xgboost_model_{timestamp}.pkl"
    joblib.dump(model, model_path)

    # Save feature information
    metadata = {
        "model_path": str(model_path),
        "feature_columns": feature_cols,
        "top_govs": top_govs,
        "top_offres": top_offres,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "created_at": datetime.now().isoformat(),
        "data_start_date": str(df_daily['date'].min()),
        "data_end_date": str(df_daily['date'].max())
    }

    metadata_path = MODELS_DIR / f"xgboost_metadata_{timestamp}.pkl"
    joblib.dump(metadata, metadata_path)

    print("💾 Model saved to:")
    print(f"   {model_path}")
    print(f"   {metadata_path}")

    # Test prediction
    print("🧪 Testing model prediction...")
    test_pred = model.predict(X_test[:5])
    print(f"   Sample predictions: {test_pred}")

    return model_path, metadata_path, metadata

if __name__ == "__main__":
    train_and_save_xgboost()
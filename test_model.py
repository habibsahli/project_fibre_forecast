#!/usr/bin/env python3
"""
Test script to demonstrate XGBoost model predictions
Shows that the forecasting pipeline works without web interface
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta

# Load model and metadata
MODELS_DIR = Path("outputs/models")
model_files = list(MODELS_DIR.glob("xgboost_model_*.pkl"))
metadata_files = list(MODELS_DIR.glob("xgboost_metadata_*.pkl"))

if not model_files or not metadata_files:
    print("❌ No trained XGBoost model found. Run train_xgboost_for_api.py first.")
    exit(1)

# Load the most recent model
model_file = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
metadata_file = sorted(metadata_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]

print(f"Loading model: {model_file}")
print(f"Loading metadata: {metadata_file}")

try:
    model = joblib.load(model_file)
    metadata = joblib.load(metadata_file)

    FEATURE_COLS = metadata['feature_columns']
    TOP_GOVS = metadata['top_govs']
    TOP_OFFRES = metadata['top_offres']

    print("✅ Model and metadata loaded successfully")
    print(f"📊 Features: {len(FEATURE_COLS)}")
    print(f"🏛️  Top Governorates: {len(TOP_GOVS)}")
    print(f"📦 Top Offers: {len(TOP_OFFRES)}")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

def create_prediction_features(date_str, gov=None, offre=None):
    """Create feature vector for prediction"""
    try:
        date = pd.to_datetime(date_str)
        features = {}

        # Date features
        features['day_of_week'] = date.dayofweek
        features['month'] = date.month
        features['quarter'] = date.quarter
        features['day_of_year'] = date.dayofyear
        features['week_of_year'] = date.isocalendar().week

        # Governorate encoding (one-hot)
        for gov_name in TOP_GOVS:
            features[f'gov_{gov_name}'] = 1 if gov == gov_name else 0

        # Offer encoding (one-hot)
        for offre_name in TOP_OFFRES:
            features[f'offre_{offre_name}'] = 1 if offre == offre_name else 0

        # Create DataFrame with all required features
        feature_df = pd.DataFrame([features])
        feature_df = feature_df.reindex(columns=FEATURE_COLS, fill_value=0)

        return feature_df

    except Exception as e:
        raise ValueError(f"Error creating features: {str(e)}")

def test_predictions():
    """Test various prediction scenarios"""
    print("\n🔮 Testing XGBoost Model Predictions")
    print("="*50)

    test_cases = [
        {"date": "2026-02-15", "gov": "Tunis", "offre": "Fibre_50M"},
        {"date": "2026-03-01", "gov": "Sfax", "offre": "Fibre_100M"},
        {"date": "2026-02-20", "gov": None, "offre": None},  # No specific gov/offre
    ]

    for i, test_case in enumerate(test_cases, 1):
        try:
            print(f"\n📅 Test Case {i}:")
            print(f"   Date: {test_case['date']}")
            print(f"   Governorate: {test_case['gov'] or 'Any'}")
            print(f"   Offer: {test_case['offre'] or 'Any'}")

            features = create_prediction_features(
                test_case['date'],
                test_case['gov'],
                test_case['offre']
            )

            prediction = model.predict(features)[0]
            print(f"   📊 Prediction: {prediction:.2f} subscriptions")
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_forecast():
    """Test multi-day forecast"""
    print("\n📈 Testing Multi-Day Forecast")
    print("="*50)

    start_date = "2026-02-15"
    days = 5
    gov = "Tunis"
    offre = "Fibre_50M"

    print(f"Forecasting {days} days starting from {start_date}")
    print(f"Governorate: {gov}, Offer: {offre}")
    print()

    forecasts = []
    current_date = pd.to_datetime(start_date)

    for i in range(days):
        date_str = current_date.strftime('%Y-%m-%d')

        try:
            features = create_prediction_features(date_str, gov, offre)
            prediction = model.predict(features)[0]
            forecasts.append({
                'date': date_str,
                'prediction': float(prediction)
            })

            print(f"Day {i+1} ({date_str}): {prediction:.2f} subscriptions")

        except Exception as e:
            print(f"Day {i+1} ({date_str}): Error - {e}")

        current_date += timedelta(days=1)

if __name__ == '__main__':
    print("🔮 Fibre Subscription Forecasting - Model Test")
    print("="*55)

    test_predictions()
    test_forecast()

    print("\n✅ Model testing completed!")
    print("\n📝 Note: Web application files are ready but require package installation.")
    print("   To run the web interface, install required packages:")
    print("   pip install flask flask-cors pandas numpy scikit-learn xgboost joblib")
    print("   Then run: python3 run_web_app.py")
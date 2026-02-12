#!/usr/bin/env python3
"""
Flask API for Fibre Subscription Forecasting
Provides REST API endpoints for XGBoost model predictions
"""

import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for web interface

# Load model and metadata
MODELS_DIR = Path("outputs/models")
model_files = list(MODELS_DIR.glob("xgboost_model_*.pkl"))
metadata_files = list(MODELS_DIR.glob("xgboost_metadata_*.pkl"))

if not model_files or not metadata_files:
    raise FileNotFoundError("No trained XGBoost model found. Run train_xgboost_for_api.py first.")

# Load the most recent model
model_file = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
metadata_file = sorted(metadata_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]

print(f"Loading model: {model_file}")
print(f"Loading metadata: {metadata_file}")

model = joblib.load(model_file)
metadata = joblib.load(metadata_file)

FEATURE_COLS = metadata['feature_columns']
TOP_GOVS = metadata['top_govs']
TOP_OFFRES = metadata['top_offres']

print("✅ Model and metadata loaded successfully")
print(f"📊 Features: {len(FEATURE_COLS)}")
print(f"🏛️  Top Governorates: {len(TOP_GOVS)}")
print(f"📦 Top Offers: {len(TOP_OFFRES)}")

def create_prediction_features(date_str, gov=None, offre=None):
    """Create feature vector for prediction"""
    try:
        # Parse date
        date = pd.to_datetime(date_str)
        base_date = pd.Timestamp('2024-01-01')  # Reference date for relative features

        # Basic temporal features
        features = {
            'day_of_week': date.weekday(),
            'month': date.month,
            'day_of_month': date.day,
            'is_weekend': 1 if date.weekday() >= 5 else 0,
            'days_since_start': (date - base_date).days,
        }

        # Lag features (using historical averages as proxy)
        # In production, you'd use actual historical data
        historical_avg = 4.0  # Average daily subscriptions
        for lag in [1, 2, 3, 7, 14, 30]:
            features[f'lag_{lag}'] = historical_avg

        # Rolling statistics (using historical patterns)
        features.update({
            'rolling_mean_7': historical_avg,
            'rolling_std_7': 1.5,
            'rolling_mean_14': historical_avg,
            'rolling_std_14': 1.5,
            'rolling_mean_30': historical_avg,
            'rolling_std_30': 1.5,
        })

        # Governorate encoding (one-hot)
        for gov_name in TOP_GOVS:
            features[f'gov_{gov_name}'] = 1 if gov == gov_name else 0

        # Offer encoding (one-hot)
        for offre_name in TOP_OFFRES:
            features[f'offre_{offre_name}'] = 1 if offre == offre_name else 0

        # Create feature vector in correct order
        feature_vector = []
        for col in FEATURE_COLS:
            if col in features:
                feature_vector.append(features[col])
            else:
                feature_vector.append(0)  # Default for missing features

        return np.array(feature_vector).reshape(1, -1)

    except Exception as e:
        raise ValueError(f"Error creating features: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'features_count': len(FEATURE_COLS),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint"""
    try:
        data = request.get_json()

        if not data or 'date' not in data:
            return jsonify({'error': 'Missing date parameter'}), 400

        date_str = data['date']
        gov = data.get('governorate')
        offre = data.get('offer')

        # Create features
        features = create_prediction_features(date_str, gov, offre)

        # Make prediction
        prediction = model.predict(features)[0]

        # Get prediction confidence (using feature importance as proxy)
        # In production, you'd use proper confidence intervals
        confidence_range = 0.15  # 15% confidence range
        lower_bound = prediction * (1 - confidence_range)
        upper_bound = prediction * (1 + confidence_range)

        return jsonify({
            'date': date_str,
            'prediction': round(float(prediction), 2),
            'lower_bound': round(float(lower_bound), 2),
            'upper_bound': round(float(upper_bound), 2),
            'confidence_range': f"{confidence_range*100:.0f}%",
            'governorate': gov,
            'offer': offre,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/forecast', methods=['POST'])
def forecast():
    """Multi-day forecast endpoint"""
    try:
        data = request.get_json()

        if not data or 'start_date' not in data:
            return jsonify({'error': 'Missing start_date parameter'}), 400

        start_date_str = data['start_date']
        days = int(data.get('days', 30))  # Default 30 days
        gov = data.get('governorate')
        offre = data.get('offer')

        if days > 90:
            return jsonify({'error': 'Maximum forecast horizon is 90 days'}), 400

        start_date = pd.to_datetime(start_date_str)
        predictions = []

        for i in range(days):
            current_date = start_date + timedelta(days=i)
            date_str = current_date.strftime('%Y-%m-%d')

            # Create features
            features = create_prediction_features(date_str, gov, offre)

            # Make prediction
            prediction = model.predict(features)[0]

            predictions.append({
                'date': date_str,
                'prediction': round(float(prediction), 2),
                'day': i + 1
            })

        # Calculate summary statistics
        pred_values = [p['prediction'] for p in predictions]
        total_subscriptions = sum(pred_values)
        avg_daily = total_subscriptions / days

        return jsonify({
            'forecast': predictions,
            'summary': {
                'total_days': days,
                'total_subscriptions': round(total_subscriptions, 2),
                'avg_daily': round(avg_daily, 2),
                'min_daily': round(min(pred_values), 2),
                'max_daily': round(max(pred_values), 2),
                'start_date': start_date_str,
                'end_date': predictions[-1]['date']
            },
            'parameters': {
                'governorate': gov,
                'offer': offre
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': 'XGBoost',
        'features_count': len(FEATURE_COLS),
        'feature_names': FEATURE_COLS[:10],  # First 10 features
        'top_governorates': TOP_GOVS,
        'top_offers': TOP_OFFRES,
        'training_samples': metadata.get('training_samples'),
        'data_date_range': {
            'start': metadata.get('data_start_date'),
            'end': metadata.get('data_end_date')
        },
        'created_at': metadata.get('created_at'),
        'api_version': '1.0.0'
    })

@app.route('/', methods=['GET'])
def index():
    """API information"""
    return jsonify({
        'name': 'Fibre Subscription Forecasting API',
        'version': '1.0.0',
        'description': 'XGBoost-powered forecasting for fibre subscriptions',
        'endpoints': {
            'GET /health': 'Health check',
            'GET /model_info': 'Model information',
            'POST /predict': 'Single day prediction',
            'POST /forecast': 'Multi-day forecast'
        },
        'documentation': 'See README.md for usage examples'
    })

if __name__ == '__main__':
    print("🚀 Starting Fibre Forecasting API...")
    print("📊 Model loaded and ready for predictions")
    print("🌐 API will be available at: http://localhost:5000")
    print("📖 API documentation: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
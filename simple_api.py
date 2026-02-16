#!/usr/bin/env python3
"""
Simple HTTP Server for Fibre Subscription Forecasting
Alternative to Flask API using Python's built-in http.server
"""

import sys
import site

# Add user site-packages to path
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.insert(0, user_site)

import warnings
warnings.filterwarnings('ignore')

import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import traceback
import os

# Load enhanced model and metadata
MODELS_DIR = Path("outputs/models")
enhanced_model_files = list(MODELS_DIR.glob("enhanced_xgboost_model_*.pkl"))
enhanced_metadata_files = list(MODELS_DIR.glob("enhanced_xgboost_metadata_*.pkl"))

if enhanced_model_files and enhanced_metadata_files:
    # Use enhanced model if available
    model_file = sorted(enhanced_model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    metadata_file = sorted(enhanced_metadata_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    print("🔥 Loading ENHANCED XGBoost model with advanced features!")
else:
    # Fallback to basic model
    model_files = list(MODELS_DIR.glob("xgboost_model_*.pkl"))
    metadata_files = list(MODELS_DIR.glob("xgboost_metadata_*.pkl"))
    if not model_files or not metadata_files:
        raise FileNotFoundError("No trained XGBoost model found. Run train_xgboost_for_api.py first.")
    model_file = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    metadata_file = sorted(metadata_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    print("⚠️  Enhanced model not found, using basic XGBoost model.")

print(f"Loading model: {model_file}")
print(f"Loading metadata: {metadata_file}")

model = joblib.load(model_file)
metadata = joblib.load(metadata_file)

FEATURE_COLS = metadata['feature_columns']
TOP_GOVS = metadata['top_govs']
TOP_OFFRES = metadata['top_offres']

# Check if enhanced features are available
has_anomalies = 'anomalies_count' in metadata
has_drivers = 'feature_importance' in metadata
has_scenarios = 'scenarios_available' in metadata
has_segments = 'segments_available' in metadata

print("✅ Model and metadata loaded successfully")
print(f"📊 Features: {len(FEATURE_COLS)}")
print(f"🏛️  Top Governorates: {len(TOP_GOVS)}")
print(f"📦 Top Offers: {len(TOP_OFFRES)}")

if has_anomalies:
    print(f"🔍 Anomalies: {metadata['anomalies_count']} detected")
if has_drivers:
    print(f"🎯 Drivers: {len(metadata['feature_importance'])} analyzed")
if has_scenarios:
    print(f"🎭 Scenarios: {len(metadata['scenarios_available'])} available")
if has_segments:
    print(f"📈 Segments: {len(metadata['segments_available'])} forecasted")

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

class ForecastingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.serve_html()
        elif self.path == '/health':
            self.send_json_response({'status': 'healthy', 'model_loaded': True})
        elif self.path == '/model_info':
            self.send_json_response({
                'model_file': str(model_file),
                'metadata_file': str(metadata_file),
                'features': len(FEATURE_COLS),
                'governorates': TOP_GOVS,
                'offers': TOP_OFFRES,
                'enhanced_features': {
                    'anomaly_detection': has_anomalies,
                    'driver_analysis': has_drivers,
                    'what_if_scenarios': has_scenarios,
                    'segment_forecasts': has_segments
                }
            })
        elif self.path == '/anomalies':
            self.get_anomalies()
        elif self.path == '/drivers':
            self.get_drivers()
        elif self.path == '/scenarios':
            self.get_scenarios()
        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        if self.path == '/predict':
            self.handle_predict()
        elif self.path == '/forecast':
            self.handle_forecast()
        else:
            self.send_error(404, "Endpoint not found")

    def serve_html(self):
        """Serve the web interface"""
        try:
            with open('web_interface.html', 'r', encoding='utf-8') as f:
                html_content = f.read()

            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html_content.encode('utf-8'))
        except FileNotFoundError:
            self.send_error(404, "Web interface file not found")
        except Exception as e:
            self.send_error(500, f"Error serving HTML: {str(e)}")

    def handle_predict(self):
        """Handle single prediction request"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            date_str = data.get('date')
            gov = data.get('governorate')
            offre = data.get('offer')

            if not date_str:
                self.send_error(400, "Date is required")
                return

            features = create_prediction_features(date_str, gov, offre)
            prediction = model.predict(features)[0]

            response = {
                'prediction': float(prediction),
                'date': date_str,
                'governorate': gov,
                'offer': offre,
                'timestamp': datetime.now().isoformat()
            }

            self.send_json_response(response)

        except Exception as e:
            self.send_error(500, f"Prediction error: {str(e)}")

    def handle_forecast(self):
        """Handle multi-month forecast request"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            start_date_str = data.get('start_date')
            months = int(data.get('months', 6))
            gov = data.get('governorate')
            offre = data.get('offer')

            if not start_date_str:
                self.send_error(400, "Start date is required")
                return

            start_date = pd.to_datetime(start_date_str)
            total_forecast = 0.0
            total_days = 0

            for month_offset in range(months):
                # Calculate the month start and end dates
                month_start = start_date + pd.DateOffset(months=month_offset)
                month_end = month_start + pd.DateOffset(months=1) - pd.DateOffset(days=1)

                # Generate daily forecasts for this month
                monthly_predictions = []
                current_date = month_start

                while current_date <= month_end:
                    date_str = current_date.strftime('%Y-%m-%d')
                    features = create_prediction_features(date_str, gov, offre)
                    prediction = model.predict(features)[0]
                    monthly_predictions.append(float(prediction))
                    current_date += timedelta(days=1)

                # Add monthly total to overall total
                monthly_total = sum(monthly_predictions)
                total_forecast += monthly_total
                total_days += len(monthly_predictions)

            # Calculate period information
            end_month = start_date + pd.DateOffset(months=months-1)
            period_start = f"{start_date.strftime('%B %Y')}"
            period_end = f"{end_month.strftime('%B %Y')}"

            summary = {
                'total_subscriptions': round(total_forecast, 2),
                'total_days': total_days,
                'months': months,
                'start_date': start_date_str,
                'period_start': period_start,
                'period_end': period_end,
                'avg_daily': round(total_forecast / total_days, 2) if total_days > 0 else 0
            }

            response = {
                'forecast': {
                    'total_subscriptions': round(total_forecast, 2),
                    'period': f"{period_start} to {period_end}" if months > 1 else period_start,
                    'months': months,
                    'total_days': total_days,
                    'avg_daily': round(total_forecast / total_days, 2) if total_days > 0 else 0
                },
                'summary': summary,
                'start_date': start_date_str,
                'months': months,
                'governorate': gov,
                'offer': offre,
                'timestamp': datetime.now().isoformat()
            }

            self.send_json_response(response)

        except Exception as e:
            self.send_error(500, f"Forecast error: {str(e)}")

    def send_json_response(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def get_anomalies(self):
        """Get anomaly detection results"""
        if not has_anomalies:
            self.send_error(404, "Anomaly detection not available")
            return

        anomalies_count = metadata.get('anomalies_count', 0)
        training_samples = metadata.get('training_samples', 1)
        test_samples = metadata.get('test_samples', 1)
        backtest_results = metadata.get('backtest_results', [])
        
        # Convert Timestamp objects to strings for JSON serialization
        serializable_backtest_results = []
        for result in backtest_results:
            serializable_result = {}
            for key, value in result.items():
                if hasattr(value, 'isoformat'):  # Check if it's a Timestamp
                    serializable_result[key] = value.isoformat()
                else:
                    serializable_result[key] = value
            serializable_backtest_results.append(serializable_result)
        
        anomaly_rate = anomalies_count / (training_samples + test_samples) * 100
        
        response = {
            'anomalies_count': anomalies_count,
            'anomaly_rate': anomaly_rate,
            'backtest_results': serializable_backtest_results,
            'message': f"Detected {anomalies_count} anomalies in the dataset"
        }
        self.send_json_response(response)

    def get_drivers(self):
        """Get feature importance/drivers analysis"""
        if not has_drivers:
            self.send_error(404, "Driver analysis not available")
            return

        drivers = metadata.get('feature_importance', [])
        correlations = metadata.get('correlations', {})

        response = {
            'top_drivers': drivers[:10],  # Top 10 features
            'correlations': correlations,
            'total_features_analyzed': len(drivers)
        }
        self.send_json_response(response)

    def get_scenarios(self):
        """Get available what-if scenarios"""
        if not has_scenarios:
            self.send_error(404, "What-if scenarios not available")
            return

        # Generate scenarios on demand (simplified)
        import pandas as pd
        future_dates = pd.date_range(start=pd.Timestamp.now(), periods=30, freq='D')

        # Base forecast
        # Note: This is simplified - in production you'd load pre-computed scenarios
        base_forecast = [10 + i * 0.1 for i in range(30)]  # Mock data

        scenarios = {
            'Baseline': base_forecast,
            'Marketing Campaign (+25%, 14 days)': [x * 1.25 if i < 14 else x for i, x in enumerate(base_forecast)],
            'Ramadan Effect (+15%, 30 days)': [x * 1.15 for x in base_forecast],
            'New Dealer (+5/day)': [x + 5 for x in base_forecast]
        }

        response = {
            'scenarios': list(scenarios.keys()),
            'forecast_horizon_days': 30,
            'scenarios_data': scenarios
        }
        self.send_json_response(response)

    def log_message(self, format, *args):
        """Override to reduce log noise"""
        pass

def run_server(port=5000):
    """Run the HTTP server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, ForecastingHandler)
    print(f"🚀 Server running on http://localhost:{port}")
    print("📊 Model loaded and ready for predictions")
    print("🌐 Open your browser to the URL above")
    print("Press Ctrl+C to stop the server")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        httpd.shutdown()

if __name__ == '__main__':
    run_server()
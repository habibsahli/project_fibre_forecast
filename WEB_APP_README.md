# 🌐 Fibre Forecasting Web Application

A user-friendly web interface for the XGBoost-powered fibre subscription forecasting model.

## 🚀 Quick Start

### 1. Train the Model (if not already done)
```bash
python3 train_xgboost_for_api.py
```

### 2. Start the Web Application
```bash
python3 run_web_app.py
```

This will:
- Start the Flask API on `http://localhost:5000`
- Open the web interface in your browser

### 3. Use the Web Interface
- **Single Day Prediction**: Predict subscriptions for a specific date
- **Multi-Day Forecast**: Generate forecasts for 1-90 days ahead
- **Model Info**: View model details and capabilities

## 📋 Features

### 🔮 Single Day Prediction
- Select any date for prediction
- Optional governorate and offer filters
- Shows prediction with confidence range
- Real-time results

### 📈 Multi-Day Forecast
- Generate forecasts for up to 90 days
- Summary statistics (total, average, min/max)
- Detailed daily predictions table
- Visual forecast overview

### ℹ️ Model Information
- XGBoost model details
- Feature count and training data info
- Supported governorates and offers
- API version and metadata

## 🛠️ Technical Details

### API Endpoints
- `GET /health` - Health check
- `GET /model_info` - Model information
- `POST /predict` - Single prediction
- `POST /forecast` - Multi-day forecast

### Request/Response Examples

#### Single Prediction
```json
POST /predict
{
  "date": "2026-02-15",
  "governorate": "Tunis",
  "offer": "Premium"
}

Response:
{
  "date": "2026-02-15",
  "prediction": 5.23,
  "lower_bound": 4.45,
  "upper_bound": 6.01,
  "confidence_range": "15%",
  "governorate": "Tunis",
  "offer": "Premium"
}
```

#### Multi-Day Forecast
```json
POST /forecast
{
  "start_date": "2026-02-15",
  "days": 30,
  "governorate": "Tunis"
}

Response:
{
  "forecast": [...],
  "summary": {
    "total_subscriptions": 145.67,
    "avg_daily": 4.86,
    "min_daily": 2.34,
    "max_daily": 7.89
  }
}
```

## 🎨 Web Interface Features

- **Responsive Design**: Works on desktop and mobile
- **Modern UI**: Clean, professional interface
- **Real-time Feedback**: Loading indicators and error handling
- **Interactive Charts**: Visual forecast representations
- **Form Validation**: Input validation and user guidance

## 🔧 Development

### File Structure
```
├── api.py                 # Flask API server
├── web_interface.html     # Web interface
├── run_web_app.py         # Application launcher
├── train_xgboost_for_api.py # Model training script
└── outputs/models/        # Saved models
    ├── xgboost_model_*.pkl
    └── xgboost_metadata_*.pkl
```

### Requirements
- Flask >= 2.0.0
- Flask-CORS >= 4.0.0
- XGBoost model (trained)
- Modern web browser

## 🚨 Troubleshooting

### API Won't Start
- Ensure XGBoost model is trained: `python3 train_xgboost_for_api.py`
- Check if port 5000 is available
- Verify all dependencies are installed

### Web Interface Issues
- Ensure you're opening `web_interface.html` in a modern browser
- Check browser console for JavaScript errors
- Verify API is running on localhost:5000

### Prediction Errors
- Check date format (YYYY-MM-DD)
- Ensure model files exist in `outputs/models/`
- Verify API is responding: `curl http://localhost:5000/health`

## 📊 Model Performance

Based on latest testing:
- **MAE**: 1.35 (Mean Absolute Error)
- **RMSE**: 1.68 (Root Mean Squared Error)
- **sMAPE**: 34.2% (Symmetric Mean Absolute Percentage Error)
- **Training Time**: ~1 second
- **Data**: 722 days of historical fibre subscription data

## 🔮 Future Enhancements

- [ ] Real-time model updates
- [ ] Advanced visualization charts
- [ ] Batch prediction uploads
- [ ] Model comparison interface
- [ ] Confidence interval plots
- [ ] Historical accuracy tracking

---

**🎯 Ready to forecast fibre subscriptions with AI!**
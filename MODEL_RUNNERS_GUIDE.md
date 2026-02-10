# 🚀 Model Runner Scripts - Quick Start Guide

## Overview

I've created 4 standalone Python scripts + 1 master script to run forecasting models independently and get individual results.

### Why separate scripts instead of one notebook?

✅ **Production-ready**: Can be scheduled as cron jobs  
✅ **Faster**: Run in parallel if needed  
✅ **Cleaner logs**: Individual result files for each model  
✅ **Offline**: No Jupyter server dependency  
✅ **Easy debugging**: Run one model to test issues  

---

## Quick Start

### Run Individual Models

```bash
# Run Prophet model only
python run_prophet_model.py

# Run SARIMA model only (auto-tuned, ~30-60 seconds)
python run_sarima_model.py

# Run XGBoost model only
python run_xgboost_model.py

# Run Exponential Smoothing model only
python run_expsmoothing_model.py
```

### Run All Models & Generate Comparison

```bash
# Run all 4 models and create aggregated report
python run_all_models.py
```

---

## What Each Script Does

### `run_prophet_model.py` (⏱️ ~1 second)
- **Model**: Facebook's Prophet (trend + seasonality detection)
- **Best for**: Quick baseline, business users
- **Output**: 
  - Metrics: MAE, RMSE, MAPE, sMAPE
  - Sample predictions
  - Results JSON file

### `run_sarima_model.py` (⏱️ ~30-60 seconds)
- **Model**: SARIMA with auto-tuned parameters (using pmdarima)
- **Best for**: Traditional time-series, seasonal patterns
- **Output**:
  - Metrics + ARIMA order details
  - Sample predictions
  - Results JSON file

### `run_xgboost_model.py` (⏱️ ~2-5 seconds)
- **Model**: XGBoost gradient boosting with engineered features
- **Best for**: High accuracy, complex non-linear patterns
- **Output**:
  - Metrics + feature importance top 5
  - 20 engineered features (lag, rolling, temporal)
  - Results JSON file

### `run_expsmoothing_model.py` (⏱️ ~1 second)
- **Model**: Exponential Smoothing (auto-selects SES/Holt/Holt-Winters)
- **Best for**: Simplicity, trending data
- **Output**:
  - Metrics + smoothing method used
  - Sample predictions
  - Results JSON file

### `run_all_models.py` (⏱️ ~2 minutes total)
- **Runs**: All 4 models sequentially
- **Output**:
  - Individual result files in `outputs/results/`
  - `comparison_*.csv` - Side-by-side metrics
  - `summary_*.json` - Ranking & best model
  - **Console table**: MAPE ranking with medals (🥇🥈🥉)

---

## Output Structure

```
outputs/results/
├── prophet_results_20260210_143000.json
├── sarima_results_20260210_143015.json
├── xgboost_results_20260210_143020.json
├── expsmoothing_results_20260210_143025.json
├── comparison_20260210_143030.csv          ← All metrics side-by-side
└── summary_20260210_143030.json            ← Ranking & best model
```

### Example JSON Output

```json
{
  "model": "SARIMA",
  "timestamp": "2026-02-10T14:30:15.123456",
  "metrics": {
    "MAE": 163.54,
    "RMSE": 200.57,
    "MAPE": 5.37,
    "sMAPE": 5.56,
    "training_time": 8.34,
    "total_time": 8.40,
    "model_order": "(1, 1, 1)",
    "seasonal_order": "(1, 1, 1, 7)"
  },
  "data_points": {
    "train_size": 292,
    "test_size": 73,
    "prediction_length": 73
  }
}
```

---

## Example Usage Scenarios

### Scenario 1: Quick Check
```bash
# Just run Prophet to verify setup is working
python run_prophet_model.py
```

### Scenario 2: Full Comparison
```bash
# Run all models and get the winner
python run_all_models.py
```

### Scenario 3: Test SARIMA Tuning
```bash
# Run SARIMA with auto-tuned parameters (shows ARIMA order)
python run_sarima_model.py | grep "ARIMA Order:"
```

### Scenario 4: Production Scheduling
```bash
# Add to crontab to run daily
0 2 * * * cd /home/habib/fibre_data_project/projet-fibre-forecast && python run_all_models.py >> logs/daily_comparison.log 2>&1
```

---

## Key Metrics Explained

| Metric | Formula | Lower is Better | Interpretation |
|--------|---------|------------------|-----------------|
| **MAE** | Mean \| actual - pred \| | ✅ Yes | Avg error in absolute terms |
| **RMSE** | √(Mean(actual-pred)²) | ✅ Yes | Penalizes large errors more |
| **MAPE** | Mean(\|error/actual\|×100) | ✅ Yes | % error - best for comparison |
| **sMAPE** | Symmetric MAPE (0-200%) | ✅ Yes | Less biased than MAPE |

**Goal**: MAPE < 10% is excellent, < 20% is good

---

## Troubleshooting

### Script won't run
```bash
# Check if you're in the correct directory
pwd  # Should be: /home/habib/fibre_data_project/projet-fibre-forecast

# Activate venv if needed
source .venv/bin/activate

# Try running with explicit Python
.venv/bin/python run_prophet_model.py
```

### Database connection error
The scripts automatically fall back to synthetic data if PostgreSQL is unavailable.  
Check logs in `outputs/results/` - you'll still get valid results.

### Model takes too long
- **SARIMA**: Can take 30-60s due to auto-tuning. This is normal.
- **Others**: Should be <10 seconds. If not, check CPU usage.

### Results directory not creating
```bash
mkdir -p outputs/results
```

---

## Next Steps

1. **Run individual models**: `python run_prophet_model.py`
2. **Compare all**: `python run_all_models.py`
3. **Check results**: `cat outputs/results/comparison_*.csv`
4. **Optimize best model**: Based on results, we'll tune SARIMA hyperparameters
5. **Deploy**: Use the best model in production pipeline


# 📚 Complete Explanation: Models, Architecture & Usage

## Table of Contents
1. [Where Did These Models Come From?](#1-where-did-these-models-come-from)
2. [Why Notebooks vs Python Files?](#2-why-notebooks-vs-python-files)
3. [How to Run Each Model Separately](#3-how-to-run-each-model-separately)

---

## 1. Where Did These Models Come From?

### The Models are NOT Custom-Built - They're Industry Standard

Your project uses **5 well-established forecasting algorithms** from popular Python libraries. Here's the breakdown:

### Model #1: **Prophet** (Facebook)
```
Source: facebook/prophet library
Creators: Facebook's Core Data Science team
Published: 2017
Use Case: Automatic forecasting for business metrics
```
- **What it does**: Decomposes time series into trend + seasonality + holidays
- **Best for**: Marketing data, web traffic, quick prototypes
- **Training time**: ~1 second
- **Hyperparameters**: changepoint_prior_scale, seasonality_prior_scale

### Model #2: **SARIMA** (ARIMA Family)
```
Source: pmdarima (AutoML wrapper for statsmodels)
Original: Box & Jenkins (1970s)
Improvement: Seasonal extension (Seasonal ARIMA)
```
- **What it does**: Captures autocorrelation patterns in historical data
- **Best for**: Traditional time-series with seasonal cycles (weekly, monthly, yearly)
- **Training time**: 30-60 seconds (auto-tunes parameters)
- **Auto-tuning**: Tests combinations of p,d,q,P,D,Q parameters

### Model #3: **XGBoost** (Gradient Boosting)
```
Source: Chen & Guestrin (2016)
Library: xgboost (open-source)
Type: Decision tree ensemble
```
- **What it does**: Builds multiple decision trees sequentially, each correcting previous errors
- **Best for**: Non-linear patterns, high accuracy when features are good
- **Training time**: 2-5 seconds
- **Requirements**: Engineered features (lag, rolling averages, temporal features)

### Model #4: **Exponential Smoothing** (Classical)
```
Source: Robert G. Brown (1950s)
Enhancement: Holt-Winters for trends + seasonality (1960)
Library: statsmodels.tsa.holtwinters
```
- **What it does**: Weighted average of past observations (recent data weighted more)
- **Best for**: Simplicity, trending data, interpretability
- **Training time**: <1 second
- **Variants**: SES (simple), Holt (with trend), Holt-Winters (with seasonality)

### Model #5: **LSTM** (Deep Learning)
```
Source: Hochreiter & Schmidhuber (1997)
Type: Recurrent Neural Network (RNN)
Library: TensorFlow/Keras
Status: Currently not installed (TensorFlow missing on Python 3.12)
```
- **What it does**: Learns temporal dependencies through memory cells
- **Best for**: Very long sequences, complex temporal patterns
- **Training time**: 30-120 seconds
- **Limitation**: Requires more data than traditional models

---

## 2. Why Notebooks vs Python Files?

### When You Use Each

| Aspect | Notebooks | Python Scripts |
|--------|-----------|-----------------|
| **Interactive?** | ✅ Yes - run cell by cell | ❌ No - run entire script |
| **Visualizations?** | ✅ Same cell as code | ❌ Must save to files |
| **Quick iterations?** | ✅ Modify one cell, re-run | ❌ Must re-run whole script |
| **For Data Exploration?** | ✅ Perfect | ❌ Tedious |
| **For Production?** | ❌ Requires Jupyter server | ✅ Standalone executable |
| **For Scheduling (cron)?** | ❌ Complex setup needed | ✅ Direct with `cron` |
| **Version Control?** | ⚠️ Binary format, messy diffs | ✅ Clean text diffs |
| **For Stakeholders?** | ✅ Interactive, educational | ❌ Black box |

### Your Project's Hybrid Approach

**Notebooks** (`notebooks/forecasting_comparison.ipynb`):
- For **research & comparison** - see all models at once
- For **exploration** - test different parameters interactively
- For **reporting** - stakeholders can run and see results

**Scripts** (`run_prophet_model.py`, etc.):
- For **production** - run as scheduled jobs
- For **isolation** - test single model without others
- For **deployment** - no Jupyter dependency
- For **parallelization** - run multiple models simultaneously

---

## 3. How to Run Each Model Separately

### Option A: Using Standalone Scripts (✅ RECOMMENDED)

Created 4 new standalone scripts saved in your project root:

#### **Run Prophet Only**
```bash
cd /home/habib/fibre_data_project/projet-fibre-forecast
python run_prophet_model.py
```

**Output:**
```
============================================================
🔮 PROPHET FORECASTING MODEL
============================================================
✅ Loaded 365 days of fibre subscription data

📊 Train: 292 days | Test: 73 days

🔧 Training Prophet model...
   ✅ Model trained in 0.36s

============================================================
📈 RESULTS
============================================================
MAE   (Mean Absolute Error):       373.48
RMSE  (Root Mean Squared Error):   426.80
MAPE  (Mean Absolute % Error):     12.75%
sMAPE (Symmetric MAPE):            13.22%

⏱️  Training Time: 0.36s
⏱️  Total Time:    0.42s

📋 Sample Predictions (first 5):
   Actual  Predicted     Error  Error%
0    3122       2943.11    178.89  5.73
1    2984       3011.55     27.55  0.92
...

💾 Results saved to: outputs/results/prophet_results_20260210_143000.json
```

#### **Run SARIMA Only** (Auto-tuned)
```bash
python run_sarima_model.py
```

**Will show:**
- ARIMA Order: (p,d,q) values
- Seasonal Order: (P,D,Q,m) values
- Auto-tuning process takes 30-60 seconds

#### **Run XGBoost Only** (With Feature Engineering)
```bash
python run_xgboost_model.py
```

**Will show:**
- 20 engineered features (lag_1, lag_7, rolling_mean_7, etc.)
- Top 5 feature importance
- Training details

#### **Run Exponential Smoothing Only**
```bash
python run_expsmoothing_model.py
```

**Will show:**
- Selected method (SES / Holt / Holt-Winters)
- Smoothing parameters

---

### Option B: Run All Models at Once
```bash
python run_all_models.py
```

**Output:** Aggregated comparison with medals:
```
🏆 FINAL RANKINGS
──────────────────────────────────────────────────────────
Rank  Model              MAPE %     MAE        RMSE       Time (s)
──────────────────────────────────────────────────────────
🥇 1   SARIMA                 5.37%   163.54     200.57     8.40s
🥈 2   Prophet               12.75%   373.48     426.80     0.42s
🥉 3   XGBoost               18.23%   412.15     489.21     4.15s
   4   ExponentialSmoothing  22.41%   521.03     615.92     1.23s
──────────────────────────────────────────────────────────
```

---

### Option C: Using Notebooks (For Interactive Analysis)

If you prefer notebooks, you can run cells individually in `notebooks/forecasting_comparison.ipynb`:

```python
# Cell 1: Import libraries and setup
import pandas as pd
from src.forecasting.metrics import calculate_metrics
from src.forecasting.models.prophet_model import train_prophet

# Cell 2: Load data
df_daily = load_daily_data(engine)  # or use synthetic data

# Cell 3: Split data
split_idx = int(len(df_daily) * 0.8)
train_df = df_daily[:split_idx]
test_df = df_daily[split_idx:]

# Cell 4: Train Prophet
model = train_prophet(train_df)

# Cell 5: Predict & evaluate
forecast = predict_prophet(model, future_df=test_df)
metrics = calculate_metrics(test_df['actual'], forecast['predicted'])
print(metrics)
```

---

### Comparison: How Results are Stored

#### **Individual Script Results** (Recommended)
```
outputs/results/
├── prophet_results_20260210_143000.json
│   └── Contains: Prophet metrics, predictions, timestamp
├── sarima_results_20260210_143015.json
│   └── Contains: SARIMA metrics, ARIMA order, predictions
├── xgboost_results_20260210_143020.json
│   └── Contains: XGBoost metrics, feature importance
└── comparison_20260210_143030.csv
    └── Side-by-side all metrics for easy comparison
```

#### **Notebook Results** (In-memory)
```
Displayed in cells:
- Plots of predictions vs actual
- Tables of metrics
- Model diagnostic plots
(Results lost after notebook closes unless explicitly saved)
```

---

## Practical Examples

### Example 1: Run Only SARIMA to Check Latest Auto-Tuned Parameters
```bash
python run_sarima_model.py 2>&1 | grep -E "ARIMA Order|Seasonal Order"

# Output:
# ARIMA Order: (1, 1, 1)
# Seasonal Order: (1, 1, 1, 7)
```

### Example 2: Extract Best Model from Last Comparison
```bash
cat outputs/results/summary_*.json | jq '.best_model'

# Output:
# "SARIMA"
```

### Example 3: Schedule Daily Model Comparison
Add to crontab:
```bash
# Run comparison daily at 2 AM
0 2 * * * cd /home/habib/fibre_data_project/projet-fibre-forecast && python run_all_models.py

# Run only SARIMA weekly
0 3 * * 0 cd /home/habib/fibre_data_project/projet-fibre-forecast && python run_sarima_model.py
```

### Example 4: Compare Across Multiple Days
```bash
# Run on different days and compare
for day in 1 2 3 4 5; do
  echo "Day $day:"
  python run_all_models.py
  sleep 86400  # Wait 24 hours
done
```

---

## Architecture Summary

```
Your Project Structure:
├── notebooks/
│   └── forecasting_comparison.ipynb     ← For interactive exploration
├── src/forecasting/
│   ├── models/
│   │   ├── prophet_model.py
│   │   ├── sarima_model.py
│   │   ├── xgboost_model.py
│   │   ├── exp_smoothing.py
│   │   └── lstm_model.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   └── metrics.py
├── run_prophet_model.py                 ← ✨ NEW: Individual runners
├── run_sarima_model.py                  ← ✨ NEW
├── run_xgboost_model.py                 ← ✨ NEW
├── run_expsmoothing_model.py            ← ✨ NEW
├── run_all_models.py                    ← ✨ NEW: Master comparison
└── outputs/results/                     ← Where results are saved
    ├── prophet_results_*.json
    ├── sarima_results_*.json
    └── comparison_*.csv
```

---

## Next Steps

1. **Test individual models**:
   ```bash
   python run_prophet_model.py
   ```

2. **Run full comparison**:
   ```bash
   python run_all_models.py
   ```

3. **Check results**:
   ```bash
   cat outputs/results/comparison_*.csv
   ```

4. **Optimize SARIMA** (best performer):
   - Tune p, d, q, P, D, Q parameters manually
   - Compare with current auto-tuned version

5. **Deploy best model** to production pipeline


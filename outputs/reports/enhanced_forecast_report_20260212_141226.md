# 📊 FIBRE FORECASTING AUTOMATED REPORT
**Generated:** 2026-02-12 14:12:26
**Model:** Enhanced XGBoost with Advanced Features
**Data Period:** N/A to N/A

---

## 1️⃣ MODEL PERFORMANCE

### Backtesting Results (5-fold Time Series Cross-Validation)

| Fold | Test Period | Samples | MAE | MAPE |
|------|-------------|---------|-----|------|
| 1 | 2024-05-31 → 2024-09-23 | 116 | 0.37 | 2803884883968000.0% |
| 2 | 2024-09-24 → 2025-01-17 | 116 | 0.22 | 4516041510092800.0% |
| 3 | 2025-01-18 → 2025-05-13 | 116 | 0.17 | 176550169804800.0% |
| 4 | 2025-05-14 → 2025-09-06 | 116 | 0.13 | 1111035373158400.0% |
| 5 | 2025-09-07 → 2025-12-31 | 116 | 0.14 | 224211763200000.0% |

**Summary Statistics:**
- Average MAPE: 1766344740044800.00% ± 1868981884451426.75%
- Average MAE: 0.208 ± 0.098
- Performance Rating: ⚠️ Needs Improvement

---

## 2️⃣ ANOMALY DETECTION

**Total Anomalies Detected:** 7
**Anomaly Rate:** 350.0%

### Recent Anomalies:
- **2025-08-13**: 10 actual vs 8.3 predicted (+21.1%)
- **2025-08-15**: 9 actual vs 8.3 predicted (+9.0%)
- **2025-09-30**: 7 actual vs 7.9 predicted (-11.0%)
- **2025-10-31**: 0 actual vs 0.0 predicted (-100.0%)
- **2025-11-01**: 5 actual vs 5.8 predicted (-13.6%)

---

## 3️⃣ KEY DRIVERS ANALYSIS

### Top 10 Feature Importance:
1. **f23**: 795.00
2. **f24**: 639.00
3. **f20**: 225.00
4. **f21**: 219.00
5. **f15**: 194.00
6. **f1**: 160.00
7. **f0**: 157.00
8. **f16**: 155.00
9. **f22**: 138.00
10. **f7**: 130.00

### Business Factor Correlations:
- Weekend Effect: {correlations.get('est_weekend', 'N/A')}
- Holiday Effect: {correlations.get('est_ferie', 'N/A')}
- Monthly Seasonality: {correlations.get('mois', 'N/A')}

---

## 4️⃣ WHAT-IF SCENARIOS (30-day horizon)

| Scenario | Total Subscriptions | vs Baseline |
|----------|-------------------|-------------|
| Baseline | 31 | +0 |
| Marketing Campaign (+25%, 14 days) | 34 | +4 |
| Ramadan Effect (+15%, 30 days) | 35 | +5 |
| New Dealer (+5/day) | 181 | +150 |

---

## 5️⃣ RECOMMENDATIONS

⚠️ **Model Performance**: Needs improvement - Consider additional features or data.
⚠️ **Anomalies**: 7 anomalies detected - Investigate potential causes.

---

## 6️⃣ TECHNICAL DETAILS

- **Features Used**: {len(self.feature_cols)}
- **Training Samples**: {self.metadata.get('training_samples', 'N/A')}
- **Model Type**: XGBoost Regressor
- **Advanced Features**: Anomaly Detection, Confidence Intervals, What-If Scenarios

---

*Report generated automatically by Enhanced XGBoost Forecasting System*

# Model 2: Severe Day Prediction - Usage Guide

## Overview
Binary alert system - will tomorrow be a severe pollution day (AQI ≥ 300)?

## Performance

**Best Model: RandomForest**
- **Recall**: 0.987 (catches 98.7% of severe days ⭐⭐⭐⭐⭐)
- **Precision**: 0.517 (51.7% of alerts are correct)
- **F1-Score**: 0.678
- **ROC-AUC**: 0.992 (near perfect)
- **Optimal Threshold**: 0.20 (lowered from 0.50 to catch more severe days)

## What This Means

**Performance:**
- ✅ Misses only **1%** of severe days (1 out of 78)
- ⚠️ Issues **72 false alarms** out of 149 total alerts (~48%)
- Trade-off: Optimized for **safety** over precision

**Real-World Impact:**
- Out of 100 alerts: ~50 are real, ~50 are false alarms
- Out of 100 severe days: Catches ~99, misses ~1
- **Priority**: Don't miss severe days (public health critical)

## Quick Start

```python
import joblib
import pandas as pd

# Load model
model = joblib.load("model2_best_RandomForest_Recall-0.987_F1-0.678.pkl")

# Load threshold
threshold = 0.20  # or read from model2_threshold.txt

# Prepare data (needs lag features, rolling stats, etc.)
X = pd.read_csv("model2_severe_day.csv")
X = X.drop(columns=["is_severe_tomorrow"], errors="ignore")

# Predict
probabilities = model.predict_proba(X)[:, 1]
predictions = (probabilities >= threshold).astype(int)

# Results
results = pd.DataFrame({
    "City": X["City"],
    "Date": X["Date"],
    "Severe_Probability": probabilities,
    "Alert_Issued": predictions,  # 1 = severe expected
})

# Filter high-risk
high_risk = results[results["Alert_Issued"] == 1]
print(f"⚠️ {len(high_risk)} cities need alerts tomorrow")
```

## Required Features

**Lag features (1-3 days):**
- AQI, PM2.5, PM10, NO2, SO2, CO, O3, NO, NOx (lag_1, lag_2, lag_3)

**Rolling stats (3-day window):**
- rolling_mean_3, rolling_max_3, rolling_std_3

**Temporal:**
- day_of_week, month, season, is_winter

**Change features:**
- AQI_change_1d, AQI_change_3d, PM2.5_change_1d, PM10_change_1d

**Severe indicators:**
- was_severe_yesterday, days_since_last_severe

**Total**: ~100 features (including City encoding)

## Outputs

**Two values per prediction:**
1. **Probability** (0.0 to 1.0): Chance of severe day tomorrow
2. **Binary alert** (0 or 1): Issue warning or not

**Interpretation:**
- Probability > 0.20 → Issue alert
- Probability > 0.50 → High confidence severe day
- Probability > 0.80 → Very high confidence

## Practical Examples

### Example 1: Daily Monitoring
```python
# Get latest data for all cities
latest = X.groupby("City").tail(1)

# Predict
latest["Severe_Prob"] = model.predict_proba(latest)[:, 1]
latest["Alert"] = (latest["Severe_Prob"] >= 0.20).astype(int)

# Cities needing alerts
alerts = latest[latest["Alert"] == 1].sort_values("Severe_Prob", ascending=False)
print(f"⚠️ {len(alerts)} cities: Issue public health alerts")
print(alerts[["City", "Severe_Prob"]])
```

### Example 2: Risk-Based Actions
```python
def get_action(probability):
    if probability >= 0.80:
        return "🚨 URGENT: Close schools, halt outdoor activities"
    elif probability >= 0.50:
        return "⚠️ HIGH RISK: Issue health advisories"
    elif probability >= 0.20:
        return "⚡ MODERATE: Monitor closely, prepare response"
    else:
        return "✅ LOW RISK: Normal operations"

# Apply to predictions
results["Action"] = results["Severe_Probability"].apply(get_action)
```

### Example 3: Multi-City Dashboard
```python
import matplotlib.pyplot as plt

# Predict for all cities
cities = X.groupby("City").tail(1).copy()
cities["Risk"] = model.predict_proba(cities)[:, 1]

# Visualize
cities_sorted = cities.sort_values("Risk", ascending=False).head(10)

plt.figure(figsize=(10, 5))
plt.barh(cities_sorted["City"], cities_sorted["Risk"])
plt.axvline(0.20, color='r', linestyle='--', label='Alert threshold')
plt.xlabel("Severe Day Probability")
plt.title("Top 10 At-Risk Cities (Tomorrow)")
plt.legend()
plt.tight_layout()
plt.show()
```

## Understanding the Trade-off

**Why 48% false alarm rate is acceptable:**

| Scenario | Cost |
|----------|------|
| **Miss severe day** (1%) | Health crisis, hospitalizations, deaths 💀 |
| **False alarm** (48%) | Unnecessary school closures, minor inconvenience ⚠️ |

**Decision**: Better to have false alarms than miss severe days.

## Confusion Matrix Explained

```
                  Predicted
                 No    Yes
Actual  No     2259    72   ← 72 false alarms
        Yes       1    77   ← Caught 77/78 severe days!
```

**Key insights:**
- Top-left (2259): Correctly predicted normal days
- Top-right (72): False alarms (said severe, was normal)
- Bottom-left (1): **MISSED severe day** (said normal, was severe) ⚠️
- Bottom-right (77): Correctly caught severe days ✅

## When to Use

✅ **Perfect for:**
- Same-day public health alerts
- School closure decisions
- Emergency response planning
- Outdoor event cancellations

❌ **Not good for:**
- 7-day forecasts (use Model 1)
- Exact AQI predictions (use Model 1)
- Non-severe pollution days (this model ignores AQI < 300)

## Comparison with Model 1

| Aspect | Model 1 | Model 2 |
|--------|---------|---------|
| **Task** | Predict AQI value | Severe day yes/no |
| **Horizon** | 7 days ahead | Tomorrow only |
| **Output** | Continuous (0-1000) | Binary (0/1) |
| **Accuracy** | R²=0.52 (±16 AQI) | Recall=98.7% |
| **Use case** | Planning | Immediate alerts |

## Files Included

- Model: `model2_best_RandomForest_Recall-0.987_F1-0.678.pkl`
- Threshold: `model2_threshold.txt` (0.20)
- Predictions: `model2_predictions.csv`
- Report: `model2_classification_report.txt`
- Plots:
  - `model2_confusion_matrix.png`
  - `model2_roc_curve.png`
  - `model2_pr_curve.png`

## Data Preparation

```python
from model2_data_prep import prepare_model2_data

# Creates model2_severe_day.csv with all features
prepare_model2_data(
    input_path="city_day.csv",
    output_path="model2_severe_day.csv"
)
```

## Key Notes

1. **Class imbalance**: Only 3.2% of days are severe (78/2409)
2. **Threshold tuning**: Lowered to 0.20 to maximize recall
3. **False alarms**: Acceptable trade-off for public health
4. **Temporal order**: Always predict chronologically
5. **City coverage**: Handles new cities via one-hot encoding

---
*Dataset: Indian cities, 2015-2020*  
*Best model: RandomForest (Recall=98.7%)*  
*Priority: Catch severe days > Avoid false alarms*  
*Optimal for: Emergency public health alerts*
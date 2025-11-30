# Model 1: AQI Forecasting (7 Days Ahead) - Usage Guide

## Overview
Predicts Air Quality Index (AQI) 7 days in advance using historical pollution data and temporal patterns. Early warning system for vulnerable populations.

## Model Performance

| Model | R² Score | RMSE | MAE | Use Case |
|-------|----------|------|-----|----------|
| **Lasso** ✓ | **0.52** | **54.4** | **27.8** | **Best overall** |
| RandomForest | 0.49 | 56.3 | 30.4 | Overfits (Train R²=0.94) |
| GradientBoosting | 0.42 | 60.1 | 31.9 | Overfits (Train R²=0.85) |
| GBR_Quantile | -0.02 | 79.5 | 55.3 | Failed completely |

**Why Lasso wins:**
- Good balance: R² = 0.52 (explains 52% of variance)
- No overfitting: Train R² ≈ Test R² (healthy gap)
- Strong regularization prevents memorization

**Accuracy Expectations:**
- **Median error**: ±16 AQI points
- **90th percentile**: ±59 AQI points
- **Percentage error**: ~19% (typical)

**Examples:**
- If actual AQI = 100 → prediction ≈ 100 ± 16 (range: 84-116)
- If actual AQI = 200 → prediction ≈ 200 ± 31 (range: 169-231)
- If actual AQI = 300 → prediction ≈ 300 ± 47 (range: 253-347)

## Quick Start

```python
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Load the complete pipeline (preprocessing + model)
model = joblib.load("model1_best_Lasso_R2-0.523.pkl")

# Load your prepared data
X = pd.read_csv("model1_aqi_forecast.csv")
X = X.drop(columns=["AQI_target"], errors="ignore")

# Add extra features (CRITICAL - must match training)
from model1_aqi_forecast import add_extra_features
X = add_extra_features(X)

# Predict AQI 7 days ahead
predictions = model.predict(X)

# Create results
results = pd.DataFrame({
    "City": X["City"],
    "Date": X["Date"],
    "Current_AQI": X["AQI"],
    "Predicted_AQI_7days": predictions
})

print(results.head())
```

## Required Input Features

**Base pollutants (current values):**
- AQI, PM2.5, PM10, NO2, SO2

**Lag features (past 7 days):**
- AQI_lag_1 through AQI_lag_7
- PM2.5_lag_1 through PM2.5_lag_7
- PM10_lag_1 through PM10_lag_7
- NO2_lag_1 through NO2_lag_7
- SO2_lag_1 through SO2_lag_7

**Rolling statistics (7-day window):**
- AQI_rolling_mean_7
- AQI_rolling_std_7
- AQI_rolling_max_7
- AQI_rolling_min_7

**Temporal features:**
- day_of_week (0-6)
- month (1-12)
- season (1-4: winter, spring, summer, monsoon)
- is_winter (0 or 1)

**Technical features:**
- AQI_ema_7 (exponential moving average)

**Extra features (added by `add_extra_features`):**
- AQI_lag_1_squared
- AQI_lag_1_log
- was_severe_last_week (1 if AQI_lag_7 > 300)
- high_days_last_week (count of days with AQI > 300)
- PM25_winter_interaction (PM2.5 × is_winter)

**Total**: ~50 features (including City encoding)

## Expected Outputs

**Single value per row**: Predicted AQI 7 days from the input date

**Interpretation:**
- 0-50: Good
- 51-100: Satisfactory
- 101-200: Moderate
- 201-300: Poor
- 301-400: Very Poor
- 400+: Severe

## Practical Examples

### Example 1: Single City Forecast
```python
# Get latest data for Delhi
delhi_data = X[X["City"] == "Delhi"].tail(1)

# Predict 7 days ahead
pred_aqi = model.predict(delhi_data)[0]

print(f"Delhi AQI forecast (7 days): {pred_aqi:.0f}")
print(f"Expected range: {pred_aqi - 16:.0f} to {pred_aqi + 16:.0f}")

# Alert if severe expected
if pred_aqi > 300:
    print("⚠️ SEVERE pollution expected - issue public health alert")
```

### Example 2: Multi-City Monitoring
```python
# Get latest for all cities
latest = X.groupby("City").tail(1)

# Predict for all
latest["Forecast_7d"] = model.predict(latest)
latest["Risk_Level"] = pd.cut(
    latest["Forecast_7d"],
    bins=[0, 50, 100, 200, 300, 400, 1000],
    labels=["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
)

# Sort by worst forecast
worst_cities = latest.sort_values("Forecast_7d", ascending=False)[
    ["City", "Current_AQI", "Forecast_7d", "Risk_Level"]
].head(10)

print(worst_cities)
```

### Example 3: Trend Analysis
```python
# Get last 30 days for Mumbai
mumbai = X[X["City"] == "Mumbai"].tail(30).copy()

# Predict for each day (rolling forecast)
mumbai["Forecast_7d"] = model.predict(mumbai)

# Plot trend
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(mumbai["Date"], mumbai["AQI"], label="Current AQI", marker='o')
plt.plot(mumbai["Date"], mumbai["Forecast_7d"], label="7-day Forecast", marker='s')
plt.axhline(300, color='r', linestyle='--', label='Severe threshold')
plt.xlabel("Date")
plt.ylabel("AQI")
plt.legend()
plt.title("Mumbai: Current vs Forecast AQI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## Data Preparation

If starting from raw `city_day.csv`:

```python
from model1_data_prep import prepare_model1_data

# Creates model1_aqi_forecast.csv with all features
prepare_model1_data(
    input_path="city_day.csv",
    output_path="model1_aqi_forecast.csv"
)
```

**What it does:**
1. Sorts data by City and Date
2. Creates 7-day lag features for each pollutant
3. Calculates rolling statistics (mean, std, max, min)
4. Extracts temporal features (day, month, season)
5. Computes exponential moving average
6. Creates target (AQI 7 days ahead)
7. Drops rows with missing lags (first 7 days per city)

## Important Notes

1. **Pipeline includes preprocessing**: No need to manually scale/encode
2. **City encoding**: Model handles new cities via `handle_unknown="ignore"`
3. **Temporal order**: Always predict chronologically (no shuffling)
4. **Missing data**: Pipeline imputes with median (numeric) and mode (categorical)
5. **Extreme events**: Model struggles with AQI > 400 (see residuals plot - larger errors)

## Limitations

1. **R² = 0.52**: Model explains ~52% of variance; other factors (weather, emissions) also matter
2. **Extreme values**: Under-predicts severe pollution events (AQI > 400)
3. **7-day horizon**: Accuracy degrades beyond 7 days
4. **Cold start**: Needs 7 days of history per city
5. **Seasonality**: Performs better in stable seasons vs transitions

## When to Use

✅ **Good for:**
- Early warning (5-7 days ahead)
- Comparative forecasts (city rankings)
- Trend detection
- Public health planning

❌ **Not good for:**
- Precise predictions (±16 error is significant)
- Extreme event prediction (under-predicts)
- Next-day forecasts (use simpler persistence models)
- Cities with <7 days of data

## Files Included

- **Model**: `model1_best_Lasso_R2-0.523.pkl` (complete pipeline)
- **Data**: `model1_aqi_forecast.csv` (prepared dataset)
- **Predictions**: `model1_predictions.csv` (test set results)
- **Comparison**: `model1_comparison.csv` (all models tested)
- **Feature importance**: `model1_feature_importance.csv` (top 30 features)
- **Plots**: 
  - `model1_actual_vs_predicted.png` (scatter plot)
  - `model1_residuals.png` (error analysis)
  - `model1_time_series.png` (temporal performance)

## Comparison: Model 1 vs Others

| Use Case | Best Model | Why |
|----------|------------|-----|
| India state-level disease | Model 3 (improved) | Higher R² (0.75), India-specific |
| Global pollutant synergy | Model 4 | Multi-country, interaction effects |
| **AQI forecasting** | **Model 1** | **Time-series specific, 7-day horizon** |
| Severe day alerts | Model 2 | Binary classification (tomorrow only) |

---
*Dataset: Indian cities, 2015-2020*  
*Best model: Lasso Regression (R² = 0.52)*  
*Typical error: ±16 AQI points (±19%)*  
*Forecast horizon: 7 days ahead*
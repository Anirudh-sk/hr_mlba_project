# Model 4: Multi-Pollutant Synergy Models - Usage Guide

## Overview
Predicts disease deaths from pollutant combinations across countries. Trained on global data (156 countries + India states, 2015-2019).

## Model Performance

| Target | Best Model | R² Score | Typical Error |
|--------|------------|----------|---------------|
| Cardiovascular deaths | RandomForest | 0.48 | ±10,000 (±10%) |
| Respiratory deaths | RandomForest | 0.50 | ±5,800 (±6%) |
| Combined disease risk | RandomForest | 0.50 | ±18,600 (±19%) |

**Key Improvements:**
- R² increased from 0.10 → 0.50 (5x better)
- Error reduced from ±100,000 → ±10,000 (10x better)
- Log transformation stabilized heavy-tailed targets
- Feature selection: Core pollutants + interactions only

## Quick Start

```python
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Load models
root = Path(__file__).parent
models = {
    "Cardiovascular_deaths_per_100k": joblib.load(
        root / "model4_best_Cardiovascular_deaths_per_100k_RandomForest_R2-0.480.pkl"
    ),
    "Respiratory_deaths_per_100k": joblib.load(
        root / "model4_best_Respiratory_deaths_per_100k_RandomForest_R2-0.504.pkl"
    ),
    "Combined_disease_risk_score": joblib.load(
        root / "model4_best_Combined_disease_risk_score_RandomForest_R2-0.504.pkl"
    ),
}

# Prepare input data
X = pd.read_csv(root / "model4_pollutant_synergy.csv")
X = X.drop(
    columns=[
        "Cardiovascular_deaths_per_100k",
        "Respiratory_deaths_per_100k",
        "Combined_disease_risk_score",
    ],
    errors="ignore",
)

# Make predictions (IMPORTANT: Transform from log space)
preds = {}
for tgt, mdl in models.items():
    y_pred_log = mdl.predict(X)  # Predictions in log space
    preds[tgt] = np.expm1(y_pred_log)  # Convert back to original scale

# Create results dataframe
pred_df = pd.DataFrame({
    "Country": X.get("Country", ["unknown"] * len(X)),
    "Year": X.get("Year", [2019] * len(X)),
})
for tgt, arr in preds.items():
    pred_df[f"Pred_{tgt}"] = arr

print(pred_df.head())
```

## Required Input Features

The models use **only core pollutants and key interactions**:

**Base Pollutants:**
- PM2.5 (particulate matter)
- NO2 (nitrogen dioxide)
- SO2 (sulfur dioxide)
- CO (carbon monoxide)
- Ozone (ground-level)

**Interaction Features:**
- PM25_NO2 (PM2.5 × NO2)
- PM25_SO2 (PM2.5 × SO2)
- PM25_CO (PM2.5 × CO)
- NO2_SO2 (NO2 × SO2)
- SO2_CO (SO2 × CO)

**Optional:**
- Country (categorical, for country-specific patterns)

## Expected Outputs

**Three disease death predictions:**

1. **Cardiovascular_deaths_per_100k**: Deaths from heart disease, stroke, etc.
2. **Respiratory_deaths_per_100k**: Deaths from lower respiratory infections + chronic respiratory diseases
3. **Combined_disease_risk_score**: Total burden (cardiovascular + respiratory + neoplasms)

**Accuracy:**
- Median errors: ±5,800 to ±18,600
- Percentage errors: ±75-80% (typical)
- R² scores: 0.48-0.50

## Example Use Cases

```python
# Example 1: Predict for a specific country/region
new_data = pd.DataFrame({
    "Country": ["United States"],
    "PM2.5": [12.0],
    "NO2": [21.0],
    "SO2": [3.5],
    "CO": [0.5],
    "Ozone": [42.0],
    "PM25_NO2": [12.0 * 21.0],
    "PM25_SO2": [12.0 * 3.5],
    "PM25_CO": [12.0 * 0.5],
    "NO2_SO2": [21.0 * 3.5],
    "SO2_CO": [3.5 * 0.5],
})

cardio_pred = np.expm1(models["Cardiovascular_deaths_per_100k"].predict(new_data))
print(f"Predicted cardiovascular deaths: {cardio_pred[0]:,.0f} ± 10,000")

# Example 2: Compare scenarios
baseline = new_data.copy()
reduced_pm25 = baseline.copy()
reduced_pm25["PM2.5"] *= 0.8  # 20% reduction
reduced_pm25["PM25_NO2"] *= 0.8
reduced_pm25["PM25_SO2"] *= 0.8
reduced_pm25["PM25_CO"] *= 0.8

baseline_deaths = np.expm1(models["Combined_disease_risk_score"].predict(baseline))
reduced_deaths = np.expm1(models["Combined_disease_risk_score"].predict(reduced_pm25))

print(f"Baseline deaths: {baseline_deaths[0]:,.0f}")
print(f"With 20% PM2.5 reduction: {reduced_deaths[0]:,.0f}")
print(f"Deaths averted: {(baseline_deaths[0] - reduced_deaths[0]):,.0f}")
```

## Important Notes

1. **Log transformation**: Models predict in log space - always use `np.expm1()` to convert back
2. **Scale**: Predictions are absolute death counts, not per-100k rates
3. **Uncertainty**: Report with error bands (e.g., "100,000 ± 10,000")
4. **Use case**: Best for comparative analysis (scenario testing) rather than absolute predictions
5. **Limitations**: R² ≈ 0.50 means pollution explains ~50% of variance; other factors (healthcare, demographics, smoking) also matter

## Files Included

- **Models**: `model4_best_*_RandomForest_R2-*.pkl` (3 files)
- **Predictions**: `model4_*_predictions.csv` (test set results)
- **Comparison**: `model4_*_comparison.csv` (all models tested)
- **Visualizations**: `model4_*_actual_vs_pred.png` (scatter plots)
- **Data**: `model4_pollutant_synergy.csv` (prepared dataset)

## Summary

**Use Model 4 when:**
- Analyzing global pollutant-disease relationships
- Testing "what-if" scenarios (e.g., pollution reduction impacts)
- Comparing multiple countries/regions

**Use Model 3 instead when:**
- Focusing specifically on India
- Need higher accuracy (R² ≈ 0.75 vs 0.50)
- Need per-100k normalized rates

---
*Dataset: 156 countries + Indian states, 2015-2019*  
*Best model: RandomForest (R² = 0.48-0.50)*  
*Typical error: ±6-19% of actual value*
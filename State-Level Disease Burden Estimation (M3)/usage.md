# Model 3: Disease Burden Estimation - Usage Guide

## Overview
Estimates respiratory/cardiovascular disease rates for Indian states using pollution as proxy. India-specific, per-100k population rates.

## Performance

**Best Model: ElasticNet (Strong Regularization)**

| Target | R² | RMSE | Gap | Median Error |
|--------|----|----- |-----|--------------|
| Cardiovascular | 0.81 | 77 | -0.07 | ±38 per 100k |
| Respiratory | 0.80 | 49 | -0.07 | ±28 per 100k |
| All Diseases | 0.81 | 122 | -0.07 | ±60 per 100k |

**Key Improvements from Original:**
- R² reduced from 1.00 → 0.81 (no longer overfitted)
- Healthy overfitting gap (~-0.07, close to zero)
- Realistic error margins (±38-60 vs ±1-2)

## What This Means

**Accuracy:**
- If actual = 100 per 100k:
  - Cardiovascular → 100 ± 38
  - Respiratory → 100 ± 28
  - All Diseases → 100 ± 60

**Reliability:**
- ✅ R² = 0.81 is **excellent** for 77 observations
- ✅ Small overfitting gap means good generalization
- ✅ Errors are honest and realistic (±25-35%)

## Quick Start

```python
import joblib
import pandas as pd

# Load models
models = {
    "Cardiovascular": joblib.load(
        "improved_best_Cardiovascular_per_100k_ElasticNet_Strong_R2-0.805_gap--0.067.pkl"
    ),
    "Respiratory": joblib.load(
        "improved_best_Respiratory_per_100k_ElasticNet_Strong_R2-0.803_gap--0.074.pkl"
    ),
    "All_Diseases": joblib.load(
        "improved_best_All_Key_Diseases_per_100k_ElasticNet_Strong_R2-0.814_gap--0.070.pkl"
    ),
}

# Load data
X = pd.read_csv("model3_disease_burden.csv")
X = X.drop(
    columns=[
        "Cardiovascular_per_100k",
        "Respiratory_per_100k", 
        "All_Key_Diseases_per_100k"
    ],
    errors="ignore"
)

# Predict
predictions = {}
for name, model in models.items():
    predictions[name] = model.predict(X)

# Results
results = pd.DataFrame({
    "State": X["State"],
    "Year": X["Year"],
    "Pred_Cardiovascular": predictions["Cardiovascular"],
    "Pred_Respiratory": predictions["Respiratory"],
    "Pred_All_Diseases": predictions["All_Diseases"],
})

print(results)
```

## Required Features

**Core pollutants (~10 features used):**
- PM2.5, PM10, NO2, SO2, CO, O3, NOx
- mean_AQI, max_AQI, std_AQI

**Note:** State is NOT used (removed to prevent overfitting)

## Outputs

**Three predictions per row:**
1. Cardiovascular deaths per 100k
2. Respiratory deaths per 100k
3. Combined disease burden per 100k

**Report with uncertainty:**
```python
cv_pred = predictions["Cardiovascular"][0]
print(f"Cardiovascular: {cv_pred:.0f} ± 38 per 100k")
```

## Practical Examples

### Example 1: State Rankings
```python
# Predict for all states
states = X.groupby("State").tail(1).copy()
states["Disease_Burden"] = models["All_Diseases"].predict(states)

# Rank by burden
ranking = states.sort_values("Disease_Burden", ascending=False)[
    ["State", "Disease_Burden"]
]

print("Top 5 States by Disease Burden:")
print(ranking.head())
```

### Example 2: Pollution Reduction Impact
```python
# Baseline scenario
baseline = X[X["State"] == "Delhi"].tail(1).copy()
baseline_burden = models["All_Diseases"].predict(baseline)[0]

# 20% PM2.5 reduction scenario
reduced = baseline.copy()
reduced["PM2.5"] *= 0.8
reduced["PM2.5_SO2"] *= 0.8  # Update interactions
reduced["PM2.5_NO2"] *= 0.8
reduced_burden = models["All_Diseases"].predict(reduced)[0]

print(f"Baseline: {baseline_burden:.0f} per 100k")
print(f"With 20% PM2.5 reduction: {reduced_burden:.0f} per 100k")
print(f"Lives saved (per 100k): {baseline_burden - reduced_burden:.0f}")
```

### Example 3: Temporal Trends
```python
import matplotlib.pyplot as plt

# Get predictions across years for one state
delhi = X[X["State"] == "Delhi"].copy()
delhi["Predicted"] = models["All_Diseases"].predict(delhi)

plt.figure(figsize=(10, 5))
plt.plot(delhi["Year"], delhi["Predicted"], marker='o')
plt.xlabel("Year")
plt.ylabel("Disease Burden (per 100k)")
plt.title("Delhi: Predicted Disease Burden (2015-2019)")
plt.grid(True, alpha=0.3)
plt.show()
```

## Why Improved Models Are Better

**Original vs Improved:**

| Metric | Original | Improved |
|--------|----------|----------|
| R² | 1.000 (suspicious) | 0.81 (realistic) |
| Overfitting | Severe (gap=0.0) | Minimal (gap=-0.07) |
| Error estimate | ±1-2 (fake) | ±38-60 (honest) |
| Features | ~50 | ~10 |
| Generalization | Poor | Good |

**The Trade-off:**
- Sacrificed apparent perfection (R²=1.0)
- Gained real-world reliability (R²=0.81)
- **For 77 observations, R²=0.81 is excellent!**

## When to Use

✅ **Perfect for:**
- India state-level analysis
- Comparative studies (which states worse)
- Policy impact scenarios
- Exploratory research

❌ **Not good for:**
- Absolute precision (expect ±25-35% error)
- Non-India regions (use Model 4)
- Individual city predictions

## Comparison: Model 3 vs Model 4

| Aspect | Model 3 | Model 4 |
|--------|---------|---------|
| **Region** | India states only | 156 countries |
| **R² Score** | 0.81 | 0.50 |
| **Error** | ±25-35% | ±75-80% |
| **Dataset** | 77 rows | 17,767 rows |
| **Targets** | Per 100k rates | Absolute deaths |
| **Use case** | India-specific | Global comparisons |

**When to use each:**
- India-focused analysis → Model 3 (higher accuracy)
- Global analysis → Model 4 (broader coverage)

## Files Included

**Models:**
- `improved_best_Cardiovascular_per_100k_ElasticNet_Strong_R2-0.805_gap--0.067.pkl`
- `improved_best_Respiratory_per_100k_ElasticNet_Strong_R2-0.803_gap--0.074.pkl`
- `improved_best_All_Key_Diseases_per_100k_ElasticNet_Strong_R2-0.814_gap--0.070.pkl`

**Analysis:**
- `improved_*_predictions.csv` (test results)
- `improved_*_comparison.csv` (all models tested)
- `improved_*_feature_importance.csv` (top features)
- `improved_*_actual_vs_pred.png` (scatter plots)
- `comprehensive_model_comparison.png` (before/after visual)

## Data Preparation

```python
from model3_data_prep import prepare_model3_data

prepare_model3_data(
    city_path="city_day.csv",
    global_path="global_air_pollution_data.csv",
    deaths_path="cause_of_deaths.csv",
    output_path="model3_disease_burden.csv"
)
```

**What it does:**
1. Aggregates city-day data to state-year
2. Computes pollution statistics per state
3. Estimates disease rates from national data
4. Creates interaction features
5. Saves 77-row dataset (23 states × 3-4 years)

## Key Notes

1. **Small dataset**: Only 77 observations limits absolute accuracy
2. **R²=0.81 is good**: For this data size, it's realistic and reliable
3. **Overfitting fixed**: Gap ~-0.07 shows model generalizes well
4. **Report uncertainty**: Always use ±error ranges
5. **Validation needed**: Test on independent data before policy use

---
*Dataset: 23 Indian states, 2015-2019 (77 observations)*  
*Best model: ElasticNet Strong (R²=0.81)*  
*Typical error: ±25-35% of actual value*  
*Optimized for: India state-level disease burden*
# Air Quality & Health Prediction: Predictive Modeling Pipeline

## Project Overview
Build 4 predictive models to forecast air quality and estimate health impacts from pollution data. Each model will have its own prepared dataset, comparison of multiple algorithms, and the best model saved as a pickle file.

---

## MODELS TO BUILD

### Model 1: City-Level AQI Forecasting (7-30 days ahead)
**Purpose**: Early warning system for vulnerable populations  
**Output**: Predicted AQI values for next 7-30 days per city

### Model 2: Severe Day Prediction (AQI ≥300)
**Purpose**: Public health alerts, school closures, outdoor activity warnings  
**Output**: Binary classification - will tomorrow be a severe pollution day?

### Model 3: State-Level Disease Burden Estimation
**Purpose**: Estimate respiratory/cardiovascular disease rates for Indian states using pollution proxies  
**Output**: Predicted disease death rates per 100k population at state level

### Model 4: Multi-Pollutant Synergy Model
**Purpose**: Predict disease risk from pollutant combinations (non-linear health impacts)  
**Output**: Disease risk scores based on pollutant interactions

---

## PHASE 1: DATA PREPARATION

### Step 1.1: Prepare Dataset for Model 1 (AQI Forecasting)
```python
"""
Create model1_aqi_forecast.csv

Input files: city_day.csv

Steps:
1. Load city_day.csv
2. Sort by City and Date
3. For each city:
   - Create lagged features:
     * AQI_lag_1 to AQI_lag_7 (previous 7 days)
     * PM2.5_lag_1 to PM2.5_lag_7
     * PM10_lag_1 to PM10_lag_7
     * NO2_lag_1 to NO2_lag_7
     * SO2_lag_1 to SO2_lag_7
   - Create rolling window features:
     * AQI_rolling_mean_7 (7-day moving average)
     * AQI_rolling_std_7 (7-day std dev)
     * AQI_rolling_max_7 (7-day max)
     * AQI_rolling_min_7 (7-day min)
   - Create temporal features:
     * day_of_week (0-6)
     * month (1-12)
     * season (1=winter, 2=spring, 3=summer, 4=monsoon)
     * is_winter (1 if Nov-Jan, else 0)
   - Create exponential moving average:
     * AQI_ema_7 (alpha=0.3)
4. Target variable:
   - AQI_target = AQI value 7 days ahead
5. Remove rows with NaN (first 7 days per city won't have lags)
6. Final columns:
   - City, Date, all lag features, rolling features, temporal features
   - Target: AQI_target
7. Save as model1_aqi_forecast.csv
"""
```

### Step 1.2: Prepare Dataset for Model 2 (Severe Day Prediction)
```python
"""
Create model2_severe_day.csv

Input files: city_day.csv

Steps:
1. Load city_day.csv
2. Sort by City and Date
3. For each city:
   - Create lagged features (1-3 days):
     * AQI_lag_1, AQI_lag_2, AQI_lag_3
     * PM2.5_lag_1, PM2.5_lag_2, PM2.5_lag_3
     * PM10_lag_1, PM10_lag_2, PM10_lag_3
     * All pollutants: NO2, SO2, CO, O3, NO, NOx
   - Create 3-day rolling statistics:
     * rolling_mean_3, rolling_max_3, rolling_std_3 for AQI and major pollutants
   - Create rate of change features:
     * AQI_change_1d = AQI_today - AQI_lag_1
     * AQI_change_3d = AQI_today - AQI_lag_3
     * PM2.5_change_1d, PM10_change_1d
   - Temporal features:
     * day_of_week, month, season, is_winter
   - Create AQI category features:
     * was_severe_yesterday (1 if AQI_lag_1 >= 300)
     * days_since_last_severe (count)
4. Target variable:
   - is_severe_tomorrow = 1 if AQI >= 300, else 0 (shift by -1 day)
5. Handle class imbalance:
   - Calculate class distribution
   - Note severe_day_percentage for reference
6. Remove rows with NaN
7. Final columns:
   - City, Date, all features
   - Target: is_severe_tomorrow (binary)
8. Save as model2_severe_day.csv
"""
```

### Step 1.3: Prepare Dataset for Model 3 (Disease Burden Estimation)
```python
"""
Create model3_disease_burden.csv

Input files: 
- city_day.csv
- global_air_pollution_data.csv  
- cause_of_deaths.csv

Steps:
1. Aggregate city_day.csv to state-year level:
   - Map cities to states (create city-to-state mapping)
   - Group by State, Year (extract year from Date)
   - Calculate mean values for:
     * PM2.5, PM10, NO2, SO2, CO, O3, NOx
   - Calculate AQI statistics:
     * mean_AQI, max_AQI, std_AQI
     * pct_severe_days (% days with AQI >= 300)
     * pct_very_poor_days (% days with AQI >= 200)
   - Time period: 2015-2019

2. Extract India data from global_air_pollution_data.csv:
   - Filter for Country = 'India'
   - Aggregate to state level if city-level
   - Keep: State, PM2.5_value, NO2_value, Ozone_value, AQI_value
   
3. Extract India disease data from cause_of_deaths.csv:
   - Filter for Country = 'India', Year = 2015-2019
   - Calculate per 100k rates (need India population by year):
     * Cardiovascular_per_100k
     * Lower_Respiratory_per_100k
     * Chronic_Respiratory_per_100k
     * All_Respiratory_per_100k = Lower + Chronic
   - Create state-level estimates using city pollution as proxy:
     * Use correlation: state_deaths = national_deaths × (state_AQI/national_AQI)^1.5

4. Merge all sources:
   - Left join city_day aggregated with global_air_pollution 
   - Join with estimated state disease rates
   - Handle missing values with median imputation

5. Create interaction features:
   - PM2.5 × SO2
   - PM2.5 × NO2  
   - AQI × pct_severe_days

6. Target variables:
   - Cardiovascular_per_100k
   - Respiratory_per_100k (combined)
   - All_Key_Diseases_per_100k

7. Save as model3_disease_burden.csv

Note: This model uses 2019 global correlations for training, applies to India states
Columns: State, Year, all pollutant features, interaction terms, targets
"""
```

### Step 1.4: Prepare Dataset for Model 4 (Multi-Pollutant Synergy)
```python
"""
Create model4_pollutant_synergy.csv

Input files:
- city_day.csv
- global_air_pollution_data.csv
- cause_of_deaths.csv

Steps:
1. Load global_air_pollution_data.csv:
   - Filter for Year = 2019 (only year with aligned pollution + disease data)
   - Keep all countries
   - Normalize pollutant values: PM2.5, NO2, Ozone, CO

2. Load cause_of_deaths.csv:
   - Filter for Year = 2019
   - Keep disease columns:
     * Cardiovascular Diseases
     * Lower Respiratory Infections  
     * Chronic Respiratory Diseases
     * Neoplasms
   - Calculate per 100k rates using country population
   
3. Merge on Country, Year=2019

4. Create pollutant interaction features:
   - PM2.5 × NO2
   - PM2.5 × Ozone
   - PM2.5 × SO2
   - NO2 × SO2
   - NO2 × Ozone
   - Three-way: PM2.5 × NO2 × SO2
   
5. Create polynomial features:
   - PM2.5_squared, PM2.5_cubed
   - NO2_squared, NO2_cubed
   - AQI_squared

6. Create ratio features:
   - PM2.5 / NO2
   - PM10 / PM2.5
   - NOx / NO2

7. Create seasonal proxies (if lat/lon available):
   - Estimate based on country location
   - Otherwise use country-level climate zone

8. For India, append city_day aggregated to yearly (2015-2019):
   - Aggregate to Year level
   - Calculate same interaction features
   - Create pseudo-state level by city groupings
   - Estimate deaths using correlation transfer from global model

9. Target variables:
   - Cardiovascular_deaths_per_100k
   - Respiratory_deaths_per_100k
   - Combined_disease_risk_score (weighted composite)

10. Final dataset:
    - Global 2019 data (primary training set)
    - India 2015-2019 (validation/application set)
    - Mark with is_india flag

11. Save as model4_pollutant_synergy.csv

Columns: Country/State, Year, all base pollutants, all interaction features, targets
"""
```

---

## PHASE 2: MODEL BUILDING

### Step 2.1: Build Model 1 - AQI Forecasting

```python
"""
File: model1_aqi_forecast.py

Steps:

1. Load model1_aqi_forecast.csv

2. Train-test split:
   - Use last 20% of data (chronologically) as test set
   - Use first 80% as training set
   - DO NOT shuffle (time series data)

3. Feature selection:
   - X = all lag features, rolling features, temporal features
   - y = AQI_target
   - Separate categorical (City) using encoding if needed

4. Preprocessing:
   - StandardScaler for numeric features
   - OneHotEncoder for City (if used as feature)
   - Save preprocessing pipeline

5. Models to try:
   A. Linear Regression (baseline)
   B. Ridge Regression (alpha: 0.1, 1, 10)
   C. Lasso Regression (alpha: 0.1, 1, 10)
   D. Random Forest Regressor (n_estimators: 100, 200, 300; max_depth: 10, 20, None)
   E. Gradient Boosting Regressor (n_estimators: 100, 200; learning_rate: 0.01, 0.1; max_depth: 5, 10)
   F. XGBoost Regressor (n_estimators: 100, 200, 300; learning_rate: 0.01, 0.05, 0.1; max_depth: 5, 7, 10)
   G. LightGBM Regressor (n_estimators: 100, 200; learning_rate: 0.01, 0.1; num_leaves: 31, 50)
   H. Support Vector Regression (kernel: rbf, poly; C: 1, 10, 100)
   I. K-Nearest Neighbors (n_neighbors: 5, 10, 15, 20)
   J. MLP Regressor (hidden_layers: (100,), (100,50), (200,100); activation: relu, tanh)

6. Evaluation metrics:
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - R² Score
   - MAPE (Mean Absolute Percentage Error)

7. Cross-validation:
   - Use TimeSeriesSplit (n_splits=5)
   - Report mean ± std for each metric

8. Hyperparameter tuning:
   - Use GridSearchCV or RandomizedSearchCV
   - For top 3 models based on initial results

9. Create comparison table:
   Columns: Model_Name, Train_RMSE, Test_RMSE, Train_MAE, Test_MAE, Train_R2, Test_R2, CV_Score_Mean, CV_Score_Std, Training_Time
   
10. Select best model:
    - Primarily based on Test_RMSE and Test_R2
    - Consider overfitting (train-test gap)
    - Consider training time for production

11. Save best model:
    - Save as model1_best_aqi_forecast.pkl
    - Save preprocessing pipeline as model1_preprocessor.pkl
    - Save feature names as model1_features.pkl

12. Generate predictions:
    - Predict on test set
    - Save predictions as model1_predictions.csv (Date, City, Actual_AQI, Predicted_AQI)

13. Create visualizations:
    - Actual vs Predicted scatter plot
    - Residual plot
    - Time series plot (actual vs predicted over time)
    - Feature importance (if applicable)

14. Print summary:
    - Best model name and hyperparameters
    - Final test metrics
    - Top 10 most important features
"""
```

### Step 2.2: Build Model 2 - Severe Day Prediction

```python
"""
File: model2_severe_day.py

Steps:

1. Load model2_severe_day.csv

2. Check class distribution:
   - Count is_severe_tomorrow = 0 vs 1
   - Calculate class imbalance ratio
   - Print statistics

3. Train-test split:
   - Last 20% as test set (chronological)
   - First 80% as training set
   - Stratify by is_severe_tomorrow if possible

4. Feature selection:
   - X = all features (lags, rolling stats, temporal, change features)
   - y = is_severe_tomorrow
   - Drop City, Date

5. Preprocessing:
   - StandardScaler for numeric features
   - Save preprocessing pipeline

6. Handle class imbalance (try multiple strategies):
   - Strategy A: No balancing (baseline)
   - Strategy B: SMOTE (Synthetic Minority Over-sampling)
   - Strategy C: Class weights (balanced)
   - Strategy D: Random undersampling of majority class
   
   Note: Try each strategy with each model

7. Models to try:
   A. Logistic Regression (C: 0.1, 1, 10; class_weight: balanced)
   B. Random Forest Classifier (n_estimators: 100, 200, 300; max_depth: 10, 20, None; class_weight: balanced)
   C. Gradient Boosting Classifier (n_estimators: 100, 200; learning_rate: 0.01, 0.1; max_depth: 5, 10)
   D. XGBoost Classifier (n_estimators: 100, 200, 300; learning_rate: 0.01, 0.05, 0.1; scale_pos_weight: auto)
   E. LightGBM Classifier (n_estimators: 100, 200; learning_rate: 0.01, 0.1; is_unbalance: True)
   F. Support Vector Classifier (kernel: rbf, poly; C: 1, 10; class_weight: balanced)
   G. K-Nearest Neighbors (n_neighbors: 5, 10, 15, 20; weights: uniform, distance)
   H. MLP Classifier (hidden_layers: (100,), (100,50); activation: relu)
   I. Decision Tree Classifier (max_depth: 10, 20, None; class_weight: balanced)
   J. AdaBoost Classifier (n_estimators: 50, 100, 200)

8. Evaluation metrics:
   - Accuracy
   - Precision (for severe class)
   - Recall (for severe class) - MOST IMPORTANT (don't miss severe days)
   - F1-Score
   - ROC-AUC
   - Confusion Matrix
   - Classification Report

9. Cross-validation:
   - StratifiedKFold (n_splits=5)
   - Report mean ± std for each metric

10. Threshold optimization:
    - For best model, tune classification threshold
    - Optimize for high recall (catch severe days)
    - Balance with precision to avoid too many false alarms

11. Create comparison table:
    Columns: Model_Name, Imbalance_Strategy, Train_Accuracy, Test_Accuracy, Precision, Recall, F1_Score, ROC_AUC, CV_Score_Mean, CV_Score_Std

12. Select best model:
    - Prioritize Recall > 0.85 (critical for public health)
    - Then optimize F1-Score
    - Consider false positive rate (public trust)

13. Save best model:
    - Save as model2_best_severe_day.pkl
    - Save preprocessing pipeline as model2_preprocessor.pkl
    - Save optimal threshold as model2_threshold.pkl

14. Generate predictions:
    - Predict on test set with probabilities
    - Apply optimal threshold
    - Save as model2_predictions.csv (Date, City, Actual, Predicted, Probability)

15. Create visualizations:
    - Confusion matrix heatmap
    - ROC curve
    - Precision-Recall curve
    - Feature importance
    - Threshold vs metrics plot

16. Print summary:
    - Best model name and hyperparameters
    - Confusion matrix
    - Classification report
    - Optimal threshold
    - Expected false alarm rate
"""
```

### Step 2.3: Build Model 3 - Disease Burden Estimation

```python
"""
File: model3_disease_burden.py

Steps:

1. Load model3_disease_burden.csv

2. Analyze data:
   - Check number of states/regions available
   - Check years covered
   - Examine target variable distributions
   - Check for missing values

3. Multiple target strategy:
   - Build separate models for each target:
     * Cardiovascular_per_100k
     * Respiratory_per_100k
     * All_Key_Diseases_per_100k
   - Also try MultiOutputRegressor for joint prediction

4. Train-test split:
   - Random split (70-30) since not purely time series
   - Or: use 2015-2018 for training, 2019 for testing
   - Stratify by State if needed

5. Feature selection:
   - X = all pollutant features + interaction terms + AQI statistics
   - y = each target separately
   - Apply feature selection:
     * Correlation analysis (remove features with |corr| < 0.1 with target)
     * Mutual information
     * SelectKBest (keep top 20-30 features)

6. Preprocessing:
   - StandardScaler for numeric features
   - Handle outliers (optional: winsorization at 1st and 99th percentile)

7. Models to try (for EACH target):
   A. Linear Regression (baseline)
   B. Ridge Regression (alpha: 0.1, 1, 10, 100)
   C. Lasso Regression (alpha: 0.1, 1, 10, 100)
   D. ElasticNet (alpha: 0.1, 1, 10; l1_ratio: 0.3, 0.5, 0.7)
   E. Random Forest Regressor (n_estimators: 100, 200; max_depth: 10, 20; min_samples_leaf: 5, 10)
   F. Gradient Boosting Regressor (n_estimators: 100, 200; learning_rate: 0.01, 0.05, 0.1; max_depth: 3, 5)
   G. XGBoost Regressor (n_estimators: 100, 200; learning_rate: 0.01, 0.05; max_depth: 3, 5, 7)
   H. LightGBM Regressor (n_estimators: 100, 200; learning_rate: 0.01, 0.05; num_leaves: 20, 31)
   I. Support Vector Regression (kernel: rbf, linear; C: 1, 10; epsilon: 0.1, 0.2)
   J. K-Nearest Neighbors (n_neighbors: 3, 5, 7, 10)

8. Evaluation metrics:
   - RMSE
   - MAE
   - R² Score
   - MAPE
   - Max Error (identify worst predictions)

9. Cross-validation:
   - KFold (n_splits=5)
   - Report mean ± std for each metric

10. Transfer learning approach:
    - Train on global 2019 data (if available in dataset)
    - Fine-tune on India 2015-2019 data
    - Compare with direct training on India data only

11. Ensemble methods:
    - Create ensemble of top 3 models
    - Weighted average based on validation performance
    - Stacking regressor

12. Create comparison table for EACH target:
    Columns: Model_Name, Target, Train_RMSE, Test_RMSE, Train_MAE, Test_MAE, Train_R2, Test_R2, CV_Score_Mean, CV_Score_Std

13. Select best model for each target:
    - Primarily based on Test_R2 and Test_RMSE
    - Check if same model works best for all targets

14. Save best models:
    - model3_best_cardiovascular.pkl
    - model3_best_respiratory.pkl
    - model3_best_all_diseases.pkl
    - model3_preprocessor.pkl

15. Generate predictions:
    - Predict on test set for all targets
    - Save as model3_predictions.csv (State, Year, Actual_Cardio, Pred_Cardio, Actual_Resp, Pred_Resp, ...)

16. Create visualizations:
    - Actual vs Predicted for each target
    - Residual plots
    - Feature importance for each model
    - Pollutant contribution analysis

17. Validation using known correlations:
    - Check if predicted correlations match EDA findings:
      * SO2 → Respiratory should be strong positive
      * SO2 → Cardiovascular should be strong positive
    - Compare correlation of predictions vs actual

18. Print summary:
    - Best model for each target
    - Key pollutants driving each disease type
    - Expected error margins
    - States with highest predicted burden
"""
```

### Step 2.4: Build Model 4 - Multi-Pollutant Synergy

```python
"""
File: model4_pollutant_synergy.py

Steps:

1. Load model4_pollutant_synergy.csv

2. Data split strategy:
   - Global 2019 data → primary training set
   - India 2015-2019 → validation set (separate evaluation)
   - Within global: 80-20 train-test split
   - Keep India separate for domain adaptation testing

3. Feature analysis:
   - Correlation matrix of all interaction features
   - Remove highly correlated pairs (|corr| > 0.95)
   - Keep most interpretable features when removing

4. Feature selection:
   - Use recursive feature elimination (RFE)
   - Select top 30-40 features
   - Ensure at least 5 interaction terms included
   - Include key base pollutants: PM2.5, NO2, SO2

5. Preprocessing:
   - RobustScaler (better for outliers than StandardScaler)
   - Optional: QuantileTransformer for heavy-tailed distributions

6. Multiple targets:
   - Cardiovascular_deaths_per_100k
   - Respiratory_deaths_per_100k  
   - Combined_disease_risk_score
   - Build models for each separately

7. Models to try (for EACH target):
   A. Linear Regression (baseline for interpretability)
   B. Ridge Regression (alpha: 0.01, 0.1, 1, 10)
   C. Lasso Regression (alpha: 0.01, 0.1, 1, 10)
   D. ElasticNet (alpha: 0.1, 1; l1_ratio: 0.3, 0.5, 0.7)
   E. Polynomial Regression (degree: 2) with Ridge
   F. Random Forest Regressor (n_estimators: 200, 300; max_depth: 15, 20; max_features: sqrt, log2)
   G. Gradient Boosting Regressor (n_estimators: 200, 300; learning_rate: 0.01, 0.05; max_depth: 4, 6; subsample: 0.8)
   H. XGBoost Regressor (n_estimators: 200, 300; learning_rate: 0.01, 0.05; max_depth: 4, 6; colsample_bytree: 0.8)
   I. LightGBM Regressor (n_estimators: 200, 300; learning_rate: 0.01, 0.05; num_leaves: 31, 50; feature_fraction: 0.8)
   J. CatBoost Regressor (iterations: 200, 300; learning_rate: 0.01, 0.05; depth: 4, 6)
   K. Neural Network - MLP (hidden_layers: (128,64), (200,100,50); activation: relu; early_stopping)
   L. Neural Network - Custom architecture with attention on interaction features

8. Interaction-specific models:
   - Multiplicative model: y = β₀ × PM2.5^β₁ × NO2^β₂ × SO2^β₃ (log-transform)
   - GAM (Generalized Additive Model) for non-linear interactions
   - Decision Tree with max_depth=5 for interpretable interactions

9. Evaluation metrics:
   - RMSE
   - MAE  
   - R² Score
   - MAPE
   - Explained Variance Score
   - Feature interaction strength score (custom metric)

10. Cross-validation:
    - KFold (n_splits=5) on global training data
    - Separate evaluation on India data (domain shift analysis)

11. Interaction importance analysis:
    - For best tree-based model:
      * Extract feature importance
      * Identify top interaction terms
    - For linear models:
      * Examine coefficients of interaction terms
    - SHAP analysis:
      * Calculate SHAP interaction values
      * Identify synergistic vs antagonistic interactions

12. Create comparison table:
    Columns: Model_Name, Target, Test_RMSE_Global, Test_R2_Global, Test_RMSE_India, Test_R2_India, Top_Interaction_Features, CV_Score_Mean, CV_Score_Std

13. Domain adaptation:
    - Check if global model performs well on India
    - If gap exists, try:
      * Fine-tuning on small India sample
      * Domain adversarial training
      * Transfer learning with frozen layers

14. Select best model for each target:
    - Best global performance
    - Acceptable India performance (R² > 0.5)
    - Interpretable interaction terms

15. Save best models:
    - model4_best_cardiovascular.pkl
    - model4_best_respiratory.pkl  
    - model4_best_combined_risk.pkl
    - model4_preprocessor.pkl
    - model4_feature_selector.pkl

16. Generate predictions:
    - Predict on global test set
    - Predict on India data (all years)
    - Save as model4_predictions.csv (Country/State, Year, Actual, Predicted, is_india flag)

17. Create visualizations:
    - Actual vs Predicted (separate for global and India)
    - Residual analysis
    - Top 10 feature importances
    - SHAP summary plot
    - Interaction effect plots (e.g., PM2.5 × SO2 heatmap)
    - Partial dependence plots for key interactions

18. Synergy analysis:
    - Identify pollutant pairs with strongest synergy:
      * Synergy score = coefficient(A×B) / (coefficient(A) + coefficient(B))
    - Rank interactions by health impact
    - Create synergy matrix heatmap

19. Validate against EDA correlations:
    - Check if model predictions preserve correlation patterns:
      * Global: weak correlations (0.1-0.25) should be matched
      * India: strong correlations (0.75-0.98) should be matched
    - Correlation of predicted vs actual should be high

20. Print summary:
    - Best model for each target
    - Top 5 most important pollutant interactions
    - Synergy effects found (e.g., "PM2.5 + SO2 amplifies cardiovascular risk by 1.4x")
    - Model performance on global vs India data
    - Recommendations for pollutant control priorities
"""
```

---

## PHASE 3: CODE STRUCTURE

### Complete Pipeline (Single Python File)

```python
"""
File: air_quality_health_models.py

This file contains the complete pipeline for all 4 models.
Each model has its own section with data prep and model building.

Usage:
    python air_quality_health_models.py --model all
    python air_quality_health_models.py --model 1
    python air_quality_health_models.py --model 2
    python air_quality_health_models.py --model 3
    python air_quality_health_models.py --model 4

Structure:
1. Imports and setup
2. Data preparation functions (one per model)
3. Model building functions (one per model)
4. Evaluation and comparison functions
5. Saving functions
6. Main execution
"""

# Required imports
import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime
import argparse

# ML imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit, StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import lightgbm as lgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: DATA PREPARATION FUNCTIONS
# ============================================================================

def prepare_model1_data():
    """Prepare AQI forecasting dataset"""
    # Implementation as per Step 1.1
    pass

def prepare_model2_data():
    """Prepare severe day prediction dataset"""
    # Implementation as per Step 1.2
    pass

def prepare_model3_data():
    """Prepare disease burden estimation dataset"""
    # Implementation as per Step 1.3
    pass

def prepare_model4_data():
    """Prepare multi-pollutant synergy dataset"""
    # Implementation as per Step 1.4
    pass

# ============================================================================
# SECTION 2: MODEL BUILDING FUNCTIONS
# ============================================================================

def build_model1():
    """Build and evaluate AQI forecasting models"""
    # Implementation as per Step 2.1
    # Returns: best_model, comparison_df, predictions_df
    pass

def build_model2():
    """Build and evaluate severe day prediction models"""
    # Implementation as per Step 2.2
    # Returns: best_model, comparison_df, predictions_df
    pass

def build_model3():
    """Build and evaluate disease burden models"""
    # Implementation as per Step 2.3
    # Returns: best_models_dict, comparison_df, predictions_df
    pass

def build_model4():
    """Build and evaluate multi-pollutant synergy models"""
    # Implementation as per Step 2.4
    # Returns: best_models_dict, comparison_df, predictions_df, synergy_analysis
    pass

# ============================================================================
# SECTION 3: EVALUATION AND VISUALIZATION
# ============================================================================

def create_comparison_table(results_dict, model_name):
    """Create formatted comparison table for all models"""
    df = pd.DataFrame(results_dict)
    df = df.sort_values('Test_R2', ascending=False)  # or appropriate metric
    df.to_csv(f'{model_name}_comparison.csv', index=False)
    print(f"\n{model_name} Comparison Table:")
    print(df.to_string())
    return df

def plot_predictions(actual, predicted, model_name, target_name=''):
    """Create actual vs predicted plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scatter plot
    ax1.scatter(actual, predicted, alpha=0.5)
    ax1.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title(f'{model_name} - Actual vs Predicted {target_name}')
    
    # Residual plot
    residuals = actual - predicted
    ax2.scatter(predicted, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Residuals')
    ax2.set_title(f'{model_name} - Residual Plot {target_name}')
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_{target_name}_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# SECTION 4: SAVING FUNCTIONS
# ============================================================================

def save_model(model, filepath):
    """Save model as pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def save_preprocessor(preprocessor, filepath):
    """Save preprocessing pipeline"""
    with open(filepath, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Preprocessor saved to {filepath}")

# ============================================================================
# SECTION 5: MAIN EXECUTION
# ============================================================================

def main(model_number):
    """
    Main execution function
    
    Args:
        model_number: 'all' or '1', '2', '3', '4'
    """
    
    print("="*80)
    print("AIR QUALITY & HEALTH PREDICTION MODELS")
    print("="*80)
    
    if model_number in ['all', '1']:
        print("\n" + "="*80)
        print("MODEL 1: AQI FORECASTING (7-30 DAYS AHEAD)")
        print("="*80)
        prepare_model1_data()
        best_model, comparison_df, predictions_df = build_model1()
        save_model(best_model, 'model1_best_aqi_forecast.pkl')
        
    if model_number in ['all', '2']:
        print("\n" + "="*80)
        print("MODEL 2: SEVERE DAY PREDICTION")
        print("="*80)
        prepare_model2_data()
        best_model, comparison_df, predictions_df = build_model2()
        save_model(best_model, 'model2_best_severe_day.pkl')
        
    if model_number in ['all', '3']:
        print("\n" + "="*80)
        print("MODEL 3: DISEASE BURDEN ESTIMATION")
        print("="*80)
        prepare_model3_data()
        best_models, comparison_df, predictions_df = build_model3()
        for target, model in best_models.items():
            save_model(model, f'model3_best_{target}.pkl')
        
    if model_number in ['all', '4']:
        print("\n" + "="*80)
        print("MODEL 4: MULTI-POLLUTANT SYNERGY")
        print("="*80)
        prepare_model4_data()
        best_models, comparison_df, predictions_df, synergy = build_model4()
        for target, model in best_models.items():
            save_model(model, f'model4_best_{target}.pkl')
    
    print("\n" + "="*80)
    print("ALL MODELS COMPLETED SUCCESSFULLY")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build air quality and health prediction models')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['all', '1', '2', '3', '4'],
                       help='Which model to build: all, 1, 2, 3, or 4')
    args = parser.parse_args()
    
    main(args.model)
```

---

## OUTPUT FILES

### For Each Model:

**Model 1 - AQI Forecasting**
- `model1_aqi_forecast.csv` - prepared dataset
- `model1_best_aqi_forecast.pkl` - best model
- `model1_preprocessor.pkl` - preprocessing pipeline
- `model1_features.pkl` - feature names
- `model1_comparison.csv` - all models comparison table
- `model1_predictions.csv` - predictions on test set
- `model1_predictions.png` - visualization

**Model 2 - Severe Day Prediction**
- `model2_severe_day.csv` - prepared dataset
- `model2_best_severe_day.pkl` - best model
- `model2_preprocessor.pkl` - preprocessing pipeline
- `model2_threshold.pkl` - optimal classification threshold
- `model2_comparison.csv` - all models comparison table
- `model2_predictions.csv` - predictions on test set
- `model2_confusion_matrix.png` - confusion matrix
- `model2_roc_curve.png` - ROC curve

**Model 3 - Disease Burden**
- `model3_disease_burden.csv` - prepared dataset
- `model3_best_cardiovascular.pkl` - best model for cardiovascular
- `model3_best_respiratory.pkl` - best model for respiratory
- `model3_best_all_diseases.pkl` - best model for all diseases
- `model3_preprocessor.pkl` - preprocessing pipeline
- `model3_comparison.csv` - all models comparison table (all targets)
- `model3_predictions.csv` - predictions on test set
- `model3_cardiovascular_predictions.png` - visualizations for each target
- `model3_respiratory_predictions.png`
- `model3_all_diseases_predictions.png`

**Model 4 - Pollutant Synergy**
- `model4_pollutant_synergy.csv` - prepared dataset
- `model4_best_cardiovascular.pkl` - best model for cardiovascular
- `model4_best_respiratory.pkl` - best model for respiratory
- `model4_best_combined_risk.pkl` - best model for combined risk
- `model4_preprocessor.pkl` - preprocessing pipeline
- `model4_feature_selector.pkl` - feature selection pipeline
- `model4_comparison.csv` - all models comparison table
- `model4_predictions.csv` - predictions on test set
- `model4_synergy_analysis.csv` - pollutant interaction analysis
- `model4_synergy_matrix.png` - interaction heatmap
- `model4_feature_importance.png` - feature importance plot

---

## SUCCESS CRITERIA

1. ✅ All 4 datasets prepared successfully with no errors
2. ✅ Each model tests at least 8-10 different algorithms
3. ✅ Comparison tables generated with all relevant metrics
4. ✅ Best model selected based on appropriate criteria for each task
5. ✅ All models saved as .pkl files
6. ✅ Predictions generated and saved as CSV
7. ✅ Visualizations created for model evaluation
8. ✅ Code runs end-to-end without manual intervention
9. ✅ Model 1: Test R² > 0.75 for AQI forecasting
10. ✅ Model 2: Recall > 0.85 for severe day detection
11. ✅ Model 3: Test R² > 0.60 for disease burden estimation
12. ✅ Model 4: Identifies at least 3 significant pollutant synergies

---

**END OF INSTRUCTIONS**
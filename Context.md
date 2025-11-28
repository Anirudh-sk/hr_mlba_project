# Comprehensive Instructions for Air Quality Health Impact Prediction System

## Project Overview
Build an end-to-end machine learning system to predict health impacts (mortality from pollution-related diseases) based on air quality data, comparing India with global trends, and supporting SDG goals 3, 11, and 13.

---

## PHASE 1: EXPLORATORY DATA ANALYSIS (EDA)

### Step 1.1: Initial Data Loading and Inspection
```
Load all three datasets:
- cause_of_deaths.csv
- city_day.csv  
- global_air_pollution_data.csv

For each dataset:
1. Display first 10 rows
2. Show dataset shape (rows, columns)
3. Display all column names with data types
4. Show memory usage
5. Display summary statistics (describe())
6. Check for duplicate rows
```

### Step 1.2: Missing Data Analysis
```
For each dataset:
1. Calculate percentage of missing values per column
2. Create a heatmap visualizing missing data patterns
3. Identify columns with >50% missing data
4. Document which columns need imputation vs deletion
5. For city_day.csv specifically:
   - Analyze missing AQI values by city
   - Analyze missing pollutant values (PM2.5, PM10, NO2, etc.)
   - Check if missing data has temporal patterns
```

### Step 1.3: Univariate Analysis
```
For cause_of_deaths.csv:
1. Plot distribution of deaths for each disease category
2. Analyze India-specific data separately:
   - Time series plots (1990-2019) for:
     * Cardiovascular Diseases
     * Lower Respiratory Infections
     * Chronic Respiratory Diseases
     * Neoplasms
3. Identify top 10 countries by total deaths
4. Calculate year-over-year growth rates for India

For city_day.csv:
1. Distribution plots for:
   - AQI values
   - PM2.5, PM10
   - NO, NO2, NOx
   - SO2, CO, O3
2. Create boxplots showing pollutant distributions by:
   - City
   - Year
   - AQI_Bucket category
3. Identify outliers in each pollutant
4. Plot AQI trends over time (2015-2020) for each city

For global_air_pollution_data.csv:
1. Distribution of AQI values globally
2. Count of cities by AQI category
3. Distribution by country
4. Compare India's cities vs global cities
```

### Step 1.4: Bivariate and Multivariate Analysis
```
For city_day.csv:
1. Correlation matrix for all pollutants
2. Heatmap showing correlations
3. Scatter plots:
   - PM2.5 vs AQI
   - PM10 vs AQI
   - NO2 vs AQI
   - PM2.5 vs PM10
4. Pairplot for top 5 pollutants
5. Check multicollinearity (VIF scores)

For cause_of_deaths.csv:
1. Correlation between different disease categories
2. Time series correlation analysis
3. For India specifically:
   - Correlation between respiratory diseases and cardiovascular diseases
   - Trend analysis for pollution-related diseases

Cross-dataset analysis:
1. Merge city_day with India deaths by Year
2. Calculate correlation between:
   - National average AQI vs total respiratory deaths
   - National average PM2.5 vs cardiovascular deaths
   - Pollutant levels vs disease-specific mortality
3. TODO: Revisit with time-aligned/lagged analysis (India 2015-2020) because AQI trends down while deaths rise; check both AQI and PM2.5/PM10 relationships with proper temporal handling
```

### Step 1.5: Temporal Analysis
```
1. Year-over-year trends:
   - AQI trends for each Indian city (2015-2019)
   - Death trends for pollution-related diseases (2015-2019)
   - Seasonal patterns in AQI (monthly aggregations)

2. Create time series decomposition plots showing:
   - Trend component
   - Seasonal component
   - Residual component

3. Analyze if AQI improvement/degradation correlates with death rate changes
4. TODO: Factor in seasonal AQI spikes (Nov–Jan) vs monsoon lows (Jul–Sep) and high “Very Poor” day cities (Ahmedabad/Delhi/Patna/Lucknow/Gurugram) when modeling temporal effects
```

### Step 1.6: Geographical Analysis
```
1. Create city-wise summary statistics:
   - Mean AQI per city
   - Days exceeding "Very Poor" AQI
   - Most problematic pollutant per city

2. Compare Indian cities from city_day.csv with Indian cities in global_air_pollution_data.csv:
   - Which cities appear in both?
   - How do AQI values compare?

3. Rank cities by pollution severity
4. TODO: For overlapping cities, reconcile AQI gaps between datasets (e.g., Delhi/Patna higher in global data); align definitions before downstream comparisons
```

---

## PHASE 2: COMPARATIVE ANALYSIS (India vs Global)

### Step 2.1: Data Preparation for Comparison
```
1. Extract India data from global_air_pollution_data.csv
2. Calculate global statistics:
   - Global mean AQI
   - Global median AQI
   - Distribution by AQI category worldwide
   - Top 20 most polluted cities globally

3. Calculate India-specific statistics from both datasets:
   - Mean AQI across all Indian cities
   - Median AQI
   - Percentage of cities in each AQI bucket
   - Top 20 most polluted Indian cities
```

### Step 2.2: Statistical Comparison
```
1. Compare India vs Global:
   - Mean AQI: India vs World average
   - Statistical significance testing (t-test)
   - Effect size calculation (Cohen's d)

2. Pollutant-specific comparison:
   - PM2.5: India average vs Global average
   - PM10: India average vs Global average
   - NO2: India average vs Global average
   - Ozone: India average vs Global average

3. Create comparison visualizations:
   - Side-by-side boxplots (India vs Rest of World)
   - Violin plots showing distribution differences
   - Bar charts for categorical AQI buckets
```

### Step 2.3: Mortality Analysis (India vs Global Context)
```
1. Calculate India's mortality rates for pollution-related diseases:
   - Deaths per 100,000 population (need to find/estimate India population 2015-2019)
   - Compare with other major countries if available

2. Calculate pollution burden metrics:
   - Estimated attributable deaths to air pollution
   - Disability-Adjusted Life Years (DALYs) if possible

3. Trend comparison:
   - Is India's AQI improving or worsening compared to global trends?
   - Are India's pollution-related deaths increasing faster than global average?
4. TODO: Requires population data per country/year to compute per-100k rates and burden metrics; supply or load population series before proceeding
```

### Step 2.4: Detailed Insights Generation
```
Create comprehensive report addressing:
1. How much worse is India compared to global average (percentage difference)
2. Which pollutants are India's biggest challenges
3. Which cities need most urgent intervention
4. Temporal trends: Is situation improving or deteriorating?
5. Best and worst performing Indian cities
6. Comparison with similarly developed countries
```

---

## PHASE 3: FEATURE ENGINEERING AND DATASET PREPARATION

### Step 3.1: Create Master Dataset
```
Objective: Build a unified dataset for ML modeling

Steps:
1. Filter cause_of_deaths.csv:
   - Keep only India data
   - Keep only years 2015-2019
   - Select columns:
     * Year
     * Cardiovascular Diseases
     * Lower Respiratory Infections
     * Chronic Respiratory Diseases
     * Neoplasms

2. Aggregate city_day.csv to yearly national level:
   - Group by Year
   - Calculate mean for each pollutant:
     * PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
   - Calculate AQI statistics:
     * Mean AQI
     * Max AQI
     * Min AQI
     * Std deviation of AQI
     * Percentage of days in each AQI_Bucket
   - Count missing values per pollutant

3. Merge datasets on Year to create master_dataset.csv
```

### Step 3.2: Feature Engineering
```
Create new features:

1. Pollution Composite Scores:
   - Particulate_Matter_Index = weighted average of PM2.5 and PM10
   - Gaseous_Pollutant_Index = weighted average of NO2, SO2, CO
   - Overall_Pollution_Score = combination of all pollutants

2. Temporal features:
   - Year as continuous variable
   - Years_since_baseline (2015 = 0)

3. Risk multipliers:
   - For each city in city_day.csv, calculate:
     * City_AQI / National_Average_AQI per year
     * City_PM2.5 / National_Average_PM2.5 per year

4. Lagged features (if using time series approach):
   - Previous year's AQI
   - Previous year's death counts
   - 2-year moving average of pollutants

5. Target variable creation:
   - Total_Pollution_Deaths = sum of (Cardiovascular + Respiratory + relevant Neoplasms)
   - Also create separate targets for each disease category
   - Create mortality rate per 100,000 (if population data available)

6. Create city-level prediction dataset:
   - For each city-year combination in city_day.csv
   - Calculate risk score = National_Deaths × (City_AQI / National_AQI)^β
   - This will be used for city-level predictions
```

### Step 3.3: Data Preprocessing
```
1. Handle missing values:
   - For pollutants: Use median imputation within same year
   - For AQI: Use forward fill then backward fill
   - Document all imputation decisions

2. Handle outliers:
   - Identify outliers using IQR method
   - Decide: keep, cap, or remove (document reasoning)
   - Create outlier flags as binary features

3. Categorical encoding:
   - AQI_Bucket: Use ordinal encoding (Good=0, Moderate=1, Poor=2, Very Poor=3, Severe=4)
   - City: Use one-hot encoding or leave-one-out encoding

4. Feature scaling:
   - Create scaled versions using:
     * StandardScaler (for linear models, neural networks)
     * MinMaxScaler (for tree-based models comparison)
   - Keep both scaled and unscaled versions

5. Feature selection:
   - Calculate feature importance using:
     * Correlation with target
     * Mutual information scores
     * Random Forest feature importance
   - Remove highly correlated features (correlation > 0.95)
   - Remove low-variance features
```

### Step 3.4: Train-Test Split Strategy
```
IMPORTANT: Use temporal split, not random split

1. Training data: 2015-2018 (4 years)
2. Test data: 2019 (1 year)

Rationale: Predicting future based on past (realistic scenario)

For city-level predictions:
1. Use same temporal split
2. Additionally create validation set from 2018 data
3. Split: Train (2015-2017), Validation (2018), Test (2019)

Save datasets:
- X_train, X_test, y_train, y_test
- X_train_scaled, X_test_scaled
- city_X_train, city_X_test, city_y_train, city_y_test
```

---

## PHASE 4: MACHINE LEARNING MODEL DEVELOPMENT

### Step 4.1: Baseline Models
```
Build simple baselines first:
1. Mean baseline: Predict mean of training deaths
2. Linear trend: Simple linear regression with Year only
3. Last-year baseline: Predict same as previous year

Calculate baseline metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)
```

### Step 4.2: Regression Models
```
Train the following models on national-level data:

1. Linear Regression
   - Standard OLS
   - Ridge Regression (L2 regularization) - try alphas: [0.1, 1, 10, 100]
   - Lasso Regression (L1 regularization) - try alphas: [0.1, 1, 10, 100]
   - ElasticNet - try alpha and l1_ratio combinations

2. Tree-based Models
   - Decision Tree Regressor
     * Try max_depth: [3, 5, 7, 10, None]
     * Try min_samples_split: [2, 5, 10]
   - Random Forest Regressor
     * n_estimators: [50, 100, 200, 500]
     * max_depth: [5, 10, 15, None]
     * Use GridSearchCV or RandomizedSearchCV
   - Gradient Boosting Regressor
     * n_estimators: [100, 200, 500]
     * learning_rate: [0.01, 0.05, 0.1]
     * max_depth: [3, 5, 7]
   - XGBoost Regressor
     * Similar hyperparameter tuning as GBM
   - LightGBM Regressor
     * Optimize num_leaves, learning_rate, n_estimators

3. Support Vector Regression (SVR)
   - Try kernels: linear, rbf, poly
   - Optimize C and epsilon parameters

4. K-Nearest Neighbors Regressor
   - Try n_neighbors: [3, 5, 7, 10, 15]
   - Try different distance metrics

For each model:
- Use cross-validation (TimeSeriesSplit with 3-5 splits)
- Record training time
- Calculate metrics on both train and test sets
- Save predictions
- Plot actual vs predicted
- Analyze residuals
```

### Step 4.3: Model Evaluation and Comparison
```
Create comprehensive comparison:

1. Metrics table:
   | Model | Train RMSE | Test RMSE | Train R² | Test R² | MAE | MAPE | Training Time |
   
2. Visualizations:
   - Bar chart comparing test R² scores
   - Bar chart comparing test RMSE
   - Scatter plots: Actual vs Predicted for each model
   - Residual plots for top 3 models

3. Statistical tests:
   - Paired t-test comparing model predictions
   - Friedman test for overall difference

4. Feature importance analysis:
   - For tree-based models: plot feature importances
   - For linear models: plot coefficients
   - SHAP values for best model
```

### Step 4.4: Ensemble Methods
```
Create ensemble models:

1. Voting Regressor:
   - Combine top 3-5 performing models
   - Try both averaging and weighted voting
   - Optimize weights using validation set

2. Stacking:
   - Base models: Top 5 individual models
   - Meta-learner: Try Linear Regression, Ridge, or Gradient Boosting
   - Use cross-validation predictions for meta-features

3. Blending:
   - Train models on train set
   - Get predictions on validation set
   - Train meta-model on validation predictions
   - Test on test set

4. Custom ensemble:
   - Time-weighted ensemble (recent years get more weight)
   - Pollutant-specific ensemble (different models for different pollutants)

Evaluate each ensemble:
- Compare against best individual model
- Check if ensemble reduces variance
- Analyze bias-variance tradeoff
```

### Step 4.5: Advanced Techniques
```
1. AutoML (optional):
   - Use TPOT or Auto-sklearn
   - Let it search for best pipeline
   - Compare with manual models

2. Bayesian Optimization:
   - For top 2-3 models
   - Optimize hyperparameters more efficiently
   - Use Optuna or Hyperopt

3. Feature engineering v2:
   - Based on model insights
   - Create polynomial features for important variables
   - Create interaction terms
   - Retrain top models

4. Time series specific models (if treating as time series):
   - ARIMA for each pollutant
   - SARIMAX with exogenous variables
   - Prophet (Facebook)
   - Compare with ML models
```

---

## PHASE 5: DEEP LEARNING IMPLEMENTATION

### Step 5.1: Data Preparation for Neural Networks
```
1. Ensure proper scaling (StandardScaler already applied)
2. Reshape data if needed for specific architectures
3. Create additional features:
   - Embeddings for categorical variables (City)
   - Sequence data if using LSTM/GRU

4. Split data:
   - Train: 2015-2017
   - Validation: 2018
   - Test: 2019
```

### Step 5.2: Feedforward Neural Network (MLP)
```
Build from scratch using TensorFlow/Keras:

Architecture 1 - Simple:
- Input layer (number of features)
- Hidden layer 1: 64 neurons, ReLU activation, Dropout(0.2)
- Hidden layer 2: 32 neurons, ReLU activation, Dropout(0.2)
- Output layer: 1 neuron (regression)

Architecture 2 - Deeper:
- Input layer
- Hidden layer 1: 128 neurons, ReLU, BatchNorm, Dropout(0.3)
- Hidden layer 2: 64 neurons, ReLU, BatchNorm, Dropout(0.3)
- Hidden layer 3: 32 neurons, ReLU, BatchNorm, Dropout(0.2)
- Hidden layer 4: 16 neurons, ReLU, Dropout(0.2)
- Output layer: 1 neuron

Architecture 3 - Wide:
- Input layer
- Hidden layer 1: 256 neurons, ReLU, Dropout(0.3)
- Hidden layer 2: 128 neurons, ReLU, Dropout(0.2)
- Output layer: 1 neuron

Training configuration:
- Loss function: MSE (Mean Squared Error)
- Optimizer: Adam with learning_rate=0.001
- Metrics: MAE, MAPE
- Epochs: 100-500 with early stopping
- Batch size: Try [8, 16, 32]
- Early stopping: patience=20, monitor='val_loss'
- Model checkpoint: save best model

Experiments to run:
1. Different activation functions (ReLU, LeakyReLU, ELU, SELU)
2. Different optimizers (Adam, SGD with momentum, RMSprop, AdamW)
3. Different learning rates: [0.0001, 0.001, 0.01]
4. Different dropout rates: [0.1, 0.2, 0.3, 0.5]
5. Batch normalization vs Layer normalization vs no normalization
6. L1/L2 regularization
```

### Step 5.3: Recurrent Neural Networks (Time Series)
```
Since we have temporal data, try RNN architectures:

Architecture 1 - LSTM:
- Reshape input to (samples, timesteps, features)
- LSTM layer 1: 50 units, return_sequences=True
- Dropout(0.2)
- LSTM layer 2: 50 units
- Dropout(0.2)
- Dense layer: 25 units, ReLU
- Output layer: 1 unit

Architecture 2 - GRU:
- GRU layer 1: 64 units, return_sequences=True
- Dropout(0.2)
- GRU layer 2: 32 units
- Dense layer: 16 units, ReLU
- Output layer: 1 unit

Architecture 3 - Bidirectional LSTM:
- Bidirectional LSTM: 50 units, return_sequences=True
- Dropout(0.3)
- Bidirectional LSTM: 50 units
- Dense layer: 25 units
- Output layer: 1 unit

Configure similarly to MLP above
```

### Step 5.4: Advanced Deep Learning Architectures
```
1. Residual Networks (ResNet-style):
   - Add skip connections
   - Deeper networks without vanishing gradients

2. Attention Mechanism:
   - Add attention layer to focus on important features/timesteps
   - Multi-head attention

3. Encoder-Decoder architecture:
   - Encode input features
   - Decode to prediction
   - Useful for complex relationships

4. Hybrid models:
   - Combine CNN (for feature extraction) + LSTM (for temporal)
   - Combine MLP for static features + LSTM for temporal
```

### Step 5.5: Neural Network Training Best Practices
```
1. Learning rate scheduling:
   - ReduceLROnPlateau
   - Cosine annealing
   - Warm-up and decay

2. Data augmentation (if applicable):
   - Add small noise to inputs
   - Bootstrap sampling

3. Regularization techniques:
   - L1/L2 weight regularization
   - Dropout variations (SpatialDropout, AlphaDropout)
   - Batch normalization

4. Training monitoring:
   - Plot training/validation loss curves
   - Plot training/validation metric curves
   - Check for overfitting/underfitting
   - Use TensorBoard for visualization

5. Hyperparameter tuning:
   - Use Keras Tuner or Optuna
   - Search space:
     * Number of layers: [2, 3, 4, 5]
     * Units per layer: [16, 32, 64, 128, 256]
     * Dropout rates: [0.1, 0.2, 0.3, 0.4]
     * Learning rates: [0.0001, 0.0005, 0.001, 0.005]
```

### Step 5.6: Neural Network Evaluation
```
For each neural network:
1. Calculate all metrics (RMSE, MAE, R², MAPE)
2. Plot actual vs predicted
3. Plot residuals
4. Analyze prediction errors by:
   - Year
   - Pollutant level
   - Disease type

5. Compare with traditional ML models
6. Ensemble neural networks:
   - Average predictions from multiple architectures
   - Weighted ensemble based on validation performance

7. Interpretation:
   - Feature importance using permutation importance
   - SHAP values for deep learning
   - Integrated gradients
```

---

## PHASE 6: FINAL MODEL SELECTION AND CITY-LEVEL PREDICTIONS

### Step 6.1: Comprehensive Model Comparison
```
Create final comparison table including:
- All regression models
- All ensemble models  
- All neural network architectures

Metrics to compare:
- Test RMSE
- Test MAE
- Test R²
- Test MAPE
- Cross-validation score (mean ± std)
- Training time
- Inference time
- Model complexity (number of parameters)
- Interpretability score (subjective: 1-5)

Select top 3 models based on:
1. Best test performance
2. Lowest overfitting (train-test gap)
3. Good balance of performance and interpretability
```

### Step 6.2: City-Level Risk Prediction
```
Using the best model:

1. National-level predictions (already done)

2. City-level risk scoring:
   For each city in city_day.csv:
   - Calculate city's average AQI and pollutants (2015-2019)
   - Calculate risk multiplier = (City_AQI / National_Average_AQI)
   - Estimate city deaths = National_Deaths × Risk_Multiplier^β
     Where β is optimized parameter (default: 1.0-1.5)

3. Create city ranking:
   - Rank cities by estimated pollution-attributable deaths
   - Rank by deaths per capita (if population available)
   - Identify highest risk cities

4. Scenario analysis:
   - "What if City X reduces PM2.5 by 30%?"
   - Estimate lives saved
   - Calculate for multiple scenarios
```

### Step 6.3: Model Deployment Preparation
```
1. Save best model:
   - Pickle file for scikit-learn models
   - SavedModel format for neural networks
   - Save preprocessing pipeline (scaler, encoder)

2. Create prediction function:
   def predict_health_impact(pollutant_data, city=None):
       # Preprocess input
       # Make prediction
       # Return result with confidence interval

3. Model documentation:
   - Model card: architecture, performance, limitations
   - Feature requirements
   - Expected input format
   - Output interpretation guide

4. Create simple API or notebook interface:
   - Input: City name + pollutant levels
   - Output: Predicted health impact + recommendations
```

---

## PHASE 7: VISUALIZATION AND REPORTING

### Step 7.1: Create Comprehensive Dashboards
```
1. EDA Dashboard:
   - Summary statistics
   - Distribution plots
   - Correlation matrices
   - Time series trends

2. Model Performance Dashboard:
   - Model comparison charts
   - Actual vs Predicted plots
   - Residual analysis
   - Feature importance

3. India vs Global Dashboard:
   - Comparative statistics
   - Map visualizations (if possible)
   - Trend comparisons

4. City Risk Dashboard:
   - City rankings
   - Risk scores by city
   - Scenario analysis results
   - Actionable recommendations
```

### Step 7.2: Final Report Generation
```
Create comprehensive report including:

1. Executive Summary
   - Key findings
   - Best model performance
   - Top insights

2. Data Analysis Section
   - EDA findings
   - India vs global comparison
   - Data quality assessment

3. Methodology
   - Feature engineering approach
   - Models tested
   - Evaluation metrics

4. Results
   - Model performance comparison
   - Best model details
   - City-level predictions

5. Insights and Recommendations
   - Cities needing urgent intervention
   - Estimated lives that could be saved
   - Policy recommendations
   - SDG alignment

6. Limitations and Future Work
   - Data limitations
   - Model limitations
   - Suggestions for improvement

7. Appendix
   - Detailed model configurations
   - Complete results tables
   - Code snippets
```

---

## DELIVERABLES CHECKLIST

```
[ ] Complete EDA notebook with all visualizations
[ ] India vs Global comparison report
[ ] Clean master dataset (master_dataset.csv)
[ ] City-level dataset for predictions
[ ] Trained models (saved files):
    [ ] Best regression model
    [ ] Best ensemble model
    [ ] Best neural network
[ ] Model performance comparison table
[ ] City risk ranking file
[ ] Prediction pipeline (functions/API)
[ ] Comprehensive final report
[ ] Visualization dashboard/notebook
[ ] README with instructions
[ ] Requirements.txt with all dependencies
```

---

## TECHNICAL REQUIREMENTS

```python
# Required libraries:
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 1.0.0
xgboost >= 1.5.0
lightgbm >= 3.3.0
tensorflow >= 2.8.0  # or pytorch >= 1.10.0
keras-tuner >= 1.1.0
shap >= 0.40.0
optuna >= 2.10.0  # for hyperparameter optimization
plotly >= 5.0.0  # for interactive plots
scipy >= 1.7.0
statsmodels >= 0.13.0
```

---

## SUCCESS CRITERIA

The project is successful if:
1. ✅ Complete EDA reveals clear patterns and insights
2. ✅ India vs Global comparison shows quantified differences
3. ✅ At least 5 different ML models trained and compared
4. ✅ Best model achieves R² > 0.70 on test set
5. ✅ Neural network matches or exceeds traditional ML performance
6. ✅ City-level predictions generated for all 26 cities
7. ✅ Clear, actionable recommendations provided
8. ✅ Complete documentation and reproducible code
9. ✅ Alignment with SDG goals clearly demonstrated
10. ✅ Model ready for deployment/further development

---

**END OF INSTRUCTIONS**

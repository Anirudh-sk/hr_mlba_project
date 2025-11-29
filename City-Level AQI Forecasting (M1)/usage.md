# Model 1 Usage

Best model: `model1_best_Lasso_R2-0.523.pkl` (saved pipeline with preprocessing + regressor).

## Predicting
1) Load the pipeline:
```python
import joblib
from pathlib import Path
from model1_aqi_forecast import add_extra_features

model_path = Path(__file__).parent / "model1_best_Lasso_R2-0.523.pkl"
model = joblib.load(model_path)
```
2) Prepare input features (use the same columns as `model1_aqi_forecast.csv`):
```python
import pandas as pd

# Start from the prepared dataset or your new data with the same columns
X = pd.read_csv(Path(__file__).parent / "model1_aqi_forecast.csv")
# Drop target if present
X = X.drop(columns=["AQI_target"], errors="ignore")
# Add derived features to match training
X = add_extra_features(X)
```
3) Get predictions (same order as rows in `X`):
```python
predictions = model.predict(X)
```

## Expected inputs
- Columns matching the training set: City, Date, lag/rolling features, temporal flags, EMA, and the extra features added by `add_extra_features` (squared/log AQI lag, extreme indicators, interactions).
- No need to encode or scale manually; the pipeline handles preprocessing.

## Outputs
- `model.predict` returns an array of forecasted AQI values 7 days ahead for each input row.

Problems with other models:
1. Random Forest - Severe Overfitting

Train R² = 0.936 vs Test R² = 0.489
Huge gap → Model memorized training data
Need more aggressive regularization

2. Gradient Boosting - Worse than Lasso

Test R² = 0.418 (worse than Lasso's 0.523)
Also overfitting (Train R² = 0.846 vs Test R² = 0.418)

3. GBR_Quantile - Failed completely

Test R² = -0.019 (negative! Worse than predicting the mean)
This model is not working at all

 LASSO WINS and has approximately 69% accuracy (basically the predicted value of the AQI will be (±28) from actual value aprox)